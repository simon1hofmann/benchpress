# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for MQT-to-Qiskit conversion used by validation and metrics."""

from types import SimpleNamespace

import pytest
from mqt.core.mlir import CompilerTarget, PayloadEncoding, QCOProgram, QCProgram
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, efficient_su2, quantum_volume
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap, Target

from benchpress.mqt_gym.circuits import mqt_bv_all_ones, mqt_QV, to_qc_program
from benchpress.mqt_gym.utils.io import (
    load_qasm_as_qc_program,
    make_compiler_target,
    mqt_compile,
    mqt_hamiltonian_circuit,
    mqt_to_qiskit_circuit,
    prepare_mqt_compile,
    program_num_qubits,
    program_op_counts,
    program_uses_classical_control,
)
from benchpress.mqt_gym.utils.mqt_backend_utils import (
    get_mqt_bench_backend,
)

_CX_MEASURED = (
    "OPENQASM 2.0;\n"
    'include "qelib1.inc";\n'
    "qreg q[2];\n"
    "creg c[2];\n"
    "h q[0];\n"
    "cx q[0],q[1];\n"
    "measure q[0] -> c[0];\n"
    "measure q[1] -> c[1];\n"
)

_BIDIRECTIONAL_CX_MEASURED = _CX_MEASURED.replace(
    "cx q[0],q[1];\n", "cx q[0],q[1];\ncx q[1],q[0];\n"
)


def test_mqt_report_records_adapter_and_runtime_options(monkeypatch):
    import benchpress.mqt_gym.conftest as hooks
    from benchpress.config import Configuration
    from benchpress.mqt_gym.conftest import pytest_benchmark_update_json

    monkeypatch.setitem(
        Configuration.options,
        "mqt",
        {"normalize_global_phases": True, "native_gates": ["sx", "rz", "cz"]},
    )
    monkeypatch.setattr(hooks.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(
        hooks.core_mlir,
        "CompilationOptions",
        lambda: SimpleNamespace(
            seed=None,
            mapping=SimpleNamespace(
                trials=None, iterations=2, lookahead=17, search_memory_limit=1024
            ),
        ),
    )
    report = {}

    pytest_benchmark_update_json(None, [], report)

    assert "mqt.core" in report["mqt_info"]
    assert len(report["mqt_build"]["mlir_extension_sha256"]) == 64
    assert report["mqt_context"] == {
        "construction_and_binding": "qiskit_construction_mqt_binding",
        "device_circsu2_parameters": "symbolic",
        "normalize_global_phases": True,
        "native_gates_override": ["sx", "rz", "cz"],
        "timeout_scope": "whole_test_preflight",
        "compilation_timing": "copy_import_lower_compile_to_qc",
        "logical_cpus": 6,
        "compiler_defaults": {
            "seed": None,
            "trials": None,
            "iterations": 2,
            "lookahead": 17,
            "search_memory_limit": 1024,
        },
    }


@pytest.mark.parametrize("backend_name", ["fake_torino", "FakeTorino"])
def test_mqt_fake_backend_discovery_accepts_supported_name_formats(backend_name):
    backend = get_mqt_bench_backend(backend_name)

    assert type(backend).__name__ == "FakeTorino"
    assert backend.num_qubits == 133
    assert backend.two_q_gate_type == "cz"
    assert len(backend.coupling_map.get_edges()) == 300


def test_mqt_qiskit_builder_uses_supported_typed_mlir_bridge():
    circuit = QuantumCircuit(2)
    circuit.global_phase = 0.375
    circuit.h(0)
    circuit.cx(0, 1)

    program = to_qc_program(circuit)
    converted = program.to_qiskit()

    assert converted.count_ops() == {"h": 1, "cx": 1}
    assert converted.global_phase == pytest.approx(0.375)


@pytest.mark.parametrize("frontend", ["qiskit", "qasm"])
def test_mqt_target_compilation_lowers_reusable_gates(frontend):
    """Preserve gate reuse on import and lower calls during target compilation."""
    from benchpress.utilities.backends import FlexibleBackend

    pair = QuantumCircuit(2, name="pair")
    pair.h(0)
    pair.cx(0, 1)
    gate = pair.to_gate()
    source = QuantumCircuit(2)
    source.append(gate, [0, 1])
    source.rz(0.25, 0)
    source.append(gate, [0, 1])
    if frontend == "qiskit":
        program = to_qc_program(source)
    else:
        program = load_qasm_as_qc_program(
            qasm_str=(
                'OPENQASM 2.0; include "qelib1.inc"; '
                "gate pair a,b { h a; cx a,b; } "
                "qreg q[2]; pair q[0],q[1]; rz(0.25) q[0]; pair q[0],q[1];"
            )
        )

    assert "qc.call" in program.ir
    assert Operator(program.to_qiskit()).equiv(Operator(source))
    setup = prepare_mqt_compile(program, FlexibleBackend(2, layout="all-to-all"))
    result = setup.compile()

    assert "qco.call" not in result.ir
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert Operator(converted).equiv(Operator(source))


@pytest.mark.parametrize("via_qco", [False, True])
def test_mqt_to_qiskit_circuit_uses_native_compatible_program(via_qco):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    program = QCProgram.from_qiskit(circuit)
    if via_qco:
        program = program.to_qco()
    original_ir = program.ir
    converted = mqt_to_qiskit_circuit(program)
    assert converted.count_ops() == circuit.count_ops()
    assert program.is_valid and program.ir == original_ir


def test_mqt_quantum_volume_preserves_dense_unitaries():
    source = quantum_volume(4, 3, seed=12345)

    program = mqt_QV(4, 3, seed=12345)
    converted = program.to_qiskit()

    assert program.ir.count("qc.unitary") == 6
    assert converted.count_ops() == {"unitary": 6}
    assert Operator(converted).equiv(Operator(source))


def test_mqt_qiskit_round_trip_preserves_parameter_vector_provenance():
    parameters = ParameterVector("theta", 4)
    circuit = QuantumCircuit(2)
    circuit.ry(parameters[2], 0)
    circuit.rz(parameters[0], 1)

    program = QCProgram.from_qiskit(circuit)
    converted = program.to_qiskit()

    assert [
        (parameter.vector.name, len(parameter.vector), parameter.index)
        for parameter in converted.parameters
    ] == [("theta", 4, 0), ("theta", 4, 2)]
    assert set(converted.parameters) == set(circuit.parameters)
    values = {parameters[0]: 0.25, parameters[2]: 0.5}
    assert Operator(converted.assign_parameters(values)).equiv(
        Operator(circuit.assign_parameters(values))
    )
    bound = program.copy()
    bound.bind_parameters(
        {parameter.name: value for parameter, value in values.items()}
    )
    assert not bound.parameters
    assert set(program.parameters) == {parameter.name for parameter in values}
    assert Operator(bound.to_qiskit()).equiv(
        Operator(circuit.assign_parameters(values))
    )


def test_mqt_synthesizes_symbolic_single_qubit_gates_for_target():
    from benchpress.utilities.backends import FlexibleBackend

    circuit = efficient_su2(2, reps=1, entanglement="circular")
    setup = prepare_mqt_compile(circuit, FlexibleBackend(3, layout="linear"))

    result = setup.compile().to_qc()

    assert "mqt.input_name" in result.ir
    assert "qc.ry" not in result.ir
    assert "qc.rz" in result.ir
    assert "qc.sx" in result.ir

    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert set(converted.parameters) == set(circuit.parameters)
    assert set(converted.count_ops()) <= {"rz", "sx", "x", "cz", "id"}


def test_mqt_to_qiskit_circuit_exports_structured_control_flow():
    circuit = QuantumCircuit(2, 1)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(1)

    assert program_uses_classical_control(circuit)
    converted = mqt_to_qiskit_circuit(QCProgram.from_qiskit(circuit))

    assert [instruction.operation.name for instruction in converted.data] == [
        "measure",
        "if_else",
    ]
    assert converted.data[1].operation.blocks[0].count_ops() == {"x": 1}


def test_mqt_to_qiskit_circuit_from_qc_program():
    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    qc = mqt_to_qiskit_circuit(prog)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 2
    assert qc.count_ops().get("measure") == 2
    assert "cx" in qc.count_ops() or "cz" in qc.count_ops() or len(qc.data) >= 1


def test_mqt_to_qiskit_circuit_rejects_unknown_type():
    with pytest.raises(TypeError, match="QCProgram|QCOProgram"):
        mqt_to_qiskit_circuit(object())


@pytest.mark.parametrize("target_aware", [False, True])
@pytest.mark.parametrize("via_qco", [False, True])
def test_mqt_to_qiskit_circuit_propagates_native_failures(
    monkeypatch, target_aware, via_qco
):
    program = QCProgram.from_qiskit(QuantumCircuit(0))
    if via_qco:
        program = program.to_qco()
    target = make_compiler_target(2, None) if target_aware else None
    native_error = RuntimeError("measurement destination must follow the measurement")

    def fail_native(*args, **kwargs):
        raise native_error

    def fail_if_rewritten(*args, **kwargs):
        pytest.fail("native export failures must not trigger OpenQASM rewriting")

    monkeypatch.setattr(type(program), "to_qiskit", fail_native)
    monkeypatch.setattr(QCProgram, "to_openqasm3", fail_if_rewritten)
    conversion = "target-aware Qiskit" if target_aware else "Qiskit"
    with pytest.raises(
        RuntimeError, match=f"MQT {conversion} conversion failed"
    ) as error:
        mqt_to_qiskit_circuit(program, target=target)
    assert error.value.__cause__ is native_error


def test_qasm3_typed_frontend_preserves_loop_and_global_phase():
    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 3.0;\n"
            'include "stdgates.inc";\n'
            "qubit[2] q;\n"
            "gphase(pi / 4);\n"
            "for int i in [0:1] {\n"
            "  h q[i];\n"
            "}\n"
        )
    )
    assert "scf.for" in prog.ir
    assert program_uses_classical_control(prog)
    assert "qc.gphase" in prog.ir


def test_mqt_compile_preserves_unmeasured_circuit_without_mutating_it():
    """Target compilation keeps unitary work without observations."""
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'
        )
    )
    assert "measure" not in prog.ir.lower()
    backend = FlexibleBackend(5, layout="linear")
    setup = prepare_mqt_compile(prog, backend)
    assert setup.program is prog
    assert "measure" not in setup.program.ir.lower()

    result = setup.compile()
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert "qco." in result.ir
    assert len(converted.data) > 0
    assert converted.count_ops().get("measure", 0) == 0
    assert converted.num_qubits == backend.num_qubits
    assert len(converted.qregs) == 1
    assert converted.qregs[0].name == "q"
    assert converted.layout is not None
    assert len(converted.layout.final_index_layout()) == 2
    expected = QuantumCircuit(backend.num_qubits)
    expected.h(0)
    expected.cx(0, 1)
    assert Operator.from_circuit(converted).equiv(Operator(expected))


@pytest.mark.parametrize("via_qco", [False, True])
def test_prepared_compile_reuses_validation_and_preserves_input(monkeypatch, via_qco):
    import benchpress.mqt_gym.utils.io as mqt_io
    from benchpress.utilities.backends import FlexibleBackend

    source = QuantumCircuit(2)
    source.h(0)
    source.cx(0, 1)
    source.rz(0.25, 1)
    program = QCProgram.from_qiskit(source)
    if via_qco:
        program = program.to_qco()
    original_ir = program.ir
    native_compile = QCOProgram.compile_for_target
    setup = prepare_mqt_compile(program, FlexibleBackend(2, layout="all-to-all"))
    assert program.ir == original_ir

    lower = mqt_io._to_qco
    lowered = []

    def lower_fresh_input(candidate, *, copy):
        assert candidate is program and copy
        result = lower(candidate, copy=copy)
        lowered.append(result)
        return result

    def compile_with_payload(candidate, environment):
        assert environment is setup.environment
        assert environment.target.num_sites == setup.target.num_sites
        payload = environment.payload_specification
        assert payload.format.format_id == "openqasm"
        assert payload.format.version == "3.0.0"
        assert payload.format.encoding == PayloadEncoding.TEXT
        assert {capability.capability_id for capability in payload.capabilities} == {
            "forward-branching",
            "counted-iteration",
            "conditional-loop",
            "multiway-branching",
        }
        assert not payload.optional_capabilities_known
        return native_compile(candidate, environment)

    def fail_if_reprepared(*args, **kwargs):
        pytest.fail("prepared compilation must not repeat validation or target setup")

    monkeypatch.setattr(mqt_io, "_compiler_target", fail_if_reprepared)
    monkeypatch.setattr(mqt_io, "TargetEnvironment", fail_if_reprepared)
    monkeypatch.setattr(mqt_io, "PayloadSpecification", fail_if_reprepared)
    monkeypatch.setattr(mqt_io, "_to_qco", lower_fresh_input)
    monkeypatch.setattr(QCOProgram, "compile_for_target", compile_with_payload)
    for _ in range(2):
        result = setup.compile().to_qc()
        assert isinstance(result, QCProgram)
        converted = mqt_to_qiskit_circuit(result, target=setup.target)
        assert Operator(converted).equiv(Operator(source))
        assert program.ir == original_ir
    assert len(lowered) == 2
    assert lowered[0] is not lowered[1]
    assert all(candidate is not program for candidate in lowered)


def test_hamiltonian_synthesis_is_repeated_inside_compilation(monkeypatch):
    from benchpress.utilities.backends import FlexibleBackend

    definitions = []
    define = PauliEvolutionGate._define

    def record_definition(gate):
        definitions.append(gate)
        define(gate)

    monkeypatch.setattr(PauliEvolutionGate, "_define", record_definition)
    source = mqt_hamiltonian_circuit(SparsePauliOp.from_list([("ZZ", 0.5)]))
    setup = prepare_mqt_compile(source, FlexibleBackend(2, layout="all-to-all"))
    assert source.count_ops() == {"PauliEvolution": 1}
    assert program_num_qubits(source) == 2
    assert not program_uses_classical_control(source)
    assert not definitions
    expected = QuantumCircuit(2)
    expected.rzz(1.0, 0, 1)
    for round_number in (1, 2):
        result = setup.compile().to_qc()
        assert len(definitions) == round_number
        assert definitions[-1] is not source.data[0].operation
        assert isinstance(result, QCProgram)
        assert Operator(result.to_qiskit(target=setup.target)).equiv(Operator(expected))


@pytest.mark.parametrize("via_qco", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_mqt_compile_rejects_consumed_program(via_qco, prepared):
    from benchpress.utilities.backends import FlexibleBackend

    program = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    if via_qco:
        program = program.to_qco()
    backend = FlexibleBackend(2, layout="all-to-all")
    setup = prepare_mqt_compile(program, backend)
    if via_qco:
        program.to_qc()
    else:
        program.to_qco()
    assert not program.is_valid

    with pytest.raises(ValueError, match="consumed MQT program"):
        setup.compile() if prepared else mqt_compile(program, backend)


@pytest.mark.parametrize("via_qco", [False, True])
def test_mqt_compile_accepts_empty_program(monkeypatch, via_qco):
    """The native compile path accepts an empty typed program."""
    program = QCProgram.from_qiskit(QuantumCircuit(0))
    if via_qco:
        program = program.to_qco()

    target = make_compiler_target(1, None, basis_gates=["sx", "x", "rz"])
    result = mqt_compile(program, target)

    assert isinstance(result, QCOProgram)
    assert result.is_valid
    assert result.to_qc().to_qiskit().num_qubits == 0


@pytest.mark.parametrize("via_qco", [False, True])
def test_target_aware_export_rejects_unmapped_dynamic_qubits(via_qco):
    prog = load_qasm_as_qc_program(
        qasm_str=('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\n')
    )

    if via_qco:
        prog = prog.to_qco()

    with pytest.raises(RuntimeError, match="statically mapped qubits"):
        mqt_to_qiskit_circuit(
            prog,
            target=CompilerTarget(
                2,
                connectivity=CompilerTarget.Connectivity.all_to_all(),
                native_operations=CompilerTarget.NativeOperations.unrestricted(),
            ),
        )


def test_load_and_preparation_do_not_add_measurements():
    """Loading and untimed target setup preserve the input circuit exactly."""
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\n')
    )
    assert "measure" not in prog.ir.lower()
    setup = prepare_mqt_compile(prog, FlexibleBackend(3, layout="linear"))
    converted = mqt_to_qiskit_circuit(setup.program)
    assert converted.num_qubits == 3
    assert converted.count_ops().get("measure", 0) == 0


def test_mqt_compile_preserves_partially_observed_quantum_work():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "creg c[1];\n"
            "measure q[0] -> c[0];\n"
            "h q[1];\n"
        )
    )
    setup = prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))
    result = setup.compile()
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert converted.count_ops().get("measure") == 1
    assert sum(converted.count_ops().values()) > 1


def test_mqt_compile_preserves_gate_after_measurement_without_remeasuring():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "creg c[2];\n"
            "measure q[0] -> c[0];\n"
            "h q[0];\n"
            "measure q[1] -> c[1];\n"
        )
    )
    setup = prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))
    result = setup.compile()
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert converted.count_ops().get("measure") == 2
    assert sum(converted.count_ops().values()) > 2


def test_target_export_preserves_delayed_measurement_stores():
    from benchpress.utilities.backends import FlexibleBackend

    prog = mqt_bv_all_ones(4)
    setup = prepare_mqt_compile(prog, FlexibleBackend(5, layout="linear"))
    result = setup.compile()

    converted = mqt_to_qiskit_circuit(result, target=setup.target)

    assert converted.num_qubits == setup.target.num_sites
    assert converted.num_clbits == 3
    assert converted.count_ops().get("measure") == 3
    assert sorted(
        converted.find_bit(instruction.clbits[0]).index
        for instruction in converted.data
        if instruction.operation.name == "measure"
    ) == [0, 1, 2]
    assert [(register.name, register.size) for register in converted.qregs] == [
        ("q", setup.target.num_sites)
    ]
    assert converted.layout is not None
    assert len(converted.layout.final_index_layout()) == 4


@pytest.mark.parametrize("via_qco", [False, True])
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("for int i in [0:1] { x q[i]; } c = measure q;", "11"),
        ("c = measure q; while (c == 0) { x q[0]; c = measure q; }", "01"),
        (
            "x q[0]; c = measure q; if (bool(c[0]) && !bool(c[1])) { x q[1]; } c = measure q;",
            "11",
        ),
    ],
    ids=["loop-index", "while", "boolean-condition"],
)
def test_control_flow_compiles_and_exports_without_allowlists(body, expected, via_qco):
    from benchpress.utilities.backends import FlexibleBackend

    program = load_qasm_as_qc_program(
        qasm_str=('OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] c; ' + body)
    )
    if via_qco:
        program = program.to_qco()
    original_ir = program.ir
    setup = prepare_mqt_compile(
        program, FlexibleBackend(2, layout="linear", control_flow=True)
    )
    result = setup.compile()
    exported = mqt_to_qiskit_circuit(result, target=setup.target)
    restored = QCProgram.from_qiskit(exported).to_qco()

    assert program.is_valid and program.ir == original_ir
    assert restored.sample(shots=1, seed=1) == {expected: 1}


def test_target_compilation_accepts_scalar_phase_feed_forward():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "creg c[1];\n"
            "h q[0];\n"
            "measure q[0] -> c[0];\n"
            "if(c == 1) u1(pi / 2) q[1];\n"
        )
    )
    backend = FlexibleBackend(2, layout="linear", control_flow=True)

    setup = prepare_mqt_compile(prog, backend)
    result = setup.compile()
    converted = mqt_to_qiskit_circuit(result, target=setup.target)

    assert converted.count_ops().get("if_else") == 1
    assert converted.count_ops().get("measure") == 1


@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("topology", ["all-to-all", "linear", "square", "heavy-hex"])
def test_target_compilation_preserves_reused_classical_destination(control, topology):
    source = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[1];
creg out[1];
{"x q[0];" if control else ""}
cx q[0],q[2];
measure q[0] -> c[0];
if(c == 1) x q[1];
x q[0];
measure q[0] -> c[0];
if(c == 1) x q[1];
measure q[1] -> out[0];
"""
    _assert_mapped_native_output(source, 3, topology, "10" if control else "11")


def test_target_preparation_accepts_proven_affine_index():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nint i = 1;\nh q[i];\n'
        )
    )
    assert "cf.assert" not in prog.ir
    setup = prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))
    result = setup.compile()

    assert "qco." in result.ir


def test_target_compilation_accepts_64_bit_qasm2_register_conditional():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "creg c[64];\n"
            "h q[0];\n"
            "measure q[0] -> c[63];\n"
            "if(c == 9223372036854775808) x q[1];\n"
            "measure q[1] -> c[0];\n"
        )
    )
    assert "cbit.read" in prog.ir
    assert "arith.cmpi eq" in prog.ir
    backend = FlexibleBackend(2, layout="all-to-all", control_flow=True)

    setup = prepare_mqt_compile(prog, backend)
    result = setup.compile()
    converted = mqt_to_qiskit_circuit(result, target=setup.target)

    assert result.ir.count("cbit.read") == 1
    assert result.ir.count("arith.cmpi eq") == 1
    assert converted.count_ops().get("if_else") == 1


def test_target_compilation_accepts_65_bit_qasm2_register_conditional():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "creg c[65];\n"
            "measure q[0] -> c[0];\n"
            "if(c == 18446744073709551616) x q[1];\n"
        )
    )
    assert "cbit.read" in prog.ir
    assert "arith.cmpi eq" in prog.ir
    backend = FlexibleBackend(2, layout="all-to-all", control_flow=True)

    setup = prepare_mqt_compile(prog, backend)
    result = setup.compile()
    converted = mqt_to_qiskit_circuit(result, target=setup.target)

    assert result.ir.count("cbit.read") == 1
    assert result.ir.count("arith.cmpi eq") == 1
    assert converted.count_ops().get("if_else") == 1


def _assert_mapped_native_output(source, num_qubits, topology, expected):
    """Check known deterministic outputs, not equivalence from sample counts."""
    from benchpress.utilities.backends import FlexibleBackend

    program = load_qasm_as_qc_program(qasm_str=source)
    backend = FlexibleBackend(num_qubits, layout=topology, control_flow=True)
    setup = prepare_mqt_compile(program, backend)
    mapped = setup.compile()
    native = mapped.to_qc(copy=True).to_qiskit(target=setup.target)
    restored = QCProgram.from_qiskit(native).to_qco()

    for result in (program.to_qco(copy=True), mapped, restored):
        assert result.sample(shots=1, seed=1) == {expected: 1}


@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("topology", ["linear", "square", "heavy-hex"])
def test_target_compilation_preserves_register_feed_forward(control, topology):
    """Routing must retain both branches and the measurement-before-read edge."""
    source = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[2];
{"x q[0];" if control else ""}
cx q[0],q[2];
measure q[0] -> c[0];
if(c == 1) x q[1];
measure q[1] -> c[1];
"""
    _assert_mapped_native_output(source, 3, topology, "11" if control else "00")


@pytest.mark.parametrize("control", [False, True])
def test_target_compilation_accepts_canonicalized_width_one_control(control):
    source = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[1];
creg out[1];
{"x q[0];" if control else ""}
x q[1];
measure q[0] -> c[0];
if(c == 1) x q[1];
measure q[1] -> out[0];
"""
    _assert_mapped_native_output(source, 2, "linear", "01" if control else "10")


def _distinct_register_source(control):
    return f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg a[2];
creg b[2];
{"x q[0];" if control else ""}
x q[1];
x q[3];
measure q[0] -> a[1];
measure q[1] -> b[0];
if(a == 2) x q[2];
if(b == 1) x q[3];
measure q[2] -> b[1];
measure q[3] -> a[0];
"""


@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("topology", ["linear", "square", "heavy-hex"])
def test_target_compilation_preserves_distinct_register_destinations(control, topology):
    """Register/index swaps and rewired conditions change these known outputs."""
    _assert_mapped_native_output(
        _distinct_register_source(control), 4, topology, "1110" if control else "0100"
    )


def test_mqt_compile_uses_backend_native_gates():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = FlexibleBackend(
        2,
        layout="linear",
        basis_gates=["id", "rz", "sx", "x", "cx"],
    )
    setup = prepare_mqt_compile(prog, backend)
    result = setup.compile()
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert set(converted.count_ops()) <= set(backend.operation_names) | {"barrier"}
    assert converted.count_ops().get("cx", 0) >= 1


@pytest.mark.parametrize("extra", ["custom", "unsupported_bounds"])
def test_mqt_backend_omits_unsupported_extras_unless_requested(monkeypatch, extra):
    from qiskit.circuit import Gate, Parameter
    from qiskit.circuit.library import RZZGate

    from benchpress.config import Configuration
    from benchpress.mqt_gym.utils.validation import mqt_circuit_validation
    from benchpress.utilities.backends import FlexibleBackend

    backend = FlexibleBackend(2, layout="linear", basis_gates=["u", "cx"])
    unsupported = {
        "custom": Gate("provider_gate", 2, []),
        "unsupported_bounds": RZZGate(Parameter("angle")),
    }[extra]
    name = unsupported.name
    backend.target.add_instruction(
        unsupported,
        angle_bounds=[(0, 0.25)] if extra == "unsupported_bounds" else None,
    )
    program = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    monkeypatch.setitem(Configuration.options, "mqt", {})
    with pytest.warns(UserWarning, match=name):
        setup = prepare_mqt_compile(program, backend)
    assert mqt_circuit_validation(setup.compile(), backend, target=setup.target)

    monkeypatch.setitem(
        Configuration.options, "mqt", {"native_gates": ["u", "cx", name]}
    )
    with pytest.raises(ValueError, match=name):
        prepare_mqt_compile(program, backend)


@pytest.mark.parametrize("extra", ["fixed_angle", "alias", "bounded_angle"])
def test_mqt_backend_preserves_supported_native_constraints(monkeypatch, extra):
    from math import pi

    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RZZGate, XGate

    from benchpress.config import Configuration
    from benchpress.mqt_gym.utils.validation import mqt_circuit_validation
    from benchpress.utilities.backends import FlexibleBackend

    instruction = {
        "fixed_angle": RZZGate(0.5),
        "alias": XGate(),
        "bounded_angle": RZZGate(Parameter("angle")),
    }[extra]
    name = "native_x" if extra == "alias" else instruction.name
    backend = FlexibleBackend(2, layout="linear", basis_gates=["u", "cx"])
    backend.target.add_instruction(
        instruction,
        name=name,
        angle_bounds=[(0, pi / 2)] if extra == "bounded_angle" else None,
    )
    monkeypatch.setitem(
        Configuration.options, "mqt", {"native_gates": ["u", "cx", name]}
    )
    setup = prepare_mqt_compile(load_qasm_as_qc_program(qasm_str=_CX_MEASURED), backend)
    operation = next(op for op in setup.target.operations if op.name == name)
    assert operation.canonical_name == instruction.name
    if extra == "fixed_angle":
        assert operation.fixed_parameters == [0.5]
    elif extra == "bounded_angle":
        assert operation.parameter_bounds == [(0, pi / 2)]
    if extra != "alias":
        assert setup.target.supports_operation(name, 2, parameters=[0.5])
        assert not setup.target.supports_operation(name, 2, parameters=[pi])
    assert mqt_circuit_validation(setup.compile(), backend, target=setup.target)


def test_mqt_compiled_standard_alias_passes_backend_validation():
    from qiskit.circuit.library import U3Gate

    from benchpress.mqt_gym.utils.validation import mqt_circuit_validation
    from benchpress.utilities.backends import FlexibleBackend

    backend = FlexibleBackend(2, layout="linear", basis_gates=["u1", "u2", "u3", "cx"])
    program = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    setup = prepare_mqt_compile(program, backend)
    exported = mqt_to_qiskit_circuit(setup.compile(), target=setup.target)
    assert any(item.operation.base_class is U3Gate for item in exported.data)
    assert mqt_circuit_validation(exported, backend, target=setup.target)


def test_mqt_compiler_target_includes_zero_qubit_global_phase():
    target = make_compiler_target(2, None, basis_gates=["rz", "sx", "x", "cz"])

    assert target.supports_operation("gphase", 0, 1)


def test_mqt_compiler_target_rejects_unknown_native_gate():
    with pytest.raises(KeyError, match="typo"):
        make_compiler_target(2, None, basis_gates=["rz", "typo"])


def test_mqt_compiler_target_rejects_disconnected_backend():
    backend = Target.from_configuration(basis_gates=["x"], num_qubits=2)
    with pytest.raises(ValueError, match="connected"):
        prepare_mqt_compile(load_qasm_as_qc_program(qasm_str=_CX_MEASURED), backend)


@pytest.mark.parametrize(
    ("num_qubits", "edges", "message"),
    [
        (2, [(0, 0)], "self-loop"),
        (3, [(0, 0), (0, 1), (0, 2)], "self-loop"),
        (3, [(0, 1), (0, 2), (0, 3)], "outside"),
    ],
)
def test_mqt_compiler_target_rejects_invalid_coupling_edges(num_qubits, edges, message):
    with pytest.raises(ValueError, match=message):
        make_compiler_target(num_qubits, edges, basis_gates=["u", "cx"])


def test_mqt_manipulation_basis_synthesis_supports_one_qubit_program():
    from benchpress.mqt_gym.manipulate.test_manipulate import (
        _basis_environment,
        _synthesize,
    )

    circuit = QuantumCircuit(1)
    circuit.x(0)
    program = QCProgram.from_qiskit(circuit)
    original_ir = program.ir
    environment = _basis_environment(program, ["rz", "sx", "x"])

    assert environment.target.num_sites == 1
    assert (
        environment.target.connectivity_kind
        == CompilerTarget.ConnectivityKind.ALL_TO_ALL
    )
    for _ in range(2):
        result = _synthesize(program, environment)
        assert isinstance(result, QCProgram)
        assert Operator(result.to_qiskit()).equiv(Operator(circuit))
        assert program.is_valid
        assert program.ir == original_ir


def test_mqt_compile_supports_backend_without_coupling_map():
    from benchpress.mqt_gym.utils.io import mqt_compile

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = Target.from_configuration(
        num_qubits=2, basis_gates=["rz", "sx", "x", "cz", "measure", "reset"]
    )
    result = mqt_compile(prog, backend)
    assert "qco." in result.ir


@pytest.mark.parametrize("gate_name", ["cx", "ecr"])
def test_mqt_compile_supports_directional_entangler_backend(gate_name):
    backend = Target.from_configuration(
        num_qubits=2,
        coupling_map=CouplingMap([(0, 1)]),
        basis_gates=["u", gate_name, "measure", "reset"],
    )
    prog = load_qasm_as_qc_program(qasm_str=_BIDIRECTIONAL_CX_MEASURED)

    setup = prepare_mqt_compile(prog, backend)
    operation = next(
        operation
        for operation in setup.target.operations
        if operation.name == gate_name
    )

    assert [tuple(site_tuple.sites) for site_tuple in operation.site_tuples] == [(0, 1)]
    assert setup.target.supports_operation(gate_name, 2, sites=[0, 1])
    assert not setup.target.supports_operation(gate_name, 2, sites=[1, 0])

    result = setup.compile()
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    entanglers = [
        tuple(converted.find_bit(qubit).index for qubit in instruction.qubits)
        for instruction in converted.data
        if instruction.operation.name == gate_name
    ]
    assert entanglers
    assert set(entanglers) == {(0, 1)}


@pytest.mark.parametrize(
    "source, width",
    [
        ("qubit[2] a; qubit[3] b; x a[0]; x b[2];", 5),
        ("qubit a; qubit b; x a; x b;", 2),
        ("qubit a; qubit[2] b; x a; x b[1];", 3),
    ],
)
def test_mqt_qubit_count_preserves_allocations_after_lowering(source, width):
    program = load_qasm_as_qc_program(
        qasm_str='OPENQASM 3.0; include "stdgates.inc"; ' + source
    )
    lowered = program.to_qco(copy=True)
    assert program_num_qubits(program) == program_num_qubits(lowered) == width
    target = CompilerTarget(
        width - 1,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    for input_program in (program, lowered, QuantumCircuit(width)):
        with pytest.raises(ValueError, match=f"Circuit has {width} qubits"):
            prepare_mqt_compile(input_program, target)


def test_mqt_compile_rejects_circuit_wider_than_backend():
    from benchpress.mqt_gym.utils.io import mqt_compile
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nx q[2];\n')
    )
    with pytest.raises(ValueError, match="3 qubits.*backend has 2"):
        mqt_compile(prog, FlexibleBackend(2, layout="linear"))


def test_mqt_compile_preserves_unobserved_qco_input():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'
        )
    ).to_qco(copy=True)
    setup = prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))
    result = setup.compile()
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert len(converted.data) > 0
    assert converted.count_ops().get("measure", 0) == 0


def test_explicit_dead_gate_pass_removes_unobserved_work():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'
        )
    )
    setup = prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))
    result = setup.compile()
    assert len(mqt_to_qiskit_circuit(result, target=setup.target).data) > 0

    result.run_pass_pipeline("remove-dead-gates")
    removed = mqt_to_qiskit_circuit(result, target=setup.target)
    assert removed.num_qubits == setup.target.num_sites
    assert len(removed.data) == 0


def test_mqt_circuit_validation_accepts_mapped_linear_circuit():
    from benchpress.mqt_gym.utils.io import mqt_to_qiskit_circuit
    from benchpress.mqt_gym.utils.validation import mqt_circuit_validation
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = FlexibleBackend(5, layout="linear")
    setup = prepare_mqt_compile(prog, backend)
    result = setup.compile()
    exported = mqt_to_qiskit_circuit(result, target=setup.target)
    assert exported.num_qubits == backend.num_qubits
    assert len(exported.qregs) == 1
    assert exported.qregs[0].name == "q"
    assert exported.layout is not None
    assert len(exported.layout.final_index_layout()) == 2
    assert len(exported.data) >= 1
    assert mqt_circuit_validation(result, backend, target=setup.target) is True


def test_target_aware_export_rejects_wrong_sparse_target():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    setup = prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))
    result = setup.compile()
    sparse_target = CompilerTarget(
        [CompilerTarget.Site(0), CompilerTarget.Site(2)],
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )

    with pytest.raises(RuntimeError, match="target-aware Qiskit conversion failed"):
        mqt_to_qiskit_circuit(result, target=sparse_target)


def test_mqt_circuit_validation_accepts_reversed_cz_edge():
    from benchpress.mqt_gym.utils.validation import mqt_circuit_validation

    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    backend = SimpleNamespace(
        coupling_map=CouplingMap([(1, 0)]),
        operation_names=["cz"],
        two_q_gate_type="cz",
    )
    assert mqt_circuit_validation(circuit, backend) is True


def test_mqt_circuit_validation_rejects_bare_edges():
    from benchpress.mqt_gym.utils.io import mqt_compile
    from benchpress.mqt_gym.utils.validation import mqt_circuit_validation
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = FlexibleBackend(2, layout="linear")
    result = mqt_compile(prog, backend)
    with pytest.raises(TypeError, match="BackendV2-compatible"):
        mqt_circuit_validation(result, [(0, 1)])


class _Bench:
    def __init__(self):
        self.extra_info = {}


def test_mqt_output_circuit_properties_named_twoq_gate():
    from benchpress.mqt_gym.utils.io import mqt_output_circuit_properties
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = FlexibleBackend(5, layout="linear")
    setup = prepare_mqt_compile(prog, backend)
    result = setup.compile()
    bench = _Bench()
    two_q = backend.two_q_gate_type
    mqt_output_circuit_properties(result, two_q, bench, target=setup.target)
    assert bench.extra_info["output_num_qubits"] == backend.num_qubits
    assert isinstance(bench.extra_info["output_circuit_operations"], dict)
    assert bench.extra_info["output_gate_count_2q"] >= 1
    assert bench.extra_info["output_depth_2q"] >= 1


@pytest.mark.parametrize("via_qco", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_mqt_inspection_preserves_exported_metrics(via_qco, dynamic):
    from benchpress.mqt_gym.utils.io import mqt_output_circuit_properties

    source = QuantumCircuit(3, 1)
    source.h(0)
    source.cx(0, 1)
    source.measure(0, 0)
    if dynamic:
        with source.if_test((source.clbits[0], 1)):
            source.x(2)
    program = QCProgram.from_qiskit(source)
    if via_qco:
        program = program.to_qco()
    original_ir = program.ir
    assert program_num_qubits(program) == source.num_qubits
    assert program_uses_classical_control(program) == dynamic
    assert program_op_counts(program)["h"] == 1
    assert program_op_counts(program)["ctrl"] == 1

    native, exported = _Bench(), _Bench()
    mqt_output_circuit_properties(program, "cx", native)
    mqt_output_circuit_properties(program.to_qiskit(), "cx", exported)
    assert native.extra_info == exported.extra_info
    assert program.ir == original_ir


def test_mqt_inspection_rejects_unknown_width():
    program = SimpleNamespace(inspect=lambda: SimpleNamespace(num_qubits=None))
    with pytest.raises(ValueError, match="unknown quantum capacity"):
        program_num_qubits(program)
