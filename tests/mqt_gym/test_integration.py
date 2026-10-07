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
from qiskit.circuit.library import PauliEvolutionGate, efficient_su2
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap, Target

from benchpress.mqt_gym.utils.io import (
    load_qasm_as_qc_program,
    make_compiler_target,
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
        setup.compile() if prepared else prepare_mqt_compile(program, backend)


@pytest.mark.parametrize("via_qco", [False, True])
def test_mqt_compile_accepts_empty_program(monkeypatch, via_qco):
    """The native compile path accepts an empty typed program."""
    program = QCProgram.from_qiskit(QuantumCircuit(0))
    if via_qco:
        program = program.to_qco()

    target = make_compiler_target(1, None, basis_gates=["sx", "x", "rz"])
    result = prepare_mqt_compile(program, target).compile()

    assert isinstance(result, QCOProgram)
    assert result.is_valid
    assert result.to_qc().to_qiskit().num_qubits == 0


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

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = Target.from_configuration(
        num_qubits=2, basis_gates=["rz", "sx", "x", "cz", "measure", "reset"]
    )
    result = prepare_mqt_compile(prog, backend).compile()
    assert "qco." in result.ir


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
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nx q[2];\n')
    )
    with pytest.raises(ValueError, match="3 qubits.*backend has 2"):
        prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear")).compile()


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
    from benchpress.mqt_gym.utils.validation import mqt_circuit_validation
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = FlexibleBackend(2, layout="linear")
    result = prepare_mqt_compile(prog, backend).compile()
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
def test_mqt_inspection_helpers(via_qco, dynamic):
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

    assert program.ir == original_ir


def test_mqt_inspection_rejects_unknown_width():
    program = SimpleNamespace(inspect=lambda: SimpleNamespace(num_qubits=None))
    with pytest.raises(ValueError, match="unknown quantum capacity"):
        program_num_qubits(program)


def test_dynamic_compile_and_export():
    from benchpress.utilities.backends import FlexibleBackend

    program = load_qasm_as_qc_program(
        qasm_str='OPENQASM 3.0; include "stdgates.inc"; '
        "qubit[2] q; bit[2] c; x q[0]; c = measure q; "
        "if (c == 1) { x q[1]; } c = measure q;"
    )
    setup = prepare_mqt_compile(
        program, FlexibleBackend(2, layout="linear", control_flow=True)
    )
    exported = mqt_to_qiskit_circuit(setup.compile(), target=setup.target)
    assert QCProgram.from_qiskit(exported).to_qco().sample(shots=1, seed=1) == {"11": 1}
