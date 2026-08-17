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

import os
from types import SimpleNamespace

import pytest
from mqt.core.ir import QuantumComputation
from mqt.core.mlir import CompilerTarget, QCOProgram, QCProgram
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from benchpress.mqt_gym.circuits import to_qc_program
from benchpress.mqt_gym.utils.io import (
    load_qasm_as_qc_program,
    mqt_compile,
    mqt_to_qiskit_circuit,
    prepare_mqt_compile,
    qasm_uses_classical_control,
    target_unsupported_control_flow_reason,
)
from benchpress.mqt_gym.utils.mqt_backend_utils import (
    coupling_edges,
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


@pytest.mark.parametrize("backend_name", ["fake_torino", "FakeTorino"])
def test_mqt_fake_backend_discovery_accepts_supported_name_formats(backend_name):
    backend = get_mqt_bench_backend(backend_name)

    assert type(backend).__name__ == "FakeTorino"
    assert backend.num_qubits == 133
    assert backend.two_q_gate_type == "cz"
    assert len(coupling_edges(backend)) == 300


def test_mqt_ir_builder_uses_supported_typed_mlir_bridge():
    computation = QuantumComputation(2)
    computation.global_phase = 0.375
    computation.h(0)
    computation.cx(0, 1)

    program = to_qc_program(computation)
    converted = program.to_qiskit()

    assert converted.count_ops() == {"h": 1, "cx": 1}
    assert converted.global_phase == pytest.approx(0.375)


def test_mqt_to_qiskit_circuit_uses_native_compatible_program():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    converted = mqt_to_qiskit_circuit(QCProgram.from_qiskit(circuit))
    assert converted.count_ops() == circuit.count_ops()


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

    result = mqt_compile(setup.program, setup.target)
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert "qco." in result.ir
    assert len(converted.data) > 0
    assert converted.count_ops().get("measure", 0) == 0
    assert converted.num_qubits == backend.num_qubits
    assert len(converted.qregs) == 1
    assert converted.qregs[0].name == "q"
    assert converted.layout is None


def test_target_aware_export_rejects_unmapped_dynamic_qubits():
    prog = load_qasm_as_qc_program(
        qasm_str=('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\n')
    )

    with pytest.raises(RuntimeError, match="mapped static qubits"):
        mqt_to_qiskit_circuit(prog, target=CompilerTarget(2))


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
    result = mqt_compile(setup.program, setup.target)
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
    result = mqt_compile(setup.program, setup.target)
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert converted.count_ops().get("measure") == 2
    assert sum(converted.count_ops().values()) > 2


def test_target_preparation_rejects_qasm3_control_flow_safely():
    """Parsing stays supported while unsafe target compilation fails early."""
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 3.0;\n"
            'include "stdgates.inc";\n'
            "qubit[2] q;\n"
            "for int i in [0:1] {\n"
            "  h q[i];\n"
            "}\n"
        )
    )
    assert "scf.for" in prog.ir
    with pytest.raises(NotImplementedError, match="classical control flow"):
        prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))


def test_target_preparation_rejects_dynamic_index_assertion():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nint i = 1;\nh q[i];\n'
        )
    )
    assert "cf.assert" in prog.ir
    with pytest.raises(NotImplementedError, match="dynamic-index"):
        prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))


def test_target_preparation_rejects_qasm2_conditional():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "creg c[1];\n"
            "measure q[0] -> c[0];\n"
            "if(c == 1) x q[1];\n"
        )
    )
    assert "scf.if" in prog.ir
    with pytest.raises(NotImplementedError, match="classical control flow"):
        prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))


def test_public_compile_rejects_control_flow_before_native_body(monkeypatch):
    import benchpress.mqt_gym.utils.io as mqt_io
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 3.0;\n"
            'include "stdgates.inc";\n'
            "qubit[1] q;\n"
            "for int i in [0:1] { h q[0]; }\n"
        )
    )

    def fail_if_called(*args, **kwargs):
        pytest.fail("native target compilation must not be entered")

    monkeypatch.setattr(mqt_io, "_mqt_compile_body", fail_if_called)
    with pytest.raises(NotImplementedError, match="classical control flow"):
        mqt_io.mqt_compile(prog, FlexibleBackend(2, layout="all-to-all"))


def test_target_preparation_rejects_structured_qco_control_flow():
    from benchpress.utilities.backends import FlexibleBackend

    prog = QCOProgram.from_mlir_str("""
        module {
          func.func @main() attributes {passthrough = ["entry_point"]} {
            %condition = arith.constant true
            %q0 = qco.alloc : !qco.qubit
            %q1 = qco.if %condition args(%arg0 = %q0) -> (!qco.qubit) {
              qco.yield %arg0 : !qco.qubit
            } else args(%arg0 = %q0) {
              qco.yield %arg0 : !qco.qubit
            }
            qco.sink %q1 : !qco.qubit
            return
          }
        }
        """)
    with pytest.raises(NotImplementedError, match="classical control flow"):
        prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))


def test_control_flow_guard_ignores_operation_names_in_attributes():
    prog = QCOProgram.from_mlir_str("""
        module attributes {test.note = "scf.for in documentation"} {
          func.func @main() attributes {passthrough = ["entry_point"]} {
            %q = qco.alloc : !qco.qubit
            qco.sink %q : !qco.qubit
            return
          }
        }
        """)
    assert target_unsupported_control_flow_reason(prog) is None


@pytest.mark.parametrize(
    "statement",
    [
        "if (c == 1) x q[0];",
        "for int i in [0:1] { x q[i]; }",
        "while (c) {}",
        "switch (c) { case 0: { x q[0]; } }",
    ],
)
def test_qasm_control_flow_detection(tmp_path, statement):
    qasm_file = tmp_path / "control.qasm"
    qasm_file.write_text(f"OPENQASM 3.0;\n{statement}\n", encoding="utf8")
    assert qasm_uses_classical_control(qasm_file)


def test_mqt_compile_uses_backend_native_gates():
    from benchpress.mqt_gym.utils.io import mqt_compile
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = FlexibleBackend(
        2,
        layout="linear",
        basis_gates=["id", "rz", "sx", "x", "cx"],
    )
    setup = prepare_mqt_compile(prog, backend)
    result = mqt_compile(setup.program, setup.target)
    converted = mqt_to_qiskit_circuit(result, target=setup.target)
    assert set(converted.count_ops()) <= set(backend.operation_names) | {"barrier"}
    assert converted.count_ops().get("cx", 0) >= 1


def test_mqt_compile_supports_backend_without_coupling_map():
    from benchpress.mqt_gym.utils.io import mqt_compile

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = SimpleNamespace(
        num_qubits=2,
        coupling_map=None,
        operation_names=["rz", "sx", "x", "cz", "measure", "reset"],
    )
    result = mqt_compile(prog, backend)
    assert "qco." in result.ir


def test_mqt_compile_rejects_directional_entangler_backend():
    from benchpress.mqt_gym.utils.io import mqt_compile

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = SimpleNamespace(
        num_qubits=2,
        coupling_map=CouplingMap([(0, 1)]),
        operation_names=["rz", "sx", "x", "ecr", "measure", "reset"],
        two_q_gate_type="ecr",
    )
    with pytest.raises(NotImplementedError, match="directional ecr"):
        mqt_compile(prog, backend)


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
    result = mqt_compile(setup.program, setup.target)
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
    result = mqt_compile(setup.program, setup.target)
    assert len(mqt_to_qiskit_circuit(result, target=setup.target).data) > 0

    result.run_pass_pipeline("remove-dead-gates")
    removed = mqt_to_qiskit_circuit(result, target=setup.target)
    assert removed.num_qubits == setup.target.num_qubits
    assert len(removed.data) == 0


@pytest.mark.skipif(os.name == "nt", reason="hard-timeout path uses spawn on Windows")
def test_mqt_compile_timeout_drains_large_success_before_join(monkeypatch):
    from benchpress.mqt_gym.utils.io import mqt_compile
    from benchpress.utilities.backends import FlexibleBackend

    width = 30
    operations = [f"h q[{qubit}];" for qubit in range(width)]
    operations.extend(
        f"cx q[{control}],q[{target}];"
        for control in range(width)
        for target in range(control + 1, width)
    )
    prog = load_qasm_as_qc_program(
        qasm_str=(
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            f"qreg q[{width}];\n" + "\n".join(operations) + "\n"
        )
    )
    monkeypatch.setenv("MQT_COMPILE_TIMEOUT", "5")
    setup = prepare_mqt_compile(prog, FlexibleBackend(width, layout="all-to-all"))
    result = mqt_compile(setup.program, setup.target)
    assert len(result.ir) > 65_536
    exported = mqt_to_qiskit_circuit(result, target=setup.target)
    assert exported.num_qubits == width
    assert len(exported.data) > 0


def test_mqt_circuit_validation_accepts_mapped_linear_circuit():
    from benchpress.mqt_gym.utils.io import mqt_compile, mqt_to_qiskit_circuit
    from benchpress.mqt_gym.utils.validation import mqt_circuit_validation
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = FlexibleBackend(5, layout="linear")
    setup = prepare_mqt_compile(prog, backend)
    result = mqt_compile(setup.program, setup.target)
    exported = mqt_to_qiskit_circuit(result, target=setup.target)
    assert exported.num_qubits == backend.num_qubits
    assert len(exported.qregs) == 1
    assert exported.qregs[0].name == "q"
    assert exported.layout is None
    assert len(exported.data) >= 1
    assert mqt_circuit_validation(result, backend, target=setup.target) is True


def test_target_aware_export_rejects_wrong_sparse_target():
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    setup = prepare_mqt_compile(prog, FlexibleBackend(2, layout="linear"))
    result = mqt_compile(setup.program, setup.target)
    sparse_target = CompilerTarget([CompilerTarget.Site(0), CompilerTarget.Site(2)])

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
    from benchpress.mqt_gym.utils.io import mqt_compile, mqt_output_circuit_properties
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = FlexibleBackend(5, layout="linear")
    setup = prepare_mqt_compile(prog, backend)
    result = mqt_compile(setup.program, setup.target)
    bench = _Bench()
    two_q = backend.two_q_gate_type
    mqt_output_circuit_properties(result, two_q, bench, target=setup.target)
    assert bench.extra_info["output_num_qubits"] == backend.num_qubits
    assert isinstance(bench.extra_info["output_circuit_operations"], dict)
    assert bench.extra_info["output_gate_count_2q"] >= 1
    assert bench.extra_info["output_depth_2q"] >= 1


def test_mqt_output_circuit_properties_placeholder_2q_gate():
    from benchpress.mqt_gym.utils.io import mqt_compile, mqt_output_circuit_properties
    from benchpress.utilities.backends import FlexibleBackend

    prog = load_qasm_as_qc_program(qasm_str=_CX_MEASURED)
    backend = FlexibleBackend(2, layout="linear")
    setup = prepare_mqt_compile(prog, backend)
    result = mqt_compile(setup.program, setup.target)
    bench = _Bench()
    mqt_output_circuit_properties(result, "2Q_GATE", bench, target=setup.target)
    assert bench.extra_info["output_gate_count_2q"] >= 1
    assert bench.extra_info["output_depth_2q"] >= 1
