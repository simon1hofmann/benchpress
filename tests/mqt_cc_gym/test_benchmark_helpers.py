"""Regression checks for mqt-cc benchmark metrics, validation, and timeouts."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from benchpress.mqt_cc_gym.conftest import _timeout_preflight
from benchpress.mqt_cc_gym.utils.io import (
    mqt_load_qasm,
    mqt_prepare_compile,
    mqt_record_input_properties,
    mqt_record_output_properties,
)
from benchpress.mqt_cc_gym.utils.validation import mqt_circuit_validation
from benchpress.utilities.backends import FlexibleBackend
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, RZGate, SXGate
from qiskit.transpiler import Target


@pytest.mark.parametrize("gyms", [("mqt_cc_gym",), ("mqt_cc_gym", "qiskit_gym")])
def test_mixed_gym_collection_is_rejected(gyms):
    from benchpress.mqt_cc_gym.conftest import pytest_collection_modifyitems

    items = [SimpleNamespace(path=Path("benchpress") / gym / "test.py") for gym in gyms]
    if len(gyms) == 1:
        pytest_collection_modifyitems(items)
    else:
        with pytest.raises(pytest.UsageError, match="separate pytest process"):
            pytest_collection_modifyitems(items)


def test_prepared_compilation_passes_configured_mapping_effort(monkeypatch):
    from benchpress.config import Configuration
    from benchpress.mqt_cc_gym.utils import io

    monkeypatch.setitem(
        Configuration.options,
        "mqt-cc",
        {"mapping_trials": 23, "mapping_iterations": 2},
    )
    compiled = SimpleNamespace(compile_for_target=Mock())
    monkeypatch.setattr(io, "_copy_to_qco", lambda _program: compiled)
    setup = mqt_prepare_compile(QuantumCircuit(3), FlexibleBackend(3, layout="linear"))
    assert setup.options.mapping.trials == 23
    assert setup.options.mapping.iterations == 2
    assert setup.compile() is compiled
    compiled.compile_for_target.assert_called_once_with(
        setup.environment, options=setup.options
    )


@pytest.mark.parametrize("width, control_flow", [(0, False), (3, True)])
def test_input_metrics_share_one_native_inspection(width, control_flow):
    inspection = SimpleNamespace(num_qubits=width, has_control_flow=control_flow)
    program = SimpleNamespace(inspect=Mock(return_value=inspection))
    benchmark = SimpleNamespace(extra_info={})
    mqt_record_input_properties(program, benchmark)
    assert benchmark.extra_info == {
        "input_num_qubits": width,
        "input_has_control_flow": control_flow,
    }
    program.inspect.assert_called_once_with()


def test_input_metrics_reject_unknown_width():
    program = SimpleNamespace(
        inspect=lambda: SimpleNamespace(num_qubits=None, has_control_flow=False)
    )
    with pytest.raises(ValueError, match="unknown quantum capacity"):
        mqt_record_input_properties(program, SimpleNamespace(extra_info={}))


def test_shared_ibm_backend_lookup_uses_current_runtime_api(monkeypatch):
    from benchpress.qiskit_gym.utils import qiskit_backend_utils as backends

    backend = SimpleNamespace(operation_names=["sx", "rz", "cz"])
    service = SimpleNamespace(backend=Mock(return_value=backend))
    monkeypatch.setattr(backends, "QiskitRuntimeService", lambda: service)
    assert backends.get_qiskit_bench_backend("ibm_test") is backend
    service.backend.assert_called_once_with("ibm_test")
    assert backend.two_q_gate_type == "cz"


@pytest.mark.parametrize("operations", [["sx", "rz"], ["cz", "cx"]])
def test_shared_backend_rejects_ambiguous_two_qubit_basis(monkeypatch, operations):
    from benchpress.mqt_cc_gym.utils.mqt_backend_utils import get_mqt_bench_backend
    from benchpress.qiskit_gym.utils import qiskit_backend_utils as backends

    monkeypatch.setattr(
        backends,
        "get_ibm_fake_backend",
        lambda _name: SimpleNamespace(operation_names=operations),
    )
    for lookup in (get_mqt_bench_backend, backends.get_qiskit_bench_backend):
        with pytest.raises(ValueError):
            lookup("fake_test")


def test_nested_control_flow_metrics_and_physical_validation():
    circuit = QuantumCircuit(3, 1)
    with (
        circuit.if_test((circuit.clbits[0], 1)) as otherwise,
        circuit.if_test((circuit.clbits[0], 1)),
    ):
        circuit.cz(0, 2)
    with otherwise:
        circuit.cz(1, 2)
    benchmark = SimpleNamespace(extra_info={})
    mqt_record_input_properties(circuit, benchmark)
    assert benchmark.extra_info["input_has_control_flow"]
    mqt_record_output_properties(circuit, "cz", benchmark)
    assert benchmark.extra_info["output_gate_count_2q"] is None
    assert benchmark.extra_info["output_static_gate_count_2q"] == 2
    assert benchmark.extra_info["output_depth_2q"] is None
    assert benchmark.extra_info["control_flow_metrics"] == (
        "static_counts_all_blocks_not_cross_tool_comparable"
    )
    with pytest.raises(ValueError, match="2Q gate edge"):
        mqt_circuit_validation(
            circuit, FlexibleBackend(3, layout="linear", control_flow=True)
        )
    assert mqt_circuit_validation(
        circuit, FlexibleBackend(3, layout="all-to-all", control_flow=True)
    )


def test_straight_line_metrics_remain_comparable_including_zero_counts():
    circuit = QuantumCircuit(2)
    for expected in (0, 1):
        benchmark = SimpleNamespace(extra_info={})
        mqt_record_input_properties(circuit, benchmark)
        assert not benchmark.extra_info["input_has_control_flow"]
        mqt_record_output_properties(circuit, "cz", benchmark)
        assert benchmark.extra_info["output_gate_count_2q"] == expected
        assert benchmark.extra_info["output_depth_2q"] == expected
        assert "output_static_gate_count_2q" not in benchmark.extra_info
        assert "control_flow_metrics" not in benchmark.extra_info
        circuit.cz(0, 1)


def test_eliminated_input_control_flow_stays_out_of_comparison(tmp_path):
    source = tmp_path / "conditional.qasm"
    source.write_text(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[1]; '
        "if(c == 1) cx q[0],q[1];"
    )
    benchmark = SimpleNamespace(extra_info={})
    mqt_load_qasm(source, benchmark)
    assert benchmark.extra_info["input_has_control_flow"]
    # Even if Core removes the branch, other compilers may retain it.
    mqt_record_output_properties(QuantumCircuit(2), "cz", benchmark)
    assert benchmark.extra_info["output_gate_count_2q"] is None
    assert benchmark.extra_info["output_depth_2q"] is None
    assert benchmark.extra_info["output_static_gate_count_2q"] == 0


def test_nested_control_flow_rejects_non_native_gate():
    circuit = QuantumCircuit(2, 1)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.h(1)
    with pytest.raises(ValueError, match="outside backend basis"):
        mqt_circuit_validation(circuit, FlexibleBackend(2, control_flow=True))


def test_validation_preserves_operation_sites_and_parameters():
    target = Target(num_qubits=2)
    target.add_instruction(SXGate(), {(0,): None})
    target.add_instruction(RZGate(0.5), {(0,): None, (1,): None})
    target.add_instruction(CZGate(), {(0, 1): None})
    backend = SimpleNamespace(
        target=target,
        operation_names=target.operation_names,
        coupling_map=target.build_coupling_map(),
        two_q_gate_type="cz",
    )
    valid = QuantumCircuit(2)
    valid.sx(0)
    valid.rz(0.5, 1)
    valid.cz(1, 0)  # CZ is symmetric, even with a directed target entry.
    assert mqt_circuit_validation(valid, backend)

    wrong_site = QuantumCircuit(2)
    wrong_site.sx(1)
    wrong_angle = QuantumCircuit(2)
    wrong_angle.rz(0.25, 0)
    for circuit in (wrong_site, wrong_angle):
        with pytest.raises(ValueError, match="Backend does not support"):
            mqt_circuit_validation(circuit, backend)


@pytest.mark.parametrize("outcome", ["success", "error", "timeout"])
def test_timeout_preflight_runs_whole_test_outside_timer(
    tmp_path, monkeypatch, outcome
):
    import benchpress.mqt_cc_gym.conftest as hooks

    skipfile = tmp_path / "skipfile.txt"
    benchmark = SimpleNamespace(
        timeout_skip_list=5, skipfile=skipfile, fullname="test_native", extra_info={}
    )
    request = SimpleNamespace(
        node=SimpleNamespace(nodeid="test_native.py::test_native"),
        config=SimpleNamespace(rootpath=tmp_path, inipath=tmp_path / "pytest.ini"),
    )
    commands = []
    killed = []

    class Process:
        pid = 123
        returncode = 1 if outcome == "error" else 0
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def communicate(self, timeout=None):
            self.calls += 1
            if outcome == "timeout" and self.calls == 1:
                raise subprocess.TimeoutExpired("pytest", timeout)
            return "child diagnostics", None

        def kill(self):
            killed.append(self.pid)

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        return Process()

    monkeypatch.setenv("PYTEST_ADDOPTS", "--timeout-skip-list=999")
    monkeypatch.setattr(hooks.subprocess, "Popen", popen)
    monkeypatch.setattr(
        hooks.os, "killpg", lambda pid, _signal: killed.append(pid), raising=False
    )
    if outcome == "success":
        _timeout_preflight(benchmark, request)
        assert benchmark.extra_info["preflight_seconds"] >= 0
    else:
        with pytest.raises(
            pytest.fail.Exception,
            match="exceeded" if outcome == "timeout" else "child diagnostics",
        ):
            _timeout_preflight(benchmark, request)
    command, kwargs = commands[0]
    assert str(tmp_path / request.node.nodeid) in command
    assert "--timeout-skip-list=0" in command and "--benchmark-disable" in command
    assert "PYTEST_ADDOPTS" not in kwargs["env"]
    assert skipfile.exists() == (outcome == "timeout")
    if outcome == "timeout":
        assert skipfile.read_text().splitlines() == [benchmark.fullname]
        assert killed == [123]
