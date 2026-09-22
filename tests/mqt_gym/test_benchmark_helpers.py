"""Regression checks for MQT benchmark metrics, validation, and timeouts."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, RZGate, SXGate
from qiskit.transpiler import Target

from benchpress.mqt_gym.conftest import _core_provenance, _timeout_preflight
from benchpress.mqt_gym.utils.io import (
    mqt_output_circuit_properties,
    mqt_qasm_loader,
)
from benchpress.mqt_gym.utils.validation import mqt_circuit_validation
from benchpress.utilities.backends import FlexibleBackend


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
    mqt_output_circuit_properties(circuit, "cz", benchmark)
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
        mqt_output_circuit_properties(circuit, "cz", benchmark)
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
    mqt_qasm_loader(source, benchmark)
    assert benchmark.extra_info["input_has_control_flow"]
    # Even if Core removes the branch, other compilers may retain it.
    mqt_output_circuit_properties(QuantumCircuit(2), "cz", benchmark)
    assert benchmark.extra_info["output_gate_count_2q"] is None
    assert benchmark.extra_info["output_depth_2q"] is None
    assert benchmark.extra_info["output_static_gate_count_2q"] == 0


@pytest.mark.parametrize(
    ("origin", "version", "revision", "source"),
    [
        ({"vcs_info": {"commit_id": "a" * 40}}, "4.0.1", "a" * 40, "vcs_metadata"),
        ({"archive_info": {}}, "4.0.1.dev1+g123abcd", "123abcd", "version"),
        ({}, "4.0.1.dev1+g123abcd.d20260922", "123abcd", "version"),
        ({}, "4.0.1", None, None),
    ],
)
def test_provenance_uses_installed_build_not_requirements(
    tmp_path, monkeypatch, origin, version, revision, source
):
    import benchpress.mqt_gym.conftest as hooks

    extension = tmp_path / "mlir.so"
    extension.write_bytes(b"installed build")
    monkeypatch.setattr(hooks.core_mlir, "__file__", str(extension))
    monkeypatch.setattr(
        hooks,
        "distribution",
        lambda _: SimpleNamespace(
            version=version, read_text=lambda _: json.dumps(origin)
        ),
    )
    assert _core_provenance() == {
        "revision": revision,
        "revision_source": source,
        "mlir_extension_sha256": hashlib.sha256(b"installed build").hexdigest(),
    }


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
    import benchpress.mqt_gym.conftest as hooks

    skipfile = tmp_path / "skipfile.txt"
    benchmark = SimpleNamespace(
        timeout_skip_list=5, skipfile=skipfile, fullname="test_native"
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
