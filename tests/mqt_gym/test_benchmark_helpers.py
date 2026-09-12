"""Regression checks for MQT benchmark metrics, validation, and timeouts."""

import subprocess
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

from benchpress.mqt_gym.conftest import _timeout_preflight
from benchpress.mqt_gym.utils.io import mqt_output_circuit_properties
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
    assert benchmark.extra_info["output_gate_count_2q"] == 2
    assert benchmark.extra_info["output_depth_2q"] is None
    assert benchmark.extra_info["control_flow_metrics"] == (
        "static_counts_all_blocks_depth_undefined"
    )
    with pytest.raises(ValueError, match="2Q gate edge"):
        mqt_circuit_validation(
            circuit, FlexibleBackend(3, layout="linear", control_flow=True)
        )
    assert mqt_circuit_validation(
        circuit, FlexibleBackend(3, layout="all-to-all", control_flow=True)
    )


def test_nested_control_flow_rejects_non_native_gate():
    circuit = QuantumCircuit(2, 1)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.h(1)
    with pytest.raises(ValueError, match="outside backend basis"):
        mqt_circuit_validation(circuit, FlexibleBackend(2, control_flow=True))


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
