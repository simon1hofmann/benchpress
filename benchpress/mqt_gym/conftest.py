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
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from functools import partial
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import mqt.core.mlir as core_mlir
import pytest

from benchpress.config import Configuration

_REPORTED_PACKAGES = {
    "mqt.core": "mqt-core",
    "qiskit": "qiskit",
    "qiskit_ibm_runtime": "qiskit-ibm-runtime",
    "qiskit_qasm3_import": "qiskit-qasm3-import",
}


@pytest.fixture
def benchmark(benchmark, request):
    # Native MLIR programs cannot be pickled by the plugin's forkserver preflight.
    if benchmark.timeout_skip_list:
        benchmark._check_timeout = partial(_timeout_preflight, benchmark, request)
    return benchmark


def _timeout_preflight(benchmark, request, *_args, **_kwargs):
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(request.config.rootpath / request.node.nodeid),
        "-c",
        str(request.config.inipath),
        "-o",
        "addopts=",
        "--benchmark-disable",
        "--timeout-skip-list=0",
        "-q",
    ]
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=os.name != "nt",
    ) as process:
        try:
            output, _ = process.communicate(timeout=benchmark.timeout_skip_list)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            skipfile = Path(benchmark.skipfile)
            entries = skipfile.read_text().splitlines() if skipfile.exists() else []
            if benchmark.fullname not in entries:
                with skipfile.open("a") as stream:
                    stream.write(benchmark.fullname + "\n")
            pytest.fail(
                f"Test exceeded {benchmark.timeout_skip_list}s; added to {skipfile}",
                pytrace=False,
            )
        if process.returncode:
            pytest.fail(f"Timeout preflight failed:\n{output}", pytrace=False)


def pytest_configure(config):
    if float(os.environ.get("MQT_COMPILE_TIMEOUT", "0") or "0") > 0:
        raise pytest.UsageError(
            "MQT_COMPILE_TIMEOUT was removed; use --timeout-skip-list=SECONDS "
            "for a whole-test preflight outside benchmark timing"
        )


def _package_version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def _reported_versions():
    return {
        name: _package_version(distribution)
        for name, distribution in _REPORTED_PACKAGES.items()
    }


def _core_provenance():
    """Identify the installed build, not the revision requested by requirements."""
    package = distribution("mqt-core")
    origin = json.loads(package.read_text("direct_url.json") or "{}")
    revision = origin.get("vcs_info", {}).get("commit_id")
    revision_source = "vcs_metadata" if revision else None
    if not revision:
        match = re.search(r"\+g([0-9a-f]+)(?:\.|$)", package.version)
        if match:
            revision = match.group(1)
            revision_source = "version"
    with Path(core_mlir.__file__).open("rb") as extension:
        digest = hashlib.file_digest(extension, "sha256").hexdigest()
    return {
        "revision": revision,
        "revision_source": revision_source,
        "mlir_extension_sha256": digest,
    }


def pytest_report_header(config):
    """Add some info about packages and backend to the pytest CLI header"""
    ret = [
        ", ".join(
            f"{name}: {package_version}"
            for name, package_version in _reported_versions().items()
        )
    ]
    if hasattr(config.known_args_namespace, "timeout_skip_list"):
        ret.append(
            f"timeout_skip_list: {config.known_args_namespace.timeout_skip_list}"
        )
    return ret


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """Adds custom sections to the pytest-benchmark report"""
    output_json["mqt_info"] = _reported_versions()
    output_json["mqt_build"] = _core_provenance()
    options = Configuration.options.get("mqt", {})
    defaults = core_mlir.CompilationOptions()
    mapping = defaults.mapping
    output_json["mqt_context"] = {
        "construction_and_binding": "qiskit_frontend_adapter",
        "device_circsu2_parameters": "symbolic",
        "normalize_global_phases": options.get("normalize_global_phases", False),
        "native_gates_override": options.get("native_gates"),
        "timeout_scope": "whole_test_preflight",
        "compilation_timing": "copy_lower_compile",
        "logical_cpus": os.cpu_count(),
        "compiler_defaults": {
            "seed": defaults.seed,
            "trials": mapping.trials,
            "iterations": mapping.iterations,
            "lookahead": mapping.lookahead,
            "search_memory_limit": mapping.search_memory_limit,
        },
    }
