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
from importlib.metadata import PackageNotFoundError, version

import pytest

from benchpress.config import Configuration
from benchpress.mqt_gym.utils.mqt_backend_utils import unsupported_backend_reason

_REPORTED_PACKAGES = {
    "mqt.core": "mqt-core",
    "qiskit": "qiskit",
    "qiskit_ibm_runtime": "qiskit-ibm-runtime",
    "qiskit_qasm3_import": "qiskit-qasm3-import",
}


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


def pytest_collection_modifyitems(config, items):
    """Skip device workouts whose directed topology MQT cannot represent."""
    device_items = [
        item for item in items if "mqt_gym/device_transpile" in str(item.path)
    ]
    if not device_items:
        return
    reason = unsupported_backend_reason(Configuration.backend())
    if reason is None:
        return
    marker = pytest.mark.skip(reason=reason)
    for item in device_items:
        item.add_marker(marker)
