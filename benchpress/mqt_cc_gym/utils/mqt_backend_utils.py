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
"""Backend helpers for the mqt-cc gym."""

from qiskit_ibm_runtime import QiskitRuntimeService

from benchpress.config import POSSIBLE_2Q_GATES
from benchpress.qiskit_gym.utils.qiskit_backend_utils import get_ibm_fake_backend


def get_mqt_bench_backend(backend_name):
    """Return an annotated Qiskit BackendV2 for mqt-cc target construction."""
    lowered_name = backend_name.lower()
    if "fake" in lowered_name:
        backend = get_ibm_fake_backend(backend_name)
    elif "ibm" in lowered_name:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
    else:
        raise ValueError(f"Backend name {backend_name} not recognized.")

    two_qubit_gates = list(set(backend.operation_names).intersection(POSSIBLE_2Q_GATES))
    if len(two_qubit_gates) > 1:
        raise ValueError("Only one 2Q gate type is currently supported")
    if not two_qubit_gates:
        raise ValueError(f"No gate in {POSSIBLE_2Q_GATES} found!")
    backend.two_q_gate_type = two_qubit_gates[0]
    return backend
