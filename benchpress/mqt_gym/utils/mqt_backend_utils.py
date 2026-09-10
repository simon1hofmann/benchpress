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
"""Backend helpers for the MQT gym."""

from qiskit_ibm_runtime import QiskitRuntimeService

from benchpress.config import POSSIBLE_2Q_GATES
from benchpress.qiskit_gym.utils.qiskit_backend_utils import get_ibm_fake_backend


def get_mqt_bench_backend(backend_name):
    """Return an annotated Qiskit BackendV2 for MQT target construction."""
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


def _raw_coupling_edges(backend):
    """Return the backend's directed coupling entries, if any."""
    cmap = getattr(backend, "coupling_map", None)
    if cmap is not None:
        if hasattr(cmap, "get_edges"):
            edges = list(cmap.get_edges())
        else:
            edges = [tuple(edge) for edge in cmap]
        if edges:
            return edges
    if hasattr(backend, "configuration"):
        conf = backend.configuration()
        if conf is not None and getattr(conf, "coupling_map", None):
            return [tuple(edge) for edge in conf.coupling_map]
    return []


def coupling_edges(backend):
    """Return undirected coupling edges, or ``None`` for all-to-all targets."""
    edges = _raw_coupling_edges(backend)
    if not edges:
        if (
            getattr(backend, "coupling_map", None) is not None
            and int(getattr(backend, "num_qubits", 0)) > 1
        ):
            raise ValueError("Backend exposes an empty coupling map")
        return None
    undirected = set()
    for u, v in edges:
        undirected.add((int(u), int(v)))
        undirected.add((int(v), int(u)))
    return sorted(undirected)


def operation_site_tuples(backend, operation, arity):
    """Return ordered operation sites, or ``None`` for global availability."""
    target = getattr(backend, "target", None)
    if target is not None and hasattr(target, "qargs_for_operation_name"):
        try:
            qargs = target.qargs_for_operation_name(operation)
        except KeyError:
            pass
        else:
            if qargs is None:
                return None
            return sorted({tuple(int(site) for site in sites) for sites in qargs})

    if arity != 2:
        return None
    edges = _raw_coupling_edges(backend)
    return sorted({tuple(int(site) for site in edge) for edge in edges}) or None
