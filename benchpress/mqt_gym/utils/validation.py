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
"""Circuit validation for MQT MLIR programs."""

from qiskit import QuantumCircuit

from benchpress.mqt_gym.utils.io import mqt_to_qiskit_circuit


def mqt_circuit_validation(circuit, backend, *, target=None):
    """Validate that a compiled MQT program matches backend basis and topology.

    Conversion failures remain validation failures. Silently falling back to
    partial IR checks would make MQT results incomparable with other gyms.

    Parameters:
        circuit: ``QCProgram``, ``QCOProgram``, or ``QuantumCircuit``
        backend: BackendV2-compatible target (must expose ``operation_names``,
            ``coupling_map``, and ``two_q_gate_type``)
        target: optional ``CompilerTarget`` used for target-aware physical
            Qiskit export
    """
    if not hasattr(backend, "operation_names") or not hasattr(backend, "coupling_map"):
        raise TypeError(
            "mqt_circuit_validation requires a BackendV2-compatible backend, "
            f"got {type(backend)!r}"
        )

    qiskit_circuit = (
        circuit
        if isinstance(circuit, QuantumCircuit)
        else mqt_to_qiskit_circuit(circuit, target=target)
    )
    backend_ops = set(backend.operation_names) | {"barrier"}
    cmap = backend.coupling_map
    edges = None if cmap is None else set(cmap.get_edges())
    if edges is not None and backend.two_q_gate_type == "cz":
        edges |= {(target, source) for source, target in edges}

    def validate(block, sites):
        for instruction in block.data:
            name = instruction.operation.name
            if name not in backend_ops:
                raise ValueError(f"Circuit has gates outside backend basis set: {name}")
            physical = tuple(sites[block.find_bit(q).index] for q in instruction.qubits)
            if (
                name == backend.two_q_gate_type
                and edges is not None
                and physical not in edges
            ):
                raise ValueError(f"2Q gate edge {physical} not in backend topology")
            for child in getattr(instruction.operation, "blocks", ()):
                validate(child, physical)

    validate(qiskit_circuit, tuple(range(qiskit_circuit.num_qubits)))
    return True
