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


def _qiskit_style_validation(circuit, backend):
    """Mirror Qiskit-gym validation, including all-to-all backends."""
    circuit_ops = set(circuit.count_ops())
    backend_ops = set(backend.operation_names) | {"barrier"}
    unsupported_ops = circuit_ops - backend_ops
    if unsupported_ops:
        raise ValueError(
            f"Circuit has gates outside backend basis set {unsupported_ops}"
        )

    coupling_map = backend.coupling_map
    if coupling_map is None:
        return True
    if coupling_map.graph.num_edges() < (
        coupling_map.graph.num_nodes() * (coupling_map.graph.num_nodes() - 1)
    ):
        edges = set(coupling_map.get_edges())
        if backend.two_q_gate_type == "cz":
            edges |= {(target, source) for source, target in edges}
        for instruction in circuit.get_instructions(backend.two_q_gate_type):
            edge = (
                circuit.find_bit(instruction.qubits[0]).index,
                circuit.find_bit(instruction.qubits[1]).index,
            )
            if edge not in edges:
                raise ValueError(f"2Q gate edge {edge} not in backend topology")
    return True


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
    return _qiskit_style_validation(qiskit_circuit, backend)
