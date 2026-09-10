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
"""Circuit construction helpers for the MQT gym (return MLIR QCPrograms)."""

import numpy as np
from mqt.core.mlir import QCProgram
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, quantum_volume


def to_qc_program(circuit: QuantumCircuit) -> QCProgram:
    """Import a Qiskit circuit into MQT Core's QC dialect."""
    return QCProgram.from_qiskit(circuit)


def mqt_QV(num_qubits, depth=None, seed=12345) -> QCProgram:
    """Construct a QV circuit and import its dense unitaries into MLIR."""
    if depth is None:
        depth = num_qubits
    return QCProgram.from_qiskit(quantum_volume(num_qubits, depth, seed=seed))


def mqt_circSU2(width, num_reps=3, seed=12345) -> QCProgram:
    """Efficient SU2 with numeric angles, returned as MLIR."""
    rng = np.random.default_rng(seed)
    circuit = efficient_su2(width, reps=num_reps, entanglement="circular")
    values = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)
    return to_qc_program(circuit.assign_parameters(values))


def dtc_unitary(num_qubits, g=0.95, seed=12345) -> QuantumCircuit:
    """Floquet unitary layer for DTC evolution."""
    rng = np.random.default_rng(seed=seed)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(g * np.pi, i)
    for i in range(0, num_qubits - 1, 2):
        phi = rng.uniform(low=np.pi / 16, high=3 * np.pi / 16)
        qc.rzz(2 * phi, i, i + 1)
    for i in range(1, num_qubits - 1, 2):
        phi = rng.uniform(low=np.pi / 16, high=3 * np.pi / 16)
        qc.rzz(2 * phi, i, i + 1)
    for i in range(num_qubits):
        h = rng.uniform(low=-np.pi, high=np.pi)
        qc.rz(h * np.pi, i)
    return qc


def multi_control_circuit(num_qubits) -> QCProgram:
    """Build the Benchpress X/CX/.../MCX control ladder."""
    qc = QuantumCircuit(num_qubits)
    if num_qubits == 0:
        return to_qc_program(qc)
    qc.x(0)
    for target in range(1, num_qubits):
        qc.mcx(list(range(target)), target)
    return to_qc_program(qc)


def mqt_bv_all_ones(N) -> QCProgram:
    """Bernstein–Vazirani for the all-ones bitstring."""
    qc = QuantumCircuit(N, N - 1)
    qc.x(N - 1)
    for i in range(N):
        qc.h(i)
    for i in range(N - 1):
        qc.cx(i, N - 1)
    for i in range(N - 1):
        qc.h(i)
        qc.measure(i, i)
    return to_qc_program(qc)


def trivial_bvlike_circuit(N) -> QCProgram:
    """BV-like circuit that should simplify under commutation."""
    qc = QuantumCircuit(N)
    for kk in range(N - 1):
        qc.cx(kk, N - 1)
    qc.x(N - 1)
    qc.z(N - 2)
    for kk in range(N - 2, -1, -1):
        qc.cx(kk, N - 1)
    return to_qc_program(qc)


def mqt_random_clifford(num_qubits, num_gates=None, seed=12345) -> QCProgram:
    """Random Clifford-generating gate sequence as MLIR."""
    rng = np.random.default_rng(seed=seed)
    out = QuantumCircuit(num_qubits)
    num_gates = num_gates or 10 * num_qubits * num_qubits
    gates = ["cx", "cz", "cy", "swap", "x", "y", "z", "s", "sdg", "h"]
    for _ in range(num_gates):
        gate = gates[rng.integers(len(gates))]
        if gate in ("cx", "cz", "cy", "swap"):
            q0, q1 = rng.choice(num_qubits, 2, replace=False)
            getattr(out, gate)(int(q0), int(q1))
        else:
            q = int(rng.integers(num_qubits))
            getattr(out, gate)(q)
    return to_qc_program(out)
