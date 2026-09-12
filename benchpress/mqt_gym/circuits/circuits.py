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

from mqt.core.mlir import QCProgram
from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume

from benchpress.qiskit_gym.circuits import (
    bv_all_ones,
    random_clifford_circuit,
)
from benchpress.qiskit_gym.circuits import (
    multi_control_circuit as qiskit_multi_control_circuit,
)
from benchpress.qiskit_gym.circuits import (
    trivial_bvlike_circuit as qiskit_trivial_bvlike_circuit,
)


def to_qc_program(circuit: QuantumCircuit) -> QCProgram:
    """Import a Qiskit circuit into MQT Core's QC dialect."""
    return QCProgram.from_qiskit(circuit)


def mqt_QV(num_qubits, depth=None, seed=12345) -> QCProgram:
    """Construct a QV circuit and import its dense unitaries into MLIR."""
    if depth is None:
        depth = num_qubits
    return QCProgram.from_qiskit(quantum_volume(num_qubits, depth, seed=seed))


def multi_control_circuit(num_qubits) -> QCProgram:
    """Import the shared Benchpress multi-control ladder."""
    return to_qc_program(qiskit_multi_control_circuit(num_qubits))


def mqt_bv_all_ones(N) -> QCProgram:
    """Import the shared Bernstein–Vazirani circuit."""
    return to_qc_program(bv_all_ones(N))


def trivial_bvlike_circuit(N) -> QCProgram:
    """Import the shared BV-like simplification circuit."""
    return to_qc_program(qiskit_trivial_bvlike_circuit(N))


def mqt_random_clifford(num_qubits, seed=12345) -> QCProgram:
    """Import the shared random Clifford gate sequence."""
    return to_qc_program(random_clifford_circuit(num_qubits, seed=seed))
