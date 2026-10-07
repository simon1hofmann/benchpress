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
"""Circuit construction helpers for the mqt-cc gym (return MLIR QCPrograms)."""

from mqt.core.mlir import QCProgram
from qiskit.circuit.library import quantum_volume

from benchpress.mqt_cc_gym.utils.io import mqt_import_qiskit
from benchpress.qiskit_gym.circuits import (
    bv_all_ones,
    random_clifford_circuit,
)
from benchpress.qiskit_gym.circuits import (
    multi_control_circuit as qiskit_multi_control_circuit,
)


def mqt_QV(num_qubits, depth=None, seed=12345) -> QCProgram:
    """Construct a QV circuit and import its dense unitaries into MLIR."""
    if depth is None:
        depth = num_qubits
    return QCProgram.from_qiskit(quantum_volume(num_qubits, depth, seed=seed))


def mqt_multi_control_circuit(num_qubits) -> QCProgram:
    """Import the shared Benchpress multi-control ladder."""
    return mqt_import_qiskit(qiskit_multi_control_circuit(num_qubits))


def mqt_bv_all_ones(N) -> QCProgram:
    """Import the shared Bernstein–Vazirani circuit."""
    return mqt_import_qiskit(bv_all_ones(N))


def mqt_random_clifford(num_qubits, seed=12345) -> QCProgram:
    """Import the shared random Clifford gate sequence."""
    return mqt_import_qiskit(random_clifford_circuit(num_qubits, seed=seed))
