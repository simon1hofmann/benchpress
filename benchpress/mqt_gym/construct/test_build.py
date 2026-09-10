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
"""Test circuit generation"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2

from benchpress.config import Configuration
from benchpress.mqt_gym.circuits import (
    dtc_unitary,
    mqt_QV,
    mqt_random_clifford,
    multi_control_circuit,
    to_qc_program,
)
from benchpress.mqt_gym.utils.io import (
    load_qasm_as_qc_program,
    program_op_counts,
)
from benchpress.utilities.io import output_circuit_properties
from benchpress.workouts.build import WorkoutCircuitConstruction
from benchpress.workouts.validation import benchpress_test_validation

SEED = 12345


@benchpress_test_validation
class TestWorkoutCircuitConstruction(WorkoutCircuitConstruction):
    def test_QV100_build(self, benchmark):
        """Build a 100Q Quantum Volume circuit with dense unitary operations."""

        @benchmark
        def result():
            return mqt_QV(100, 100, seed=SEED)

        assert program_op_counts(result).get("unitary", 0) == 5000

    def test_DTC100_set_build(self, benchmark):
        """Build a set of 100Q DTC circuits out to 100 layers."""
        max_cycles = 100
        num_qubits = 100

        @benchmark
        def result():
            circuits = [QuantumCircuit(num_qubits)]
            dtc_circ = dtc_unitary(num_qubits, seed=SEED)
            for cycle in range(max_cycles):
                circuits.append(circuits[cycle].compose(dtc_circ))
            return to_qc_program(circuits[-1])

        output_circuit_properties(result, "rzz", benchmark)
        assert benchmark.extra_info["output_gate_count_2q"] == 9900

    def test_clifford_build(self, benchmark):
        """Build a 100Q Clifford circuit from scratch."""

        @benchmark
        def result():
            return mqt_random_clifford(100, seed=SEED)

        assert result

    def test_multi_control_circuit(self, benchmark):
        """Build a circuit with a multi-controlled X-gate."""
        ITER_CIRCUIT_WIDTH = 16

        @benchmark
        def result():
            return multi_control_circuit(ITER_CIRCUIT_WIDTH)

        assert program_op_counts(result).get("ctrl", 0) == 15

    def test_param_circSU2_100_build(self, benchmark):
        """Build a parameterized efficient SU2 circuit (1000 parameters)."""
        N = 100

        @benchmark
        def result():
            return to_qc_program(efficient_su2(N, reps=4, entanglement="circular"))

        assert result.to_qiskit().num_parameters == 1000

    def test_param_circSU2_100_bind(self, benchmark):
        """Bind 1000 parameters on efficient SU2 over 100Q."""
        N = 100
        qc = efficient_su2(N, reps=4, entanglement="circular")
        assert qc.num_parameters == 1000
        values = np.linspace(0, 2 * np.pi, qc.num_parameters)

        @benchmark
        def result():
            return to_qc_program(qc.assign_parameters(values))

        assert result.to_qiskit().num_parameters == 0

    def test_QV100_qasm2_import(self, benchmark):
        """QASM import of QV100 circuit into MLIR."""

        @benchmark
        def result():
            path = Configuration.get_qasm_dir("qv") + "qv_N100_12345.qasm"
            return load_qasm_as_qc_program(path)

        output_circuit_properties(result, "cx", benchmark)
        operations = benchmark.extra_info["output_circuit_operations"]
        assert operations.get("rz", 0) == 120000
        assert operations.get("rx", 0) == 80000
        assert operations.get("cx", 0) == 15000

    def test_bigint_qasm2_import(self, benchmark):
        """QASM import circuit with bigint."""

        @benchmark
        def result():
            path = Configuration.get_qasm_dir("bigint") + "bigint.qasm"
            return load_qasm_as_qc_program(path)

        output_circuit_properties(result, "cx", benchmark)
        assert result
