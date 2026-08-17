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
import pytest
from mqt.core.ir import QuantumComputation

from benchpress.config import Configuration
from benchpress.mqt_gym.circuits import (
    dtc_unitary,
    mqt_circSU2_ir,
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
        """MQT Core cannot represent the arbitrary two-qubit QV unitaries."""
        pytest.skip("MQT Core has no arbitrary-unitary circuit construction API")

    def test_DTC100_set_build(self, benchmark):
        """Build a set of 100Q DTC circuits out to 100 layers."""
        max_cycles = 100
        num_qubits = 100

        @benchmark
        def result():
            circ = QuantumComputation(num_qubits)
            dtc_op = dtc_unitary(num_qubits, seed=SEED).to_operation()
            for _ in range(max_cycles):
                circ.append(dtc_op)
            return to_qc_program(circ)

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
            return mqt_circSU2_ir(N, 4)

        assert len(result.variables) == 1000

    def test_param_circSU2_100_bind(self, benchmark):
        """Bind 1000 parameters on efficient SU2 over 100Q."""
        N = 100
        qc = mqt_circSU2_ir(N, 4)
        assert len(qc.variables) == 1000

        @benchmark
        def result():
            values = np.linspace(0, 2 * np.pi, len(qc.variables))
            params = dict(zip(qc.variables, values, strict=True))
            return qc.instantiate(params)

        assert len(result.variables) == 0

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

        # MQT can import this classical-integer program, but its result-bearing
        # classical control flow cannot yet be exported for standard metrics.
        assert result
