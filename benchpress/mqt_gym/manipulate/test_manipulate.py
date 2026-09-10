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
"""Test circuit manipulation"""

from benchpress.config import Configuration
from benchpress.mqt_gym.circuits import multi_control_circuit
from benchpress.mqt_gym.utils.io import (
    make_compiler_target,
    mqt_to_qiskit_circuit,
    prepare_mqt_compile,
    program_num_qubits,
)
from benchpress.utilities.io import qasm_circuit_loader
from benchpress.workouts.manipulate import WorkoutCircuitManipulate
from benchpress.workouts.validation import benchpress_test_validation


def _basis_setup(program, gates):
    """Compile onto an all-to-all target with the requested native gates."""
    nq = program_num_qubits(program)
    target = make_compiler_target(max(nq, 1), None, basis_gates=gates)
    return prepare_mqt_compile(program, target)


@benchpress_test_validation
class TestWorkoutCircuitManipulate(WorkoutCircuitManipulate):
    def test_DTC100_twirling(self, benchmark):
        """Perform Pauli twirling on a 100-qubit DTC circuit."""
        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("dtc") + "dtc_100_cx_12345.qasm", benchmark
        )
        source = circuit.to_qco()

        @benchmark
        def result():
            twirled = source.copy()
            twirled.run_pass_pipeline("pauli-twirl-2q-gates")
            return twirled

        assert result

    def test_multi_control_decompose(self, benchmark):
        """Decompose a multi-control gate into the basis [rx, ry, rz, cz]."""
        circ = multi_control_circuit(16)
        setup = _basis_setup(circ, ["rx", "ry", "rz", "cz"])

        @benchmark
        def result():
            return setup.compile()

        qc = mqt_to_qiskit_circuit(result, target=setup.target)
        gate_count_2q = qc.count_ops().get("cz", 0)
        benchmark.extra_info["gate_count_2q"] = gate_count_2q
        assert gate_count_2q > 0

    def test_QV100_basis_change(self, benchmark):
        """Change a QV100 circuit basis from [rx, ry, rz, cx] to [sx, x, rz, cz]."""
        circ = qasm_circuit_loader(
            Configuration.get_qasm_dir("qv") + "qv_N100_12345.qasm", benchmark
        )
        setup = _basis_setup(circ, ["sx", "x", "rz", "cz"])

        @benchmark
        def result():
            return setup.compile()

        qc = mqt_to_qiskit_circuit(result, target=setup.target)
        gate_count_2q = qc.count_ops().get("cz", 0)
        benchmark.extra_info["gate_count_2q"] = gate_count_2q
        assert gate_count_2q > 0

    def test_random_clifford_decompose(self, benchmark):
        """Decompose a random clifford into basis [rz, sx, x, cz]."""
        cliff_circ = qasm_circuit_loader(
            Configuration.get_qasm_dir("clifford") + "clifford_20_12345.qasm",
            benchmark,
        )
        setup = _basis_setup(cliff_circ, ["rz", "sx", "x", "cz"])

        @benchmark
        def result():
            return setup.compile()

        qc = mqt_to_qiskit_circuit(result, target=setup.target)
        benchmark.extra_info["gate_count_2q"] = qc.count_ops().get("cz", 0)
        benchmark.extra_info["depth_2q"] = qc.depth(
            filter_function=lambda x: x.operation.name == "cz"
        )
        assert benchmark.extra_info["gate_count_2q"] > 0
