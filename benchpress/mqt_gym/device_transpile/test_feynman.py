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
"""Test transpilation against a device"""

import os

import pytest

from benchpress.config import Configuration
from benchpress.mqt_gym.utils.io import (
    mqt_compile,
    prepare_mqt_compile,
    program_num_qubits,
    qasm_uses_classical_control,
)
from benchpress.utilities.io import output_circuit_properties, qasm_circuit_loader
from benchpress.utilities.validation import circuit_validator
from benchpress.workouts.device_transpile import WorkoutDeviceFeynman
from benchpress.workouts.validation import benchpress_test_validation

BACKEND = Configuration.backend()
TWO_Q_GATE = BACKEND.two_q_gate_type


def pytest_generate_tests(metafunc):
    directory = Configuration.get_qasm_dir("feynman")
    file_list = [x for x in os.listdir(directory) if x.endswith(".qasm")]
    metafunc.parametrize("filename", file_list)


@benchpress_test_validation
class TestWorkoutDeviceFeynman(WorkoutDeviceFeynman):
    def test_feynman_transpile(self, benchmark, filename):
        """Transpile a feynman benchmark qasm file against a target device"""
        qasm_file = f"{Configuration.get_qasm_dir('feynman')}{filename}"
        if qasm_uses_classical_control(qasm_file):
            pytest.skip("MQT target compilation cannot map QASM2 classical control")
        prog = qasm_circuit_loader(qasm_file, benchmark)
        if program_num_qubits(prog) > BACKEND.num_qubits:
            pytest.skip("Circuit too large for given backend.")
        setup = prepare_mqt_compile(prog, BACKEND)

        @benchmark
        def result():
            return mqt_compile(setup.program, setup.target)

        output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
        assert circuit_validator(result, BACKEND, target=setup.target)
