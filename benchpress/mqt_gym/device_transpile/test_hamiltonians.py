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

import json

import pytest
from qiskit.quantum_info import SparsePauliOp

from benchpress.config import Configuration
from benchpress.mqt_gym.utils.io import prepare_mqt_compile
from benchpress.utilities.io import input_circuit_properties, output_circuit_properties
from benchpress.utilities.io.hamiltonians import generate_hamiltonian_circuit
from benchpress.utilities.validation import circuit_validator
from benchpress.workouts.device_transpile import WorkoutDeviceHamlibHamiltonians
from benchpress.workouts.validation import benchpress_test_validation

BACKEND = Configuration.backend()
TWO_Q_GATE = BACKEND.two_q_gate_type


def pytest_generate_tests(metafunc):
    directory = Configuration.get_hamiltonian_dir("hamlib")
    with open(directory + "100_representative.json", encoding="utf8") as input_file:
        ham_records = json.load(input_file)
    for h in ham_records:
        terms = h.pop("ham_hamlib_hamiltonian_terms")
        coefficients = h.pop("ham_hamlib_hamiltonian_coefficients")
        h["ham_hamlib_hamiltonian"] = SparsePauliOp(terms, coefficients)
    metafunc.parametrize(
        "hamiltonian_info", ham_records, ids=lambda x: "ham_" + x["ham_instance"][1:-1]
    )


@benchpress_test_validation
class TestWorkoutDeviceHamlibHamiltonians(WorkoutDeviceHamlibHamiltonians):
    def test_hamlib_hamiltonians_transpile(self, benchmark, hamiltonian_info):
        """Transpile a Hamiltonian against a target device"""
        if hamiltonian_info["ham_qubits"] > BACKEND.num_qubits:
            pytest.skip("Circuit too large for given backend.")

        prog = generate_hamiltonian_circuit(
            hamiltonian_info.pop("ham_hamlib_hamiltonian"), benchmark
        )
        input_circuit_properties(prog, benchmark)
        setup = prepare_mqt_compile(prog, BACKEND)

        @benchmark
        def result():
            return setup.compile()

        benchmark.extra_info.update(hamiltonian_info)
        output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
        assert circuit_validator(result, BACKEND, target=setup.target)
