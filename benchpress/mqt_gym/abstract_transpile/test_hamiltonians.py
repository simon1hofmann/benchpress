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
"""Test Hamiltonians against abstract backend topologies"""

import pytest

from benchpress.mqt_gym.utils.io import (
    mqt_compile,
    prepare_mqt_compile,
    program_num_qubits,
)
from benchpress.utilities.backends import FlexibleBackend
from benchpress.utilities.io import input_circuit_properties, output_circuit_properties
from benchpress.utilities.io.hamiltonians import generate_hamiltonian_circuit
from benchpress.utilities.validation import circuit_validator
from benchpress.workouts.abstract_transpile.hamlib_hamiltonians import (
    HAM_TOPO,
    HAM_TOPO_NAMES,
    WorkoutAbstractHamiltonians,
)
from benchpress.workouts.validation import benchpress_test_validation


@benchpress_test_validation
class TestWorkoutAbstractHamiltonians(WorkoutAbstractHamiltonians):
    @pytest.mark.parametrize("circ_and_topo", HAM_TOPO, ids=HAM_TOPO_NAMES)
    def test_hamiltonians(self, benchmark, circ_and_topo):
        prog = generate_hamiltonian_circuit(
            circ_and_topo[0].pop("ham_hamlib_hamiltonian"), benchmark
        )
        input_circuit_properties(prog, benchmark)
        backend = FlexibleBackend(program_num_qubits(prog), layout=circ_and_topo[1])
        setup = prepare_mqt_compile(prog, backend)

        @benchmark
        def result():
            return mqt_compile(setup.program, setup.target)

        benchmark.extra_info.update(circ_and_topo[0])
        output_circuit_properties(
            result, backend.two_q_gate_type, benchmark, target=setup.target
        )
        assert circuit_validator(result, backend, target=setup.target)
