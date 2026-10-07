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
"""Test qasmbench against abstract backend topologies"""

import pytest

from benchpress.mqt_gym.utils.io import (
    prepare_mqt_compile,
    program_num_qubits,
    program_uses_classical_control,
)
from benchpress.utilities.backends import FlexibleBackend
from benchpress.utilities.io import output_circuit_properties, qasm_circuit_loader
from benchpress.utilities.validation import circuit_validator
from benchpress.workouts.abstract_transpile import (
    WorkoutAbstractQasmBenchLarge,
    WorkoutAbstractQasmBenchMedium,
    WorkoutAbstractQasmBenchSmall,
)
from benchpress.workouts.abstract_transpile.qasmbench import (
    LARGE_CIRC_TOPO,
    LARGE_NAMES,
    MEDIUM_CIRC_TOPO,
    MEDIUM_NAMES,
    SMALL_CIRC_TOPO,
    SMALL_NAMES,
)
from benchpress.workouts.validation import benchpress_test_validation


@benchpress_test_validation
class TestWorkoutAbstractQasmBenchSmall(WorkoutAbstractQasmBenchSmall):
    @pytest.mark.parametrize("circ_and_topo", SMALL_CIRC_TOPO, ids=SMALL_NAMES)
    def test_QASMBench_small(self, benchmark, circ_and_topo):
        prog = qasm_circuit_loader(circ_and_topo[0], benchmark)
        uses_control_flow = program_uses_classical_control(prog)
        backend = FlexibleBackend(
            program_num_qubits(prog),
            layout=circ_and_topo[1],
            control_flow=uses_control_flow,
        )
        setup = prepare_mqt_compile(prog, backend)

        @benchmark
        def result():
            return setup.compile().to_qc()

        result = result.to_qiskit(target=setup.target)

        output_circuit_properties(
            result, backend.two_q_gate_type, benchmark, target=setup.target
        )
        assert circuit_validator(result, backend, target=setup.target)


@benchpress_test_validation
class TestWorkoutAbstractQasmBenchMedium(WorkoutAbstractQasmBenchMedium):
    @pytest.mark.parametrize("circ_and_topo", MEDIUM_CIRC_TOPO, ids=MEDIUM_NAMES)
    def test_QASMBench_medium(self, benchmark, circ_and_topo):
        prog = qasm_circuit_loader(circ_and_topo[0], benchmark)
        uses_control_flow = program_uses_classical_control(prog)
        backend = FlexibleBackend(
            program_num_qubits(prog),
            layout=circ_and_topo[1],
            control_flow=uses_control_flow,
        )
        setup = prepare_mqt_compile(prog, backend)

        @benchmark
        def result():
            return setup.compile().to_qc()

        result = result.to_qiskit(target=setup.target)

        output_circuit_properties(
            result, backend.two_q_gate_type, benchmark, target=setup.target
        )
        assert circuit_validator(result, backend, target=setup.target)


@benchpress_test_validation
class TestWorkoutAbstractQasmBenchLarge(WorkoutAbstractQasmBenchLarge):
    @pytest.mark.parametrize("circ_and_topo", LARGE_CIRC_TOPO, ids=LARGE_NAMES)
    def test_QASMBench_large(self, benchmark, circ_and_topo):
        prog = qasm_circuit_loader(circ_and_topo[0], benchmark)
        uses_control_flow = program_uses_classical_control(prog)
        backend = FlexibleBackend(
            program_num_qubits(prog),
            layout=circ_and_topo[1],
            control_flow=uses_control_flow,
        )
        setup = prepare_mqt_compile(prog, backend)

        @benchmark
        def result():
            return setup.compile().to_qc()

        result = result.to_qiskit(target=setup.target)

        output_circuit_properties(
            result, backend.two_q_gate_type, benchmark, target=setup.target
        )
        assert circuit_validator(result, backend, target=setup.target)
