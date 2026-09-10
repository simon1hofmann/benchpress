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

from pathlib import Path

import pytest

from benchpress.mqt_gym.utils.io import (
    UnsupportedTargetControlFlowError,
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

_VERIFIED_REGISTER_FEED_FORWARD = {
    (filename, topology)
    for filename in (
        "inverseqft_n4.qasm",
        "ipea_n2.qasm",
        "qec_sm_n5.qasm",
        "shor_n5.qasm",
        "cc_n12.qasm",
        "cc_n32.qasm",
        "cc_n64.qasm",
        "cc_n151.qasm",
        "cc_n301.qasm",
    )
    for topology in ("all-to-all", "square", "heavy-hex", "linear")
}


def _verified_register_feed_forward(circ_and_topo):
    """Whether this exact benchmark/topology passed the pinned Core matrix."""
    # Structural payload checks do not establish native exporter compatibility.
    return (
        Path(circ_and_topo[0]).name,
        circ_and_topo[1],
    ) in _VERIFIED_REGISTER_FEED_FORWARD


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
        try:
            setup = prepare_mqt_compile(
                prog,
                backend,
                verified_register_feed_forward=_verified_register_feed_forward(
                    circ_and_topo
                ),
            )
        except UnsupportedTargetControlFlowError as exc:
            pytest.skip(str(exc))

        @benchmark
        def result():
            return setup.compile()

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
        try:
            setup = prepare_mqt_compile(
                prog,
                backend,
                verified_register_feed_forward=_verified_register_feed_forward(
                    circ_and_topo
                ),
            )
        except UnsupportedTargetControlFlowError as exc:
            pytest.skip(str(exc))

        @benchmark
        def result():
            return setup.compile()

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
        try:
            setup = prepare_mqt_compile(
                prog,
                backend,
                verified_register_feed_forward=_verified_register_feed_forward(
                    circ_and_topo
                ),
            )
        except UnsupportedTargetControlFlowError as exc:
            pytest.skip(str(exc))

        @benchmark
        def result():
            return setup.compile()

        output_circuit_properties(
            result, backend.two_q_gate_type, benchmark, target=setup.target
        )
        assert circuit_validator(result, backend, target=setup.target)
