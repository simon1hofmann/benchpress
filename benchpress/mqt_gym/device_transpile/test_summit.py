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
"""Test summit benchmarks"""

import pytest
from qiskit.circuit.library import QuantumVolume

from benchpress.config import Configuration
from benchpress.mqt_gym.circuits import (
    mqt_bv_all_ones,
    mqt_circSU2,
    to_qc_program,
    trivial_bvlike_circuit,
)
from benchpress.mqt_gym.utils.io import (
    prepare_mqt_compile,
    program_num_qubits,
)
from benchpress.utilities.io import (
    input_circuit_properties,
    output_circuit_properties,
    qasm_circuit_loader,
)
from benchpress.utilities.validation import circuit_validator
from benchpress.workouts.device_transpile import WorkoutDeviceTranspile100Q
from benchpress.workouts.validation import benchpress_test_validation

BACKEND = Configuration.backend()
TWO_Q_GATE = BACKEND.two_q_gate_type


def _skip_if_too_large(prog):
    if program_num_qubits(prog) > BACKEND.num_qubits:
        pytest.skip("Circuit too large for given backend.")


def _prepare(prog):
    return prepare_mqt_compile(prog, BACKEND)


def _transpile_circsu2(width, benchmark):
    # ponytail: bind outside timing until Core can export synthesized atan2.
    prog = mqt_circSU2(width, 3, seed=12345)
    input_circuit_properties(prog, benchmark)
    _skip_if_too_large(prog)
    setup = _prepare(prog)

    @benchmark
    def result():
        return setup.compile()

    output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
    assert circuit_validator(result, BACKEND, target=setup.target)


@benchpress_test_validation
class TestWorkoutDeviceTranspile100Q(WorkoutDeviceTranspile100Q):
    def test_QFT_100_transpile(self, benchmark):
        prog = qasm_circuit_loader(
            Configuration.get_qasm_dir("qft") + "qft_N100.qasm", benchmark
        )
        _skip_if_too_large(prog)
        setup = _prepare(prog)

        @benchmark
        def result():
            return setup.compile()

        output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
        assert circuit_validator(result, BACKEND, target=setup.target)

    def test_QV_100_transpile(self, benchmark):
        prog = to_qc_program(QuantumVolume(100, 100, seed=12345))
        input_circuit_properties(prog, benchmark)
        _skip_if_too_large(prog)
        setup = _prepare(prog)

        @benchmark
        def result():
            return setup.compile()

        output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
        assert circuit_validator(result, BACKEND, target=setup.target)

    def test_circSU2_89_transpile(self, benchmark):
        _transpile_circsu2(89, benchmark)

    def test_circSU2_100_transpile(self, benchmark):
        _transpile_circsu2(100, benchmark)

    def test_BV_100_transpile(self, benchmark):
        prog = mqt_bv_all_ones(100)
        input_circuit_properties(prog, benchmark)
        _skip_if_too_large(prog)
        setup = _prepare(prog)

        @benchmark
        def result():
            return setup.compile()

        output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
        assert circuit_validator(result, BACKEND, target=setup.target)

    def test_square_heisenberg_100_transpile(self, benchmark):
        prog = qasm_circuit_loader(
            Configuration.get_qasm_dir("square-heisenberg")
            + "square_heisenberg_N100.qasm",
            benchmark,
        )
        _skip_if_too_large(prog)
        setup = _prepare(prog)

        @benchmark
        def result():
            return setup.compile()

        output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
        assert circuit_validator(result, BACKEND, target=setup.target)

    def test_QAOA_100_transpile(self, benchmark):
        prog = qasm_circuit_loader(
            Configuration.get_qasm_dir("qaoa") + "qaoa_barabasi_albert_N100_3reps.qasm",
            benchmark,
        )
        _skip_if_too_large(prog)
        setup = _prepare(prog)

        @benchmark
        def result():
            return setup.compile()

        output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
        assert circuit_validator(result, BACKEND, target=setup.target)

    def test_BVlike_simplification_transpile(self, benchmark):
        prog = trivial_bvlike_circuit(100)
        input_circuit_properties(prog, benchmark)
        _skip_if_too_large(prog)
        setup = _prepare(prog)

        @benchmark
        def result():
            return setup.compile()

        output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
        assert circuit_validator(result, BACKEND, target=setup.target)

    def test_clifford_100_transpile(self, benchmark):
        prog = qasm_circuit_loader(
            Configuration.get_qasm_dir("clifford") + "clifford_100_12345.qasm",
            benchmark,
        )
        _skip_if_too_large(prog)
        setup = _prepare(prog)

        @benchmark
        def result():
            return setup.compile()

        output_circuit_properties(result, TWO_Q_GATE, benchmark, target=setup.target)
        assert circuit_validator(result, BACKEND, target=setup.target)
