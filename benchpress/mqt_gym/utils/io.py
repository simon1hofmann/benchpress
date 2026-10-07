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
"""I/O and compile helpers for the MQT gym."""

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from mqt.core.mlir import (
    CompilerTarget,
    OutputFormat,
    PayloadFormat,
    PayloadSpecification,
    ProgramCapability,
    QCOProgram,
    QCProgram,
    TargetEnvironment,
    compile_program,
)
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler import CouplingMap, Target

from benchpress.config import Configuration


def load_qasm_as_qc_program(qasm_file=None, *, qasm_str=None) -> QCProgram:
    """Load OpenQASM 2 or 3 through MQT Core's typed frontend."""
    if (qasm_file is None) == (qasm_str is None):
        raise ValueError("Provide exactly one of qasm_file or qasm_str")
    if qasm_file is not None:
        return QCProgram.from_openqasm_file(qasm_file)
    return QCProgram.from_openqasm_str(qasm_str)


def program_uses_classical_control(program) -> bool:
    """Whether a parsed program contains structured classical control flow."""
    if isinstance(program, QuantumCircuit):
        return any(isinstance(item.operation, ControlFlowOp) for item in program.data)
    return program.inspect().has_control_flow


def mqt_to_qiskit_circuit(program, *, target=None) -> QuantumCircuit:
    """Convert an MLIR program through MQT Core's native Qiskit exporter.

    Parameters:
        program: ``QCProgram`` or ``QCOProgram``
        target: optional ``CompilerTarget`` used to emit a canonical physical
            circuit after target compilation

    Returns:
        Converted ``QuantumCircuit``.

    Raises:
        TypeError: if ``program`` is not a QC/QCO program.
        RuntimeError: if native conversion fails.
    """
    if not isinstance(program, (QCProgram, QCOProgram)):
        raise TypeError(
            f"mqt_to_qiskit_circuit expects QCProgram or QCOProgram, got {type(program)!r}"
        )

    try:
        return program.to_qiskit(target=target)
    except Exception as exc:
        conversion = "target-aware Qiskit" if target is not None else "Qiskit"
        raise RuntimeError(f"MQT {conversion} conversion failed: {exc!r}") from exc


def mqt_qasm_loader(qasm_file, benchmark):
    """Load OpenQASM 2 or 3 into a QCProgram via the typed frontend."""
    start = perf_counter()
    program = load_qasm_as_qc_program(qasm_file)
    stop = perf_counter()
    benchmark.extra_info["qasm_load_time"] = stop - start
    mqt_input_circuit_properties(program, benchmark)
    return program


def mqt_hamiltonian_circuit(sparse_op, label=None, evo_time=1):
    """Keep Hamiltonian evolution opaque until import inside timed compilation."""
    qc = QuantumCircuit(sparse_op.num_qubits)
    qc.append(
        PauliEvolutionGate(sparse_op, time=evo_time, label=label),
        qargs=range(sparse_op.num_qubits),
    )
    return qc


def program_num_qubits(program) -> int:
    """Return declared quantum capacity; reject unknown widths."""
    if isinstance(program, QuantumCircuit):
        return program.num_qubits
    width = program.inspect().num_qubits
    if width is None:
        raise ValueError("MQT program has unknown quantum capacity")
    return width


def program_op_counts(program) -> dict:
    """Count QC/QCO operations using Core's full-module histogram."""
    counts = Counter()
    for full_name, count in program.operation_counts().items():
        dialect, _, name = full_name.partition(".")
        if dialect in ("qc", "qco") and name not in (
            "static",
            "alloc",
            "qubit",
            "yield",
            "return",
        ):
            counts[name] += count
    return dict(counts)


def mqt_input_circuit_properties(circuit, benchmark):
    benchmark.extra_info["input_num_qubits"] = program_num_qubits(circuit)
    benchmark.extra_info["input_has_control_flow"] = program_uses_classical_control(
        circuit
    )


def mqt_output_circuit_properties(circuit, two_qubit_gate, benchmark, *, target=None):
    """Record native output metrics, counting each control-flow block once."""
    qc = (
        circuit
        if isinstance(circuit, QuantumCircuit)
        else mqt_to_qiskit_circuit(circuit, target=target)
    )
    operations = Counter()
    has_control_flow = False

    def count(block):
        nonlocal has_control_flow
        operations.update(block.count_ops())
        for instruction in block.data:
            blocks = getattr(instruction.operation, "blocks", ())
            if blocks:
                has_control_flow = True
                for child in blocks:
                    count(child)

    count(qc)
    exclude_comparison = has_control_flow or benchmark.extra_info.get(
        "input_has_control_flow", False
    )
    benchmark.extra_info["output_num_qubits"] = qc.num_qubits
    benchmark.extra_info["output_circuit_operations"] = dict(operations)
    gate_count = operations.get(two_qubit_gate, 0)
    benchmark.extra_info["output_gate_count_2q"] = (
        None if exclude_comparison else gate_count
    )
    benchmark.extra_info["output_depth_2q"] = (
        None
        if exclude_comparison
        else qc.depth(filter_function=lambda x: x.operation.name == two_qubit_gate)
    )
    if exclude_comparison:
        # Other gyms do not consistently count nested blocks. Keep the raw
        # count separate so standard cross-tool metrics fail closed.
        benchmark.extra_info["output_static_gate_count_2q"] = gate_count
        benchmark.extra_info["control_flow_metrics"] = (
            "static_counts_all_blocks_not_cross_tool_comparable"
        )


def _basis_gate_names(backend=None):
    """Resolve the explicit basis, or let Core select usable backend operations."""
    gates = Configuration.options.get("mqt", {}).get("native_gates")
    if gates is None:
        if backend is not None:
            return None
        return list(
            Configuration.options.get("general", {}).get(
                "basis_gates", ["sx", "x", "rz", "cz"]
            )
        )
    gates = (
        [gate.strip() for gate in gates.split(",") if gate.strip()]
        if isinstance(gates, str)
        else list(gates)
    )
    if backend is not None:
        gates += [
            gate for gate in ("measure", "reset") if gate in backend.operation_names
        ]
    return gates


def make_compiler_target(num_qubits, edges, basis_gates=None, name=None):
    """Describe an abstract Qiskit target and let Core convert its contract."""
    num_qubits = int(num_qubits)
    if edges is not None:
        edges = [(int(source), int(target)) for source, target in edges]
        # CouplingMap drops self-loops and grows its width for out-of-range sites.
        for source, target in edges:
            if source == target:
                raise ValueError(f"Coupling edge ({source}, {target}) is a self-loop")
            if not 0 <= source < num_qubits or not 0 <= target < num_qubits:
                raise ValueError(
                    f"Coupling edge ({source}, {target}) is outside "
                    f"the {num_qubits}-site target"
                )
    gates = list(basis_gates) if basis_gates is not None else _basis_gate_names()
    target = Target.from_configuration(
        basis_gates=list(dict.fromkeys([*gates, "measure", "reset"])),
        num_qubits=num_qubits,
        coupling_map=None if edges is None else CouplingMap(list(edges)),
    )
    return CompilerTarget.from_qiskit(target, name=name)


def _compiler_target(program, backend):
    """Resolve and validate the immutable target used by timed compilation."""
    logical_qubits = (
        program_num_qubits(program)
        if isinstance(program, (QCProgram, QCOProgram, QuantumCircuit))
        else 0
    )
    if isinstance(backend, CompilerTarget):
        target = backend
        if logical_qubits > target.num_sites:
            raise ValueError(
                f"Circuit has {logical_qubits} qubits, but target has "
                f"{target.num_sites}"
            )
        return target

    num_qubits = int(backend.num_qubits)
    if logical_qubits > num_qubits:
        raise ValueError(
            f"Circuit has {logical_qubits} qubits, but backend has {num_qubits}"
        )
    return CompilerTarget.from_qiskit(
        backend, operation_names=_basis_gate_names(backend)
    )


@dataclass(frozen=True)
class PreparedMQTCompile:
    """Validated input and target for repeated compilation.

    The caller must not mutate or consume ``program`` while this setup is in
    use. Each compilation copies the input; lowering remains inside the timer.
    """

    program: object
    environment: TargetEnvironment
    normalize_global_phases: bool

    @property
    def target(self):
        return self.environment.target

    def compile(self, *, copy=True):
        """Compile without repeating validation; copy the input by default."""
        if (
            isinstance(self.program, (QCProgram, QCOProgram))
            and not self.program.is_valid
        ):
            raise ValueError("Cannot compile a consumed MQT program")
        qco = _to_qco(self.program, copy=copy)
        if self.normalize_global_phases:
            qco.normalize_global_phases()
        qco.compile_for_target(self.environment)
        return qco


def prepare_mqt_compile(program, backend) -> PreparedMQTCompile:
    """Validate input and prepare target metadata outside a benchmark timer."""
    if isinstance(program, (QCProgram, QCOProgram)) and not program.is_valid:
        raise ValueError("Cannot compile a consumed MQT program")
    target = _compiler_target(program, backend)
    # Describe the output format, not the backend's dynamic execution support.
    # Qiskit export outside the timer checks actual output compatibility.
    environment = TargetEnvironment(
        target,
        PayloadSpecification(
            PayloadFormat("openqasm", "3.0"),
            capabilities=[
                ProgramCapability(capability)
                for capability in (
                    ProgramCapability.FORWARD_BRANCHING,
                    ProgramCapability.COUNTED_ITERATION,
                    ProgramCapability.CONDITIONAL_LOOP,
                    ProgramCapability.MULTIWAY_BRANCHING,
                )
            ],
        ),
    )
    normalize_phases = Configuration.options.get("mqt", {}).get(
        "normalize_global_phases", False
    )
    return PreparedMQTCompile(program, environment, normalize_phases)


def _to_qco(program, copy=True):
    """Lower a fresh copy inside each timed compilation."""
    if isinstance(program, QCOProgram):
        return program.copy() if copy else program
    if isinstance(program, QCProgram):
        return program.to_qco(copy=copy)
    if isinstance(program, QuantumCircuit) and copy:
        # Import may materialize lazy gate definitions. Do not cache synthesis
        # on the original input across benchmark rounds.
        program = program.copy()
    return compile_program(program, output=OutputFormat.QCO)
