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

import os
import re
from collections import Counter
from dataclasses import dataclass
from multiprocessing import get_context
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
from qiskit.circuit.library import PauliEvolutionGate

from benchpress.config import Configuration
from benchpress.mqt_gym.utils.mqt_backend_utils import (
    coupling_edges,
    operation_site_tuples,
)

_STATIC_QUBIT_RE = re.compile(r"qco\.static\s+(\d+)\s*:")
# Prefer alloc sites so load/store type annotations are not double-counted.
_MEMREF_ALLOC_QUBIT_RE = re.compile(r"memref\.alloc\(\)[^\n]*memref<(\d+)x!qc\.qubit>")
_QTENSOR_ALLOC_QUBIT_RE = re.compile(
    r"qtensor\.alloc\b[^\n]*:\s*tensor<(\d+)x!qco\.qubit>"
)
_SCALAR_ALLOC_QUBIT_RE = re.compile(
    r"^\s*%[-\w.$]+\s*=\s*(?:qc|qco)\.alloc\b", flags=re.MULTILINE
)
_MEMREF_QUBIT_RE = re.compile(r"memref<(\d+)x!qc\.qubit>")
_QTENSOR_QUBIT_RE = re.compile(r"tensor<(\d+)x!qco\.qubit>")
_TARGET_UNSUPPORTED_CONTROL_FLOW_RE = re.compile(
    r"^\s*(?:%[^=\n]+\s*=\s*)?"
    r"(?:(?:scf|cf)\.[A-Za-z_]\w*|qco\.(?:if|index_switch))\b",
    flags=re.MULTILINE,
)
_STRUCTURED_CONTROL_FLOW_OPERATION_RE = re.compile(
    r"^\s*(?:%[^=\n]+\s*=\s*)?"
    r"(?P<dialect>scf|cf)\.(?P<name>[A-Za-z_]\w*)\b",
    flags=re.MULTILINE,
)
_CBIT_ALLOC_RE = re.compile(
    r"^\s*(?P<result>%[-\w.$]+)\s*=\s*cbit\.alloc\b[^\n]*"
    r":\s*!cbit\.reg<(?P<width>\d+)>",
    flags=re.MULTILINE,
)
_INTEGER_CONSTANT_RE = re.compile(
    r"^\s*(?P<result>%[-\w.$]+)\s*=\s*arith\.constant\s+"
    r"(?P<value>true|false|-?\d+)(?:\s*:\s*i(?P<width>\d+))?\s*$",
    flags=re.MULTILINE,
)
_CBIT_READ_RE = re.compile(
    r"^\s*(?P<result>%[-\w.$]+)\s*=\s*cbit\.read\s+"
    r"(?P<register>%[-\w.$]+)\s*:\s*!cbit\.reg<(?P<register_width>\d+)>"
    r"\s*->\s*i(?P<result_width>\d+)\s*$",
    flags=re.MULTILINE,
)
_ARITH_COMPARISON_RE = re.compile(
    r"^\s*(?P<result>%[-\w.$]+)\s*=\s*arith\.cmpi\s+"
    r"(?P<predicate>\w+)\s*,\s*(?P<lhs>%[-\w.$]+)\s*,\s*"
    r"(?P<rhs>%[-\w.$]+)\s*:\s*i(?P<width>\d+)\s*$",
    flags=re.MULTILINE,
)
_REGISTER_CONDITION_RE = re.compile(
    r"^\s*(?:%[^=\n]+\s*=\s*)?(?:scf|qco)\.if\s+"
    r"(?P<condition>%[-\w.$]+)\b",
    flags=re.MULTILINE,
)
_INDEX_CONSTANT_RE = re.compile(
    r"^\s*(?P<result>%[-\w.$]+)\s*=\s*arith\.constant\s+"
    r"(?P<value>-?\d+)\s*:\s*index\b",
    flags=re.MULTILINE,
)
_CBIT_STORE_DESTINATION_RE = re.compile(
    r"^\s*cbit\.store\s+(?P<source>%[-\w.$]+)\s*,\s*"
    r"(?P<register>%[-\w.$]+)"
    r"\[(?P<index>%[-\w.$]+)\]",
    flags=re.MULTILINE,
)
# (num_qubits, num_parameters) for CompilerTarget.OperationCapability construction.
_TARGET_GATE_SPECS = {
    "gphase": (0, 1),
    "u": (1, 3),
    "u1": (1, 1),
    "u2": (1, 2),
    "u3": (1, 3),
    "p": (1, 1),
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "h": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "sx": (1, 0),
    "sxdg": (1, 0),
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "r": (1, 2),
    "cx": (2, 0),
    "cnot": (2, 0),
    "cz": (2, 0),
    "cy": (2, 0),
    "ecr": (2, 0),
    "swap": (2, 0),
    "iswap": (2, 0),
    "rxx": (2, 1),
    "ryy": (2, 1),
    "rzz": (2, 1),
    "rzx": (2, 1),
    "measure": (1, 0),
    "reset": (1, 0),
}


class UnsupportedTargetControlFlowError(NotImplementedError):
    """Target compilation cannot safely preserve this classical control flow."""


def load_qasm_as_qc_program(qasm_file=None, *, qasm_str=None) -> QCProgram:
    """Load OpenQASM 2 or 3 through MQT Core's typed frontend."""
    if (qasm_file is None) == (qasm_str is None):
        raise ValueError("Provide exactly one of qasm_file or qasm_str")
    if qasm_file is not None:
        return QCProgram.from_openqasm_file(qasm_file)
    return QCProgram.from_openqasm_str(qasm_str)


def program_uses_classical_control(program) -> bool:
    """Whether a parsed program contains structured classical control flow."""
    return _TARGET_UNSUPPORTED_CONTROL_FLOW_RE.search(program.ir) is not None


def target_unsupported_control_flow_reason(
    program, *, verified_register_feed_forward=False
) -> str | None:
    """Explain why a program cannot safely enter target compilation.

    Core checks structural payload capabilities, but not every classical form
    supported by native Qiskit export. Callers may opt an exact benchmark profile
    into the direct register feed-forward path after it has been verified end to
    end on the pinned Core revision.
    """
    if not isinstance(program, (QCProgram, QCOProgram)):
        return None
    ir = program.ir
    if _TARGET_UNSUPPORTED_CONTROL_FLOW_RE.search(ir) is not None and not (
        verified_register_feed_forward and _supports_verified_register_feed_forward(ir)
    ):
        return (
            "MQT benchmark export path does not support this classical control flow "
            "or dynamic-index control-flow lowering"
        )
    return None


def _classical_register_identities(ir: str):
    """Map CBit SSA values to stable source register identities."""
    registers = {}
    for ordinal, match in enumerate(_CBIT_ALLOC_RE.finditer(ir)):
        name_match = re.search(r'mqt\.register_name\s*=\s*"([^"]+)"', match.group(0))
        name = name_match.group(1) if name_match else f"#{ordinal}"
        registers[match.group("result")] = (name, int(match.group("width")))
    return registers


def _supports_verified_register_feed_forward(ir: str) -> bool:
    """Recognize direct classical register feed-forward supported by Core."""
    comparisons = _comparison_signatures(ir)
    if not comparisons or any(
        register == ("unknown", -1) or register[1] != width
        for register, _predicate, _rhs, width, _epoch in comparisons.values()
    ):
        return False
    conditions = [
        match.group("condition") for match in _REGISTER_CONDITION_RE.finditer(ir)
    ]
    if not conditions or any(condition not in comparisons for condition in conditions):
        return False

    index_constants = {
        match.group("result"): int(match.group("value"))
        for match in _INDEX_CONSTANT_RE.finditer(ir)
    }
    store_destinations = []
    for match in _CBIT_STORE_DESTINATION_RE.finditer(ir):
        index = index_constants.get(match.group("index"))
        if index is None:
            return False
        store_destinations.append((match.group("register"), index))
    if not store_destinations or len(store_destinations) != len(
        set(store_destinations)
    ):
        return False

    if re.search(r"^\s*(?:%[^=\n]+\s*=\s*)?qco\.index_switch\b", ir, re.MULTILINE):
        return False
    for match in _STRUCTURED_CONTROL_FLOW_OPERATION_RE.finditer(ir):
        dialect = match.group("dialect")
        operation = match.group("name")
        if dialect != "scf" or operation not in {"if", "yield"}:
            return False
    return True


def _comparison_signatures(ir: str):
    """Map comparison SSA values to stable register comparison facts."""
    registers = _classical_register_identities(ir)
    store_epochs = Counter()
    constants = {}
    reads = {}
    comparisons = {}
    for line in ir.splitlines():
        if store := _CBIT_STORE_DESTINATION_RE.match(line):
            store_epochs[registers.get(store.group("register"), ("unknown", -1))] += 1
        if constant := _INTEGER_CONSTANT_RE.match(line):
            value = constant.group("value")
            constants[constant.group("result")] = (
                value == "true" if value in {"true", "false"} else int(value),
                1 if constant.group("width") is None else int(constant.group("width")),
            )
        if read := _CBIT_READ_RE.match(line):
            register = registers.get(read.group("register"), ("unknown", -1))
            register_width = int(read.group("register_width"))
            result_width = int(read.group("result_width"))
            reads[read.group("result")] = (
                register,
                result_width,
                store_epochs[register],
                register_width == result_width,
            )
            if register_width == result_width == 1:
                comparisons[read.group("result")] = (
                    register,
                    "eq",
                    True,
                    1,
                    store_epochs[register],
                )
        if comparison := _ARITH_COMPARISON_RE.match(line):
            read = reads.get(comparison.group("lhs"))
            constant = constants.get(comparison.group("rhs"))
            width = int(comparison.group("width"))
            if (
                read is None
                or constant is None
                or not read[3]
                or read[1] != width
                or constant[1] != width
            ):
                continue
            comparisons[comparison.group("result")] = (
                read[0],
                comparison.group("predicate"),
                constant[0],
                width,
                read[2],
            )
    return comparisons


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
    benchmark.extra_info["input_num_qubits"] = program_num_qubits(program)
    return program


def mqt_hamiltonian_circuit(sparse_op, label=None, evo_time=1):
    """Build a Trotterized Hamiltonian circuit as a QCProgram."""
    qc = QuantumCircuit(sparse_op.num_qubits)
    qc.append(
        PauliEvolutionGate(sparse_op, time=evo_time, label=label),
        qargs=range(sparse_op.num_qubits),
    )
    return QCProgram.from_qiskit(qc)


def program_num_qubits(program) -> int:
    """Best-effort qubit count from an MLIR program.

    Sum QC ``memref.alloc`` and QCO ``qtensor.alloc`` register sizes, plus
    scalar ``qc.alloc`` and ``qco.alloc`` qubits.
    Fall back to ``qco.static`` indices after target compilation.
    """
    ir = program.ir
    allocs = [
        int(match.group(1))
        for pattern in (_MEMREF_ALLOC_QUBIT_RE, _QTENSOR_ALLOC_QUBIT_RE)
        for match in pattern.finditer(ir)
    ]
    scalar_qubits = len(_SCALAR_ALLOC_QUBIT_RE.findall(ir))
    if allocs or scalar_qubits:
        return sum(allocs) + scalar_qubits
    # Fallback if alloc sites are not present in the textual dump.
    memrefs = [int(m.group(1)) for m in _MEMREF_QUBIT_RE.finditer(ir)]
    if memrefs:
        return max(memrefs)
    qtensors = [int(m.group(1)) for m in _QTENSOR_QUBIT_RE.finditer(ir)]
    if qtensors:
        return max(qtensors)
    statics = {int(m.group(1)) for m in _STATIC_QUBIT_RE.finditer(ir)}
    if statics:
        return max(statics) + 1
    m = re.search(r"qubit\[(\d+)\]", ir)
    if m:
        return int(m.group(1))
    return 0


def program_op_counts(program) -> dict:
    """Count simple op names appearing in textual MLIR."""
    ir = program.ir
    counts = Counter()
    for match in re.finditer(r"\b(?:qco|qc)\.([A-Za-z_][A-Za-z0-9_]*)\b", ir):
        name = match.group(1)
        if name in ("static", "alloc", "qubit", "ctrl", "yield", "return"):
            if name == "ctrl":
                counts["ctrl"] += 1
            continue
        counts[name] += 1
    return dict(counts)


def mqt_input_circuit_properties(circuit, benchmark):
    benchmark.extra_info["input_num_qubits"] = program_num_qubits(circuit)


def mqt_output_circuit_properties(circuit, two_qubit_gate, benchmark, *, target=None):
    """Record Qiskit-equivalent output metrics for an MQT program."""
    qc = (
        circuit
        if isinstance(circuit, QuantumCircuit)
        else mqt_to_qiskit_circuit(circuit, target=target)
    )
    benchmark.extra_info["output_num_qubits"] = qc.num_qubits
    benchmark.extra_info["output_circuit_operations"] = qc.count_ops()
    if two_qubit_gate == "2Q_GATE":

        def _is_twoq(inst):
            return getattr(inst.operation, "num_qubits", len(inst.qubits)) == 2

        benchmark.extra_info["output_gate_count_2q"] = sum(
            1 for inst in qc.data if _is_twoq(inst)
        )
        benchmark.extra_info["output_depth_2q"] = qc.depth(
            filter_function=lambda inst: _is_twoq(inst)
        )
    else:
        name = str(two_qubit_gate)
        benchmark.extra_info["output_gate_count_2q"] = qc.count_ops().get(name, 0)
        benchmark.extra_info["output_depth_2q"] = qc.depth(
            filter_function=lambda inst: inst.operation.name == name
        )


def _basis_gate_names(backend=None):
    """Resolve native gates from an override, backend, or abstract defaults."""
    opts = Configuration.options.get("mqt", {})
    if "native_gates" in opts:
        gates = opts["native_gates"]
        if isinstance(gates, str):
            return [g.strip() for g in gates.split(",") if g.strip()]
        return list(gates)
    if backend is not None:
        basis = list(getattr(backend, "operation_names", ()))
    else:
        basis = list(
            Configuration.options.get("general", {}).get(
                "basis_gates", ["sx", "x", "rz", "cz"]
            )
        )
    gates = [g for g in basis if g in _TARGET_GATE_SPECS and g not in ("id", "delay")]
    if backend is not None and not gates:
        raise ValueError("Backend exposes no MQT-supported native gates")
    return gates or ["sx", "x", "rz", "cz"]


def make_compiler_target(
    num_qubits, edges, basis_gates=None, name=None, *, backend=None
):
    """Build a dense ``CompilerTarget`` for Benchpress backends."""
    num_sites = int(num_qubits)
    gates = list(basis_gates) if basis_gates is not None else _basis_gate_names()
    unknown_gates = sorted(set(gates) - _TARGET_GATE_SPECS.keys() - {"id", "delay"})
    if unknown_gates:
        raise ValueError(f"Unsupported MQT native gates: {unknown_gates}")
    gates = [gate for gate in gates if gate in _TARGET_GATE_SPECS]
    backend_operations = (
        None if backend is None else set(getattr(backend, "operation_names", ()))
    )
    if backend_operations is not None:
        unavailable_gates = sorted(set(gates) - backend_operations - {"gphase"})
        if unavailable_gates:
            raise ValueError(
                "Backend does not expose requested MQT native gates: "
                f"{unavailable_gates}"
            )
    operations = []
    seen = set()
    for gate in [*gates, "gphase", "measure", "reset"]:
        if (
            backend_operations is not None
            and gate in {"measure", "reset"}
            and gate not in backend_operations
        ):
            continue
        spec = _TARGET_GATE_SPECS.get(gate)
        if spec is None or gate in seen:
            continue
        arity = CompilerTarget.OperationArity.fixed(0) if spec[0] == 0 else spec[0]
        sites = (
            operation_site_tuples(backend, gate, spec[0])
            if backend is not None
            else None
        )
        if sites is not None and not sites:
            continue
        operations.append(
            CompilerTarget.OperationCapability(
                gate,
                arity,
                spec[1],
                site_tuples=sites,
            )
        )
        seen.add(gate)
    couplings = None
    if edges is not None:
        normalized_couplings = set()
        for source, target in edges:
            source = int(source)
            target = int(target)
            if source == target:
                raise ValueError(f"Coupling edge ({source}, {target}) is a self-loop")
            if not 0 <= source < num_sites or not 0 <= target < num_sites:
                raise ValueError(
                    f"Coupling edge ({source}, {target}) is outside "
                    f"the {num_sites}-site target"
                )
            normalized_couplings.add((min(source, target), max(source, target)))
        couplings = sorted(normalized_couplings)
    connectivity = (
        CompilerTarget.Connectivity.all_to_all()
        if couplings is None or len(couplings) == num_sites * (num_sites - 1) // 2
        else CompilerTarget.Connectivity(couplings)
    )
    native_operations = CompilerTarget.NativeOperations(operations)
    if name is not None:
        return CompilerTarget(
            str(name),
            num_sites,
            connectivity=connectivity,
            native_operations=native_operations,
        )
    return CompilerTarget(
        num_sites,
        connectivity=connectivity,
        native_operations=native_operations,
    )


def _mqt_options():
    opts = Configuration.options.get("mqt", {})
    return {
        "do_normalize_phases": opts.get("normalize_global_phases", False),
    }


def _compiler_target(program, backend_or_edges):
    """Resolve and validate the immutable target used by timed compilation."""
    logical_qubits = (
        program_num_qubits(program)
        if isinstance(program, (QCProgram, QCOProgram))
        else 0
    )
    if isinstance(backend_or_edges, CompilerTarget):
        target = backend_or_edges
        if logical_qubits > target.num_sites:
            raise ValueError(
                f"Circuit has {logical_qubits} qubits, but target has "
                f"{target.num_sites}"
            )
        return target

    is_backend = hasattr(backend_or_edges, "num_qubits")
    if is_backend:
        edges = coupling_edges(backend_or_edges)
        num_qubits = int(backend_or_edges.num_qubits)
        if logical_qubits > num_qubits:
            raise ValueError(
                f"Circuit has {logical_qubits} qubits, but backend has {num_qubits}"
            )
        return make_compiler_target(
            num_qubits,
            edges,
            basis_gates=_basis_gate_names(backend_or_edges),
            backend=backend_or_edges,
        )

    edges = [tuple(edge) for edge in backend_or_edges]
    if not edges:
        raise ValueError("A non-empty coupling edge list is required")
    num_qubits = max(max(source, target) for source, target in edges) + 1
    num_qubits = max(num_qubits, logical_qubits)
    return make_compiler_target(num_qubits, edges)


@dataclass(frozen=True)
class PreparedMQTCompile:
    """Validated input and target for repeated compilation.

    The caller must not mutate or consume ``program`` while this setup is in
    use. Each compilation copies the input; lowering remains inside the timer.
    """

    program: object
    target: CompilerTarget
    verified_register_feed_forward: bool

    def compile(self):
        """Compile a fresh copy without repeating input and target validation."""
        return _run_mqt_compile(
            self.program,
            self.target,
            verified_register_feed_forward=self.verified_register_feed_forward,
        )


def prepare_mqt_compile(
    program, backend_or_edges, *, verified_register_feed_forward=False
) -> PreparedMQTCompile:
    """Validate input and prepare target metadata outside a benchmark timer."""
    if isinstance(program, (QCProgram, QCOProgram)) and not program.is_valid:
        raise ValueError("Cannot compile a consumed MQT program")
    target = _compiler_target(program, backend_or_edges)
    unsupported_reason = target_unsupported_control_flow_reason(
        program, verified_register_feed_forward=verified_register_feed_forward
    )
    if unsupported_reason is not None:
        raise UnsupportedTargetControlFlowError(unsupported_reason)
    return PreparedMQTCompile(program, target, verified_register_feed_forward)


def _mqt_compile_body(
    program,
    target,
    copy=True,
    opts=None,
    *,
    verified_register_feed_forward=False,
):
    """Lower validated input to QCO and run ``compile_for_target``."""
    opts = opts or _mqt_options()
    do_normalize_phases = opts["do_normalize_phases"]

    if isinstance(program, QCOProgram):
        qco = program.copy() if copy else program
    elif isinstance(program, QCProgram):
        qco = program.to_qco(copy=copy)
    else:
        qco = compile_program(program, output=OutputFormat.QCO)

    # Lowering can introduce control flow even when the input passed validation.
    unsupported_reason = target_unsupported_control_flow_reason(
        qco, verified_register_feed_forward=verified_register_feed_forward
    )
    if unsupported_reason is not None:
        raise UnsupportedTargetControlFlowError(unsupported_reason)

    # Keep phase normalization outside Core's target pipeline and off by default
    # so the optional preprocessing remains an explicit configuration choice.
    if do_normalize_phases:
        qco.normalize_global_phases()

    # This models the validated export path, not backend execution support.
    environment = TargetEnvironment(
        target,
        PayloadSpecification(
            PayloadFormat("openqasm", "3.0"),
            capabilities=(
                [ProgramCapability(ProgramCapability.FORWARD_BRANCHING)]
                if verified_register_feed_forward
                else []
            ),
        ),
    )
    qco.compile_for_target(environment)
    return qco


def _target_spec(target):
    """Serialize a dense Benchpress target for a timeout worker."""
    if tuple(int(site.id) for site in target.sites) != tuple(
        range(target.num_sites)
    ) or any(
        site.name is not None or site.t1 is not None or site.t2 is not None
        for site in target.sites
    ):
        raise ValueError(
            "MQT_COMPILE_TIMEOUT supports only homogeneous dense CompilerTarget values"
        )
    duration_unit = target.duration_unit
    return {
        "name": target.name,
        "num_sites": target.num_sites,
        "edges": (
            target.couplings
            if target.connectivity_kind == CompilerTarget.ConnectivityKind.EXPLICIT
            else None
        ),
        "operations": (
            [
                {
                    "name": operation.name,
                    "arity": operation.arity.value,
                    "variadic": (
                        operation.arity.kind
                        == CompilerTarget.OperationArityKind.VARIADIC
                    ),
                    "num_parameters": operation.num_parameters,
                    "site_tuples": [
                        {
                            "sites": tuple(int(site) for site in site_tuple.sites),
                            "duration": site_tuple.duration,
                            "fidelity": site_tuple.fidelity,
                        }
                        for site_tuple in operation.site_tuples
                    ],
                    "duration": operation.duration,
                    "fidelity": operation.fidelity,
                }
                for operation in target.operations
            ]
            if target.native_operations_kind
            == CompilerTarget.NativeOperationsKind.EXPLICIT
            else None
        ),
        "duration_unit": (
            None
            if duration_unit is None
            else (duration_unit.unit, duration_unit.scale_factor)
        ),
    }


def _target_from_spec(target_spec):
    """Reconstruct a compiler target serialized by ``_target_spec``."""
    operation_specs = target_spec["operations"]
    operations = None
    if operation_specs is not None:
        operations = []
        for operation in operation_specs:
            arity = operation["arity"]
            if operation["variadic"]:
                arity = CompilerTarget.OperationArity.variadic(arity)
            elif arity == 0:
                arity = CompilerTarget.OperationArity.fixed(0)
            site_tuples = [
                CompilerTarget.SiteTuple(
                    site_tuple["sites"],
                    site_tuple["duration"],
                    site_tuple["fidelity"],
                )
                for site_tuple in operation["site_tuples"]
            ]
            operations.append(
                CompilerTarget.OperationCapability(
                    operation["name"],
                    arity,
                    operation["num_parameters"],
                    site_tuples=site_tuples,
                    duration=operation["duration"],
                    fidelity=operation["fidelity"],
                )
            )

    connectivity = (
        CompilerTarget.Connectivity.all_to_all()
        if target_spec["edges"] is None
        else CompilerTarget.Connectivity(target_spec["edges"])
    )
    native_operations = (
        CompilerTarget.NativeOperations.unrestricted()
        if operations is None
        else CompilerTarget.NativeOperations(operations)
    )
    duration_unit_spec = target_spec["duration_unit"]
    duration_unit = (
        None
        if duration_unit_spec is None
        else CompilerTarget.DurationUnit(*duration_unit_spec)
    )
    target_args = (
        (target_spec["name"], target_spec["num_sites"])
        if target_spec["name"] is not None
        else (target_spec["num_sites"],)
    )
    return CompilerTarget(
        *target_args,
        connectivity=connectivity,
        native_operations=native_operations,
        duration_unit=duration_unit,
    )


def _mqt_compile_worker(
    conn, kind, mlir_text, target_spec, opts, verified_register_feed_forward
):
    """Child-process worker for hard compile timeouts (pickle-safe args only)."""
    try:
        if kind == "qco":
            program = QCOProgram.from_mlir_str(mlir_text)
        else:
            program = QCProgram.from_mlir_str(mlir_text)
        target = _target_from_spec(target_spec)
        result = _mqt_compile_body(
            program,
            target,
            copy=False,
            opts=opts,
            verified_register_feed_forward=verified_register_feed_forward,
        )
        conn.send(("ok", result.ir))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - surfaced to parent
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def mqt_compile(
    program,
    backend_or_edges,
    copy=True,
    *,
    verified_register_feed_forward=False,
):
    """Compile a program for a coupling graph via ``compile_for_target``.

    Parameters:
        program: QCProgram (or compatible) input
        backend_or_edges: BackendV2 / FlexibleBackend, ``CompilerTarget``, or
            iterable of edges. The optional hard-timeout transport supports
            homogeneous dense compiler targets.
        copy: whether to copy the QC program before lowering
        verified_register_feed_forward: whether this exact benchmark/target
            profile is approved for direct register feed-forward compilation

    Returns:
        QCOProgram after target compilation (map + native synthesis)

    Timing note:
        The default path is **in-process**. Timing includes QC-to-QCO lowering,
        ``compile_for_target``, and the adapter's safety checks. Benchmarks use
        ``prepare_mqt_compile(...).compile()`` to keep input validation and
        immutable backend-to-target setup outside the timer.

        ``MQT_COMPILE_TIMEOUT`` (seconds) enables a **forked** hard-kill for
        native hangs (``pytest --timeout`` cannot interrupt MLIR C++). That
        fork adds ~10 ms per call and **must not** be set during fair
        timing runs. Apply any per-test watchdog outside the measured worker
        instead.
    """
    setup = prepare_mqt_compile(
        program,
        backend_or_edges,
        verified_register_feed_forward=verified_register_feed_forward,
    )
    return _run_mqt_compile(
        setup.program,
        setup.target,
        copy=copy,
        verified_register_feed_forward=setup.verified_register_feed_forward,
    )


def _run_mqt_compile(
    program, target, copy=True, *, verified_register_feed_forward=False
):
    """Execute prepared compilation in process or with a hard timeout."""
    if isinstance(program, (QCProgram, QCOProgram)) and not program.is_valid:
        raise ValueError("Cannot compile a consumed MQT program")
    opts = _mqt_options()

    timeout = float(os.environ.get("MQT_COMPILE_TIMEOUT", "0") or "0")
    if timeout <= 0 or not isinstance(program, (QCProgram, QCOProgram)):
        return _mqt_compile_body(
            program,
            target,
            copy=copy,
            opts=opts,
            verified_register_feed_forward=verified_register_feed_forward,
        )

    kind = "qco" if isinstance(program, QCOProgram) else "qc"
    mlir_text = program.ir
    # fork is much cheaper than spawn (no interpreter relaunch); required so
    # short timeouts reflect compile time rather than process startup.
    # Not used for fair benchmarks — see docstring.
    ctx = get_context("fork" if os.name != "nt" else "spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_mqt_compile_worker,
        args=(
            child_conn,
            kind,
            mlir_text,
            _target_spec(target),
            opts,
            verified_register_feed_forward,
        ),
    )
    proc.start()
    child_conn.close()
    # Drain the pipe before joining. Joining first deadlocks once a successful
    # result is larger than the OS pipe buffer.
    if not parent_conn.poll(timeout):
        proc.kill()
        proc.join()
        parent_conn.close()
        raise TimeoutError(f"mqt_compile exceeded {timeout}s")
    status, payload = parent_conn.recv()
    parent_conn.close()
    proc.join()
    if status == "err":
        raise RuntimeError(payload)
    return QCOProgram.from_mlir_str(payload)
