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
from pathlib import Path
from time import perf_counter

from mqt.core.mlir import (
    CompilerTarget,
    OutputFormat,
    QCOProgram,
    QCProgram,
    compile_program,
)
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import PauliEvolutionGate

from benchpress.config import Configuration
from benchpress.mqt_gym.utils.mqt_backend_utils import (
    coupling_edges,
    unsupported_backend_reason,
)

_STATIC_QUBIT_RE = re.compile(r"qco\.static\s+(\d+)\s*:")
# Prefer alloc sites so load/store type annotations are not double-counted.
_MEMREF_ALLOC_QUBIT_RE = re.compile(r"memref\.alloc\(\)[^\n]*memref<(\d+)x!qc\.qubit>")
_MEMREF_QUBIT_RE = re.compile(r"memref<(\d+)x!qc\.qubit>")
_QTENSOR_QUBIT_RE = re.compile(r"tensor<(\d+)x!qco\.qubit>")
_DYNAMIC_QUBIT_ALLOCATION_RE = re.compile(
    r"\b(?:qc|qco)\.alloc\b|memref\.alloc\(\)[^\n]*!qc\.qubit"
)
_TWOQ_NAMES = ("cx", "cz", "ecr", "swap", "rzz", "cp", "crx", "cry", "crz")
_TARGET_UNSUPPORTED_CONTROL_FLOW_RE = re.compile(
    r"^\s*(?:%[^=\n]+\s*=\s*)?"
    r"(?:(?:scf|cf)\.[A-Za-z_]\w*|qco\.(?:if|index_switch))\b",
    flags=re.MULTILINE,
)
_QASM_CONTROL_FLOW_RE = re.compile(
    r"^\s*(?:if\s*\(|for\b|while\s*\(|switch\b)", flags=re.MULTILINE
)

# (num_qubits, num_parameters) for CompilerTarget.Operation construction.
_TARGET_GATE_SPECS = {
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


def load_qasm_as_qc_program(qasm_file=None, *, qasm_str=None) -> QCProgram:
    """Load OpenQASM 2 or 3 through MQT Core's typed frontend."""
    if (qasm_file is None) == (qasm_str is None):
        raise ValueError("Provide exactly one of qasm_file or qasm_str")
    if qasm_file is not None:
        return QCProgram.from_qasm_file(qasm_file)
    return QCProgram.from_qasm_str(qasm_str)


def qasm_uses_classical_control(qasm_file) -> bool:
    """Whether QASM uses control flow unsupported by target compilation."""
    source = Path(qasm_file).read_text(encoding="utf8")
    return _QASM_CONTROL_FLOW_RE.search(source) is not None


def target_unsupported_control_flow_reason(program) -> str | None:
    """Explain why a program cannot safely enter target compilation.

    Current MQT target compilation has no target-capability model for classical
    control flow and may abort natively when passed ``scf``/``cf`` operations
    or structured QCO control operations.
    Keep parsing support separate from target-compilation support and fail
    before entering the native pass pipeline.
    """
    if isinstance(program, (QCProgram, QCOProgram)) and (
        _TARGET_UNSUPPORTED_CONTROL_FLOW_RE.search(program.ir) is not None
    ):
        return (
            "MQT target compilation does not support classical control flow "
            "or dynamic-index control-flow lowering"
        )
    return None


def _sanitize_openqasm3_for_qiskit(source: str) -> str:
    """Drop output declarations/assignments unsupported by Qiskit's importer.

    Keep local measurement declarations so qubit width and measurement counts
    remain faithful to the MQT program.
    """
    lines = []
    output_names = set()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("output "):
            match = re.search(r"\b([A-Za-z_]\w*)\s*;\s*$", stripped)
            if match:
                output_names.add(match.group(1))
            continue
        if any(
            re.match(rf"{re.escape(name)}(?:\[\d+\])?\s*=", stripped)
            for name in output_names
        ):
            continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def mqt_to_qiskit_circuit(program, *, target=None) -> QuantumCircuit:
    """Convert an MLIR program to a Qiskit circuit.

    Prefer MQT Core's native conversion. Programs with classical execution are
    not supported there yet, so use sanitized OpenQASM 3 as a metrics and
    validation fallback for measured programs.

    Parameters:
        program: ``QCProgram`` or ``QCOProgram``
        target: optional ``CompilerTarget`` used to emit a canonical physical
            circuit after target compilation

    Returns:
        Converted ``QuantumCircuit``.

    Raises:
        TypeError: if ``program`` is not a QC/QCO program.
        RuntimeError: if both native and OpenQASM conversion fail.
    """
    if isinstance(program, QCOProgram):
        qc_prog = program.to_qc(copy=True)
    elif isinstance(program, QCProgram):
        qc_prog = program
    else:
        raise TypeError(
            f"mqt_to_qiskit_circuit expects QCProgram or QCOProgram, got {type(program)!r}"
        )

    if target is not None and _DYNAMIC_QUBIT_ALLOCATION_RE.search(program.ir):
        raise RuntimeError(
            "MQT target-aware Qiskit conversion requires mapped static qubits"
        )

    native_error = None
    if hasattr(qc_prog, "to_qiskit"):
        try:
            if target is None:
                return qc_prog.to_qiskit()
            return qc_prog.to_qiskit(target=target)
        except Exception as exc:  # noqa: BLE001 - fall back for classical execution
            native_error = exc

    if not hasattr(qc_prog, "to_openqasm3"):
        raise RuntimeError(
            "MQT-to-Qiskit conversion is unavailable: native conversion failed "
            f"with {native_error!r}, and QCProgram.to_openqasm3 is missing"
        )
    if target is not None and "does not support classical execution" not in str(
        native_error
    ):
        raise RuntimeError(
            f"MQT target-aware Qiskit conversion failed; native error: {native_error!r}"
        ) from native_error

    try:
        source = qc_prog.to_openqasm3().source
        circuit = qasm3.loads(_sanitize_openqasm3_for_qiskit(source))
        if target is not None:
            sites = tuple(int(site.id) for site in target.sites)
            if sites != tuple(range(target.num_qubits)):
                raise RuntimeError(
                    "OpenQASM fallback cannot canonicalize a sparse compiler target"
                )
            canonical = QuantumCircuit(target.num_qubits, circuit.num_clbits)
            canonical.compose(
                circuit,
                qubits=range(circuit.num_qubits),
                clbits=range(circuit.num_clbits),
                inplace=True,
            )
            circuit = canonical
        return circuit
    except Exception as exc:
        raise RuntimeError(
            "MQT-to-Qiskit conversion failed; "
            f"native error: {native_error!r}; OpenQASM error: {exc}"
        ) from exc


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
    qc = qc.decompose().decompose()
    return QCProgram.from_qiskit(qc)


def program_num_qubits(program) -> int:
    """Best-effort qubit count from an MLIR program.

    Sum QC-dialect ``memref.alloc`` qubit-register sizes (multi-register
    OpenQASM circuits are common). Fall back to ``qco.static`` indices after
    target compilation.
    """
    ir = program.ir
    allocs = [int(m.group(1)) for m in _MEMREF_ALLOC_QUBIT_RE.finditer(ir)]
    if allocs:
        return sum(allocs)
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


def program_twoq_count(program, two_qubit_gate=None) -> int:
    """Count two-qubit-ish ops in textual MLIR."""
    counts = program_op_counts(program)
    if two_qubit_gate and two_qubit_gate != "2Q_GATE":
        key = str(two_qubit_gate).lower()
        return counts.get(key, 0) + counts.get("ctrl", 0)
    total = counts.get("ctrl", 0)
    for name in _TWOQ_NAMES:
        total += counts.get(name, 0)
    return total


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


def make_compiler_target(num_qubits, edges, basis_gates=None, name=None):
    """Build a dense ``CompilerTarget`` for Benchpress backends."""
    gates = list(basis_gates) if basis_gates is not None else _basis_gate_names()
    operations = []
    seen = set()
    for gate in gates:
        spec = _TARGET_GATE_SPECS.get(gate)
        if spec is None or gate in seen:
            continue
        operations.append(CompilerTarget.Operation(gate, spec[0], spec[1]))
        seen.add(gate)
    for extra in ("measure", "reset"):
        if extra not in seen:
            operations.append(CompilerTarget.Operation(extra, 1, 0))
            seen.add(extra)
    couplings = (
        None
        if edges is None
        else sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges})
    )
    if name is not None:
        return CompilerTarget(
            str(name),
            int(num_qubits),
            couplings=couplings,
            operations=operations,
        )
    return CompilerTarget(
        int(num_qubits),
        couplings=couplings,
        operations=operations,
    )


def _mqt_options(backend=None):
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
        if logical_qubits > target.num_qubits:
            raise ValueError(
                f"Circuit has {logical_qubits} qubits, but target has "
                f"{target.num_qubits}"
            )
        return target

    is_backend = hasattr(backend_or_edges, "num_qubits")
    if is_backend:
        unsupported_reason = unsupported_backend_reason(backend_or_edges)
        if unsupported_reason is not None:
            raise NotImplementedError(unsupported_reason)
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
        )

    edges = [tuple(edge) for edge in backend_or_edges]
    if not edges:
        raise ValueError("A non-empty coupling edge list is required")
    num_qubits = max(max(source, target) for source, target in edges) + 1
    num_qubits = max(num_qubits, logical_qubits)
    return make_compiler_target(num_qubits, edges)


@dataclass(frozen=True)
class PreparedMQTCompile:
    """Untimed validation and target setup for one MQT compile workload."""

    program: object
    target: CompilerTarget


def prepare_mqt_compile(program, backend_or_edges) -> PreparedMQTCompile:
    """Validate input and prepare target metadata outside a benchmark timer."""
    unsupported_reason = target_unsupported_control_flow_reason(program)
    if unsupported_reason is not None:
        raise NotImplementedError(unsupported_reason)
    target = _compiler_target(program, backend_or_edges)
    return PreparedMQTCompile(program, target)


def _mqt_compile_body(program, target, copy=True, opts=None):
    """Lower to QCO and run ``compile_for_target``."""
    opts = opts or _mqt_options()
    do_normalize_phases = opts["do_normalize_phases"]

    unsupported_reason = target_unsupported_control_flow_reason(program)
    if unsupported_reason is not None:
        raise NotImplementedError(unsupported_reason)

    if isinstance(program, QCOProgram):
        qco = program.copy() if copy else program
    elif isinstance(program, QCProgram):
        qco = program.to_qco(copy=copy)
    else:
        qco = compile_program(program, output=OutputFormat.QCO)

    unsupported_reason = target_unsupported_control_flow_reason(qco)
    if unsupported_reason is not None:
        raise NotImplementedError(unsupported_reason)

    # Keep phase normalization outside Core's target pipeline and off by default
    # so this optional preprocessing remains explicit in benchmark metadata.
    if do_normalize_phases and hasattr(qco, "normalize_global_phases"):
        qco.normalize_global_phases()

    qco.compile_for_target(target)
    return qco


def _target_spec(target):
    """Serialize the homogeneous Benchpress target for a timeout worker."""
    return {
        "name": target.name,
        "num_qubits": target.num_qubits,
        "edges": target.couplings if target.has_explicit_topology else None,
        "operations": (
            [
                (
                    operation.name,
                    operation.num_qubits,
                    operation.num_parameters,
                )
                for operation in target.operations
            ]
            if target.has_explicit_operations
            else None
        ),
    }


def _mqt_compile_worker(conn, kind, mlir_text, target_spec, opts):
    """Child-process worker for hard compile timeouts (pickle-safe args only)."""
    try:
        if kind == "qco":
            program = QCOProgram.from_mlir_str(mlir_text)
        else:
            program = QCProgram.from_mlir_str(mlir_text)
        operations = target_spec["operations"]
        if operations is not None:
            operations = [
                CompilerTarget.Operation(*operation) for operation in operations
            ]
        target_args = (
            (target_spec["name"], target_spec["num_qubits"])
            if target_spec["name"] is not None
            else (target_spec["num_qubits"],)
        )
        target = CompilerTarget(
            *target_args,
            couplings=target_spec["edges"],
            operations=operations,
        )
        result = _mqt_compile_body(program, target, copy=False, opts=opts)
        conn.send(("ok", result.ir))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - surfaced to parent
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _require_nonempty_qco(program):
    """Fail closed when target compilation produced no quantum program."""
    payload = program.ir
    if "qco." not in payload and "!qco." not in payload:
        raise RuntimeError(
            "mqt_compile produced empty/non-QCO IR after target compilation"
        )
    return program


def mqt_compile(program, backend_or_edges, copy=True):
    """Compile a program for a coupling graph via ``compile_for_target``.

    Parameters:
        program: QCProgram (or compatible) input
        backend_or_edges: BackendV2 / FlexibleBackend, ``CompilerTarget``, or
            iterable of edges
        copy: whether to copy the QC program before lowering

    Returns:
        QCOProgram after target compilation (map + native synthesis)

    Timing note:
        The default path is **in-process** so pytest-benchmark / fair SDK
        comparisons measure ``compile_for_target`` itself.

        ``MQT_COMPILE_TIMEOUT`` (seconds) enables a **forked** hard-kill for
        native hangs (``pytest --timeout`` cannot interrupt MLIR C++). That
        fork adds ~10 ms per call and **must not** be set during fair
        timing runs. Prefer killing a per-test worker process instead (see
        ``scripts/qasmbench_small_compare.py``).
    """
    unsupported_reason = target_unsupported_control_flow_reason(program)
    if unsupported_reason is not None:
        raise NotImplementedError(unsupported_reason)
    target = _compiler_target(program, backend_or_edges)
    opts = _mqt_options()

    timeout = float(os.environ.get("MQT_COMPILE_TIMEOUT", "0") or "0")
    if timeout <= 0 or not isinstance(program, (QCProgram, QCOProgram)):
        result = _mqt_compile_body(program, target, copy=copy, opts=opts)
        return _require_nonempty_qco(result)

    kind = "qco" if isinstance(program, QCOProgram) else "qc"
    mlir_text = program.ir
    # fork is much cheaper than spawn (no interpreter relaunch); required so
    # short timeouts reflect compile time rather than process startup.
    # Not used for fair benchmarks — see docstring.
    ctx = get_context("fork" if os.name != "nt" else "spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_mqt_compile_worker,
        args=(child_conn, kind, mlir_text, _target_spec(target), opts),
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
    # Fail closed if target compilation produces no quantum program.
    if "qco." not in payload and "!qco." not in payload:
        raise RuntimeError(
            "mqt_compile produced empty/non-QCO IR after target compilation"
        )
    return QCOProgram.from_mlir_str(payload)
