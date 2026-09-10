# MQT Gym: Native Export for Validation and Metrics

**Date:** 2026-08-05
**Status:** Implemented, source-reviewed, and focused-runtime-tested against MQT
Core `main` at `6a928c868871e1fd0706778edbc68aa00703cc49` on 2026-09-10.
**Goal:** Replace fragile QCO-IR regex validation and IR text scraping for
output metrics with native MQT→Qiskit conversion, so gate-set/topology checks
and reported metrics match other gyms. Core owns measurement scheduling and
classical snapshot preservation; Benchpress does not rewrite exported stores.

## Background

MQT Core now exposes OpenQASM 3 export on the MLIR surface:

- `QCProgram.to_openqasm3() → OpenQASMProgram` (`.source`)
- `QCOProgram.to_qc()` then the same export path

Before this integration, Benchpress `mqt_gym`:

- Validates mapped circuits by regex over textual QCO IR (`qco.static` / `qco.ctrl`), skipping when SSA→device indices cannot be recovered
- Records output metrics via regex op counts and an approximate “2Q depth” (sum of 2Q-ish op counts), not true circuit depth

That diverges from `qiskit_gym` (basis + coupling on a real `QuantumCircuit`; `count_ops` / filtered `depth`) and breaks whenever IR shape changes.

## Decisions

| Topic | Choice |
| --- | --- |
| Compile result type | Unchanged: `mqt_compile` returns `QCOProgram` |
| Validation | Native conversion, then basis + coupling checks (qiskit-gym style) |
| Output metrics | Same exported circuit → same fields as `qiskit_output_circuit_properties` |
| Regex IR validation | Remove; no silent fallback |
| Shared conversion | One helper used by validation and metrics; mapped output also receives its `CompilerTarget` |
| Prerequisite | Benchpress must run against the pinned MQT Core baseline, which provides target-aware `to_qiskit` and handles delayed stores |
| Native export failures | Fail with the original diagnostic; no OpenQASM recovery path |

## Architecture

```
QCOProgram / QCProgram  (timed compile still ends on QCOProgram)
        │
        ▼
  mqt_to_qiskit_circuit(program, target=None)
        │  QCO: to_qc(copy=True)
        │  QCProgram.to_qiskit(target=target)
        ▼
  QuantumCircuit
        ├── mqt_circuit_validation(circuit, backend, target=target)
        │     ops ⊆ backend.operation_names ∪ {barrier}
        │     each backend.two_q_gate_type edge ∈ coupling_map
        └── mqt_output_circuit_properties(..., target=target)
              output_num_qubits, output_circuit_operations,
              output_gate_count_2q, output_depth_2q
```

Tests still pass the MLIR program into `circuit_validation` /
`output_circuit_properties`; conversion happens inside the MQT gym helpers.
Mapped call sites must additionally retain and forward the same
`CompilerTarget` used for compilation. Unmapped construction/manipulation paths
may omit it.

## Components

### `mqt_to_qiskit_circuit(program, *, target=None)` (helper in `mqt_gym/utils/io.py`)

1. If `QCOProgram`: `qc = program.to_qc(copy=True)` (do not mutate the stored result).
2. Else if `QCProgram`: use `program` as-is.
3. Else: raise `TypeError` with a clear message.
4. Use `qc.to_qiskit(target=target)`. For a mapped program, `target` is the
   `CompilerTarget` used for compilation; Core then creates Qiskit's canonical
   full-width physical circuit.
5. Let the native exporter reject target-aware conversion when dynamic qubit
   allocations remain; do not duplicate that check over textual MLIR.
6. On native export failure, retain the exception as the cause and include its
   diagnostic. Do not rewrite OpenQASM or fall back to IR regex.

Double export in one test (validate then metrics) is acceptable; compile cost dominates. Caching is out of scope.

### `mqt_circuit_validation(circuit, backend, *, target=None)`

- Convert via `mqt_to_qiskit_circuit(..., target=target)` when `circuit` is still
  an MLIR program. Mapped callers must supply their compilation target.
- Mirror `qiskit_circuit_validation`:
  - Reject ops outside `backend.operation_names` (always allow `barrier`).
  - If the coupling map is not all-to-all, require every instruction named `backend.two_q_gate_type` to use an edge present in `backend.coupling_map`.
- Remove `_CTRL_BLOCK_RE` / `_STATIC_RE` and the “skip if no SSA map” behavior. Failed export or illegal gates/edges fail the run.

Validation always takes a BackendV2-compatible object (`operation_names`, `coupling_map`, `two_q_gate_type`), including abstract-transpile `FlexibleBackend` instances. Do not accept a bare edge list; callers that currently pass edges must wrap them in `FlexibleBackend` (overview script included).

### `mqt_output_circuit_properties(circuit, two_qubit_gate, benchmark, *, target=None)`

- Convert via `mqt_to_qiskit_circuit(..., target=target)`. Mapped callers must
  supply their compilation target so `output_num_qubits` reports the target
  device width.
- Fill the same `benchmark.extra_info` keys as `qiskit_output_circuit_properties`:
  - `output_num_qubits`
  - `output_circuit_operations` (`count_ops()`)
  - `output_gate_count_2q` / `output_depth_2q` using `two_qubit_gate` as the filter name

**`two_qubit_gate == "2Q_GATE"`:** many mqt tests pass this placeholder. Keep compatibility by counting/depth-filtering all two-qubit instructions (ops with two qubit args), not a literal gate name `"2Q_GATE"`. When a real name (e.g. `cz`, `ecr`) is passed, match qiskit semantics exactly (that name only).

### What stays IR-based (for now)

- `program_num_qubits` for skip-if-too-wide and input sizing (construct / pre-compile paths).
- `mqt_input_circuit_properties` may keep IR qubit count unless a later change switches it to export.

### Docs / overview scripts

- Update any overview script that relied on soft regex validation to expect hard failures on illegal topology/basis.
- Amend the mqt-core-gym design note that validation is “best-effort IR” once this lands (short pointer in that doc or in this spec’s status only—no large rewrite required).

## Error handling

| Case | Behavior |
| --- | --- |
| Stale or incompatible wheel | Fail; do not silently skip the gym |
| Native export error | Fail the workout and retain the underlying diagnostic |
| Gate outside basis / bad 2Q edge | Raise like qiskit gym (`Exception` with message) |

## Out of scope

- Changing `mqt_compile` to return OpenQASM or `QuantumCircuit`
- Timing export inside the compile benchmark (export is post-compile bookkeeping)
- Equivalence / fidelity checks beyond basis + topology
- Reworking call sites to stop passing `"2Q_GATE"` (compat handled in properties)

## Success criteria

1. Device and abstract transpile workouts validate with qiskit-style basis + coupling on the exported circuit.
2. Reported output metrics come from that circuit (true 2Q depth when a real gate name is used).
3. No QCO IR regex remains in `mqt_gym/utils/validation.py`.
4. A small smoke path (unit or scripted): compile a tiny mapped circuit → export → validate → properties without error against a known backend/topology.

## Dependencies

- Local/CI MQT Core must use the exact snapshot pinned in
  `requirements-mqt.txt`. It includes OpenQASM 3 export and target-aware
  `QCProgram.to_qiskit(target=target)`.
