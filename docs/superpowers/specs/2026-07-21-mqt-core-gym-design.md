# MQT Core Gym Design

**Date:** 2026-07-21
**Status:** Implemented, source-reviewed, and focused-runtime-tested against MQT
Core `main` at `6a928c868871e1fd0706778edbc68aa00703cc49` on 2026-09-10.
**Goal:** Add MQT Core as a first-class Benchpress SDK gym, using the newest `mqt.core` MLIR Python API end-to-end wherever practical, including device-aware transpile.

## Background

Benchpress benchmarks quantum SDKs via per-SDK `*_gym` packages that implement shared workouts (`construct`, `manipulate`, `abstract_transpile`, `device_transpile`). Unsupported workouts remain skipped by default.

MQT Core provides an MLIR compiler surface (`mqt.core.mlir`) with typed programs
(`QCProgram`, `QCOProgram`), `CompilerTarget`, `compile_program`, optimization
helpers, and `QCOProgram.compile_for_target(...)` for target-aware mapping and
native synthesis. Core v4 removes the former `mqt.core.ir` circuit surface.

## Decisions

| Topic | Choice |
| --- | --- |
| Scope | Full gym (all workout categories; implement what APIs support, skip the rest) |
| Install | Exact compatible Git revision (or matching MLIR-enabled source build): PyPI wheels do not ship `mqt.core.mlir` yet |
| Transpile | Device-aware via `TargetEnvironment` containing a `CompilerTarget` and selected payload |
| Representation | MLIR programs as the gym’s native circuit type |
| Approach | MLIR-native gym (not IR-primary, not Qiskit-plugin wrapper) |
| Package / gym name | `benchpress/mqt_gym/`, `Configuration.gym_name = "mqt"` |

## Architecture

```
mqt_gym/
  __init__.py          # Configuration.gym_name = "mqt"
  conftest.py
  circuits/            # helpers that produce MLIR programs
  construct/
  manipulate/
  abstract_transpile/
  device_transpile/
  utils/               # io, backend, validation
```

**Circuit type:** `mqt.core.mlir.QCProgram` / `QCOProgram`. Construct helpers use
Qiskit circuits as the supported builder frontend, then immediately import them
through `QCProgram.from_qiskit`; QASM inputs use `QCProgram.from_qasm_*`.

**Integration points:** wire `"mqt"` into:

- `benchpress/config.py` (backend allow-list)
- `benchpress/utilities/backends/backend_utils.py`
- `benchpress/utilities/io/{qasm_loader,circuit_input,circuit_output,hamiltonians}.py`
- `benchpress/utilities/validation/validation.py`

**Deps / docs:**

- `requirements-mqt.txt` — install a pinned MQT Core revision containing the required MLIR APIs; Qiskit + `qiskit-ibm-runtime` for backend/HamLib plumbing; existing pytest-benchmark skiplist pin
- Document that a system/portable MLIR (LLVM 23.1+) install is required for the source build (`MLIR_DIR` / `setup-mlir`), matching MQT Core’s install docs
- `[mqt]` section in `default.conf`
- README supported-SDKs entry for MQT Core, noting the MLIR-enabled install path

Qiskit supplies construction/import and backend plumbing, while QC/QCO programs
remain the gym's native stored and compiled representation.

**Install note (reviewed 2026-09-06):** Latest stable PyPI `mqt.core` (3.10.0)
does not include `mqt.core.mlir`. The gym therefore depends on the exact
MLIR-enabled revision pinned in `requirements-mqt.txt` until compatible wheels
ship.

## Components and data flow

### Construct

- Helpers in `circuits/` build Qiskit circuits, then import them into a
  `QCProgram` before returning.
- Timed work measures that build/conversion path.
- Benchpress QASM corpora are largely OpenQASM 2. The newest typed frontend, `QCProgram.from_qasm_file` / `from_qasm_str`, handles both OpenQASM 2 and 3 directly. Timed import tests measure that source→MLIR path without adding operations to the input.
- Implement every construct workout MLIR/IR can support; leave unsupported ones skipped.

### Manipulate

- Load/build → `QCProgram` → `to_qco()` → apply matching QCO operations (`fuse_single_qubit_unitary_runs`, `merge_single_qubit_rotation_gates`, `run_pass_pipeline`, etc.).
- Pauli twirling uses Core's `pauli-twirl-2q-gates` pass; operations without an
  MLIR equivalent stay skipped.

### Abstract + device transpile

```
QASM/HamLib → QCProgram → untimed compatibility/target preparation → to_qco()
  → compile_for_target(TargetEnvironment) → result program
```

- Device backends come from `default.conf` (`backend_name`, e.g. `fake_torino`) via Qiskit fake/runtime backends, exposing `operation_names`, `coupling_map`, `num_qubits`, and `two_q_gate_type`.
- Abstract topologies build a `CompilerTarget` from `FlexibleBackend`; device targets use each backend's native operations and topology.
- `CompilerTarget` is immutable and prepared outside the benchmark closure, matching other gyms' pass-manager/model setup.
- The shared compile helper wraps it in a `TargetEnvironment` with an OpenQASM
  3 payload specification. Verified register-feed-forward profiles explicitly
  enable `forward-branching`; other optional capabilities remain unknown. The result
  remains QCO, and native Qiskit export stays outside timing. The environment
  metadata enables Core's structural control-flow checks but does not replace
  the integration's exporter eligibility checks.
- Target compilation preserves unobserved quantum operations by default, retaining unitary work without injecting measurements. Core's former preservation keyword is not part of the final API. A workflow that opts into DCE must explicitly run the `remove-dead-gates` pass and accept that unobserved work can be erased and performance/output can differ.
- Retain the `CompilerTarget` used for mapping and pass it to mapped native export as `QCProgram.to_qiskit(target=target)`. This produces Qiskit's canonical full-width physical circuit for validation and metrics.
- General `scf`/`cf` and QCO control flow is rejected before native target
  compilation. Direct whole-register comparisons use an explicit
  benchmark/topology allowlist plus structural checks for the canonical
  `cbit.read` + same-width `arith.constant` + `arith.cmpi` form; the integration
  also recognizes its direct one-bit `cbit.read` canonicalization. Input and
  lowered-IR checks restrict eligible control forms; deterministic regressions
  test mapped behavior without comparing textual operation order. This direct
  form supports widths above 64 bits.
- Ordered Qiskit target qargs are stored as `CompilerTarget` operation site
  tuples. Connectivity remains undirected for routing, while Core selects and
  validates the supported gate direction.
- Skip when circuit width exceeds backend qubit count (existing Benchpress pattern).

### Utils

- `mqt_qasm_loader`, Hamiltonian loader, and input/output property extractors (op counts / depth from the program or its textual `ir` as needed).
- `mqt_circuit_validation` / output metrics: target-aware native MQT→Qiskit
  conversion for mapped programs. Core handles safely delayed measurement
  destinations; export errors fail the workout without a Benchpress OpenQASM
  rewrite (see `2026-08-05-mqt-openqasm-validation-design.md`).
- `[mqt]` config knobs for an optional native-gate override and phase
  normalization. Target compilation preserves unobserved work by default; DCE
  is available through Core's explicit `remove-dead-gates` pass, not a
  Benchpress compile option.

## Error handling and test status

Follow existing Benchpress semantics:

| Status | When |
| --- | --- |
| Skipped | No MLIR equivalent, unsupported target control, or circuit too large for backend |
| Failed | API exists but target compilation, validation, export, or timeout fails |
| XFAIL | Only for known irrecoverable crashes (e.g. OOM), not “not implemented” |

- Unrecognized backend name → clear `ValueError` from the backend helper.
- If PyPI `mqt.core` lacks an expected MLIR symbol at import time, fail fast in gym `conftest` / utils with an explicit message (do not silently skip the whole gym).

## Verification

- Smoke: run a small construct subset (e.g. QASM import) after `pip install -r requirements-mqt.txt`.
- Spot-check one small device transpile (Feynman QASM) against `fake_torino` coupling and one abstract Hamiltonian.
- Match existing gym formatting (Black) and Apache copyright headers.
- No new CI matrix required in this change unless the repo already runs per-gym CI; follow the same convention as qpanda/staq.

## In scope

- Full `mqt_gym` skeleton and workout subclasses
- Utilities wiring for gym name `"mqt"`
- `requirements-mqt.txt`, `default.conf` `[mqt]`, README entry
- Implement construct / manipulate / transpile tests where the MLIR-enabled source build clearly supports them

## Out of scope

- Depending on MQT QMAP as a separate package
- Custom LLVM/MLIR from-source builds
- Changing shared workouts or other gyms’ behavior
- Publishing benchmark result numbers

## Success criteria

1. `python -m pytest benchpress/mqt_gym` discovers the full workout surface.
2. At least construct QASM import and one device-aware transpile path run successfully against the pinned compatible MLIR-enabled revision.
3. Unsupported workouts remain explicitly skipped (not failed).
4. README lists MQT Core among supported SDKs with install via `requirements-mqt.txt`.
