# MQT Core Gym Design

**Date:** 2026-07-21  
**Status:** Approved for planning  
**Goal:** Add MQT Core as a first-class Benchpress SDK gym, using the PyPI `mqt.core` MLIR Python API end-to-end wherever practical, including device-aware transpile.

## Background

Benchpress benchmarks quantum SDKs via per-SDK `*_gym` packages that implement shared workouts (`construct`, `manipulate`, `abstract_transpile`, `device_transpile`). Unsupported workouts remain skipped by default.

MQT Core (`mqt.core` on PyPI) provides:

- An IR (`mqt.core.ir.QuantumComputation`) for circuit construction
- An MLIR compiler surface (`mqt.core.mlir`) with typed programs (`QCProgram`, `QCOProgram`), `compile_program`, optimization helpers, and `QCOProgram.place_and_route(coupling=...)` for device mapping

## Decisions

| Topic | Choice |
| --- | --- |
| Scope | Full gym (all workout categories; implement what APIs support, skip the rest) |
| Install | Normal PyPI: `pip install mqt.core` |
| Transpile | Device-aware via MLIR `place_and_route` against Benchpress backends/topologies |
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

**Circuit type:** `mqt.core.mlir.QCProgram` / `QCOProgram`. Construct helpers may briefly use `QuantumComputation` or QASM only as a builder, then convert immediately with `QCProgram.from_quantum_computation` / `from_qasm_*` so timed work and downstream APIs stay on MLIR.

**Integration points:** wire `"mqt"` into:

- `benchpress/config.py` (backend allow-list)
- `benchpress/utilities/backends/backend_utils.py`
- `benchpress/utilities/io/{qasm_loader,circuit_input,circuit_output,hamiltonians}.py`
- `benchpress/utilities/validation/validation.py`

**Deps / docs:**

- `requirements-mqt.txt` — `mqt.core`, Qiskit + `qiskit-ibm-runtime` for backend/HamLib plumbing (same role as in qpanda/staq), and the existing pytest-benchmark skiplist pin
- `[mqt]` section in `default.conf`
- README supported-SDKs entry for MQT Core

Qiskit is plumbing only (fake/runtime backends, Pauli evolution → QASM), not the gym’s primary circuit API.

## Components and data flow

### Construct

- Helpers in `circuits/` build gates (via `QuantumComputation` or QASM), then convert to `QCProgram` before returning.
- Timed work measures that build/conversion path.
- Benchpress QASM corpora are largely OpenQASM 2. Loader strategy: prefer `QuantumComputation.from_qasm` / `from_qasm_str` (full QASM2 support) then `QCProgram.from_quantum_computation`; use `QCProgram.from_qasm_*` only when the input is already QASM3-compatible. Timed import tests measure the full load→MLIR path.
- Implement every construct workout MLIR/IR can support; leave unsupported ones skipped.

### Manipulate

- Load/build → `QCProgram` → `to_qco()` → apply matching QCO operations (`fuse_single_qubit_unitary_runs`, `merge_single_qubit_rotation_gates`, `run_pass_pipeline`, etc.).
- Twirling or other ops with no MLIR equivalent stay skipped.

### Abstract + device transpile

```
QASM/HamLib → (QASM2 via IR if needed) → QCProgram → to_qco()
  → optimize (default or config pipeline)
  → place_and_route(coupling=backend edges) → result program
```

- Device backends come from `default.conf` (`backend_name`, e.g. `fake_torino`) via Qiskit fake/runtime backends, exposing `coupling_map`, `num_qubits`, and `two_q_gate_type`.
- Abstract topologies build coupling edges from `FlexibleBackend` / configured topologies, then use the same `place_and_route` path.
- Skip when circuit width exceeds backend qubit count (existing Benchpress pattern).

### Utils

- `mqt_qasm_loader`, Hamiltonian loader, and input/output property extractors (op counts / depth from the program or its textual `ir` as needed).
- `mqt_circuit_validation`: after place-and-route, best-effort check that two-qubit ops respect the coupling map.
- `[mqt]` config knobs (e.g. QCO pipeline name, `place_and_route` seed / lookahead).

## Error handling and test status

Follow existing Benchpress semantics:

| Status | When |
| --- | --- |
| Skipped | No MLIR equivalent for the workout, or circuit too large for backend |
| Failed | API exists but compile / place-and-route errors or timeout |
| XFAIL | Only for known irrecoverable crashes (e.g. OOM), not “not implemented” |

- Unrecognized backend name → clear `ValueError` from the backend helper.
- If PyPI `mqt.core` lacks an expected MLIR symbol at import time, fail fast in gym `conftest` / utils with an explicit message (do not silently skip the whole gym).

## Verification

- Smoke: run a small construct subset (e.g. QASM import) after `pip install -r requirements-mqt.txt`.
- Spot-check one small device transpile (Feynman QASM) against `fake_torino` coupling.
- Match existing gym formatting (Black) and Apache copyright headers.
- No new CI matrix required in this change unless the repo already runs per-gym CI; follow the same convention as qpanda/staq.

## In scope

- Full `mqt_gym` skeleton and workout subclasses
- Utilities wiring for gym name `"mqt"`
- `requirements-mqt.txt`, `default.conf` `[mqt]`, README entry
- Implement construct / manipulate / transpile tests where PyPI MLIR APIs clearly support them

## Out of scope

- Depending on MQT QMAP as a separate package
- Custom LLVM/MLIR from-source builds
- Changing shared workouts or other gyms’ behavior
- Publishing benchmark result numbers

## Success criteria

1. `python -m pytest benchpress/mqt_gym` discovers the full workout surface.
2. At least construct QASM import and one device-aware transpile path run successfully against PyPI `mqt.core`.
3. Unsupported workouts remain explicitly skipped (not failed).
4. README lists MQT Core among supported SDKs with install via `requirements-mqt.txt`.
