# MQT Core × Benchpress integration gap report

**Review date:** 2026-08-17

**Newest upstream `main` and pinned baseline:**
`b401a064c7d2e5668cedf8aaeb185ed971b9de19` (merge of
[#2133](https://github.com/munich-quantum-toolkit/core/pull/2133), including
[#2118](https://github.com/munich-quantum-toolkit/core/pull/2118))

**Runtime-tested source tree:** the pinned `b401a064` tree
(`57d47d1b2acce7773f4da8984ce02c69ab42a6af`)

**Last observed collection:** 1,066 MQT cases, exactly matching Qiskit's 1,066;
the per-file workout counts also match exactly

Core #2118 and #2133 have landed. `requirements-mqt.txt` pins their combined,
immutable `main` revision. The final #2118 API preserves unobserved quantum
operations during target compilation by default and exposes dead-gate removal
as the explicit `remove-dead-gates` pass; it does not accept the earlier
`preserve_unobserved_quantum_operations` keyword. Core #2133 makes Qiskit export
target-aware: mapped output must call `to_qiskit(target=target)` with the
corresponding `CompilerTarget` to preserve the device qubit count.

## Executive summary

The merged source tree reproduces the same workout surface and per-file counts
as Qiskit. Focused construction, manipulation, abstract transpilation, device
transpilation, validation, metrics, backend-basis selection, target-aware
export, explicit dead-gate removal, and timeout handling pass. The historical
bounded Feynman sweep has not yet been repeated on the merged baseline.

Core #2118 resolves the largest comparability gap without synthetic terminal
measurements: target compilation now retains unmeasured and partially measured
unitary work by default. If a workflow intentionally wants dead-code
elimination, it must explicitly run the `remove-dead-gates` pass and accept that
unobserved work may be erased.

## Merged Core integration prerequisites

| Core change | Final behavior | Required Benchpress use |
| --- | --- | --- |
| [#2118](https://github.com/munich-quantum-toolkit/core/pull/2118) | Target compilation preserves unobserved quantum operations by default; dead-gate elimination is the explicit `remove-dead-gates` pass | Do not pass the removed preservation keyword. Only run `remove-dead-gates` when DCE is explicitly requested. |
| [#2133](https://github.com/munich-quantum-toolkit/core/pull/2133) | `QCProgram.to_qiskit` accepts a `CompilerTarget` and creates Qiskit's canonical full-width physical circuit | Retain the target used for compilation and pass it as `to_qiskit(target=target)` for mapped validation and metrics. |

Both adapter contracts are covered by the current integration regressions.

## Remaining MQT Core capabilities needed

| Priority | Missing capability or release step | Benchpress impact | Current handling |
| --- | --- | --- | --- |
| P1 | Represent directional two-qubit constraints in `CompilerTarget` and honor them during mapping/synthesis ([#2132](https://github.com/munich-quantum-toolkit/core/issues/2132)) | A directional CX/ECR target cannot be modeled safely because couplings and routing are currently undirected | Default `fake_torino`/CZ is supported. Directional device configurations skip all 162 device cases rather than risk invalid output. This needs a target/mapping design change, not a one-line adapter relaxation. |
| P1 | Describe and compile classical control-flow capabilities through a target ([#2131](https://github.com/munich-quantum-toolkit/core/issues/2131)) | Nine QASMBench inputs across four topologies (36 cases) and five Feynman inputs use QASM2 classical control | The source pre-check and an IR-level guard for `scf.*`, `cf.*`, `qco.if`, and `qco.index_switch` skip before native target compilation. QASM3 parsing remains supported. Without the guard, some loop/dynamic-index inputs abort in the native pipeline. |
| P1 | Preserve free parameters through target compilation | Device `circSU2_89` and `circSU2_100` cannot compile the same symbolic workload as the other gyms | Both cases skip explicitly. This is tracked upstream in [#2067](https://github.com/munich-quantum-toolkit/core/issues/2067). |
| P1 | Import or represent arbitrary two-qubit unitaries for Quantum Volume | Native QV construction and QV100 device transpilation cannot use the shared workload faithfully | Both cases skip explicitly. This is tracked upstream in [#2068](https://github.com/munich-quantum-toolkit/core/issues/2068). |
| P1 | Export classical execution and result-bearing control flow | The bigint import benchmark succeeds, but its result-bearing `scf.if` cannot be exported for output metrics | Benchpress omits metrics only for that import case. Measured straight-line programs use the guarded dense-target fallback; dynamic-control target compilation remains skipped. |
| P2 | Expose Pauli twirling through the MLIR API | The DTC100 twirling workout has no equivalent operation | One manipulation case skips explicitly. |
| P2 | Improve target-mapping and final-liveness performance on HWB inputs ([#2134](https://github.com/munich-quantum-toolkit/core/issues/2134)) | `hwb10`, `hwb11`, and `hwb12` exceed a 10-second bound | These are timeout/performance results, not correctness failures. Historical profiling found both routing and final dead-value/liveness cleanup bottlenecks; #2134 tracks a production-representative rerun. |

## Expected skip inventory on the default backend

The last bounded run on the pre-merge candidate had 68 expected skips among
1,066 cases for the default `fake_torino` configuration:

- 41 MQT control-flow skips: 36 QASMBench cases and five Feynman cases.
- Four construction/device feature skips: QV construction, QV100 device
  transpilation, and two symbolic circSU2 cases.
- One Pauli-twirling skip.
- 22 backend-capacity skips shared in concept with the other gyms: 19 HamLib
  and three Feynman inputs wider than 133 qubits.

The first 46 are MQT capability gaps. The final 22 are workload/backend-size
constraints, not compiler defects.

## Integration defects corrected during this review

- Switched QASM loading to `QCProgram.from_qasm_file` / `from_qasm_str`; QASM2
  bigint/QV and QASM3 loop/global-phase inputs now parse through the typed
  frontend.
- Derived target operations from `backend.operation_names` rather than the
  global default basis.
- Rejected unsupported directional CX/ECR targets and circuits wider than a
  backend; supported a missing coupling map as all-to-all.
- Built immutable compiler targets outside timed benchmark closures.
- Added a fail-closed IR guard for unsupported control flow and dynamic-index
  assertions before entering the native target pipeline.
- Removed fail-open textual validation and approximate depth metrics; conversion
  and validation now fail closed.
- Preferred native MQT-to-Qiskit conversion. Measured mapped programs still
  require sanitized OpenQASM 3 because Core's native exporter rejects classical
  execution; for dense Benchpress targets the fallback is rebuilt as a
  canonical physical circuit with the exact target qubit count.
- Adapted MQT IR circuit construction to Core main's supported
  MQT-to-Qiskit-to-typed-MLIR bridge after the legacy direct conversion was
  removed, while explicitly retaining the MQT circuit's global phase.
- Fixed the hard-timeout pipe deadlock by draining large results before joining
  the worker.
- Matched the shared MCX ladder, DTC layer reuse, and circSU2 binding values used
  by the other gyms.
- Kept all integration/unit checks outside `benchpress/mqt_gym`, preserving the
  benchmark collection.

## Verification evidence

Current focused evidence uses a build of the exact pinned `b401a064` source
tree. The PR head used for the local build and the squash-merge commit have the
same Git tree (`57d47d1b2acce7773f4da8984ce02c69ab42a6af`).

- Combined focused run: **47 passed, 2 expected skips**. This includes all
  **35/35** MQT integration regressions, construction and manipulation, and one
  representative abstract and device transpilation case.
- Target-aware native and measured-fallback exports both report the exact target
  width, one canonical `q` register, and no Qiskit layout metadata.
- Collection: **1,066 MQT** and **1,066 Qiskit** cases with identical per-file
  counts (400 + 492 + 8 + 53 + 100 + 9 + 4).
- Python lint/format checks and `git diff --check` pass for the integration
  files.

Historical bounded evidence from the pre-merge #2118 candidate:

- Complete bounded Feynman sweep: **42 passed, 8 expected skips, 3 ten-second
  timeouts**; `hwb10` separately passed in **53.80 seconds** with a 60-second
  bound.

This was a bounded functional review, not a serial run of every expensive
HamLib and QASMBench case. Before publishing benchmark results, build the exact
pinned revision on the target machine, generate the normal timeout skip list,
and run all 1,066 MQT cases without the internal `MQT_COMPILE_TIMEOUT` timing
wrapper.
