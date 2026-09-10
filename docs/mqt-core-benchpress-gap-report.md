# MQT Core × Benchpress: main refresh and historical v4 review

## Integration review before fork update (2026-09-10)

Reviewed the adapter against Core `main` at
`b24dd0ca7bb4421fc5456b1131a90780b3fe4e82`. The only commit after the tested
`6a928c868871e1fd0706778edbc68aa00703cc49` pin adopts LLVM-style documentation
comments; it does not change the Python API used here. Keep the verified
Release wheel and source pin. No unmerged Core PR is installed.

### Fixes from this review

- Count scalar `qc.alloc` and `qco.alloc` qubits alongside register allocations.
  Two OpenQASM 3 scalar qubits previously reported zero; a scalar plus a
  two-qubit register reported two instead of three. Extend the existing
  QC/QCO width and target-capacity regression to cover both forms.
- Use native target-aware Qiskit export for the two remaining manipulation
  gate counts. Remove the approximate helper that counted every `ctrl` region
  as a two-qubit gate regardless of its width or gate. Export remains outside
  the benchmark timer, as in the other MQT output metrics.

### Remaining comparison and API boundaries

- Construction and binding still measure Qiskit work plus Core import, not
  native Core Python builders. Device `circSU2` is numeric while Qiskit's is
  symbolic; these are not input-equivalent timing comparisons.
- Basis-change workloads use the full Core target pipeline, while Qiskit uses
  its translation stage. Report that difference rather than interpreting equal
  benchmark names as equal optimization work.
- Width inspection still depends on printed IR. A typed logical-width query
  on QC/QCO would remove this adapter logic; it needs defined behavior for
  dynamic sizes, helper functions, and physical-site spans.
- The composed Python target-compilation API does not expose mapping seed or
  trial-count overrides. [#2500](https://github.com/munich-quantum-toolkit/core/pull/2500)
  changes the native default, but does not add those Python options. Do not
  silently install it or the identity-layout branch for a main-based run.
- Backend adaptation still maintains gate metadata and ordered placements;
  it does not transfer Qiskit calibration data. Native inbound target
  conversion remains useful Core API work.
- MQT and Qiskit's workout validators/metrics remain top-level checks, not
  recursive branch validation or dynamic execution counts. Their agreement
  does not establish semantic equivalence. The separate native control-flow
  matrix below uses stronger checks.

### Verification of the reviewed integration

| Check | Result |
| --- | --- |
| Integration regressions | 91 passed |
| QASMBench-small | 168 passed |
| Construction and manipulation | 12 passed |
| Feynman register-feed-forward smoke tests | 5 passed |
| One HamLib input on four abstract topologies and FakeTorino | 5 passed |
| MQT versus Qiskit collection | 1,066 each; collected case names match |
| Ruff lint, formatting, and diff whitespace | Passed |

All 281 executed checks passed. Benchmark repetitions were disabled. The
271-check main set used in-process compilation (`MQT_COMPILE_TIMEOUT=0`);
the ten device/Hamiltonian smoke checks used a 60-second hard compile limit.
This is not a full-suite rerun or a new timing comparison. Reproduce the main
set with the pinned environment:

```sh
MQT_COMPILE_TIMEOUT=0 python -m pytest \
  tests/mqt_gym/test_integration.py \
  benchpress/mqt_gym/abstract_transpile/test_qasmbench.py \
  benchpress/mqt_gym/construct benchpress/mqt_gym/manipulate \
  -k 'not QASMBench_medium and not QASMBench_large' \
  --benchmark-disable --timeout=0 --timeout-skip-list=0
```

## Current main refresh and API review (2026-09-10)

The integration and `.venv-mqt` now use upstream `main` at
`6a928c868871e1fd0706778edbc68aa00703cc49`. The clean Release wheel reports
`0.1.dev42+g6a928c868`; no Core source patches are applied. This includes
payload control-flow legalization [#2162](https://github.com/munich-quantum-toolkit/core/pull/2162)
and decomposition/fusion improvements [#2478](https://github.com/munich-quantum-toolkit/core/pull/2478).

### Integration changes

- Explicitly declare `forward-branching` in the selected payload for verified
  register-feed-forward profiles. Without it, the new native legalization pass
  rejects a live `qco.if`. Other optional capabilities remain unknown. The
  declaration models this benchmark's validated export path, not a claim that
  FakeTorino accepts arbitrary OpenQASM or classical programs.
- Keep exporter-specific eligibility checks. Structural control-flow legality
  alone does not establish supported measurement/register provenance.
- Remove the duplicate dynamic-qubit regex check before target-aware export:
  Core already rejects unmapped allocations. Test both QC and QCO inputs.
- Fix QCO register sizing: two allocations of widths 2 and 3 now report 5,
  rather than the largest type annotation, 3. The capacity guard is tested.
- Recognize complete connectivity by the number of validated unique edges,
  without constructing another quadratic-size complete graph. Remove an unused
  options argument and an obsolete optional-method check.

### Verification scope

This refresh uses functional checks with benchmark repetitions disabled, not a
new full-suite timing run.

| Check | Result |
| --- | --- |
| Integration regressions | 89 passed |
| QASMBench-small | 168 passed |
| Construction and manipulation | 12 passed |
| Full gym collection | 1,066 cases |
| Ruff lint and formatting | 19 integration/test files and the API probe passed |
| Historical native-export matrix | 39 passed, 3 timed out; 240-second limit per case |

The larger `cc_n301` square, heavy-hex, and linear checks exceeded that limit
during heavy host memory pressure. They remain enabled based on their prior
successful validation; these bounded checks do not revalidate those three
profiles on this revision, establish a correctness regression, or prove memory
pressure caused the timeout. The all-to-all `cc_n301` profile passed.

All four Shor topologies passed. The 39 successful matrix cases retain 2,812
conditional operations and pass native Qiskit export plus recursive basis and
connectivity checks. These checks are not a general semantic-equivalence proof.

### Small Core API improvements

1. **Expose the existing capability and constraint constants in Python.**
   Benchpress must spell `ProgramCapability("forward-branching")` manually.
   C++ already defines `FORWARD_BRANCHING`, `COUNTED_ITERATION`, and the other
   recognized IDs, but the Python classes expose none of them. A misspelling
   such as `forward_branching` survives payload construction and fails only at
   compilation. Bind those existing constants; do not replace extensible string
   IDs with a closed enum or reject legitimate vendor extensions.
   [Constants](https://github.com/munich-quantum-toolkit/core/blob/6a928c868871e1fd0706778edbc68aa00703cc49/mlir/include/mlir/Compiler/TargetEnvironment.h#L50),
   [bindings](https://github.com/munich-quantum-toolkit/core/blob/6a928c868871e1fd0706778edbc68aa00703cc49/bindings/mlir/register_mlir.cpp#L508).

2. **Include native compiler diagnostics in the Python exception.** Missing or
   misspelled branching capabilities both raise only
   `RuntimeError("MLIR operation failed; see diagnostics for details.")`.
   The useful operation/location diagnostic goes to stderr, which the optional
   timeout worker cannot carry in its exception payload. Reuse the binding's
   existing scoped diagnostic-capture pattern at the compilation boundary. A
   small message-preserving wrapper is useful without a new exception hierarchy.
   [Diagnostic helper](https://github.com/munich-quantum-toolkit/core/blob/6a928c868871e1fd0706778edbc68aa00703cc49/bindings/mlir/register_mlir.cpp#L163),
   [compile binding](https://github.com/munich-quantum-toolkit/core/blob/6a928c868871e1fd0706778edbc68aa00703cc49/bindings/mlir/register_mlir.cpp#L1240).

3. **Add non-consuming `QCOProgram.to_qiskit(target=...)` convenience.** Every
   compiled result currently requires `to_qc(copy=True).to_qiskit(target=...)`.
   A wrapper can reuse that implementation and make ownership explicit. It is
   an ergonomics improvement, not a claim that conversion becomes faster. Keep
   target selection explicit unless Core provides a retained-target accessor.
   [Current conversion API](https://github.com/munich-quantum-toolkit/core/blob/6a928c868871e1fd0706778edbc68aa00703cc49/python/mqt/core/mlir.pyi#L590).

These three improvements are now proposed in open
[#2497](https://github.com/munich-quantum-toolkit/core/pull/2497). The pinned
main integration does not yet use them.
[#2495](https://github.com/munich-quantum-toolkit/core/pull/2495) is
related work: it adds device-directed compilation and a `CompiledProgram` that
retains its target, but not direct QCO-to-Qiskit export. Its device submission
path is not a drop-in replacement for this QCO-ending benchmark timer.

### Useful but larger API work

- **Typed qubit/resource inspection on QC and QCO.** The confirmed 2+3 register
  sizing bug illustrates why the adapter should not infer widths from text.
  Define logical register width versus physical-site span and dynamic/unknown
  sizes before adding the query. Existing QC gate counters do not answer this.
  [#2260](https://github.com/munich-quantum-toolkit/core/pull/2260) adds QC
  `gate_counts()` and `static_depth()`, not qubit width or QCO inspection.
- **Inbound Qiskit target conversion.** Benchpress still maintains gate
  arities, names, directed placements, and connectivity conversion. A Core
  `BackendV2`/`Target` adapter could centralize that policy and calibration/unit
  handling, but this is more than a convenience binding. QDMI-to-Core
  [#2227](https://github.com/munich-quantum-toolkit/core/pull/2227) and device
  compilation #2495 do not implement that conversion direction.

### Boundaries that must not be reported as new missing Core APIs

- A two-iteration OpenQASM loop now compiles, exports, and matches the expected
  unitary through Core directly. Benchpress's conservative input guard still
  rejects it. That is an integration eligibility restriction, not missing loop
  synthesis. The default tested OpenQASM 2 corpus does not need this extension.
- Symbolic two-qubit `efficient_su2` compilation still produces `math.atan2`
  (12 occurrences in the focused probe), which native Qiskit export rejects.
  Numeric binding remains explicit for the device `circSU2` workloads. This is
  not established to be a small exporter-only change.
- The full-suite timing comparison belongs to `4e39f11c4`, not this new build.
  No performance gain or routing-quality improvement is claimed from these
  functional checks.

The [API probe script](/Users/simonhofmann/benchpress/.benchmarks/mqt-main-6a928c8-refresh-20260910/api_probes.py)
and its JSON/stderr outputs preserve the small reproductions. The same artifact
directory contains the build manifest, wheel/extension hashes, JUnit reports,
and native-matrix results. Full build and test logs are in
[/private/tmp/benchpress-main-20260910.kHxuUa](/private/tmp/benchpress-main-20260910.kHxuUa).
Ponytail guided the removal of duplicate checks and reuse of the existing native
pipeline; this refresh does not implement the proposed Core API changes.

## Previous main refresh: completed full-suite run (2026-09-09–10)

The integration pins upstream `main` at
`4e39f11c494df573da9b6e88225554ae2ce579d5`, including the merged tensor-wire
routing fix [#2489](https://github.com/munich-quantum-toolkit/core/pull/2489).
The workspace `.venv-mqt` uses a fresh Release wheel reporting
`0.1.dev36+g4e39f11c4`. No Core source patches are applied.

The Core API used by the adapter has not changed. The three constrained Shor
profiles are enabled again, and the validated-profile regression now includes
Shor on all four topologies. The control-capability guard and capacity skips
remain unchanged.

| Check | Result |
| --- | --- |
| Integration regressions | 86 passed |
| Shor: all-to-all, square, heavy-hex, linear | 4 passed |
| Python lint and formatting | 19 files passed |
| Full MQT gym collection | 1,066 cases |
| Full benchmark suite | 1,044 passed, 22 capacity skips, 0 failures; 19 h 2 min |

The full run uses normal benchmark repetitions, not `--benchmark-disable`.
`MQT_COMPILE_TIMEOUT=0`, `--timeout=0`, and `--timeout-skip-list=0` disable
execution timeouts. There is no session deadline and no timeout skip file.
The benchmark repetition budget remains at its default; it does not interrupt
an individual compilation. Capacity skips still apply to oversized inputs.

The [run directory](/Users/simonhofmann/benchpress/.benchmarks/mqt-main-4e39f11-full-20260909)
contains the reproduction script, exact command and revision in `manifest.json`,
and `full-suite.log`, `benchmarks.json`, `junit.xml`, and `completion.json`.
The run finished on September 10. Preflight test reports and the wheel
build log are in
[/private/tmp/benchpress-full-main-20260909.Tm5QWk](/private/tmp/benchpress-full-main-20260909.Tm5QWk).
The installed Release extension's SHA-256 is
`7bc31d07edda7a91a7a6f93544cd7d87ebf282390b93f745828797d69593ff02`.

## Historical main refresh before the routing fix (2026-09-09)

The integration now pins upstream `main` at
`91a9e0ba514af938680cdd394d6d63195872dc9a`. This includes merged
[#2351](https://github.com/munich-quantum-toolkit/core/pull/2351),
[#2452](https://github.com/munich-quantum-toolkit/core/pull/2452),
[#2461](https://github.com/munich-quantum-toolkit/core/pull/2461), and
[#2467](https://github.com/munich-quantum-toolkit/core/pull/2467).
The tested Release wheel reports `0.1.dev28+g91a9e0ba5`; the commit hash,
not the local development-version prefix, identifies this source build.
The workspace `.venv-mqt` also uses this wheel.

### Integration changes

- Adapted the new `compile_for_target(TargetEnvironment)` API. The environment
  pairs the existing quantum target with an OpenQASM 3 payload specification;
  optional capabilities remain unknown. Compilation still returns QCO, and
  validation/metrics still use native Qiskit export outside timing. This is
  not a claim that FakeTorino accepts arbitrary OpenQASM or dynamic programs.
- Removed the delayed-measurement OpenQASM rewrite and recovery path. Native
  export errors now fail directly with their diagnostic retained.
- Replaced the old stale-snapshot rejection with a positive deterministic
  round-trip test. All six distinct-register/control/topology combinations
  now preserve their known outputs.
- Retained input/lowered-IR control guards. Core's new capability metadata
  does not yet enforce classical payload capabilities in target compilation.
- Guarded `shor_n5` again on square, heavy-hex, and linear targets after
  reproducing routing timeouts. Its all-to-all profile remains enabled.

### Verification

| Check | Result |
| --- | --- |
| Historical feed-forward/BV100 gap matrix | 39 passed; 3 Shor routing timeouts |
| Integration regressions | 86 passed |
| Full QASMBench-small, final guards | 165 passed, 3 Shor skips |
| Construction and manipulation | 12 passed |
| Capacity-only selection | 22 expected skips |
| MQT versus Qiskit collection | 1,066 each; identical normalized IDs and per-file counts |
| Python lint and formatting | All 19 MQT integration/test files passed |

All 22 capacity skips were executed: 19 HamLib inputs and three Feynman circuits
exceed FakeTorino's 133 qubits. Those inputs require a larger target, not a
compiler fix.

All nine exporter failures from the September 8 small-QASM run now pass:
`inverseqft_n4`, `qec_sm_n5`, and `bb84_n8` on square, heavy-hex, and linear.
Before restoring the three Shor guards, the full small-QASM workout produced
165 passes and three routing timeouts, with no export failures.

The final combined integration/small-QASM run has 251 passes and three skips.
Construction/manipulation plus integration also pass together (98 cases) in
the updated workspace environment. Benchmarks were disabled, and bounded
compilation used a subprocess, so these are correctness results, not timing
comparisons with the September 8 performance run.

The native gap matrix covers all 41 historical control profiles plus BV100.
The 38 passing control profiles retain 4,609 conditional operations. Direct
native export and Qiskit's recursive basis/connectivity checks pass; the
matrix does not use the removed OpenQASM fallback. Condition counts are not
a general semantic-equivalence proof.

The initial 30-second matrix limit also expired on the three constrained
`cc_n301` profiles. All three passed a separate 240-second-limit rerun and
remain enabled. These larger cases need more than the short smoke-test limit;
their functional run times include import, export and validation and must not
be compared with compile-only benchmark timings.

### Remaining correctness gap: Shor routing

`shor_n5` does not finish routing on the three constrained topologies. The
square profile exceeded 240 seconds in both direct native validation and the
actual benchmark. All three profiles exceeded 30 seconds in a bounded rerun.
These same profiles passed the earlier PR snapshot, so they cannot remain on
the validated-profile list for current main. The final configured skips are
these three routing cases plus the 22 capacity cases.

An isolated reproducer imports, lowers, cleans up, decomposes, and merges
rotations in 14 ms, then times out in `place-and-route`: 30 seconds with four
trials and 15 seconds with one. It does not run two-qubit fusion, native
synthesis, or export. This localizes the defect to routing but does not prove
the exact non-progress cause. Fix the routing pass in Core and rerun these
three profiles before restoring their opt-ins; no Core fix is included here.

The earlier results below are historical evidence, not current-main results.
This refresh is a bounded correctness check, not a full-suite performance run
or a fresh general v4 API audit.

### Artifacts

Results, logs, exact inputs and the isolated routing reproducer are in
[/private/tmp/benchpress-main-20260909.Tq4ldH](/private/tmp/benchpress-main-20260909.Tq4ldH).
A compact [results archive](/Users/simonhofmann/benchpress/.benchmarks/mqt-main-refresh-2026-09-09-Tq4ldH.tar.gz)
retains the test reports, logs and probe scripts without build dependencies.
The local Release extension's SHA-256 is
`bc0ff544de044db4eb7ec606c85ecf795193b9ca8efd241b95c5d4cc3ba181c0`.
The source checkout is clean; no Core patches, commits or pushes were made
for this refresh.

## Historical PR snapshot (2026-09-07)

The previous tested pin was `7dad9e19e2028bfdef6e746f19fc0f442fccdadd`, the
[Core #2452](https://github.com/munich-quantum-toolkit/core/pull/2452)
delayed-measurement exporter fix stacked on
[Core #2351](https://github.com/munich-quantum-toolkit/core/pull/2351).
This was a PR snapshot, not upstream `main`. Its 42-profile native
feed-forward/BV100 matrix and 80 integration regressions pass. It is based on #2351's
`3f801880a` main snapshot; later main changes are not implicitly included.

The historical main review below used
`f4093cbdc435254d08c544e5a033d464a2635d15`; its wheel reported
`3.3.4.dev1662+gf4093cbdc` and contained no open PR patches. The older
verification table and API-gap analysis apply to that revision, not to the
historical candidate or newest main. That targeted refresh was not a full API
or cross-tool audit.

## Outcome

The historical main review established the full reference workout surface:
27 methods and 1,066 collected cases, with exactly the same normalized case
IDs as Qiskit. That baseline worked with the existing Python compiler API.

The baseline update tested reusable Qiskit/OpenQASM gates through
target compilation, accepted valid empty programs on the native compile path,
and recorded important execution variants in benchmark JSON. The working-tree
feed-forward refresh below retains input-eligibility restrictions and the
delayed-measurement export fallback, but retires the temporary post-compile
textual-order check. It pins the combined mapping/exporter candidate above;
the native gap matrix passes. A separate unsupported stale-snapshot export
pattern remains explicitly covered by a rejection regression below.

Before v4, prioritize known mapping/interchange correctness defects and small
public contracts that remove unsafe or ambiguous adapter behavior. Do not make
every peer SDK feature a release requirement.

## Historical main changes (`39538ecc` to `f4093cbdc`)

The previous baseline was `39538ecc292c4a487c4bea8974245bffc2e11da8`.
The relevant additions are:

| Main change | Integration consequence |
| --- | --- |
| [#2338](https://github.com/munich-quantum-toolkit/core/pull/2338): reusable OpenQASM gates | Imports can retain helper functions and calls instead of flattening every definition. |
| [#2342](https://github.com/munich-quantum-toolkit/core/pull/2342): reusable Qiskit gates | Repeated custom gates retain their call boundary on import and export. |
| [#2344](https://github.com/munich-quantum-toolkit/core/pull/2344): standard interprocedural lowering | Target compilation lowers the retained calls without a Benchpress-side inlining workaround. |
| [#2421](https://github.com/munich-quantum-toolkit/core/pull/2421): CMake baseline | C++ builds require CMake 3.28; the Python source build still requests CMake 4.4.1 or newer. |
| [#2437](https://github.com/munich-quantum-toolkit/core/pull/2437), [#2438](https://github.com/munich-quantum-toolkit/core/pull/2438) | DD/QDMI internal refactors require no Benchpress API adaptation. |

The public `mlir.pyi` and `register_mlir.cpp` surfaces are unchanged between
these two pins. Reusable gates, not new Python method availability, are the
main integration change.

Runtime probes distinguish the remaining representation differences:

- A custom two-qubit Qiskit gate used twice imports as one helper and two
  `qc.call` operations. Qiskit export retains the calls.
- `PauliEvolutionGate` now retains a call boundary. The old report's claim
  that all Hamiltonians are flattened before timed compilation is obsolete.
  Import still has to materialize the gate definition.
- The legacy `QuantumVolume` circuit used by the Qiskit device workout wraps
  its body in an `Instruction`. Core still expands this particular wrapper
  into dense unitary operations during import. This is not evidence that all
  composite gates are flattened.

## Historical verification (`f4093cbdc`)

The following checks used the historical main wheel and Qiskit 2.5.0.
These are correctness/smoke runs with benchmarking disabled, not speed
comparisons.

| Check | Result |
| --- | --- |
| Integration regression suite | 68 passed, including two reusable-gate frontends, empty native QC/QCO compilation, and benchmark-context metadata |
| Construction and manipulation | 12 passed |
| Summit device workloads on FakeTorino | 9 passed |
| Summit device workloads on directional FakeSherbrooke | 9 passed |
| Controlled QASMBench profiles | 10 passed, 26 expected skips |
| Controlled Feynman profiles | 5 expected skips |
| First nine HamLib inputs on all four abstract topologies | 36 passed |
| MQT versus Qiskit collection | 1,066 versus 1,066; normalized case IDs and per-file counts identical |
| Python lint and formatting | Passed for all MQT integration and regression files |

The device smoke runs used a 240-second hard compile limit; HamLib used
60 seconds. Their timings include the optional process transport and must
not be used as performance results. No limit expired in these runs.
The retained legacy Quantum Volume constructor emits Qiskit's deprecation
warning; both reference integrations should migrate together.

Per-file collection counts are 400 abstract Hamiltonians, 492 QASMBench,
8 construction, 53 Feynman, 100 device Hamiltonians, 9 Summit, and
4 manipulation cases. Cirq, staq and Braket implement only part of this
workout surface; their inherited skips are not missing MQT capabilities.

This was not a complete execution of all 1,066 cases or a fresh all-tool
performance sweep. Large-workload throughput and the full remaining corpus
still need release/CI validation.

## Feed-forward refresh (`7dad9e19e`)

The working tree opts all four configured topologies (`all-to-all`, `square`,
`heavy-hex`, `linear`) into direct register feed-forward for `inverseqft_n4`,
`ipea_n2`, `qec_sm_n5`, `shor_n5`, `cc_n12`, `cc_n32`, `cc_n64`,
`cc_n151`, and `cc_n301`: 36 QASMBench profiles. It also opts `inverseqft1`,
`inverseqft2`, `qec`, `teleport`, and `teleportv2` into the FakeTorino Feynman
workout. Other device backends and unrecognized control-flow shapes remain
guarded. This adds the 31 formerly skipped control profiles to the ten
previously enabled QASMBench profiles; it does not relax capacity checks.

The temporary post-compile comparison of textual measurement/store/condition
order has been removed. Valid scheduling can change that order and measurement
ordinals without changing the program. It is not replaced with a counter-only
check or another regex equivalence checker. Integration regressions instead
check known deterministic outputs for both control outcomes, distinct
registers, and destination indices through mapping and native Qiskit
export/reimport. General semantic equivalence is not inferred from sample
histograms.

The new pin contains the combined mapping and delayed-measurement exporter
fixes; the older `f4093cbdc` baseline does not. Its fresh wheel reports
`3.3.4.dev1684+g7dad9e19e`. The 31 newly opted-in profiles, ten existing
profiles, and BV100 all pass direct native Qiskit export, with no OpenQASM
fallback. This includes all 25 profiles that still failed on the refreshed
mapping-only revision. Native Qiskit basis and mapping analysis also checks
operations inside control-flow bodies.

| Candidate check | Result |
| --- | --- |
| Native feed-forward/BV100 matrix | 42 passed; 4,621 control operations retained |
| Benchpress integration regressions | 80 passed; no skips or xfails |
| Core Qiskit translation regressions | 330 passed |
| Core stub generation, repository lint, and whole-file C++ lint | Passed; no C++ findings |
| Edited Python lint/format and whitespace checks | Passed |

These are bounded functional checks, not a full 1,066-case run, performance
measurements, or general semantic-equivalence proofs. The earlier verification
table records the historical baseline separately.

The configured remaining expected benchmark skips are the 22
backend-capacity cases: 19 HamLib and three Feynman inputs exceed FakeTorino's
133-qubit width. These are not Core defects.

### Remaining native-export limitation: captured classical snapshots

The new distinct-register integration regression has five successful native
round-trip combinations and one explicit unsupported combination: the
heavy-hex target with the false control input. The identical source program
still compiles to valid QCO and samples its known deterministic output
`0100`. Its mapped schedule can contain this sequence:

```text
snapshot = read b
condition = snapshot == 1
measured = measure qubit
if condition: ...
store measured -> b[1]
```

Fusing the last store into the measurement would expose a write before the
conditional. A Qiskit condition that re-reads `b` would no longer represent
the captured snapshot. Native export therefore raises
`Qiskit control-flow export cannot preserve a stale classical snapshot`.
The regression asserts both the mapped output and this specific rejection;
it is not skipped or marked xfail. The source is shared with the five positive
cases, rather than changed to avoid the limitation.

Supporting this general pattern needs runtime materialization of captured
classical values and a clear target capability contract for the resulting
classical storage. That is a follow-up beyond the 25 benchmark-profile fixes;
the passing matrix does not establish general dynamic-circuit export support.

## Historical v4 recommendations (`f4093cbdc`, not a current gap inventory)

The following findings and PR coverage describe the older review. Later main
changes may already resolve entries; they have not been comprehensively
re-audited here. Do not treat this table as the remaining release blockers for
the candidate or newest main. In particular, the one-site synthesis fix in
[#2444](https://github.com/munich-quantum-toolkit/core/pull/2444) was validated
in the subsequent focused checks, and the empty-loader fix in
[#2443](https://github.com/munich-quantum-toolkit/core/pull/2443) has now merged
on main at `f601`. The latter main update is outside the candidate's base.
Current feed-forward work is described in its separate section above.

### Correctness and supported interchange

| Priority | Gap at the historical baseline | Recommended action / PR coverage at that time |
| --- | --- | --- |
| P1 | Constrained routing can lose register-controlled behavior. An independent four-qubit reproducer also creates a dependency cycle through a later measurement-controlled operation. | Finish mapping correctness before enabling the 31 profiles. [#2351](https://github.com/munich-quantum-toolkit/core/pull/2351) contains the crossed-control routing correction; [#2436](https://github.com/munich-quantum-toolkit/core/pull/2436) targets side-effect ordering. Revalidate their combination on current main and then run the complete guarded matrix. |
| P1 | Native Qiskit export rejects a measurement separated from its classical destination by quantum operations. Benchpress still needs its narrow OpenQASM recovery path, including for mapped BV100. | [#2439](https://github.com/munich-quantum-toolkit/core/pull/2439) fixes the exporter boundary. Keep the quantum-only safety restriction: fusion exposes the classical write earlier, and recursive memory effects alone miss deferred classical reads. |
| P2 | A truthful one-site target with `sx/x/rz` cannot synthesize `ry(0.123)`: the basis resolver also requires a two-qubit entangler. | Decouple the one-qubit synthesis family from the optional entangler. Direct target construction succeeds, but synthesis reports “no usable synthesis basis.” No covering open implementation PR found. |
| P2 | Symbolic target synthesis succeeds but can produce `math.atan2`, which native Qiskit parameter export rejects. | Provide export-compatible symbolic synthesis or explicitly document the compile-only boundary. Qiskit 2.5's public parameter surface does not provide an obvious `atan2` operation, so this is not necessarily a one-line exporter addition. No covering open implementation PR found. |
| P2 | A valid QCO program can become empty after compilation, yet `QCOProgram.from_mlir_str(program.ir)` rejects it for lacking a QCO dialect operation. | Ensure explicitly typed QC/QCO loaders accept valid empty outputs. This is a concrete round-trip defect, separate from designing a general pickle API. It still breaks Benchpress's optional timeout transport for empty results. No covering open implementation PR found. |

The preceding routing review used main plus #2436/#2439 in a separate C++
test build: 95 existing mapping tests passed, but the isolated routing test
aborted at the sorter's acyclicity assertion. The routing-only correction
from #2351 passed that test 25 times without its measurement-adjacency
workaround. This is distinct from the exact-main Python checks above.

That review also found a new #2436 SSA-edge omission affecting `qco.pow`
and a pre-existing gap for unknown-effect classical calls. The newer
`3216248` head adds only a changelog update relative to the reviewed
`b49e2ca` implementation. These findings need resolution before counting
PR #2436 as complete coverage. [#2435](https://github.com/munich-quantum-toolkit/core/pull/2435)
tightens unitary/effect contracts, but its inspected `9397096` head does
not fix those cases or the exporter's deferred-read issue.

### Public contracts worth settling before the API freezes

| Area | Historical boundary | Small useful contract / coverage at that time |
| --- | --- | --- |
| Classical target capabilities | `CompilerTarget` describes quantum operations, connectivity and calibration, not the classical payload the device accepts. Successful lowering is not a device capability check. | [#2219](https://github.com/munich-quantum-toolkit/core/pull/2219) then [#2162](https://github.com/munich-quantum-toolkit/core/pull/2162) address payload specification/legalization. This is complementary to routing correctness and necessary before claiming general target-controlled execution. |
| QC/QCO inspection | Python QC has static gate counters; neither compiler wrapper supplies the width, register/control tree and physical-site queries used by this integration. QCO lacks a corresponding operation histogram/depth surface. | Add a small typed inspection surface with explicit static versus expanded semantics. [#2260](https://github.com/munich-quantum-toolkit/core/pull/2260) adds QC-only `gate_counts()` and `static_depth()`; it does not cover QCO, qubit width, control traversal or filtered two-qubit depth. |
| Compiler diagnostics | Target construction already raises `ValueError`, but many compiler failures become a generic `RuntimeError` while details go only to stderr. Export recovery matches an English message. | Preserve diagnostic text and expose stable categories/codes. A structured diagnostic object may suffice; a large exception hierarchy is unnecessary. No covering open implementation PR found. |
| Inbound Qiskit target adaptation | Benchpress owns gate aliases, arities, ordered operation placements and connectivity conversion. It does not transfer backend calibration data even though Core can represent it. | A shared `BackendV2`/`Target` to `CompilerTarget` adapter could centralize unit conversions and unsupported-fact policy. Existing QDMI-to-Qiskit and QDMI-to-Core adapters are different directions. No covering open implementation PR found. |
| Physical result ownership | `compile_for_target` mutates the QCO program, returns `None`, and exposes no Python retained-target/layout query. Callers must keep the target for canonical physical export. | A target/layout accessor would remove the common mistake; a new general result framework is not required. |
| Compilation effort and pass discovery | The composed target pipeline offers no optimization-level/effort options at its Python call boundary. Some passes, including twirling, are selected by string pipelines; other transformations already have typed methods. | Document the fixed pipeline and reproducibility knobs. Add typed options/discovery where there is a real caller need; do not imply that equal numeric levels mean equal effort across SDKs. |
| Frontend documentation | The QASM loader handles the QASM2 corpus, while its Python documentation describes OpenQASM 3. | Document supported versions and unsupported subsets explicitly. |

Existing QC counters are not absent, but they are also not interchangeable
with physical benchmark metrics. They count entry-point operations, visit
control-flow bodies statically, count modifiers/calls as operations, and do
not expand callee bodies. They exclude barriers. An imported Bell circuit
with `h` and `cx` reports `num_gates() == 3` because its zero global-phase
operation is counted; its 1Q and 2Q counters are each one. Define those
semantics clearly rather than silently treating static counts as executed
gate counts.

### Features that need a scope decision, not an automatic release block

- **Native Python construction/composition/binding:** useful if v4 is intended
  as a native Python circuit SDK. Native C++ QC/QCO builders already exist.
  Without Python builders/binding, label these Benchpress workouts as Qiskit
  frontend-adapter measurements. #2196 concerns C++ reusable-function building,
  not this Python gap.
- **Disconnected devices:** Core rejects even a graph with two nonempty
  connected components, not just an empty map. Either support component-aware
  placement/routing or document connected-only targets. The default Benchpress
  devices do not need this feature.
- **Compiler cancellation and general process transport:** helpful for robust
  services, but not required for fair in-process Benchpress timing. Jeff byte
  serialization and textual QC/QCO transport already exist; “Core lacks
  serialization” is too broad. Fix the concrete empty-output round trip
  separately.
- **Packaging:** ship compiler-enabled v4 wheels and test installation without
  a local LLVM development tree. The inspected current stable 3.10.0 wheel
  lacks `mqt.core.mlir`; that is a release delivery gap, not evidence that an
  eventual v4 wheel is broken. Verify version identity and wheel contents in
  release CI.

## Historical comparison with other integrations (`f4093cbdc`)

| Workload / boundary | MQT behavior at the historical baseline | Interpretation |
| --- | --- | --- |
| Construction and parameter binding | Qiskit builds or binds, then Core imports, inside the timer. DTC retains the same progressive Qiskit circuit sequence but imports only the final result. | Not native MQT builder/binder performance. Native Python APIs would enable that comparison; metadata now labels the frontend path. |
| Device circSU2 | Deterministic numeric parameters, bound before timing. Qiskit and Tket retain symbols. | Not strictly input-equivalent: binding changes optimization opportunities. BQSKit's numeric model is a separate intentional SDK limitation. |
| Hamiltonians | Direct `PauliEvolutionGate` import now preserves a call boundary, and the target pipeline lowers it. | The former blanket pre-flattening gap is closed. Definition construction/import cost still belongs to the untimed frontend setup. |
| Device Quantum Volume | Same legacy Qiskit source circuit and seed; Core expands its instruction wrapper before compilation. | Source semantics agree, but the timed representation still differs. Migrate both gyms to the functional generator together, not just MQT. |
| Basis-change manipulation | Full MQT/BQSKit target compilation; Qiskit translation stage; Tket decomposition/rebase. | Recommended-path comparisons, not identical compiler algorithms. |
| Clifford manipulation | MQT/BQSKit compile raw QASM; Qiskit first builds a `Clifford`; Tket performs Clifford optimization in its timed path. | Different timed scopes should be reported, not attributed to a missing Core API. |
| Compile timer | Copying, QC-to-QCO lowering, the lowered control-flow check, and native compilation are included. Input validation and immutable target preparation are outside timing. | Prepared workloads require an unchanged input and compile a fresh copy each time. The direct API still validates each input; lowering retains its own safety check. |
| staq | External process startup and parsing are part of its timing. | Not an in-process compiler baseline. |
| Hardware facts | MQT preserves ordered operation-site availability; some peers use an undirected flattened model. | Stricter MQT handling is intentional, not an inconsistency to remove. |

## Historical Benchpress findings outside the Core API (`f4093cbdc`)

These peer/shared changes were reported during the older review, not applied
as part of it. They have not all been re-audited for the current candidate.

1. **Wrong peer optimization settings.** BQSKit's abstract and device
   Hamiltonian tests read `Configuration.options["qiskit"]["optimization_level"]`,
   as does Tket's abstract Hamiltonian test. The default Qiskit level is two
   while BQSKit is configured for one. This changes actual work and invalidates
   a comparison claiming that BQSKit uses its configured level.
2. **Incomplete physical validation.** The MQT/Qiskit-style validator checks
   top-level operations and aggregate coupling. It accepts an X on a site
   where the backend Target disallows X, and accepts an off-coupling CX inside
   an `if`. Recursively map body operands to physical sites and use per-operation
   target support. These probes demonstrate validator gaps, not Core
   miscompilation.
3. **Control-flow metrics omit branch work.** One CX inside an `if` reports
   zero two-qubit gates and zero filtered two-qubit depth in the current
   top-level metric path. Agree on structural, worst-path or execution-weighted
   metrics across gyms; blindly summing loop bodies is not a solution.
4. **Printed-IR counters are fragile.** MQT width inference depends on allocation
   spellings; its helper counts every `ctrl` as two-qubit work and scans helper
   bodies once rather than expanding calls. The current focused workloads pass,
   but these are not general semantic counters.
5. **Shared input generators differ.** Seeded Clifford generators draw
   instructions in different orders. BQSKit DTC angles omit Qiskit/MQT factors
   of two/pi; Tket uses a half-turn convention. Exact semantic comparisons need
   a shared gate/parameter trace.
6. **Control opt-ins identify filenames, not content.** The ten verified
   profiles have structural postchecks but no fixture digest. Replacing a
   corpus file under the same name can retain its opt-in. Version or hash the
   fixtures before broadening control support.
7. **Repeated output conversion.** Metrics and validation each convert QCO to
   QC/Qiskit. Share a validated output artifact if this untimed overhead matters.
8. **Remaining timeout limits.** The optional worker supports dense homogeneous
   targets and flattens child exception types. Non-QC/QCO input forms bypass it.
   Document this restricted contract; do not present these runs as normal
   in-process compiler timing.

The refresh fixes the empty-program rejection on the native path and adds
`mqt_context` metadata for frontend construction/binding, numeric circSU2,
phase normalization, native-gate overrides and the hard compile timeout.
The optional timeout's empty-output parse failure remains a reported Core
round-trip limitation.

## Historical source anchors and follow-up (`f4093cbdc`)

The older Core evidence is pinned to its tested revision:

- [Target validation and synthesis-basis resolution](https://github.com/munich-quantum-toolkit/core/blob/f4093cbdc435254d08c544e5a033d464a2635d15/mlir/lib/Compiler/Target.cpp#L599)
- [Single-qubit synthesis checks the full basis first](https://github.com/munich-quantum-toolkit/core/blob/f4093cbdc435254d08c544e5a033d464a2635d15/mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp#L477)
- [Python compiler API, including existing QC counters](https://github.com/munich-quantum-toolkit/core/blob/f4093cbdc435254d08c544e5a033d464a2635d15/python/mqt/core/mlir.pyi#L393)
- [Compiler diagnostic adaptation](https://github.com/munich-quantum-toolkit/core/blob/f4093cbdc435254d08c544e5a033d464a2635d15/bindings/mlir/register_mlir.cpp#L71)
- [Static counting implementation](https://github.com/munich-quantum-toolkit/core/blob/f4093cbdc435254d08c544e5a033d464a2635d15/mlir/lib/Compiler/Programs.cpp#L310)

Then-open heads checked during the historical review: #2351 `6e8708c`,
PR #2436 `3216248`, #2439 `a3e71e2`, #2435 `9397096`, #2260 `922ad25`,
and #2219 `9e2339c`, #2162 `bf0431c`. PR scope was checked, but their combined
stack was not rebuilt as part of that historical main refresh.

The historical recommended order was: repair mapping and native measurement
export; fix one-qubit synthesis and empty-output round trips; settle diagnostics
and typed inspection;
then decide whether symbolic end-to-end export and native Python circuit
building are part of the v4 release promise. Backend convenience adapters and
general cancellation/transport can follow without blocking the default
Benchpress integration.

This list is historical context, not a current release checklist. Use the
separate feed-forward section for this candidate's validation status.
