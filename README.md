![Benchpress Image - © 2024 IBM. All Rights Reserved.](https://github.com/user-attachments/assets/67b82edb-52d6-47ac-a513-1b7129c20ea5)

Quantum software benchmarking
___

## What is Benchpress?

Benchpress is an open-source tool for benchmarking quantum software.

The Benchpress open-source benchmarking suite comprises over 1,000 different tests. These are standardized benchmarking tests designed by other members of the quantum community. For example, Benchpress compares SDKs’ abilities to generate [QASMBench circuits](https://github.com/pnnl/QASMBench), [Feynman circuits](https://github.com/meamy/feynman), and [Hamiltonian circuits](https://arxiv.org/pdf/2306.13126) (https://portal.nersc.gov/cfs/m888/dcamps/hamlib/). It also includes tests designed to test a language's ability to transpiler circuits for specific hardware, including the heavy hex architecture of IBM quantum processors and other generic qubit layouts.

If you find an issue with the testing or how we completed it, we encourage you to make a pull request.

## Citing Benchpress

 > [Benchmarking the performance of quantum computing software for quantum circuit creation, manipulation, and compilation](https://doi.org/10.1038/s43588-025-00792-y),
Paul D. Nation, Abdullah Ash Saki, Sebastian Brandhofer, Luciano Bello, Shelly Garion, Matthew Treinish & Ali Javadi-Abhari, Nat. Comput. Sci. (2025).

## Supported SDKs

Benchpress currently supports the following SDKs:

- **BQSKit** (https://github.com/BQSKit/bqskit)
- **Braket** (https://github.com/amazon-braket/amazon-braket-sdk-python)
- **Cirq** (https://github.com/quantumlib/Cirq)
- **MQT Compiler Collection (mqt-cc)** (https://github.com/munich-quantum-toolkit/core)
- **Qiskit** (https://github.com/Qiskit/qiskit)
- **Qiskit IBM transpiler** (https://github.com/Qiskit/qiskit-ibm-transpiler)
- **pyqpanda3** (https://pypi.org/project/pyqpanda3/)
- **Staq** (https://github.com/softwareQinc/staq)
- **Tket** (https://github.com/CQCL/tket)

## Testing resource requirements

Running Benchpress is resource intensive.  Although the exact requirements depend on the SDK in question, a full execution of all the SDKs requires a system with 96+Gb of memory and, in some cases, will consume as many CPU resources as are available / assigned.  In addition, each suite of tests takes a non-negligible about of time, typically several hours or more depending on the machine and timeout specified.

## Installation

Benchpress itself requires no installation.  However running it requires the tools in `requirements.txt`.  In addition, running each of the frameworks has its own dependencies in the corresponding `*-requirements.txt` file

### [pre-running] Create a skiplist

With the parameter `--timeout-skip-list=<SECs>`, a  *skiplist* (a list of tests to skip, given they take too long) is created.
For example, the following line runs the tests in `benchpress/tket_gym/construct` with a 1 hour timeout:

```bash
python -m pytest  --timeout-skip-list=3600 benchpress/tket_gym/construct
```

This will create a `skipfile.txt` file.
The mere existence of this file skips the tests listed there in the following executions.
No modifier needed.

## Running the benchmark tests

To run the benchmarks in the default configuration from inside the environment in which you want to perform the tests run:

```bash
python -m pytest benchpress/*_gym
```
where `*` is one of the frameworks that you want to test, and which matches the environment you are in.

To run the benchmarks and save to JSON one can do:

```bash
python -m pytest --benchmark-save=SAVED_NAME  benchpress/*_gym
```
which will save the file to the CWD in the `.benchmarks` folder

Further details on using `pytest-benchmark` can be found here: https://pytest-benchmark.readthedocs.io/en/latest/usage.html

### MQT Compiler Collection

Install `requirements.txt` and `requirements-mqt-cc.txt` in the same environment.
The latter pins a source revision of MQT Core main, including native support for
Qiskit 2.5 and 2.6. Building Core requires its C++/LLVM toolchain; follow
[Core's development instructions](https://mqt.readthedocs.io/projects/core/en/latest/development.html).
Update the source pin deliberately when comparing new compiler revisions.

Run one gym per pytest process: shared loaders and validators select their
implementation when imported. Mixed collection with the MQT gym is rejected.
For example, a short evaluation with complete JSON dumps is:

```bash
python -m pytest benchpress/mqt_cc_gym/abstract_transpile/test_qasmbench.py::TestWorkoutAbstractQasmBenchSmall --benchmark-json=mqt-small.json --benchmark-save-data --timeout-skip-list=3600
python -m pytest benchpress/qiskit_gym/abstract_transpile/test_qasmbench.py::TestWorkoutAbstractQasmBenchSmall --benchmark-json=qiskit-small.json --benchmark-save-data --timeout-skip-list=3600
```

Replace `Small` with `Medium` or `Large`, select `device_transpile`, or select the
whole gym. Run the integration checks with `python -m pytest tests/mqt_cc_gym`.

The MQT timer includes input copying/import, QC-to-QCO lowering, target
compilation and conversion back to QC. Target setup, Qiskit export, metrics and
validation remain outside the timer. Qiskit inputs are copied each round because
import can cache lazy gate definitions; omitting the copy would skip synthesis
in later samples. Exports consume their native output; callers that need to
reuse it must request `copy=True`.

`mapping_trials` and `mapping_iterations` in `[mqt-cc]` control placement effort.
The supplied 23 starts and two refinement rounds approximate Qiskit level 2's
layout effort, but Core has no separate final-routing restart control. They do
not establish identical search work. Remove these options to use Core defaults.
JSON reports record overrides, compiler defaults, CPU affinity and the installed
Core revision and extension hash.

MQT uses a fresh-process, whole-test timeout preflight because native MLIR
programs cannot be pickled by the benchmark plugin. Its duration is recorded as
`preflight_seconds`; it is not compilation time. Other gyms use the plugin's
callable timeout, so equal numeric limits alone do not give equal timeout scope.
Control-flow gate/depth metrics are marked non-comparable, including when
compilation removes the control flow. Static nested counts are retained separately.


## :construction: Running the memory tests :construction:

Benchmarking the amount of memory a test uses can be very costly in terms of time and memory.  Here we use the `pytest-memray` plugin.  Calling the memory bechmark looks like:

```bash

python -m pytest --memray --trace-python-allocators --native --most-allocations=100 --benchmark-disable benchpress/*_gym
```

Here `--memray` turns on the memory profiler, `--trace-python-allocators` tracks all the memoryu allocations from Python, `--native` track C/C++/Rust memory, `--most-allocations=N` shows only the top `N` tests in terms of memory consuption, and finally `--benchmark-disable` turns off the timing benchmarks.

### Histogram issues

The `pytest-memray` plugin will sometimes raise on building the histrogram included in the report by default.  Currently the only way around this error, which does not affect the tests, is to manually comment out L322 and L323 from the `plugin.py` file:

```python
#histogram_txt = cli_hist(sizes, bins=min(len(sizes), N_HISTOGRAM_BINS))
#writeln(f"\t 📊 Histogram of allocation sizes: |{histogram_txt}|")
```
## Testing details

We have designed Benchpress in a manner to allow all tests to be executed on each SDK, regardless of whether that functionality is supported or not.  This is facilitated by the use of "workouts" that define abstract base classes that define each set of tests.  This design choice has the advantage of explicitly measuring the breadth of SDK functionality

### Test status description

In Benchpress each test status has a well defined meaning:

- **PASSED** - Indicates that the SDK has the functionality required to run the test, and doing so completed without error, and within the desired time-limit.

- **SKIPPED** - The SDK does not have the required functionality to execute the test.  This is the default for all tests defined in the workouts.

- **FAILED** - The SDK has the necessary functionality, but the test failed or the test did not complete within the set time-limit.

- **XFAIL** - The test fails in an irrecoverable manner, and is therefore tagged as failed rather than being executed. E.g. the test tries to use more memory than is available.

### Test runtime

Running the full suite of tests will easily take a week or more if executed in serial, e.g. so that memory bandwidth or multiprocessing usage does no skew results.  Users can always select a subset of tests to reduce this overall time.

## Open-source packages

Benchpress makes use of files from the following open-source packages under terms of their licenses. License files are included in the corresponding directories.

- [Feynman](https://github.com/meamy/feynman)

- [QasmBench](https://github.com/pnnl/QASMBench)

- [HamLib](https://portal.nersc.gov/cfs/m888/dcamps/hamlib/)


## Previous results

Results obtained by running Benchpress over multiple versions of SDKs can be found in the [previous results](https://github.com/Qiskit/benchpress/tree/previous_results) branch.


## License

[Apache License 2.0](LICENSE.txt)
