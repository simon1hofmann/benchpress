from functools import partial
from weakref import WeakSet

import pytest

from benchpress.utilities import sampling


def test_measured_result_is_reused_after_three_samples(benchmark):
    benchmark._min_rounds = 3
    benchmark._min_time = 1e-12
    benchmark._max_time = 1e-12
    benchmark._raw = partial(sampling.sample_and_return, benchmark)
    calls = []

    def compile(value, *, scale):
        result = {"value": value * scale, "invocation": len(calls)}
        calls.append(result)
        return result

    result = benchmark(compile, 7, scale=2)
    assert result == {"value": 14, "invocation": 3}
    assert result is calls[-1]
    assert len(calls) == 4  # One calibration plus three measured calls.
    assert benchmark.stats.stats.rounds == 3
    assert benchmark.stats.iterations == 1


def test_disabled_benchmark_still_executes_once(benchmark):
    benchmark.disabled = True
    benchmark._raw = partial(sampling.sample_and_return, benchmark)
    calls = []
    result = benchmark(lambda: calls.append(1))
    assert result is None
    assert calls == [1]


@pytest.mark.parametrize("warmup", [False, 1])
def test_previous_output_is_released_before_the_next_compile(benchmark, warmup):
    benchmark._warmup = warmup
    benchmark._min_rounds = 3
    benchmark._min_time = 1e-12
    benchmark._max_time = 1e-12
    benchmark._raw = partial(sampling.sample_and_return, benchmark)
    live = WeakSet()

    class Output:
        pass

    def compile():
        assert not live
        output = Output()
        live.add(output)
        return output

    result = benchmark(compile)
    assert set(live) == {result}


def test_sampling_propagates_errors(benchmark):
    benchmark._raw = partial(sampling.sample_and_return, benchmark)

    def invalid():
        raise ValueError("invalid program")

    with pytest.raises(ValueError, match="invalid program"):
        benchmark(invalid)


def test_profiling_keeps_original_sampler(benchmark):
    original = benchmark._raw
    benchmark.cprofile = True
    assert sampling.configure_sampling(benchmark) is benchmark
    assert benchmark._raw == original
    benchmark._min_rounds = 1
    benchmark._min_time = 1e-12
    benchmark._max_time = 1e-12
    assert benchmark(lambda: 42) == 42
    assert benchmark.stats.cprofile_stats is not None


def test_batched_sampling_preserves_return_and_iterations(benchmark, monkeypatch):
    benchmark._min_rounds = 2
    benchmark._max_time = 1e-12
    monkeypatch.setattr(benchmark, "_calibrate_timer", lambda runner: (1, 2, range(2)))
    sampling.configure_sampling(benchmark)
    calls = []
    result = benchmark(lambda: calls.append(len(calls)) or len(calls))
    assert result == 5  # Two rounds of two iterations, then the return call.
    assert benchmark.stats.iterations == 2
    assert (
        benchmark.extra_info["evaluation_timing"]["sampling_protocol"]
        == "batched-return-extra-v1"
    )
