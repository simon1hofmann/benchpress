"""Shared sampling for the pinned pytest-benchmark fork."""

import sys
from functools import partial
from math import ceil
from time import perf_counter


def configure_sampling(benchmark):
    """Retain measured outputs; preserve the plugin profiling path."""
    if not benchmark.cprofile:
        benchmark._raw = partial(sample_and_return, benchmark)
    return benchmark


def sample_and_return(benchmark, function, *args, **kwargs):
    """Keep Benchpress calibration and rounds; reuse the measured result."""
    if benchmark.disabled:
        return function(*args, **kwargs)
    started = perf_counter()
    runner = benchmark._make_runner(function, args, kwargs)
    duration, iterations, _ = benchmark._calibrate_timer(runner)
    calibration = perf_counter() - started
    rounds = min(
        sys.maxsize, max(ceil(benchmark._max_time / duration), benchmark._min_rounds)
    )
    warmup = (
        min(rounds, max(1, int(benchmark._warmup / iterations)))
        if benchmark._warmup
        else 0
    )
    target, setup = function, None
    retained = []
    if iterations == 1:
        # Release the previous native output before the next compilation/timer.
        def target(*args, **kwargs):
            retained.append(function(*args, **kwargs))

        setup = retained.clear
    result = benchmark._raw_pedantic(
        target,
        args=args,
        kwargs=kwargs,
        setup=setup,
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup,
    )
    if iterations == 1:
        result = retained.pop()
    benchmark.extra_info["evaluation_timing"] = {
        "calibration_seconds": calibration,
        "sampling_seconds": perf_counter() - started - calibration,
        "sampling_protocol": (
            "return-last-sample-v2" if iterations == 1 else "batched-return-extra-v1"
        ),
    }
    return result
