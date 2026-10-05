"""Time one iteration of the TSE flip-angle optimization, split into its parts.

Every iteration builds the 16-echo TSE of ``main.py`` from a new flip-angle
tensor and converts it with to_mr0(), once without and once with the cache.
Reported per iteration: building the sequence, converting it, and the total.
The last row shows what leaving check_timing() in the loop costs.

    uv run demo/benchmark_tse.py [--iterations 20] [--threads 4]
"""

import argparse
import io
import sys
from contextlib import redirect_stdout
from time import perf_counter

import numpy as np
import torch

import pulseqzero as pp
sys.modules["pypulseq"] = pp
from write_tse import main as build_tse  # noqa: E402
from benchmark_to_mr0 import CACHED, FULL, clear_cache  # noqa: E402

N_ECHO = 16


def flips(i):
    return torch.full((N_ECHO,), 2.8 - 0.01 * i, requires_grad=True)


def build(i, check_timing=False):
    with redirect_stdout(io.StringIO()):
        return build_tse(refoc_flips=flips(i), check_timing=check_timing)


def run(iterations, to_mr0_args, check_timing=False):
    """Median build, to_mr0 and total time in ms over an optimization loop."""
    clear_cache()
    build(0).to_mr0(**to_mr0_args)  # first iteration fills the cache
    times = []
    for i in range(1, iterations + 1):
        start = perf_counter()
        seq = build(i, check_timing)
        built = perf_counter()
        seq.to_mr0(**to_mr0_args)
        converted = perf_counter()
        times.append((built - start, converted - built, converted - start))
    return 1e3 * np.median(times, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    rows = {
        "no cache": run(args.iterations, FULL),
        "cache": run(args.iterations, CACHED),
        "cache + check_timing in loop": run(args.iterations, CACHED, check_timing=True),
    }
    print(f"TSE, {N_ECHO} echoes, median of {args.iterations} iterations, ms per iteration")
    print(f"{'':30s} {'build':>8s} {'to_mr0':>8s} {'total':>8s}")
    for name, (build_ms, convert_ms, total_ms) in rows.items():
        print(f"{name:30s} {build_ms:8.1f} {convert_ms:8.1f} {total_ms:8.1f}")


if __name__ == "__main__":
    main()
