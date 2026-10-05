"""Time to_mr0() in an optimization loop that rebuilds the sequence every iteration.

Every case rebuilds a sequence from new tensor values, as an optimizer would,
and converts it with and without reusing the previous conversion. The results
must agree. Runs against any pulseq-zero version that has a cached to_mr0(),
so different implementations can be compared on the same cases:

    uv run demo/benchmark_to_mr0.py
"""

import inspect
import io
import sys
from contextlib import redirect_stdout
from time import perf_counter

import numpy as np
import torch

import pulseqzero as pp
sys.modules["pypulseq"] = pp
from write_tse import main as build_tse  # noqa: E402

N_REPEAT = 10


# Both cached to_mr0() APIs: this branch and the constant-structure PR.
# Versions without a cache convert from scratch in both columns.
TO_MR0_PARAMETERS = inspect.signature(pp.Sequence.to_mr0).parameters
if "speed_up_by_assuming_const_seq_structure" in TO_MR0_PARAMETERS:
    from pulseqzero.convert_cache import clear_structure_cache as clear_cache
    CACHED = {"speed_up_by_assuming_const_seq_structure": True}
    FULL = {}
elif "cached" in TO_MR0_PARAMETERS:
    try:
        from pulseqzero.seq_convert_cached import clear_cache
    except ImportError:  # cached conversion inside seq_convert
        from pulseqzero.seq_convert import clear_cache
    CACHED = {}
    FULL = {"cached": False}
else:
    def clear_cache():
        pass
    CACHED = {}
    FULL = {}


def tse(k):
    flips = torch.full((16,), 2.8 - 0.01 * k, requires_grad=True)
    with redirect_stdout(io.StringIO()):
        return build_tse(refoc_flips=flips, check_timing=False)


def gre(p, n_lines=32):
    seq = pp.Sequence()
    for i in range(n_lines):
        seq.add_block(pp.make_block_pulse(
            flip_angle=p.get("flip", 0.3), duration=1e-3, delay=p.get("rf_delay", 1e-4)))
        seq.add_block(
            pp.make_trapezoid("x", area=p.get("pre_area", -100.0), duration=1e-3),
            pp.make_trapezoid("y", area=(i - n_lines / 2) * 10.0, duration=1e-3),
        )
        if "arb_ro" in p:
            gx = pp.make_arbitrary_grad("x", waveform=p["arb_ro"])
            adc = pp.make_adc(num_samples=64, duration=float(gx.shape_dur))
        else:
            gx = pp.make_trapezoid("x", flat_area=p.get("ro_area", 200.0), flat_time=2e-3)
            adc = pp.make_adc(num_samples=64, duration=2e-3, delay=gx.rise_time)
        seq.add_block(adc, gx)
        if "arb_sp" in p:
            seq.add_block(pp.make_arbitrary_grad("z", waveform=p["arb_sp"]))
        else:
            seq.add_block(pp.make_trapezoid("z", area=p.get("sp_area", 1000.0), duration=2e-3))
        seq.add_block(pp.make_delay(p.get("tr_fill", 5e-3)))
    return seq


def T(value):
    return torch.tensor(float(value), requires_grad=True)


def waveform(k):
    ramp = np.concatenate([np.linspace(0, 1e5, 20), np.full(160, 1e5), np.linspace(1e5, 0, 20)])
    return torch.tensor(ramp * (1 + 0.01 * k), dtype=torch.float32, requires_grad=True)


CASES = {
    "TSE flip angles (16 echoes)": tse,
    "GRE flip angle": lambda k: gre({"flip": T(0.3 + 0.01 * k)}),
    "GRE TR fill delay": lambda k: gre({"tr_fill": T(5e-3 + 1e-4 * k)}),
    "GRE spoiler area": lambda k: gre({"sp_area": T(1000 + k)}),
    "GRE prephaser area": lambda k: gre({"pre_area": T(-100 - k)}),
    "GRE RF delay": lambda k: gre({"rf_delay": T(1e-4 + 1e-5 * k)}),
    "GRE readout area (ADC block)": lambda k: gre({"ro_area": T(200 + k)}),
    "GRE free gradient, own block": lambda k: gre({"arb_sp": waveform(k)}),
    "GRE free gradient during ADC": lambda k: gre({"arb_ro": waveform(k)}),
}


def values(seq):
    return torch.cat([
        torch.cat([rep.pulse.angle.reshape(-1), rep.pulse.phase.reshape(-1),
                   rep.event_time, rep.gradm.reshape(-1), rep.adc_phase])
        for rep in seq
    ]).detach()


def median_ms(fn, sequences):
    times = []
    for seq in sequences:
        start = perf_counter()
        out = fn(seq)
        times.append(perf_counter() - start)
    return 1e3 * float(np.median(times)), out


def main():
    torch.set_num_threads(4)
    print(f"{'case':32s} {'build':>8s} {'full':>8s} {'cached':>8s}  match")
    for name, build in CASES.items():
        clear_cache()
        build(0).to_mr0(**CACHED)  # first conversion fills the cache

        start = perf_counter()
        sequences = [build(k) for k in range(1, N_REPEAT + 1)]
        build_ms = 1e3 * (perf_counter() - start) / N_REPEAT

        full_ms, full = median_ms(lambda s: s.to_mr0(**FULL), sequences)
        cached_ms, cached = median_ms(lambda s: s.to_mr0(**CACHED), sequences)
        match = torch.allclose(values(full), values(cached), rtol=1e-4, atol=1e-6)
        print(f"{name:32s} {build_ms:8.1f} {full_ms:8.1f} {cached_ms:8.1f}  {match}")
    print("times in ms per iteration, median of", N_REPEAT)


if __name__ == "__main__":
    main()
