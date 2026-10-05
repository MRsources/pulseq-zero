"""to_mr0() reuses the previous conversion and gives the same result as a full one.

The optimization loop pattern: rebuild the sequence from new tensors every
iteration and convert it. Values and gradients must match `convert`, whatever
the tensors control.
"""

import numpy as np
import pytest
import torch

import pulseqzero as pp
from pulseqzero.seq_convert import convert
from pulseqzero.seq_convert_cached import clear_cache


def setup_function():
    clear_cache()


def _gre(flip, tr, samples=8):
    seq = pp.Sequence()
    rf = pp.make_block_pulse(flip_angle=flip, duration=1e-3)
    seq.add_block(rf)
    seq.add_block(pp.make_delay(tr))
    gx = pp.make_trapezoid(channel="x", flat_area=200.0, flat_time=1e-3)
    adc = pp.make_adc(num_samples=samples, duration=1e-3, delay=gx.rise_time)
    seq.add_block(adc, gx)
    return seq


def _angles(seq):
    return torch.stack([rep.pulse.angle.reshape(-1)[0] for rep in seq])


def _times(seq):
    return torch.stack([rep.event_time.sum() for rep in seq])


def _values(seq):
    """Everything the simulation reads from a sequence, as one vector."""
    return torch.cat([
        torch.cat([
            rep.pulse.angle.reshape(-1),
            rep.pulse.phase.reshape(-1),
            rep.pulse.freq_offset.reshape(-1),
            rep.pulse.duration.reshape(-1),
            rep.pulse.grad.reshape(-1),
            rep.event_time.reshape(-1),
            rep.gradm.reshape(-1),
            rep.adc_phase.reshape(-1),
            rep.adc_usage.reshape(-1).float(),
        ])
        for rep in seq
    ])


# ---------------------------------------------------------------------------
# The cases of the constant-structure tests (PR #59)
# ---------------------------------------------------------------------------

def test_replay_matches_full_conversion_and_keeps_gradients():
    fa = torch.tensor(0.4, requires_grad=True)
    tr = torch.tensor(0.03, requires_grad=True)
    _gre(fa, tr).to_mr0()

    fa2 = torch.tensor(0.7, requires_grad=True)
    tr2 = torch.tensor(0.05, requires_grad=True)
    fast = _gre(fa2, tr2).to_mr0()
    full = convert(_gre(fa2.detach(), tr2.detach()), 1, 1, 1)
    assert torch.allclose(_angles(fast), _angles(full))
    assert torch.allclose(_times(fast), _times(full))

    (_angles(fast).sum() + _times(fast).sum()).backward()
    assert fa2.grad is not None and float(fa2.grad) != 0.0
    assert tr2.grad is not None and float(tr2.grad) != 0.0


def test_previous_conversion_keeps_its_values():
    fa = torch.tensor(0.2, requires_grad=True)
    first = _gre(fa, torch.tensor(0.02)).to_mr0()
    kept = first[0].pulse.angle.detach().clone()

    _gre(torch.tensor(0.9, requires_grad=True), torch.tensor(0.02)).to_mr0()
    assert torch.allclose(first[0].pulse.angle, kept)


def test_layout_change_is_not_served_from_the_old_cache():
    fa = torch.tensor(0.3, requires_grad=True)
    _gre(fa, torch.tensor(0.02), samples=8).to_mr0()
    fa2 = torch.tensor(0.6, requires_grad=True)
    fast = _gre(fa2, torch.tensor(0.04), samples=32).to_mr0()
    full = convert(_gre(fa2.detach(), torch.tensor(0.04), samples=32), 1, 1, 1)
    assert len(fast) == len(full)
    assert torch.allclose(_angles(fast), _angles(full))
    assert torch.allclose(_times(fast), _times(full))


def test_uncached_still_converts():
    fa = torch.tensor(0.25, requires_grad=True)
    seq = pp.Sequence()
    seq.add_block(pp.make_block_pulse(flip_angle=fa, duration=1e-3))
    seq.add_block(pp.make_delay(0.01))
    out = seq.to_mr0(cached=False)
    (_angles(out).sum()).backward()
    assert fa.grad is not None


# ---------------------------------------------------------------------------
# Every kind of optimized parameter
# ---------------------------------------------------------------------------

def _ramp_waveform():
    return np.concatenate([np.linspace(0, 1e5, 20), np.full(60, 1e5), np.linspace(1e5, 0, 20)])


def _probe(p, n_lines=4):
    """A small GRE; `p` replaces any of its parameters."""
    seq = pp.Sequence()
    for i in range(n_lines):
        seq.add_block(pp.make_sinc_pulse(
            flip_angle=p.get("flip", 0.3),
            duration=1e-3,
            delay=p.get("rf_delay", 1e-4),
            phase_offset=p.get("rf_phase", 0.0),
            freq_offset=p.get("rf_freq", 0.0),
        ))
        seq.add_block(
            pp.make_trapezoid("x", area=p.get("pre_area", -100.0), duration=1e-3),
            pp.make_trapezoid("y", area=(i - n_lines / 2) * 10.0, duration=1e-3),
        )
        if "arb_readout" in p:
            gx = pp.make_arbitrary_grad("x", waveform=p["arb_readout"])
            adc = pp.make_adc(num_samples=16, duration=float(gx.shape_dur))
        else:
            gx = pp.make_trapezoid("x", flat_area=p.get("ro_area", 200.0), flat_time=2e-3)
            adc = pp.make_adc(
                num_samples=16,
                duration=2e-3,
                delay=gx.rise_time,
                phase_offset=p.get("adc_phase", 0.0),
            )
        seq.add_block(adc, gx)
        if "arb_spoiler" in p:
            seq.add_block(pp.make_arbitrary_grad("z", waveform=p["arb_spoiler"]))
        else:
            seq.add_block(pp.make_trapezoid("z", area=p.get("sp_area", 1000.0), duration=2e-3))
        seq.add_block(pp.make_delay(p.get("tr_fill", 5e-3)))
    return seq


PARAMETERS = {
    "flip angle": (0.3, 0.05),
    "rf_phase": (0.2, 0.1),
    "rf_freq": (100.0, 20.0),
    "adc_phase": (0.1, 0.3),
    "tr_fill": (5e-3, 1e-3),
    "sp_area": (1000.0, 100.0),
    "pre_area": (-100.0, -20.0),
    "rf_delay": (2e-4, 5e-5),
    "ro_area": (200.0, 20.0),
    "arb_spoiler": (_ramp_waveform(), 1e4),
    "arb_readout": (_ramp_waveform(), 1e4),
}


@pytest.mark.parametrize("name", PARAMETERS)
def test_cached_values_and_gradients_match_full_conversion(name):
    key = "flip" if name == "flip angle" else name
    start, step = PARAMETERS[name]

    def tensor(k):
        return torch.tensor(start + k * step, dtype=torch.float32, requires_grad=True)

    _probe({key: tensor(0)}).to_mr0()  # fills the cache

    for k in (1, 2):
        cached_param, full_param = tensor(k), tensor(k)
        cached = _probe({key: cached_param}).to_mr0()
        full = _probe({key: full_param}).to_mr0(cached=False)
        assert torch.allclose(_values(cached), _values(full), rtol=1e-5, atol=1e-6)

        # A fixed random weighting reaches every value the simulation reads
        weights = torch.rand(_values(full).numel(), generator=torch.Generator().manual_seed(k))
        (weights * _values(cached)).sum().backward()
        (weights * _values(full)).sum().backward()
        assert full_param.grad is not None and full_param.grad.abs().sum() > 0
        assert torch.allclose(cached_param.grad, full_param.grad, rtol=1e-4, atol=1e-6)


def test_changed_constants_are_converted_again():
    flip = torch.tensor(0.3, requires_grad=True)
    _probe({"flip": flip, "sp_area": 1000.0}).to_mr0()
    cached = _probe({"flip": flip, "sp_area": 1500.0}).to_mr0()
    full = _probe({"flip": flip, "sp_area": 1500.0}).to_mr0(cached=False)
    assert torch.allclose(_values(cached), _values(full))


def test_unchanged_sequence_without_tensors():
    first = _probe({}).to_mr0()
    second = _probe({}).to_mr0()
    full = _probe({}).to_mr0(cached=False)
    assert torch.allclose(_values(first), _values(full))
    assert torch.allclose(_values(second), _values(full))
