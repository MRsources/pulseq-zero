"""Shapes of RF pulses.

An `RfPulse` stores the parameters its waveform is generated from in
`rf.waveform` instead of the waveform itself (not `rf.shape`, which numpy and
user code read as an array shape). The shape classes are frozen dataclasses:
they compare by value and can be used as keys of cached functions, so
`to_mr0()` generates each distinct waveform once (see `seq_convert.pulse_shape`).

`to_pulseq` builds the matching pypulseq pulse for writing, plotting etc. The
time-dependent and differentiable parameters (flip angle, delay, duration,
offsets) are taken from the `RfPulse`, not from the shape.
"""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING
import inspect as _inspect
import numpy as np
import pypulseq as pp
from pypulseq import Opts

from . import FREUDENSPRUNG_PTX
from .wrapper import _n, _r

if TYPE_CHECKING:
    from .events import RfPulse

# pypulseq 1.5+ added freq_ppm / phase_ppm to all pulse factories
_PP_HAS_PPM = "freq_ppm" in _inspect.signature(pp.make_block_pulse).parameters
# pypulseq 1.5+ added center and no_signal_scaling to make_arbitrary_rf
_PP_ARB_HAS_CENTER = "center" in _inspect.signature(pp.make_arbitrary_rf).parameters
_PP_ARB_HAS_NO_SIGNAL_SCALING = "no_signal_scaling" in _inspect.signature(pp.make_arbitrary_rf).parameters


def _pulse_args(rf: RfPulse, system: Opts) -> dict:
    """Arguments shared by all pypulseq pulse factories, taken from the pulse."""
    return dict(
        flip_angle=_n(rf.flip_angle),
        delay=_r(_n(rf.delay), system.rf_raster_time),
        freq_offset=_n(rf.freq_offset),
        phase_offset=_n(rf.phase_offset),
        system=system,
        use=rf.use,
        **({"freq_ppm": _n(rf.freq_ppm), "phase_ppm": _n(rf.phase_ppm)} if _PP_HAS_PPM else {}),
        **({"shim_array": rf.shim_array} if FREUDENSPRUNG_PTX else {}),
    )


# Gradients are constructed separately by pulseq-zero, these disable them
_NO_GRADIENT = dict(max_grad=0.0, max_slew=0.0, return_gz=False, slice_thickness=0.0)


@dataclass(frozen=True)
class BlockShape:
    def to_pulseq(self, rf: RfPulse, system: Opts) -> SimpleNamespace:
        return pp.make_block_pulse(
            duration=_r(_n(rf.shape_dur), system.rf_raster_time),
            bandwidth=None,
            time_bw_product=None,
            **_pulse_args(rf, system),
        )


@dataclass(frozen=True)
class GaussShape:
    apodization: float
    bandwidth: float
    center_pos: float
    dwell: float
    time_bw_product: float

    def to_pulseq(self, rf: RfPulse, system: Opts) -> SimpleNamespace:
        return pp.make_gauss_pulse(
            apodization=self.apodization,
            bandwidth=self.bandwidth,
            center_pos=self.center_pos,
            dwell=_r(self.dwell, system.rf_raster_time),
            duration=_r(_n(rf.shape_dur), system.rf_raster_time),
            time_bw_product=self.time_bw_product,
            **_NO_GRADIENT,
            **_pulse_args(rf, system),
        )


@dataclass(frozen=True)
class SincShape:
    apodization: float
    center_pos: float
    dwell: float
    time_bw_product: float

    def to_pulseq(self, rf: RfPulse, system: Opts) -> SimpleNamespace:
        return pp.make_sinc_pulse(
            apodization=self.apodization,
            center_pos=self.center_pos,
            dwell=_r(self.dwell, system.rf_raster_time),
            duration=_r(_n(rf.shape_dur), system.rf_raster_time),
            time_bw_product=self.time_bw_product,
            **_NO_GRADIENT,
            **_pulse_args(rf, system),
        )


@dataclass(frozen=True)
class ArbitraryShape:
    signal: tuple[float, ...]  # already scaled, a tuple to stay hashable
    dwell: float
    center: float

    def to_pulseq(self, rf: RfPulse, system: Opts) -> SimpleNamespace:
        return pp.make_arbitrary_rf(
            signal=np.asarray(self.signal),
            bandwidth=0.0,  # for grads only
            dwell=_r(self.dwell, system.rf_raster_time),
            time_bw_product=0.0,  # for grads only
            **({"no_signal_scaling": True} if _PP_ARB_HAS_NO_SIGNAL_SCALING else {}),
            **({"center": self.center} if _PP_ARB_HAS_CENTER else {}),
            **_NO_GRADIENT,
            **_pulse_args(rf, system),
        )


RfShape = BlockShape | GaussShape | SincShape | ArbitraryShape
