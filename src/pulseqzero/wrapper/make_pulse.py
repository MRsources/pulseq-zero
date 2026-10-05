from typing import Optional, cast
from warnings import warn
from copy import copy
import numpy as np
import torch

from .. import get_supported_rf_uses, Opts
from ..events import RfPulse, TrapGrad, Scalar, Array
from ..rf_shapes import ArbitraryShape, BlockShape, GaussShape, SincShape
from .make_grad import make_trapezoid
from . import _n


def make_block_pulse(
    flip_angle: Scalar,
    delay: Scalar = 0.0,
    duration: Optional[Scalar] = None,
    bandwidth: Optional[Scalar] = None,
    time_bw_product: Optional[Scalar] = None,
    freq_offset: Scalar = 0.0,
    phase_offset: Scalar = 0.0,
    system: Optional[Opts] = None,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    # Martins pTx extension
    shim_array: Optional[Array] = None,
) -> RfPulse:
    if system is None:
        system = Opts.default

    valid_uses = get_supported_rf_uses()
    if use != "" and use not in valid_uses:
        raise ValueError(
            f"Invalid use parameter. Must be one of {valid_uses}. Passed: {use}"
        )

    if delay < system.rf_dead_time:
        delay = system.rf_dead_time

    if duration is None:
        if bandwidth is None:
            warn("Using default 4 ms duration for block pulse.")
            duration = 4e-3
        else:
            if time_bw_product is None:
                time_bw_product = 0.25
            duration = time_bw_product / bandwidth
    elif bandwidth is not None:
        raise ValueError("One of bandwidth or duration must be defined, but not both.")

    return RfPulse(
        flip_angle=flip_angle,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=delay,
        shape_dur=duration,
        center=duration / 2,
        ringdown_time=system.rf_ringdown_time,
        use=use,
        shim_array=shim_array,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        waveform=BlockShape(),
    )


def make_gauss_pulse(
    flip_angle: float,
    apodization: float = 0.0,
    bandwidth: float = 0.0,
    center_pos: float = 0.5,
    delay: float = 0.0,
    dwell: float = 0.0,
    duration: float = 4e-3,
    freq_offset: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    phase_offset: float = 0.0,
    return_gz: bool = False,
    slice_thickness: float = 0.0,
    system: Optional[Opts] = None,
    time_bw_product: float = 4.0,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    # Martins pTx extension
    shim_array: Optional[Array] = None,
) -> RfPulse | tuple[RfPulse, TrapGrad, TrapGrad]:
    if system is None:
        system = Opts.default

    valid_uses = get_supported_rf_uses()
    if use != "" and use not in valid_uses:
        raise ValueError(
            f"Invalid use parameter. Must be one of {valid_uses}. Passed: {use}"
        )

    if delay < system.rf_dead_time:
        delay = system.rf_dead_time

    if bandwidth == 0:
        bandwidth = time_bw_product / duration

    rf = RfPulse(
        flip_angle=flip_angle,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=delay,
        shape_dur=duration,
        center=duration * center_pos,
        ringdown_time=system.rf_ringdown_time,
        use=use,
        shim_array=shim_array,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        waveform=GaussShape(
            apodization=_n(apodization),
            bandwidth=_n(bandwidth),
            center_pos=_n(center_pos),
            dwell=_n(dwell),
            time_bw_product=_n(time_bw_product),
        ),
    )

    if not return_gz:
        return rf
    else:
        if slice_thickness == 0:
            raise ValueError("Slice thickness must be provided")

        if max_grad > 0:
            system = copy(system)
            system.max_grad = max_grad

        if max_slew > 0:
            system = copy(system)
            system.max_slew = max_slew

        bandwidth = time_bw_product / duration
        amplitude = bandwidth / slice_thickness
        area = amplitude * duration

        gz = make_trapezoid(
            channel="z", system=system, flat_time=duration, flat_area=area
        )
        gzr = make_trapezoid(
            channel="z",
            system=system,
            area=-area * (1 - center_pos) - 0.5 * (gz.area - area),
        )

        if rf.delay > gz.rise_time:
            gz.delay = rf.delay - gz.rise_time
        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

        return rf, gz, gzr


def make_sinc_pulse(
    flip_angle: Scalar,
    apodization: float = 0.0,
    delay: Scalar = 0.0,
    duration: Scalar = 4e-3,
    dwell: float = 0.0,
    center_pos: float = 0.5,
    freq_offset: Scalar = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    phase_offset: Scalar = 0.0,
    return_gz: bool = False,
    slice_thickness: float = 0.0,
    system: Optional[Opts] = None,
    time_bw_product: float = 4.0,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    # Martins pTx extension
    shim_array: Optional[Array] = None,
) -> RfPulse | tuple[RfPulse, TrapGrad, TrapGrad]:
    if system is None:
        system = Opts.default

    valid_uses = get_supported_rf_uses()
    if use != "" and use not in valid_uses:
        raise ValueError(
            f"Invalid use parameter. Must be one of {valid_uses}. Passed: {use}"
        )

    if delay < system.rf_dead_time:
        delay = system.rf_dead_time

    if duration <= 0:
        raise ValueError("RF pulse duration must be positive.")

    rf = RfPulse(
        flip_angle=flip_angle,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=delay,
        shape_dur=duration,
        center=duration * center_pos,
        ringdown_time=system.rf_ringdown_time,
        use=use,
        shim_array=shim_array,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        waveform=SincShape(
            apodization=_n(apodization),
            center_pos=_n(center_pos),
            dwell=_n(dwell),
            time_bw_product=_n(time_bw_product),
        ),
    )

    if not return_gz:
        return rf
    else:
        if slice_thickness == 0:
            raise ValueError("Slice thickness must be provided")

        if max_grad > 0:
            system = copy(system)
            system.max_grad = max_grad

        if max_slew > 0:
            system = copy(system)
            system.max_slew = max_slew

        bandwidth = time_bw_product / duration
        amplitude = bandwidth / slice_thickness
        area = amplitude * duration

        gz = make_trapezoid(
            channel="z", system=system, flat_time=duration, flat_area=area
        )
        gzr = make_trapezoid(
            channel="z",
            system=system,
            area=-area * (1 - center_pos) - 0.5 * (gz.area - area),
        )

        if rf.delay > gz.rise_time:
            gz.delay = rf.delay - gz.rise_time
        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

        return rf, gz, gzr


def make_arbitrary_rf(
    signal: Array,
    flip_angle: Scalar,
    bandwidth: Scalar = 0.0,
    delay: Scalar = 0.0,
    dwell: Scalar = 0.0,
    freq_offset: Scalar = 0.0,
    no_signal_scaling: bool = False,
    max_grad: Scalar = 0.0,
    max_slew: Scalar = 0.0,
    phase_offset: Scalar = 0.0,
    return_gz: bool = False,
    slice_thickness: Scalar = 0.0,
    system: Optional[Opts] = None,
    time_bw_product: Scalar = 0.0,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    center: Optional[Scalar] = None,
    # Martins pTx extension
    shim_array: Optional[Array] = None,
) -> RfPulse | tuple[RfPulse, TrapGrad]:
    if system is None:
        system = cast(Opts, Opts.default)

    valid_uses = get_supported_rf_uses()
    if use != "" and use not in valid_uses:
        raise ValueError(
            f"Invalid use parameter. Must be one of {valid_uses}. Passed: {use}"
        )

    if delay < system.rf_dead_time:
        delay = system.rf_dead_time
    if dwell == 0:
        dwell = system.rf_raster_time
    duration = len(signal) * dwell

    if not no_signal_scaling:
        if isinstance(signal, torch.Tensor):
            total = signal.sum().abs()
        else:
            total = np.abs(np.sum(signal))
        signal = cast(Array, signal / (total * dwell) * flip_angle / (2 * np.pi))

    if center is not None:
        center = min(max(center, 0), duration)
    else:
        # TODO: improve performance by inlining the function and optimizing away
        # the interpolation over the completely regular time array
        time = (np.arange(1, len(signal) + 1) - 0.5) * dwell
        center = _calc_shape_center(signal, time)

    rf = RfPulse(
        flip_angle=flip_angle,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=delay,
        shape_dur=duration,
        center=center,
        ringdown_time=system.rf_ringdown_time,
        use=use,
        shim_array=shim_array,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        waveform=ArbitraryShape(
            signal=tuple(np.atleast_1d(_n(signal)).tolist()),
            dwell=_n(dwell),
            center=_n(center),
        ),
    )

    if not return_gz:
        return rf
    else:
        if slice_thickness <= 0:
            raise ValueError("Slice thickness must be provided.")
        if bandwidth <= 0:
            raise ValueError("Bandwidth of pulse must be provided.")

        if max_grad > 0:
            system = copy(system)
            system.max_grad = max_grad  # ty: ignore[invalid-assignment]

        if max_slew > 0:
            system = copy(system)
            system.max_slew = max_slew

        if time_bw_product > 0:
            bandwidth = time_bw_product / duration
        amplitude = bandwidth / slice_thickness
        area = amplitude * duration

        gz = make_trapezoid(
            channel="z", system=system, flat_time=duration, flat_area=area
        )

        if rf.delay > gz.rise_time:
            gz.delay = rf.delay - gz.rise_time
        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

        return rf, gz


# ==============================================================================
# Helper to compute the center of arbitrary pulses
# ==============================================================================


def _calc_shape_center(signal: Array, time: Array) -> Scalar:
    """Detect the excitation peak; if i is a plateau take its center"""
    if isinstance(signal, torch.Tensor):
        rf_max = torch.max(torch.abs(signal))
        i_peak = torch.where(torch.abs(signal) >= rf_max * 0.99999)[0]
        time_center = cast(Scalar, (time[i_peak[0]] + time[i_peak[-1]]) / 2)
    else:
        rf_max = np.max(np.abs(signal))
        i_peak = np.where(np.abs(signal) >= rf_max * 0.99999)[0]
        time_center = (time[i_peak[0]] + time[i_peak[-1]]) / 2

    return time_center
