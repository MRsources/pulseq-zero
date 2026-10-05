from functools import lru_cache

import numpy as np
import torch
import MRzeroCore as mr0
from pypulseq import Opts

from . import calc_duration
from .events import Adc, Delay, SoftDelay, RfPulse, TrapGrad, ExtTrapGrad, ArbitraryGrad
from .rf_shapes import RfShape


def convert_tensors_to_float32(obj):
    if hasattr(obj, "__dataclass_fields__"):
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                setattr(obj, field_name, value.to(dtype=torch.float32))
    return obj


def convert(
    pp0, samples_offres: int, samples_slicesel: int, samples_onres: int
) -> mr0.Sequence:
    seq = []

    for block in pp0.blocks:
        delay = None
        adc = None
        rf = None
        grad_x = None
        grad_y = None
        grad_z = None
        for ev in block:
            ev = convert_tensors_to_float32(ev)
            if isinstance(ev, (Delay, SoftDelay)):
                assert delay is None
                delay = ev
            if isinstance(ev, Adc):
                assert adc is None
                adc = ev
            if isinstance(ev, RfPulse):
                assert rf is None
                rf = ev
            if isinstance(ev, (TrapGrad, ExtTrapGrad, ArbitraryGrad)):
                assert ev.channel in ["x", "y", "z"]
                if ev.channel == "x":
                    assert grad_x is None
                    grad_x = ev
                elif ev.channel == "y":
                    assert grad_y is None
                    grad_y = ev
                elif ev.channel == "z":
                    assert grad_z is None
                    grad_z = ev

        if rf:
            assert adc is None
            # Use pulse sub-samples according to the type of pulse
            if rf.freq_offset != 0:
                samples = samples_offres
            elif grad_x or grad_y or grad_z:
                samples = samples_slicesel
            else:
                samples = samples_onres

            seq += parse_pulse(delay, rf, grad_x, grad_y, grad_z, samples)
        elif adc:
            seq += parse_adc(delay, adc, grad_x, grad_y, grad_z)
        else:
            seq += parse_spoiler(delay, grad_x, grad_y, grad_z)

    reps = []
    rep = []
    for ev in seq:
        if isinstance(ev, TmpPulse):
            rep = []
            reps.append(rep)
        rep.append(ev)

    seq = mr0.Sequence()
    for rep_in in reps:
        event_count = 0
        for ev in rep_in[1:]:  # the pulse is not an event
            if isinstance(ev, TmpAdc):
                event_count += len(ev.event_time)
            else:
                event_count += 1

        rep_out = seq.new_rep(event_count)
        # dtype is pinned: a flip angle given as a numpy scalar (np.deg2rad(90),
        # ubiquitous in pypulseq scripts) would otherwise make pulse.angle
        # float64 and execute_graph fail against a float32 phantom.
        rep_out.pulse.angle = torch.as_tensor(rep_in[0].angle, dtype=torch.float32)
        rep_out.pulse.phase = torch.as_tensor(rep_in[0].phase, dtype=torch.float32)
        if rep_in[0].shim_array is not None:
            rep_out.pulse.shim_array = rep_in[0].shim_array

        rep_out.pulse.freq_offset = torch.as_tensor(
            rep_in[0].freq_offset, dtype=torch.float32
        )
        rep_out.pulse.duration = torch.as_tensor(
            rep_in[0].duration, dtype=torch.float32
        )
        rep_out.pulse.grad = torch.stack(
            [
                torch.as_tensor(rep_in[0].grad_x, dtype=torch.float32),
                torch.as_tensor(rep_in[0].grad_y, dtype=torch.float32),
                torch.as_tensor(rep_in[0].grad_z, dtype=torch.float32),
            ]
        )
        rep_out.pulse.off_res = bool(
            (rep_out.pulse.freq_offset != 0).any().item()
            or (rep_out.pulse.grad != 0).any().item()
        )
        # pulse_freq = ω₁ = angle/duration (legacy field, to be removed upstream)
        rep_out.pulse.pulse_freq = rep_out.pulse.angle / rep_out.pulse.duration

        # pulse.usage: honour the rf.use tag when it's explicit; only fall back
        # to the flip-angle heuristic when the tag was not set ('undefined'/UNDEF).
        if rep_in[0].use != mr0.PulseUsage.UNDEF:
            rep_out.pulse.usage = rep_in[0].use
        elif rep_out.pulse.angle > 100 * torch.pi / 180:
            rep_out.pulse.usage = mr0.PulseUsage.REFOC
        else:
            rep_out.pulse.usage = mr0.PulseUsage.EXCIT

        # pulse.selective: True when a z-gradient is active during any sub-pulse
        rep_out.pulse.selective = any(
            isinstance(ev, TmpPulse) and ev.grad_z != 0 for ev in rep_in
        )

        i = 0
        for ev in rep_in[1:]:
            if isinstance(ev, TmpSpoiler):
                rep_out.event_time[i] = ev.duration
                rep_out.gradm[i, :] = ev.gradm
                i += 1
            else:
                assert isinstance(ev, TmpAdc)
                num = len(ev.event_time)
                rep_out.event_time[i : i + num] = torch.as_tensor(ev.event_time)
                rep_out.gradm[i : i + num, :] = torch.as_tensor(ev.gradm)
                rep_out.adc_phase[i : i + num] = torch.pi / 2 - ev.phase
                rep_out.adc_usage[i : i + num] = 1
                i += num

    seq.normalized_grads = False
    return seq


class TmpPulse:
    def __init__(
        self,
        angle,
        phase,
        duration,
        freq_offset,
        grad_x,
        grad_y,
        grad_z,
        shim_array,
        use: mr0.PulseUsage,
    ) -> None:
        self.angle = angle
        self.phase = phase
        self.freq_offset = freq_offset
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.grad_z = grad_z
        self.duration = duration
        self.shim_array = shim_array
        self.use = use

    def __repr__(self) -> str:
        from math import pi

        return f"Pulse(angle={self.angle * 180 / pi}°, phase={self.phase * 180 / pi}°, shim_array={self.shim_array}, use={self.use})"


class TmpSpoiler:
    def __init__(self, duration, gx, gy, gz) -> None:
        self.duration = torch.as_tensor(duration)
        self.gradm = torch.cat(
            [
                torch.as_tensor(gx).view(1),
                torch.as_tensor(gy).view(1),
                torch.as_tensor(gz).view(1),
            ]
        )

    def __repr__(self) -> str:
        return f"Spoiler(gradm={self.gradm}, duration={self.duration})"


class TmpAdc:
    def __init__(self, event_time, gradm, phase) -> None:
        self.event_time = event_time
        self.gradm = gradm
        self.phase = phase

    def __repr__(self) -> str:
        from math import pi

        return f"Adc(phase={self.phase * 180 / pi}°, total_gradm={self.gradm.sum(0)}, total_time={self.event_time.sum(0)})"


def parse_pulse(
    delay, rf, grad_x, grad_y, grad_z, samples: int
) -> list[TmpPulse | TmpSpoiler]:
    if rf.use == "excitation":
        use = mr0.PulseUsage.EXCIT
    elif rf.use == "refocusing":
        use = mr0.PulseUsage.REFOC
    else:
        use = mr0.PulseUsage.UNDEF

    # time points edges of the pulse buckets which are integrated over
    duration = calc_duration(delay, rf, grad_x, grad_y, grad_z)
    step = rf.shape_dur / samples

    # Adjusted to cover whole block
    # Integration windows used to divide the RF waveform.
    # The first and last windows include the regions before and after the
    # RF waveform. integrate_pulse() evaluates the RF as zero there.
    t_rf = [0] + [rf.delay + step * i for i in range(1, samples)] + [duration]
    
    # Effective time of each instantaneous MRzero sub-pulse.
    # These must be centred within the actual RF shape, without including
    # RF dead time, RF delay, or ringdown in the centring operation.
    pulse_times = [
        rf.delay + (i + 0.5) * step
        for i in range(samples)
    ]
    # grads are integrated from one pulse center to the next
    t_grad = torch.stack(
        [torch.as_tensor(t, dtype=torch.float64) for t in [0, *pulse_times, duration]]
    )
    gradm = torch.stack([
        integrate(grad, t_grad) if grad else torch.zeros_like(t_grad)
        for grad in (grad_x, grad_y, grad_z)
    ], dim=1)
    gradm = torch.diff(gradm, dim=0)
    event_time = torch.diff(t_grad)

    # Alternate spoiler from one pulse center to next with pulse itself
    events: list[TmpPulse | TmpSpoiler] = []

    phase_increment = torch.tensor(0.0)
    for i in range(samples):
        events.append(TmpSpoiler(event_time[i], *gradm[i]))
        flip, phase = integrate_pulse(rf, t_rf[i], t_rf[i + 1])

        # phase profile due to off-resonance
        phase += phase_increment
        phase_increment += 2 * torch.pi * rf.freq_offset * step

        rf_dur = rf.delay + rf.shape_dur  # rf duration without ringdown

        # handle gradients in parallel to pulse
        if grad_x:
            if rf_dur <= grad_x.delay:  # gradient in block starts after pulse has ended
                grad_ampl_x = 0
            else:
                # distinguish between TrapGrad and FreeGrad
                if isinstance(grad_x, ExtTrapGrad | ArbitraryGrad):
                    # find closest waveform point to gradient timepoint
                    grad_ampl_x = grad_x.waveform[
                        torch.argmin(torch.abs(torch.as_tensor(grad_x.tt) - t_grad[i + 1]))
                    ]
                else:
                    grad_ampl_x = grad_x.amplitude
        else:
            grad_ampl_x = 0

        if grad_y:
            if rf_dur <= grad_y.delay:
                grad_ampl_y = 0
            else:
                if isinstance(grad_y, ExtTrapGrad | ArbitraryGrad):
                    grad_ampl_y = grad_y.waveform[
                        torch.argmin(torch.abs(torch.as_tensor(grad_y.tt) - t_grad[i + 1]))
                    ]
                else:
                    grad_ampl_y = grad_y.amplitude
        else:
            grad_ampl_y = 0

        if grad_z:
            if rf_dur <= grad_z.delay:
                grad_ampl_z = 0
            else:
                if isinstance(grad_z, ExtTrapGrad | ArbitraryGrad):
                    grad_ampl_z = grad_z.waveform[
                        torch.argmin(torch.abs(torch.as_tensor(grad_z.tt) - t_grad[i + 1]))
                    ]
                else:
                    grad_ampl_z = grad_z.amplitude
        else:
            grad_ampl_z = 0

        events.append(
            TmpPulse(
                flip,
                phase,
                step,
                rf.freq_offset,
                grad_ampl_x,
                grad_ampl_y,
                grad_ampl_z,
                rf.shim_array,
                use,
            )
        )
    events.append(TmpSpoiler(event_time[-1], *gradm[-1]))

    return events


def parse_spoiler(delay, grad_x, grad_y, grad_z) -> tuple[TmpSpoiler]:
    duration = calc_duration(delay, grad_x, grad_y, grad_z)
    gx = grad_x.area if grad_x is not None else 0.0
    gy = grad_y.area if grad_y is not None else 0.0
    gz = grad_z.area if grad_z is not None else 0.0
    return (TmpSpoiler(duration, gx, gy, gz),)


# TODO: why does only adc have typing?
def parse_adc(delay, adc: Adc, grad_x, grad_y, grad_z) -> tuple[TmpAdc, TmpSpoiler]:
    duration = calc_duration(delay, adc, grad_x, grad_y, grad_z)
    time = torch.cat(
        [
            torch.as_tensor(0.0).view((1,)),
            adc.delay + (torch.arange(adc.num_samples) + 0.5) * adc.dwell,
            torch.as_tensor(duration).view((1,)),
        ]
    )

    gradm = torch.zeros((adc.num_samples + 2, 3))
    if grad_x:
        gradm[:, 0] = integrate(grad_x, time)
    if grad_y:
        gradm[:, 1] = integrate(grad_y, time)
    if grad_z:
        gradm[:, 2] = integrate(grad_z, time)

    event_time = torch.diff(time)
    gradm = torch.diff(gradm, dim=0)
    # Per-sample ADC phase: constant phase_offset plus time-varying freq_offset term.
    # t_samples are the centre times of each ADC dwell interval.
    t_samples = (
        adc.delay
        + (torch.arange(adc.num_samples, dtype=torch.float32) + 0.5) * adc.dwell
    )
    adc_phase = adc.phase_offset + 2 * torch.pi * adc.freq_offset * t_samples
    return (
        TmpAdc(event_time[:-1], gradm[:-1, :], adc_phase),
        TmpSpoiler(event_time[-1], gradm[-1, 0], gradm[-1, 1], gradm[-1, 2]),
    )


def split_gradm(grad, t):
    before = integrate(grad, t)
    total = integrate(grad, 1e9)  # Infinity produces 0*inf = NaNs internally
    return (before, total - before)


def integrate(grad, t):
    """Gradient moment from the start of the block up to t (scalar or 1D tensor)."""
    if isinstance(grad, TrapGrad):
        # The Heaviside terms only select which piece of the piecewise integral
        # is active; they are not part of the integrand. torch.heaviside has no
        # backward implementation, so the step is evaluated detached. Without
        # this, optimizing anything that ends up in a time shape raises
        # "derivative for aten::heaviside is not implemented".
        def h(x):
            x = torch.as_tensor(x)
            return torch.heaviside(x.detach(), torch.tensor(0.5, dtype=x.dtype))

        # https://www.desmos.com/calculator/0q5co02ecm

        d = grad.delay
        t1 = grad.rise_time
        t2 = grad.flat_time
        t3 = grad.fall_time
        T1 = d + t1
        T12 = d + t1 + t2
        T123 = d + t1 + t2 + t3

        # Trapezoid, could be provided as derivative:
        # f1 = h(t - d) * h(T1 - t) * (t - d) / t1
        # f2 = h(t - T1) * h(T12 - t)
        # f3 = h(t - T12) * h(T123 - t) * (T123 - t) / t3
        # f = grad.amplitude * (f1 + f2 + f3)

        F_inf = t1 / 2 + t2 + t3 / 2
        F1 = h(t - d) * h(T1 - t) * 0.5 * (t - d) ** 2 / t1
        F2 = h(t - T1) * h(T12 - t) * (t1 / 2 + t - T1)
        F3 = h(t - T12) * h(T123 - t) * (F_inf - 0.5 * (T123 - t) ** 2 / t3)
        F = grad.amplitude * (F1 + F2 + F3 + h(t - T123) * F_inf)

        return F
    elif isinstance(grad, ExtTrapGrad | ArbitraryGrad):
        # To stay differentiable, we don't want dynamic indexing, but instead
        # calculate, how much of every segment of the gradient contributes
        # https://www.desmos.com/calculator/j2vopzhb2z

        d = grad.delay

        tt = torch.as_tensor(grad.tt)
        waveform = torch.as_tensor(grad.waveform).reshape(-1)

        if isinstance(grad, ArbitraryGrad):
            tt = torch.cat((
                torch.zeros(1, dtype=tt.dtype),
                tt,
                torch.as_tensor(grad.shape_dur, dtype=tt.dtype).reshape(1),
            ))

            waveform = torch.cat((
                torch.as_tensor(grad.first, dtype=waveform.dtype).reshape(1),
                waveform,
                torch.as_tensor(grad.last, dtype=waveform.dtype).reshape(1),
            ))

        # Start and end time point and amplitude of all line segments
        t1 = d + tt[:-1]
        t2 = d + tt[1:]
        c1 = waveform[:-1]
        c2 = waveform[1:]

        # One row of segments per time point
        t = torch.as_tensor(t)[..., None]
        # This is how much of every segment contributes, clamped to [0, width]
        t_rel = torch.clamp(t - t1, 0 * t1, t2 - t1)
        # The amplitude of the segment at t, will be clamped to the amplitude
        # of the right point for segments before t and the left point for
        # segments after; only one segment where t lies in will be interpolated
        c_end = c1 + t_rel / (t2 - t1) * (c2 - c1)
        # For integration, we calculate the area of the rectangle with the
        # average height of the left and right side of the actual shape
        c_avg = 0.5 * (c1 + c_end)
        return (t_rel * c_avg).sum(-1)
    else:
        raise NotImplementedError


def integrate_pulse(rf: RfPulse, t_start, t_end):
    # The fraction of the pulse area inside [t_start, t_end] only depends on
    # the (detached) shape and timing. Multiplying by the live rf.flip_angle
    # tensor keeps the gradient of the user's flip-angle parameter.
    fraction = pulse_fraction(
        rf.waveform, float(rf.shape_dur), float(rf.delay), float(t_start), float(t_end)
    )
    flip = torch.as_tensor(rf.flip_angle) * fraction
    phase = (
        rf.phase_offset + 0.0
    )  # not returned by the _generate_shape() function - extend!

    return flip, phase


@lru_cache(maxsize=256)
def pulse_shape(shape: RfShape, shape_dur: float) -> tuple[np.ndarray, np.ndarray]:
    """Time points (from the shape start) and amplitudes of the pulse waveform."""
    unit_pulse = RfPulse(
        flip_angle=1.0,
        freq_offset=0.0,
        phase_offset=0.0,
        delay=0.0,
        shape_dur=shape_dur,
        center=0.0,
        ringdown_time=0.0,
        use="undefined",
        shim_array=None,
        freq_ppm=0.0,
        phase_ppm=0.0,
        waveform=shape,
    )
    pp_rf = unit_pulse.to_pulseq(Opts.default)
    return np.asarray(pp_rf.t), np.asarray(pp_rf.signal)


@lru_cache(maxsize=4096)
def pulse_fraction(
    shape: RfShape, shape_dur: float, delay: float, t_start: float, t_end: float
) -> float:
    """Fraction of the pulse area between t_start and t_end (block time)."""
    t_rel, amp_shape = pulse_shape(shape, shape_dur)
    time_shape = t_rel + delay

    # Clamp the window to the support of the RF shape. Outside of it the RF is
    # zero, but a trapezoidal segment joining a zero pad point to a non-zero
    # boundary sample would contribute area that the pulse does not have
    # (block pulses and most arbitrary RF are non-zero at their boundaries).
    lo = max(t_start, float(time_shape[0]))
    hi = min(t_end, float(time_shape[-1]))
    if hi <= lo:
        return 0.0
    # Find where lo and hi are placed in time_shape
    i_start = np.searchsorted(time_shape, lo, side="left")
    i_end = np.searchsorted(time_shape, hi, side="right")
    # Find the interpolated shape values at lo and hi
    v_start = np.interp(lo, time_shape, amp_shape)
    v_end = np.interp(hi, time_shape, amp_shape)
    # Construct the shape of the integrated part of the pulse
    time = [lo] + time_shape[i_start:i_end].tolist() + [hi]
    amp = [v_start] + amp_shape[i_start:i_end].tolist() + [v_end]

    window_area = np.trapezoid(amp, time)
    full_area = np.trapezoid(amp_shape, time_shape)
    return float(window_area / full_area) if full_area != 0 else 0.0
