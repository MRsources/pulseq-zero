from typing import Optional
from ..events import Delay, Scalar, Array, Label, SoftDelay, Adc
from . import _n
import pypulseq as pp
from pypulseq import Opts, get_supported_labels


def make_delay(d: Scalar) -> Delay:
    return Delay(delay=d, _pp_event=pp.make_delay(_n(d)))


def make_trigger(
    channel: str,
    delay: Scalar = 0.0,
    duration: float = 0.0,
    system: Optional[Opts] = None,
) -> Delay:
    return Delay(
        delay=delay, _pp_event=pp.make_trigger(channel, _n(delay), duration, system)
    )


def make_digital_output_pulse(
    channel: str,
    delay: Scalar = 0.0,
    duration: float = 4e-3,
    system: Optional[Opts] = None,
) -> Delay:
    return Delay(
        delay=delay,
        _pp_event=pp.make_digital_output_pulse(channel, _n(delay), duration, system),
    )


def make_label(label: str, type: str, value: int) -> Label:
    if label not in get_supported_labels():
        raise ValueError("invalid label")

    if type == "INC":
        inc = True
    elif type == "SET":
        inc = False
    else:
        raise ValueError("Invalid type. Must be one of 'SET' or 'INC'.")

    return Label(label, inc, int(value))


def make_soft_delay(
    hint: str,
    numID: Optional[int] = None,
    offset: float = 0.0,
    factor: float = 1.0,
    default_duration: float = 10e-6,
) -> SoftDelay:
    if any(c.isspace() for c in hint):
        raise ValueError("Parameter 'hint' may not contain white space characters.")
    if default_duration <= 0:
        raise ValueError("Default duration must be greater than 0.")
    if factor == 0:
        raise ValueError("Parameter 'factor' cannot be zero.")
    if numID is not None and (not isinstance(numID, int) or numID < 0):
        raise ValueError("Parameter 'numID' must be a non-negative integer or None.")

    return SoftDelay(hint, numID, offset, factor, default_duration)


def make_adc(
    num_samples: int,
    delay: Scalar = 0.0,
    duration: Scalar = 0.0,
    dwell: Scalar = 0.0,
    freq_offset: Scalar = 0.0,
    phase_offset: Scalar = 0.0,
    system: Optional[Opts] = None,
    freq_ppm: Scalar = 0.0,
    phase_ppm: Scalar = 0.0,
    phase_modulation: Optional[Array] = None,
) -> Adc:
    if phase_modulation is not None and len(phase_modulation) != num_samples:
        raise ValueError(
            "ADC Phase modulation vector must have the same length as the number of samples"
        )

    if (dwell == 0 and duration == 0) or (dwell > 0 and duration > 0):
        raise ValueError("Either dwell or duration must be defined")
    if duration > 0:
        dwell = duration / num_samples

    if system is None:
        system = Opts.default
    delay = max(delay, system.adc_dead_time)

    if phase_modulation is not None:
        raise NotImplementedError(
            "ADC phase modulation is not yet supported by pulseq-zero"
        )

    return Adc(
        num_samples,
        dwell,
        delay,
        freq_offset,
        phase_offset,
        freq_ppm,
        phase_ppm,
        system.adc_dead_time,
    )
