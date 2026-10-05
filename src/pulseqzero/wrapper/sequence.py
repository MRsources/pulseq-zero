from warnings import warn
from ..wrapper import _n
from copy import copy, deepcopy
import MRzeroCore
import pypulseq
import numpy as np

from typing import Optional
from types import SimpleNamespace
from .. import calc_duration, Opts, seq_convert, seq_convert_cached, __version__
from ..events import Event, SoftDelay, TrapGrad, ExtTrapGrad, ArbitraryGrad


class Sequence:
    version_major = int(__version__.split(".")[0])
    version_minor = int(__version__.split(".")[1])
    version_revision = __version__.split(".")[2]

    def __init__(self, system: Optional[Opts] = None, use_block_cache=True):
        self.definitions = {}
        self.blocks: list[list[Event]] = []
        self.system = system if system else Opts.default
        self._pp_use_block_cache = use_block_cache

    def __str__(self):
        return f"mr0 sequence adapter; ({len(self.blocks)}) blocks"

    @property
    def rf_raster_time(self) -> float:
        return self.system.rf_raster_time

    @property
    def grad_raster_time(self) -> float:
        return self.system.grad_raster_time

    @property
    def adc_raster_time(self) -> float:
        return self.system.adc_raster_time

    @property
    def block_duration_raster(self) -> float:
        return self.system.block_duration_raster

    # =========================================================================
    # Ported functions
    # =========================================================================

    def add_block(self, *args: Event):
        self.blocks.append([copy(arg) for arg in args])

    def apply_soft_delay(self, **kwargs):
        # In pypulseq, this sets an extra block_duration cache of the sequence.
        # Here we will instead set default_duration to the computed values.
        for block in self.blocks:
            for event in block:
                if isinstance(event, SoftDelay):
                    if event.hint in kwargs:
                        event.default_duration = (
                            kwargs[event.hint] / event.factor + event.offset
                        )

    def duration(self):
        duration = sum(calc_duration(*block) for block in self.blocks)
        num_blocks = len(self.blocks)
        event_count = sum(len(b) for b in self.blocks)
        return duration, num_blocks, event_count

    def find_block_by_time(self, t: float) -> int | None:
        time = np.cumsum([calc_duration(*block) for block in self.blocks])
        index = np.searchsorted(time, t, side="right").item()

        if index > len(self.blocks):
            return None
        else:
            return index

    def flip_grad_axis(self, axis: str) -> None:
        self.mod_grad_axis(axis, modifier=-1)

    def get_block(self, block_index: int) -> list[Event]:
        return self.blocks[block_index]

    def get_definition(self, key):
        if key in self.definitions:
            return self.definitions[key]
        else:
            return ""

    def mod_grad_axis(self, axis: str, modifier: float) -> None:
        if axis not in ["x", "y", "z"]:
            raise ValueError(
                f"Invalid axis. Must be one of 'x', 'y','z'. Passed: {axis}"
            )

        for block in self.blocks:
            for event in block:
                if isinstance(event, TrapGrad) and event.channel == axis:
                    event.amplitude = event.amplitude * modifier
                elif isinstance(event, ExtTrapGrad) and event.channel == axis:
                    event.waveform = event.waveform * modifier
                elif isinstance(event, ArbitraryGrad) and event.channel == axis:
                    event.waveform = event.waveform * modifier
                    event.first = event.first * modifier
                    event.last = event.last * modifier

    def remove_duplicates(self, in_place=False):
        warn(
            "remove_duplicates() does nothing in pulseq-zero: It has no block "
            "cache (generates data lazily) that could be de-duplicated."
        )
        if in_place:
            return self
        else:
            return deepcopy(self)

    def set_block(self, block_index: int, *args: Event):
        if block_index >= len(self.blocks):
            self.blocks.append([copy(arg) for arg in args])
        else:
            self.blocks[block_index] = [copy(arg) for arg in args]

    def set_definition(self, key, value):
        self.definitions[key] = value

    # =========================================================================
    # Non-differentiable functions - pypulseq pass-through
    # =========================================================================

    def adc_times(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self.to_pypulseq().adc_times(*args, **kwargs)

    def calculate_gradient_spectrum(
        self, *args, **kwargs
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        return self.to_pypulseq().calculate_gradient_spectrum(*args, **kwargs)

    def calculate_kspace(
        self, *args, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, list[float], list[float], np.ndarray]:
        return self.to_pypulseq().calculate_kspace(*args, **kwargs)

    def calculate_kspacePP(self, *args, **kwargs):
        raise NotImplementedError("calculate_kspacePP has been deprecated")

    def calculate_pns(self, *args, **kwargs):
        return self.to_pypulseq().calculate_pns(*args, **kwargs)

    def check_timing(self, *args, **kwargs) -> tuple[bool, list[SimpleNamespace]]:
        return self.to_pypulseq().check_timing(*args, **kwargs)

    def evaluate_labels(self, *args, **kwargs) -> dict:
        return self.to_pypulseq().evaluate_labels(*args, **kwargs)

    def get_gradients(self, *args, **kwargs):
        raise NotImplementedError("No scipy support")

    def get_raw_block_content_IDs(self, *args, **kwargs):
        raise NotImplementedError(
            "Pulseq-zero has different internals: cannot return IDs"
        )

    def install(self, *args, **kwargs):
        return self.to_pypulseq().install(*args, **kwargs)

    def paper_plot(self, *args, **kwargs):
        return self.to_pypulseq().paper_plot(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return self.to_pypulseq().plot(*args, **kwargs)

    def read(self, *args, **kwargs):
        raise NotImplementedError("Cannot read .seq files in pulseq-zero")

    def register_adc_event(self, *args, **kwargs):
        raise NotImplementedError("Pulseq-zero does not use internal libraries")

    def register_grad_event(self, *args, **kwargs):
        raise NotImplementedError("Pulseq-zero does not use internal libraries")

    def register_label_event(self, *args, **kwargs):
        raise NotImplementedError("Pulseq-zero does not use internal libraries")

    def register_rf_event(self, *args, **kwargs):
        raise NotImplementedError("Pulseq-zero does not use internal libraries")

    def register_soft_delay_event(self, *args, **kwargs):
        raise NotImplementedError("Pulseq-zero does not use internal libraries")

    def rf_from_lib_data(self, *args, **kwargs):
        raise NotImplementedError("Pulseq-zero does not use internal libraries")

    def rf_times(
        self, *args, **kwargs
    ) -> tuple[list[float], np.ndarray, list[float], np.ndarray, np.ndarray]:
        return self.to_pypulseq().rf_times(*args, **kwargs)

    def test_report(self, *args, **kwargs) -> str:
        return self.to_pypulseq().test_report(*args, **kwargs)

    def waveforms(self, *args, **kwargs) -> tuple[np.ndarray]:
        return self.to_pypulseq().waveforms(*args, **kwargs)

    def waveforms_and_times(
        self, *args, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.to_pypulseq().waveforms_and_times(*args, **kwargs)

    def write(self, *args, **kwargs) -> str | None:
        return self.to_pypulseq().write(*args, **kwargs)

    # =========================================================================
    # PULSEQ-ZERO ADAPTER TO LIB CONVERSIONS
    #
    # The whole purpose of pulseq is to write differentiable code and only
    # swapping pypulseq -> pulseqzero imports. The wrapper then tracks all
    # calls and can convert the sequence to pypulseq or MR-zero. Pypulseq
    # conversion is done on-demand for applications like writing or plotting.
    # to_mr0 is the actual feature: convert to MR-zero while keeping gradients
    # intact, allowing sequence optimizations with pulseq-in-the-loop.
    # =========================================================================

    def to_pypulseq(self) -> pypulseq.Sequence:
        import warnings

        warnings.warn(
            "pulseqzero.Sequence: translating to PyPulseq — avoid calling "
            "this inside a hot loop.",
            stacklevel=2,
        )
        if self.version_minor >= 5:
            pp_seq = pypulseq.Sequence(
                system=self.system, use_block_cache=self._pp_use_block_cache
            )
        else:
            pp_seq = pypulseq.Sequence(system=self.system)
        for block in self.blocks:
            pp_events = [ev.to_pulseq(self.system) for ev in block]
            pp_seq.add_block(*pp_events)
        for key, value in self.definitions.items():
            pp_seq.set_definition(key=key, value=_n(value) if not isinstance(value, str) else value)
        return pp_seq

    def to_mr0(
        self,
        samples_offres: int = 1,
        samples_slicesel: int = 1,
        samples_onres: int = 1,
        cached: bool = True,
    ) -> MRzeroCore.Sequence:
        """Convert to an MR-zero sequence, keeping gradients of tensor parameters.

        With ``cached``, the result of the last call with the same block layout
        is reused: only blocks that changed or hold tensors are converted
        again. ``seq_convert_cached.clear_cache()`` frees the cached sequences.
        """
        if cached:
            return seq_convert_cached.convert_cached(self, samples_offres, samples_slicesel, samples_onres)
        return seq_convert.convert(self, samples_offres, samples_slicesel, samples_onres)
