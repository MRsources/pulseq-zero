# Changelog

All notable changes to pulseq-zero are recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0]

### Changed

- **Cached `to_mr0()`.** Previous MR-zero sequences are reused. Blocks whose
  non-tensor content changed or have torch tensors are converted again. flip angle, phase and frequency offsets only update the corresponding properties to avoid re-
  computations while correctly passing autograd gradients.
  The cached conversion lives in `seq_convert_cached`, next to
  `seq_convert`: `to_mr0(cached=False)` converts from scratch with
  `seq_convert.convert`, `seq_convert_cached.clear_cache()` frees the cached
  sequences.
- **Faster `seq_convert`.** Pulse shapes and their integrals are cached, gradient
  moments are integrated for all time points of a block at once. The conversion is now
  roughly 2.5x faster, cached conversion up to 20x - see ([demo/benchmark_to_mr0.py](demo/benchmark_to_mr0.py)).
- **RF shapes.** `RfPulse.waveform` stores the parameters of its waveform as a
  `BlockShape`, `GaussShape`, `SincShape` or `ArbitraryShape` instead of a
  pypulseq factory function, and `freq_ppm` / `phase_ppm` as fields.
- **TSE demo.** [demo/main.py](demo/main.py) checks the timing once before
  the optimization loop instead of in every rebuild, and prints the time of
  every iteration.

## [1.0.4]

### Added

- **Test suite** (`tests/`). 3 CI checks: the `.seq` file written through
  pulseq-zero is byte-identical to one PyPulseq; `seq.to_mr0()` agrees with
  writing the `.seq` and importing it into MR-zero; derivatives reach every
  parameter documented as differentiable and match a finite difference of the
  same loss.
- **Raster times on `Sequence`** (`grad_raster_time`, `rf_raster_time`,
  `adc_raster_time`, `block_duration_raster`), as pypulseq exposes them.
  Example scripts read them off the sequence to round their delays.

### Fixed

- **ADC delay was rounded to the gradient raster on export.**
- **`TrapGrad.area` / `flat_area` accept assignment.** They are derived
  properties, scripts like `write_epi_se_rs` trigger an `AttributeError` on
  write. Assignment now sets the amplitude the requested area implies, which
  keeps amplitude, area and flat_area from ever disagreeing.

## [1.0.3]

### Fixed

- **Heaviside:** use a differentiable heaviside during `seq_convert`

## [1.0.2]

### Fixed

- **RF dead time:** Clamp rf `delay` to be at least `system.rf_dead_time`.
- **seq_convert torch tensors:** Convert parameters to tensors if needed.
- **seq_convert pulse integration:** Fix in shape -> flip angle conversion.

## [1.0.1]

### Fixed

- **Arbitrary gradients: endpoints now included when integrating.**
  `seq_convert.integrate` previously integrated only over the internal
  `grad.tt` / `grad.waveform` samples of an `ArbitraryGrad`, silently
  dropping the `first` and `last` boundary samples. The waveform is now
  padded with `first` at `t=0` and `last` at `t=shape_dur` before
  integration, so the ramps into and out of the free waveform are
  accounted for.
- **`parse_pulse`: sub-pulse centring matches PyPulseq's RF timing.**
  The effective time of each MRzero sub-pulse is now centred within the
  RF shape itself (`rf.delay + (i + 0.5) * step`), rather than being
  derived from the midpoints of the integration windows used to bucket
  the waveform. This changes (and corrects) the timing of ADC samples
  during simulation to match PyPulseq's definition of RF timing.

## [1.0.0]

The 1.0 release unifies pulseq-zero into a single, always-on facade. The
mode-switching surface (`pp_impl`, `mr0_mode()`, `is_mr0_mode()`) is gone
— the same script now writes `.seq` files *and* feeds MR-zero, with no
context manager and no mode flag. Compatible with **PyPulseq 1.4.x and
1.5.x**; the correct API surface is detected at import time.

### Added (post-unification fixes)

- **PyPulseq 1.4 compatibility.** Pulseq-zero now runs against both
  PyPulseq 1.4.x and 1.5.x. All 1.5-only kwargs are guarded by
  `inspect.signature` probes at module load time:
  - `freq_ppm` / `phase_ppm` on all four pulse factories and `make_adc`.
  - `no_signal_scaling` and `center` on `make_arbitrary_rf`.
  - `first` / `last` / `oversampling` on `make_arbitrary_grad`.
  - `use_block_cache` on `pypulseq.Sequence.__init__`.
  - `Opts.default = Opts()` is injected once at import time so code that
    relies on the class attribute works with both versions.
  - `demo/test_pypulseq_sequence_all_funcs.py` now runs cleanly under 1.4
    (one known row mismatch: pypulseq 1.4's `test_report` has an
    off-by-one index bug, reported upstream).

- **MRzeroCore 0.4.11 pulse fields.** `seq.to_mr0()` now populates all
  fields introduced in MRzeroCore 0.4.11:
  - `pulse.freq_offset` — RF carrier offset in Hz.
  - `pulse.duration` — sub-pulse interval duration in seconds.
  - `pulse.grad` — gradient amplitude vector `[gx, gy, gz]` in Hz/m
    during the RF window.
  - `pulse.off_res` — `True` when `freq_offset ≠ 0` or any gradient
    component is non-zero during the pulse.
  - `pulse.pulse_freq` — legacy `ω₁ = angle / duration` field (marked
    upstream as "to be removed").

### Fixed (post-unification fixes)

- **numpy deprecated type aliases.** `numpy.bool`, `numpy.float`,
  `numpy.int`, and `numpy.complex` are now mapped to their correct numpy
  scalar types (`bool_`, `float64`, `int_`, `complex128`), preventing
  `AttributeError: 'bool' object has no attribute 'view'` with
  numpy ≥ 1.24.

- **`seq_convert`: `pulse.usage` now respects `rf.use`.** When a pulse
  factory is called with an explicit `use=` string (`'excitation'`,
  `'refocusing'`, …), that tag is passed through to `pulse.usage` on the
  MRzeroCore side. The flip-angle threshold heuristic (`> 100°` →
  `REFOC`) only fires when `use` is `'undefined'` / `UNDEF`.

- **`seq_convert`: `pulse.selective` set from z-gradient.** `pulse.selective`
  is now `True` whenever a z-gradient is active during any sub-pulse of
  the RF event, correctly signalling slice-selective excitation to
  MR-zero.

- **`seq_convert`: ADC `freq_offset` contributes per-sample phase.** The
  ADC phase passed to MR-zero is now `φ(t) = phase_offset + 2π ×
  freq_offset × t_sample` (where `t_sample` is the centre time of each
  ADC dwell interval), matching the physical per-sample demodulation
  model. Previously `freq_offset` was silently ignored in simulation.

### Breaking changes

- Imports: `import pulseqzero; pp = pulseqzero.pp_impl` → `import pulseqzero as pp`.
- The `with pulseqzero.mr0_mode():` context manager has been removed.
  `seq.to_mr0()` and `seq.write()` both work unconditionally.
- `pulseqzero.is_mr0_mode()`, `pulseqzero.pp_impl`, `use_pypulseq()`,
  `use_pulseqzero()`, `torch_to_numpy`, and `convert_tensor` are gone.
  Unsupported PyPulseq functions now raise `NotImplementedError` with a
  named workaround instead of asking user code to branch on the mode.
- There is no deprecation shim. Scripts migrating from 0.3 need the two
  mechanical edits above (import swap + drop the `mr0_mode` wrapper).

### Added

- `Sequence.write(path)` / `Sequence.to_pypulseq()` on the adapter. Both
  lazily translate the in-memory event graph through PyPulseq and emit
  `warnings.warn(...)` on every call so usage inside a hot optimization
  loop is visible. `seq.to_pypulseq()` is the explicit escape hatch for
  one-off PyPulseq-native calls.
- Forwarding stubs on `Sequence`: `calculate_pns`, `test_report`,
  `paper_plot` all route through `self.to_pypulseq().foo(...)`.
- Clear `NotImplementedError` raisers for PyPulseq entry points we
  don't (yet) wrap: `make_adiabatic_pulse`, `sigpy_n_seq`, `make_slr`,
  `make_sms`, `SigpyPulseOpts`, `align`, `calc_ramp`, `rotate`,
  `points_to_waveform`, `traj_to_grad`. Each message names the function
  and points at the workaround.
- [demo/](demo/): a `uv`-managed workspace member targeting PyPulseq
  1.5.0.post1.
  - [demo/write_tse.py](demo/write_tse.py): TSE sequence ported from the
    PyPulseq 1.5 upstream example. The refocusing flip angles are an
    input (`refoc_flips`, length `n_echo`, radians) and the refocusing
    pulse is rebuilt per echo.
  - [demo/main.py](demo/main.py): end-to-end demo that simulates a
    target image from a fully-180° TSE refocusing train and optimizes
    the refocusing flip-angle vector with an RMS image-fidelity loss
    plus a `sum(flips**2)` SAR penalty.

### Changed

- Pulse-shape generation is delegated to PyPulseq. `make_sinc_pulse`,
  `make_gauss_pulse`, `make_block_pulse`, and `make_arbitrary_rf` now
  call the corresponding `pypulseq.make_*` factory once at construction
  time and store the returned `(t, signal)` tuple on the adapter
  `Pulse`. The hand-rolled envelope math and companion `gz`/`gzr`
  trapezoid construction are gone (~150 LoC deleted).
  - `flip_angle`, `phase_offset`, `freq_offset`, and `delay` stay as
    live torch tensors on `Pulse`; autograd on `flip_angle` is
    re-established in `seq_convert.integrate_pulse` via the
    `window_area / full_area` ratio trick.
  - Pulse-shape parameters (`duration` when used to shape the envelope,
    `time_bw_product`, `apodization`, `center_pos`, `slice_thickness`)
    are no longer differentiable — they flow through PyPulseq as numpy.
    These were never optimized in practice; see README §4 for the full
    differentiability list.
- `pulseqzero.Opts` is now a direct re-export of `pypulseq.Opts`. All
  fields (including 1.5 additions: `B0`, `adc_samples_limit`,
  `adc_samples_divisor`) come along for free.
- `Sequence.check_timing()` is a `(True, [])` stub. Real validation
  happens on `write()` — PyPulseq runs `check_timing` internally. Users
  wanting on-demand validation call `seq.to_pypulseq().check_timing()`
  explicitly. The stub keeps hot-loop callers warning-free.

### Fixed

- `make_sinc_pulse` / `make_gauss_pulse`: the old `generate_shape`
  closures called `float(flip_angle)` on a torch tensor, raising a
  scalar-conversion warning and silently breaking autograd. With the
  PyPulseq delegation in this release the code path is gone entirely.
- `seq_convert.integrate_pulse`: previously funneled the flip through
  numpy (`np.trapz` / `np.interp` / `.tolist()`), which severed
  gradients and triggered `RuntimeError: element 0 of tensors does not
  require grad and does not have a grad_fn` on `loss.backward()`. The
  windowed-flip computation now forms a grad-free `window_area /
  full_area` ratio and multiplies it by the live `rf.flip_angle`
  tensor. Switched the deprecated `np.trapz` to `np.trapezoid` at the
  same time.

### Acceptance

- `uv run demo/main.py` completes 30 Adam iterations end-to-end with
  monotonically-decreasing SAR and non-NaN data loss (matches
  pre-unification behavior).
- `demo/write_tse.py`, run once under `import pulseqzero as pp` and
  once under `import pypulseq as pp`, produces byte-identical `.seq`
  output (after stripping the `[SIGNATURE]` block). The translator
  round-trips.
