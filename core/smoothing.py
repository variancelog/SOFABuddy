"""Fractional-octave smoothing of magnitude spectra (fixed and variable).

One smoother, :func:`smooth_spectrum`, serves every smoothing path in the app:
fixed 1/N-octave smoothing, frequency-dependent ("variable") smoothing, the
diffuse-field curves and the stereo curves.

Window (``window``)
    Centred on each output frequency on a log2-frequency axis, with nominal
    full width ``w(f)`` octaves. Fixed 1/N smoothing is ``w = 1/N`` everywhere;
    variable smoothing is ``w(f) = smoothing_profile(f)``. The same window is used
    for both, so a variable profile whose points all share one N is identical to
    fixed 1/N.

    ``"gaussian"`` (default): Gaussian kernel with sigma = w/sqrt(12), i.e. the
    same variance as the flat window, so "1/N" means the same amount of
    smoothing in either window. Its gain on a ripple of period P octaves is
    exp(-(pi*w/P)^2/6), which is never negative, so peaks and dips never
    invert. Cut at 5 sigma and integrated exactly against the piecewise-linear
    spectrum.

    ``"flat"``: a rectangular window of full width ``w``. Its ripple gain is
    sinc(w/P), which is NEGATIVE for P between w/2 and w: peaks and dips of a
    regular ripple in that range come out inverted. Kept so older output can be
    reproduced exactly.

Integration
    The spectrum is treated as piecewise linear between FFT bins on the
    log2(f) axis and integrated exactly over the window (fractional bins at the
    window ends). Every octave inside the window carries equal weight regardless
    of how many FFT bins fall in it, so there is no bin-count jitter at low
    frequencies, and a window narrower than the bin spacing simply returns the
    linearly interpolated value.

Band limit
    Only data with ``f <= f_max`` (default :data:`BAND_LIMIT_HZ`, the top of the
    DFHRTF output grid) is used, so the measurement's anti-alias roll-off above
    it never leaks into the band. A window that crosses the limit is clipped
    there and renormalised by the mass left inside the band (the clipped width
    for the flat window, the Gaussian mass for the Gaussian): at the limit it
    is one-sided. The same applies at the first positive-frequency bin; DC is
    excluded.

Averaging domain (``mean_mode``)
    ``"power"`` (default) mean of 10^(dB/10); ``"db"`` arithmetic mean of dB
    (favours nulls); ``"cubic"`` REW-style cube-root of the mean of cubed
    linear magnitude (favours peaks). Always ``db <= power <= cubic``.

Variable profile
    A list of points ``(freq, n, gamma)``. Below the first point the width is
    held at the first point's 1/N, above the last point at the last point's.
    Between two points the width 1/N is interpolated over log2(f), warped by
    t**gamma; ``gamma`` belongs to the point a ramp arrives at. Two points
    with the same N make a hold. One point is fixed 1/N.

Evaluation is vectorised and done directly at the requested output
frequencies, so callers need not smooth every FFT bin and interpolate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Top of the DFHRTF output grid (Hz). Data above this is excluded from every
# smoothing window; generate_dfhrtf.py builds its output grid up to this value.
BAND_LIMIT_HZ = 20000.0


def generate_fractional_octave_frequencies(start_freq, end_freq, fraction=6):
    """Log-spaced frequencies start, start*2^(1/fraction), ... up to end_freq.

    The DFHRTF output grid (see generate_dfhrtf.py, which re-exports this) and
    the Hybrid smoother's canonical grid. Built by repeated multiplication, so
    it differs by one boundary point from core/averaging.py's generator.
    """
    freqs = []
    f = start_freq
    while f <= end_freq:
        freqs.append(f)
        f *= 2**(1/fraction)
    return np.array(freqs)


WINDOWS = ("gaussian", "flat")

# mean_mode -> (dB -> averaging domain, averaging domain -> dB)
_MODES = {
    "power": (lambda db: 10.0 ** (db / 10.0), lambda v: 10.0 * np.log10(v)),
    "db": (lambda db: db, lambda v: v),
    "cubic": (lambda db: 10.0 ** (db * 3.0 / 20.0), lambda v: (20.0 / 3.0) * np.log10(v)),
}
MEAN_MODES = tuple(_MODES)

# Allowed ranges for a variable-profile point (also enforced by core/config.py
# and written into the settings JSON schema).
PROFILE_FREQ_RANGE = (20.0, 20000.0)
PROFILE_N_RANGE = (1.0, 48.0)
PROFILE_GAMMA_RANGE = (0.1, 10.0)


def _check_profile(points) -> tuple:
    """Validate and normalise a profile to a tuple of (freq, n, gamma) floats.

    Raises ValueError with a readable reason on any violation.
    """
    pts = tuple(tuple(float(v) for v in p) for p in points)
    if not pts:
        raise ValueError("variable smoothing profile needs at least one point")
    last = None
    for i, p in enumerate(pts):
        if len(p) != 3:
            raise ValueError(f"profile point {i + 1} must be (freq, n, gamma)")
        freq, n, gamma = p
        if not (PROFILE_FREQ_RANGE[0] <= freq <= PROFILE_FREQ_RANGE[1]):
            raise ValueError(
                f"profile point {i + 1}: freq {freq:g} Hz outside "
                f"{PROFILE_FREQ_RANGE[0]:g}-{PROFILE_FREQ_RANGE[1]:g}")
        if not (PROFILE_N_RANGE[0] <= n <= PROFILE_N_RANGE[1]):
            raise ValueError(
                f"profile point {i + 1}: n {n:g} outside "
                f"{PROFILE_N_RANGE[0]:g}-{PROFILE_N_RANGE[1]:g}")
        if not (PROFILE_GAMMA_RANGE[0] <= gamma <= PROFILE_GAMMA_RANGE[1]):
            raise ValueError(
                f"profile point {i + 1}: gamma {gamma:g} outside "
                f"{PROFILE_GAMMA_RANGE[0]:g}-{PROFILE_GAMMA_RANGE[1]:g}")
        if i == 0 and gamma != 1.0:
            raise ValueError("the first profile point cannot have a gamma")
        if last is not None and freq <= last:
            raise ValueError("profile frequencies must be strictly increasing")
        last = freq
    return pts


def profile_from_dicts(items) -> tuple:
    """Settings-file form (list of {freq, n, gamma?}) -> points tuple.

    Unknown keys, a missing freq/n, a non-list, or a gamma on the first point
    raise ValueError (as do the range/ordering rules of the profile).
    """
    if not isinstance(items, (list, tuple)):
        raise ValueError("profile must be a list of points")
    pts = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"profile point {i + 1} must be an object")
        extra = set(item) - {"freq", "n", "gamma"}
        if extra:
            raise ValueError(f"profile point {i + 1} has unknown keys {sorted(extra)}")
        if "freq" not in item or "n" not in item:
            raise ValueError(f"profile point {i + 1} needs 'freq' and 'n'")
        if i == 0 and "gamma" in item:
            raise ValueError("the first profile point cannot have a gamma")
        vals = (item["freq"], item["n"], item.get("gamma", 1.0))
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            raise ValueError(f"profile point {i + 1}: freq, n and gamma must be numbers")
        pts.append(vals)
    try:
        return _check_profile(pts)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from None


def profile_to_dicts(points) -> list:
    """Points tuple -> settings-file form (gamma omitted when 1.0)."""
    out = []
    for i, (freq, n, gamma) in enumerate(points):
        d = {"freq": freq, "n": n}
        if i > 0 and gamma != 1.0:
            d["gamma"] = gamma
        out.append(d)
    return out


def _num(x: float) -> str:
    s = repr(float(x))
    return s[:-2] if s.endswith(".0") else s


def format_profile(points) -> str:
    """Compact CLI form: ``"350:2,6000:12:0.88,14000:12,17500:4:1.6"``."""
    parts = []
    for i, (freq, n, gamma) in enumerate(points):
        s = f"{_num(freq)}:{_num(n)}"
        if i > 0 and gamma != 1.0:
            s += f":{_num(gamma)}"
        parts.append(s)
    return ",".join(parts)


def parse_profile(text: str) -> tuple:
    """Parse the compact ``freq:n[:gamma],...`` form into a points tuple.

    Raises ValueError on malformed text or any profile rule violation.
    """
    pts = []
    for i, chunk in enumerate(str(text).split(",")):
        fields = chunk.strip().split(":")
        if len(fields) not in (2, 3) or not all(f.strip() for f in fields):
            raise ValueError(f"bad profile point {chunk!r}; expected freq:n[:gamma]")
        try:
            nums = [float(f) for f in fields]
        except ValueError:
            raise ValueError(f"bad profile point {chunk!r}; expected numbers") from None
        if len(nums) == 3 and i == 0:
            raise ValueError("the first profile point cannot have a gamma")
        pts.append((nums[0], nums[1], nums[2] if len(nums) == 3 else 1.0))
    return _check_profile(pts)


@dataclass(frozen=True)
class SmoothingParams:
    """Parameters for variable smoothing.

    Attributes:
        points: The width profile as ``(freq_hz, n, gamma)`` tuples with strictly
            increasing frequencies (see the module docstring). ``gamma`` on the
            first point must be 1.0 (unused).
        mean_mode: How values inside each window are combined: ``"power"``,
            ``"db"`` or ``"cubic"`` (see the module docstring).
        window: one of :data:`WINDOWS`.
    """

    # NOTE: these mirror the `dfhrtf.advanced.variable_smoothing_profile`,
    # `smoothing_mean_mode` and `smoothing_window` factory defaults in
    # core/default_settings.json, the --var-smooth-profile / --smoothing-mean-mode
    # / --smoothing-window argparse defaults in core/scripts/generate_dfhrtf.py,
    # and the fallbacks in core/dfhrtf_processor.py.  Keep all four in sync
    # (tests/test_defaults_parity.py::test_variable_smoothing_four_way_sync).
    points: tuple = (
        (350.0, 2.0, 1.0),
        (6000.0, 12.0, 0.88),
        (14000.0, 12.0, 1.0),
        (17500.0, 4.0, 1.6),
    )
    mean_mode: str = "power"
    window: str = "gaussian"

    def __post_init__(self):
        object.__setattr__(self, "points", _check_profile(self.points))
        if self.window not in WINDOWS:
            raise ValueError(f"unknown window {self.window!r}; expected one of {WINDOWS}")


def smoothing_profile(freq: np.ndarray, params: SmoothingParams) -> np.ndarray:
    """Return the octave-fraction *width* (1/N) applied at each input frequency.

    Per segment the width (1/N, not N, which keeps the transition perceptually
    even) is interpolated over log2(f) and warped by t**gamma, where gamma is the
    one on the segment's upper point. Outside the first/last point the width is
    held. Returns an array the same shape as ``freq``.
    """
    freq = np.asarray(freq, dtype=float)
    pts = params.points
    width = np.array([1.0 / p[1] for p in pts])
    if len(pts) == 1:
        return np.full(freq.shape, width[0])
    xs = np.log2([p[0] for p in pts])
    gamma = np.maximum([p[2] for p in pts], 1e-6)

    lf = np.log2(np.clip(freq, 1e-9, None))
    seg = np.clip(np.searchsorted(xs, lf, side="right") - 1, 0, len(pts) - 2)
    t = np.clip((lf - xs[seg]) / (xs[seg + 1] - xs[seg]), 0.0, 1.0)
    t_warped = t ** gamma[seg + 1]
    return width[seg] + (width[seg + 1] - width[seg]) * t_warped


# The Gaussian window processes outputs in blocks of neighbouring points and only
# visits the source nodes its kernel reaches: nodes further than _GAUSS_REACH
# sigma are skipped (the tail mass beyond 5 sigma is ~5.7e-7 per side).
_BLOCK = 64
_GAUSS_REACH = 5.0


def _box_mean(x, v, xc, width):
    """Mean of the piecewise-linear ``v(x)`` over boxes of full ``width`` centred
    on ``xc``, clipped to [x[0], x[-1]] and renormalised. Domain-space result."""
    dx = np.diff(x)
    slope = np.diff(v) / dx
    # cum[i] = integral of the piecewise-linear v from x[0] to x[i]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dx)])

    def integral_to(xq):
        xq = np.clip(xq, x[0], x[-1])
        i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
        d = xq - x[i]
        return cum[i] + v[i] * d + 0.5 * slope[i] * d * d

    a = np.maximum(xc - width / 2.0, x[0])      # clip the window to the band
    b = np.minimum(xc + width / 2.0, x[-1])
    span = b - a
    ok = span > 1e-12
    return np.where(
        ok,
        (integral_to(b) - integral_to(a)) / np.where(ok, span, 1.0),
        np.interp(xc, x, v),                    # window outside/degenerate: edge value
    )


def _gauss_mean(x, v, xc, width):
    """Mean of piecewise-linear `v(x)` against a Gaussian of sigma `width/sqrt(12)`
    centred on `xc`, clipped to [x[0], x[-1]] and renormalised. Exact per segment.
    Each block of output points only visits the nodes within ``_GAUSS_REACH`` sigma
    (tail mass beyond that is ~6e-7 per side), so cost scales with the window
    width rather than with the whole grid. Domain-space result."""
    from scipy.special import ndtr      # lazy: keeps scipy out of GUI-process imports

    shape = np.shape(xc)
    xc = np.asarray(xc, dtype=float).ravel()
    sig = np.broadcast_to(np.asarray(width, dtype=float) / np.sqrt(12.0), shape).ravel()
    slope = np.diff(v) / np.diff(x)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    def dPhi(z):
        # Phi(z[:, 1:]) - Phi(z[:, :-1]) from one tail evaluation per node, so the
        # difference keeps full precision in both tails.
        tail = ndtr(-np.abs(z))
        t = np.where(z > 0, -tail, tail)
        return t[:, 1:] - t[:, :-1] + ((z[:, :-1] <= 0) & (z[:, 1:] > 0))

    order = np.argsort(xc, kind="stable")       # neighbours share nodes -> narrow blocks
    out = np.empty(xc.size)
    for s in range(0, xc.size, _BLOCK):
        idx = order[s:s + _BLOCK]
        c = xc[idx, None]
        sg = sig[idx, None]
        reach = _GAUSS_REACH * sg
        lo = max(int(np.searchsorted(x, (c - reach).min(), side="right")) - 1, 0)
        hi = min(int(np.searchsorted(x, (c + reach).max(), side="left")) + 1, x.size)
        if hi - lo < 2:                         # block entirely outside the band
            lo, hi = (0, 2) if lo == 0 else (x.size - 2, x.size)
        xn = x[lo:hi]
        z = (xn - c) / sg
        dphi = inv_sqrt_2pi * np.diff(np.exp(-0.5 * z * z), axis=1)
        sl = slope[lo:hi - 1]
        num = ((v[lo:hi - 1] + sl * (c - xn[:-1])) * dPhi(z) - sl * sg * dphi).sum(axis=1)
        zb = np.stack([(x[0] - c[:, 0]) / sg[:, 0], (x[-1] - c[:, 0]) / sg[:, 0]], axis=1)
        den = dPhi(zb)[:, 0]
        ok = den > 1e-12
        out[idx] = np.where(ok, num / np.where(ok, den, 1.0), np.interp(c[:, 0], x, v))
    return out.reshape(shape)


def smooth_spectrum(
    freq: np.ndarray,
    mag_db: np.ndarray,
    out_freq: np.ndarray,
    width_oct: float | np.ndarray,
    mean_mode: str = "power",
    f_max: float = BAND_LIMIT_HZ,
    window: str = "gaussian",
) -> np.ndarray:
    """Smooth a magnitude spectrum with a log-frequency window (see :data:`WINDOWS`).

    Args:
        freq: Source frequencies in Hz, strictly increasing. Shape (M,). Bins at
            or below 0 Hz (DC) and above ``f_max`` are ignored.
        mag_db: Magnitudes in dB at ``freq``. Shape (M,).
        out_freq: Positive frequencies (Hz) at which to evaluate the result.
            Evaluate at or below ``f_max``; points above it get a one-sided
            window that shrinks to nothing, then the last bin's value.
        width_oct: Nominal full window width in octaves (1/N for fixed
            smoothing), a scalar or an array broadcastable to ``out_freq``.
            Must be > 0.
        mean_mode: ``"power"``, ``"db"`` or ``"cubic"``.
        f_max: Upper band limit in Hz.
        window: ``"gaussian"`` or ``"flat"``.

    Returns:
        Smoothed magnitude in dB with the shape of ``out_freq``.

    Raises:
        ValueError: unknown ``mean_mode`` or ``window``, mismatched or
            non-increasing ``freq``, fewer than two usable bins, non-positive
            ``out_freq`` or ``width_oct``.
    """
    if mean_mode not in _MODES:
        raise ValueError(f"unknown mean_mode {mean_mode!r}; expected one of {MEAN_MODES}")
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; expected one of {WINDOWS}")
    freq = np.asarray(freq, dtype=float)
    mag_db = np.asarray(mag_db, dtype=float)
    if freq.ndim != 1 or freq.shape != mag_db.shape:
        raise ValueError("freq and mag_db must be 1-D arrays of the same length")
    if np.any(np.diff(freq) <= 0):
        raise ValueError("freq must be strictly increasing")
    keep = (freq > 0) & (freq <= f_max)
    if np.count_nonzero(keep) < 2:
        raise ValueError(f"need at least two bins in (0, {f_max}] Hz")
    out_freq = np.asarray(out_freq, dtype=float)
    if np.any(out_freq <= 0):
        raise ValueError("out_freq must be positive")
    width = np.broadcast_to(np.asarray(width_oct, dtype=float), out_freq.shape)
    if np.any(width <= 0):
        raise ValueError("width_oct must be positive")

    to_domain, to_db = _MODES[mean_mode]
    x = np.log2(freq[keep])
    v = to_domain(mag_db[keep])
    xc = np.log2(out_freq)
    mean_fn = _gauss_mean if window == "gaussian" else _box_mean
    return to_db(mean_fn(x, v, xc, width))


def fixed_smooth(
    freq: np.ndarray,
    mag_db: np.ndarray,
    fraction: float,
    out_freq: np.ndarray | None = None,
    mean_mode: str = "power",
    f_max: float = BAND_LIMIT_HZ,
    window: str = "gaussian",
) -> np.ndarray:
    """Fixed 1/``fraction``-octave smoothing; see :func:`smooth_spectrum`.

    ``out_freq`` defaults to ``freq`` (which must then be positive).
    """
    if fraction <= 0:
        raise ValueError("fraction must be positive")
    out = freq if out_freq is None else out_freq
    return smooth_spectrum(freq, mag_db, out, 1.0 / fraction, mean_mode, f_max, window)


def variable_smooth(
    freq: np.ndarray,
    mag_db: np.ndarray,
    params: SmoothingParams,
    out_freq: np.ndarray | None = None,
    f_max: float = BAND_LIMIT_HZ,
) -> np.ndarray:
    """Frequency-dependent smoothing: width at each point is ``smoothing_profile``.

    ``out_freq`` defaults to ``freq`` (which must then be positive); an empty
    ``freq``/``mag_db`` pair returns an empty array. The averaging domain is
    ``params.mean_mode`` and the window is ``params.window``. See
    :func:`smooth_spectrum`.
    """
    freq = np.asarray(freq, dtype=float)
    mag_db = np.asarray(mag_db, dtype=float)
    if freq.size != mag_db.size:
        raise ValueError("freq and mag_db must have the same length")
    if freq.size == 0 and out_freq is None:
        return mag_db.copy()
    out = freq if out_freq is None else np.asarray(out_freq, dtype=float)
    return smooth_spectrum(freq, mag_db, out, smoothing_profile(out, params),
                           params.mean_mode, f_max, params.window)
