"""Variable (frequency-dependent) fractional-octave smoothing.

Ported from the standalone frequency_smoother project. The smoothing bandwidth
at a given frequency is expressed as a fractional octave ``1/N``. ``N = 1`` is a
full octave (heavy smoothing); larger ``N`` is lighter smoothing. The bandwidth
varies with frequency between two anchors so that low and high frequencies can
be smoothed differently, in the spirit of REW's psychoacoustic smoothing.

All smoothing happens in log-frequency space with a Gaussian window, averaging
the magnitude in dB. Inputs must be dB magnitudes over a strictly increasing,
positive frequency axis. See ``SmoothingParams`` for the transfer-function
definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Window is evaluated out to this many sigma on each side before weights are
# negligible.  3 sigma captures ~99.7% of a Gaussian.
_WINDOW_SIGMA = 3.0


@dataclass(frozen=True)
class SmoothingParams:
    """Parameters for variable smoothing.

    Attributes:
        freq_low: Frequency (Hz) at/below which smoothing is fixed at ``n_low``.
        freq_hi: Frequency (Hz) at/above which smoothing is fixed at ``n_hi``.
        n_low: Octave-fraction denominator at ``freq_low``. ``1/n_low`` octave.
        n_hi: Octave-fraction denominator at ``freq_hi``. ``1/n_hi`` octave.
        gamma: Warps the transition midpoint. 1.0 = linear in log-frequency,
            >1 pushes the transition toward ``freq_hi``, <1 toward ``freq_low``.
        mean_mode: How values inside each window are combined. ``"db"`` averages
            in dB (a geometric mean of linear magnitude, which favors nulls).
            ``"cubic"`` uses REW's psychoacoustic cubic mean (cube root of the
            mean of cubed linear magnitudes), weighting peaks over nulls.
    """

    freq_low: float = 350.0
    freq_hi: float = 3000.0
    n_low: float = 1.0
    n_hi: float = 12.0
    gamma: float = 0.88
    mean_mode: str = "db"


def smoothing_profile(freq: np.ndarray, params: SmoothingParams) -> np.ndarray:
    """Return the octave-fraction *width* (1/N) applied at each input frequency.

    Interpolation is on the octave width (``1/N``), not on ``N`` itself, which keeps
    the transition perceptually even.  Returns an array the same shape as ``freq``.
    """
    freq = np.asarray(freq, dtype=float)
    width_low = 1.0 / params.n_low
    width_hi = 1.0 / params.n_hi

    lo = np.log2(max(params.freq_low, 1e-9))
    hi = np.log2(max(params.freq_hi, 1e-9))
    span = hi - lo

    if span <= 0:
        # Degenerate band: everything below the crossover gets the low width.
        return np.where(freq <= params.freq_low, width_low, width_hi)

    t = (np.log2(np.clip(freq, 1e-9, None)) - lo) / span
    t = np.clip(t, 0.0, 1.0)
    t_warped = t ** max(params.gamma, 1e-6)
    return width_low + (width_hi - width_low) * t_warped


def variable_smooth(
    freq: np.ndarray, mag_db: np.ndarray, params: SmoothingParams
) -> np.ndarray:
    """Apply frequency-dependent fractional-octave smoothing to a magnitude curve.

    Args:
        freq: Frequencies in Hz, strictly increasing. Shape (M,).
        mag_db: Magnitudes in dB at ``freq``. Shape (M,).
        params: Smoothing parameters.

    Returns:
        Smoothed magnitude in dB, same shape as ``mag_db``.
    """
    freq = np.asarray(freq, dtype=float)
    mag_db = np.asarray(mag_db, dtype=float)
    n = freq.size
    if n == 0:
        return mag_db.copy()
    if n != mag_db.size:
        raise ValueError("freq and mag_db must have the same length")

    log_f = np.log2(np.clip(freq, 1e-9, None))

    # Per-point Gaussian sigma in octaves. The octave fraction defines the window's
    # full width; treat that as the ~full-width-at-tenth-max so the visible window
    # spans roughly +/- half an octave-fraction. sigma = width / 2.355 maps the
    # fraction to a FWHM, a common smoothing convention.
    width_oct = smoothing_profile(freq, params)
    sigma = np.maximum(width_oct / 2.355, 1e-6)

    cubic = params.mean_mode == "cubic"
    # For the cubic mean we work in linear amplitude (always positive), cube it,
    # take the weighted mean, then the cube root, and convert back to dB.
    lin = 10.0 ** (mag_db / 20.0) if cubic else None

    out = np.empty(n, dtype=float)
    for i in range(n):
        s = sigma[i]
        half = _WINDOW_SIGMA * s
        lo_f = log_f[i] - half
        hi_f = log_f[i] + half
        # Window bounds via search on the monotonic log-frequency axis.
        lo_idx = np.searchsorted(log_f, lo_f, side="left")
        hi_idx = np.searchsorted(log_f, hi_f, side="right")
        if hi_idx - lo_idx <= 1:
            out[i] = mag_db[i]
            continue
        d = log_f[lo_idx:hi_idx] - log_f[i]
        w = np.exp(-0.5 * (d / s) ** 2)
        if cubic:
            seg = lin[lo_idx:hi_idx]
            cube_mean = np.dot(w, seg ** 3) / w.sum()
            out[i] = 20.0 * np.log10(cube_mean ** (1.0 / 3.0))
        else:
            out[i] = np.dot(w, mag_db[lo_idx:hi_idx]) / w.sum()
    return out
