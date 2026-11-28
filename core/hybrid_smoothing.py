"""
Hybrid smoothing ("Option 4"): an adaptive two-regime HRTF smoother.

Below the crossover (default 7.35 kHz) narrow dips are measurement artefacts
(rig reflections, concha jitter), so a TV-residual-weighted smoothing spline
bridges them. Above it, narrow high-Q peaks and notches are real pinna cues, so
a Huber-robust local cubic fit rejects 1-bin glitches while keeping multi-bin
notches and peaks. The two curves are blended with a logistic crossfade in
log-frequency.

The spline's smoothness s is chosen per curve from its own noise level (see
adaptive_spline_s()), so each ear, and each stereo curve, gets its own s.

hybrid_smooth() is the app entry point and is a two-stage pipeline:

  0. fixed 1/48-octave smoothing of the raw FFT spectrum onto the canonical
     1/48-octave grid (the DFHRTF output grid), reproducing the "1-48" curves
     the algorithm was validated on. The spline's smoothing factor is a
     whole-curve residual budget, so it must see that grid, not a dense FFT.
  1. smooth_option4(): the algorithm itself, on that grid.
  2. PCHIP in log-frequency back onto the requested output grid.

numpy only at import time; scipy is imported inside the functions that need it
so the backend scripts that never smooth do not pay for it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from core.smoothing import BAND_LIMIT_HZ, fixed_smooth, generate_fractional_octave_frequencies

# -- Algorithm constants not exposed as settings ------------------------------
CANONICAL_PPO = 48.0          # internal grid: 1/48 octave
RESIDUAL_SCALE_DB = 0.18      # TV-residual weight: 1 / (1 + (r / 0.18)^2)
HF_WEIGHT_BOOST = 2.2         # spline weight multiplier above the crossover
MAD_THRESHOLD_MULT = 1.8      # Huber threshold: max(delta, 1.8 * MAD)
MAD_SCALE = 1.4826            # MAD -> sigma
LEVEL_REF_HZ = 1000.0         # hybrid_smooth() works relative to this bin's level

# Exposed-parameter ranges (inclusive); the settings schema mirrors these.
XOVER_HZ_RANGE = (5500.0, 9000.0)
XOVER_WIDTH_OCT_RANGE = (0.01, 0.5)
TV_WEIGHT_RANGE = (0.05, 5.0)
S_FACTOR_RANGE = (0.5, 10.0)
S_BOUND_RANGE = (0.5, 30.0)       # s_min and s_max
HUBER_WINDOW_CHOICES = (5, 7, 9, 11, 13, 15)
HUBER_DELTA_RANGE = (0.05, 0.30)
HUBER_ITER_RANGE = (1, 20)


@dataclass(frozen=True)
class HybridParams:
    """Tunable parameters of the Hybrid smoother. Defaults are the validated ones.

    Keep in sync with dfhrtf.advanced.hybrid_* in default_settings.json, the
    --hybrid-* defaults in generate_dfhrtf.py and the fallbacks in
    dfhrtf_processor.py (tests/test_defaults_parity.py).
    """

    xover_hz: float = 7350.0
    xover_width_oct: float = 0.08
    tv_weight: float = 0.65
    s_factor: float = 3.0         # s = s_factor * N * sigma^2 (see adaptive_spline_s)
    s_min: float = 4.0
    s_max: float = 12.0
    huber_window: int = 9
    huber_delta: float = 0.12
    huber_iter: int = 6

    def __post_init__(self):
        def check(name, rng):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
            if not rng[0] <= v <= rng[1]:
                raise ValueError(f"{name} must be in [{rng[0]}, {rng[1]}], got {v!r}")

        check("xover_hz", XOVER_HZ_RANGE)
        check("xover_width_oct", XOVER_WIDTH_OCT_RANGE)
        check("tv_weight", TV_WEIGHT_RANGE)
        check("s_factor", S_FACTOR_RANGE)
        check("s_min", S_BOUND_RANGE)
        check("s_max", S_BOUND_RANGE)
        if self.s_min > self.s_max:
            raise ValueError(f"s_min ({self.s_min!r}) must not exceed s_max ({self.s_max!r})")
        check("huber_delta", HUBER_DELTA_RANGE)
        for name in ("huber_window", "huber_iter"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{name} must be an integer, got {v!r}")
        if self.huber_window % 2 == 0 or not (
                HUBER_WINDOW_CHOICES[0] <= self.huber_window <= HUBER_WINDOW_CHOICES[-1]):
            raise ValueError(
                f"huber_window must be an odd integer in "
                f"[{HUBER_WINDOW_CHOICES[0]}, {HUBER_WINDOW_CHOICES[-1]}], got {self.huber_window!r}")
        check("huber_iter", HUBER_ITER_RANGE)


# -- 1-D total-variation denoising (port of skimage's Chambolle) --------------

def _tv_denoise_1d(y, weight, eps=1e-6, max_num_iter=1000):
    """1-D Chambolle TV denoising; numerically identical to
    skimage.restoration.denoise_tv_chambolle on a 1-D float64 array.

    Chambolle's dual projection with step tau = 1/(2*ndim) = 0.5, stopping when
    the relative energy change |E_prev - E| < eps * E_init (never on the first
    iteration). The returned array is the one computed before the last dual
    update, as in skimage.
    """
    image = np.asarray(y, dtype=np.float64)
    if image.ndim != 1:
        raise ValueError("_tv_denoise_1d expects a 1-D array")
    n = image.size
    if n < 2:
        return image.copy()
    tau = 0.5
    p = np.zeros(n)
    g = np.zeros(n)
    out = image
    E_init = E_prev = 0.0
    for i in range(max_num_iter):
        if i > 0:
            d = -p
            d[1:] += p[:-1]          # negative divergence of p
            out = image + d
        else:
            d = np.zeros(n)
            out = image
        g[:-1] = np.diff(out)
        g[-1] = 0.0
        norm = np.abs(g)
        E = ((d ** 2).sum() + weight * norm.sum()) / n
        if i == 0:
            E_init = E_prev = E
            if E_init == 0.0:        # constant input: already a fixed point
                return image.copy()
        else:
            if abs(E_prev - E) < eps * E_init:
                break
            E_prev = E
        p -= tau * g
        p /= 1.0 + norm * (tau / weight)
    return out


def _median3_zero_pad(y):
    """3-point running median with zero padding (scipy.signal.medfilt(y, 3))."""
    p = np.concatenate(([0.0], y, [0.0]))
    return np.median(np.stack((p[:-2], p[1:-1], p[2:])), axis=0)


# -- Option 4 building blocks -------------------------------------------------

def _prebin_dense_fft(freqs, mag_db, max_ppo=96.0):
    """Average magnitude within 1/max_ppo-octave buckets where the input is
    denser than that; grids at or sparser than that pass through untouched."""
    log_f = np.log2(freqs)
    step = 1.0 / max_ppo
    if len(log_f) < 2 or np.min(np.diff(log_f)) >= step * 0.95:
        return freqs, mag_db

    edges = np.arange(log_f[0] - 0.5 * step, log_f[-1] + step, step)
    bin_idx = np.digitize(log_f, edges) - 1
    n_bins = len(edges) - 1

    counts = np.bincount(bin_idx, minlength=n_bins)
    valid_bins = counts > 0
    f_binned = 2.0 ** (
        np.bincount(bin_idx, weights=log_f, minlength=n_bins)[valid_bins]
        / counts[valid_bins]
    )
    m_binned = (
        np.bincount(bin_idx, weights=mag_db, minlength=n_bins)[valid_bins]
        / counts[valid_bins]
    )
    return f_binned, m_binned


def huber_local_cubic_1d(y, window_length=9, delta=0.12, n_iter=6):
    """Vectorised Huber IRLS local cubic regression (outlier-robust Savitzky-Golay).

    Down-weights isolated 1-2 bin spikes without flattening multi-bin peaks or
    deep notches. Returns the fitted value at each bin (the cubic's constant term).
    """
    if window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer.")

    half = window_length // 2
    n = len(y)
    y_pad = np.pad(y, half, mode="reflect")
    idx = np.arange(n)[:, None] + np.arange(window_length)[None, :]
    Y = y_pad[idx]  # (n, W)

    t = np.arange(-half, half + 1, dtype=float)
    V = np.vstack([np.ones_like(t), t, t**2, t**3]).T  # (W, 4)

    # Median-precleaned seed keeps 1-2 bin outliers out of the initial residuals.
    y_seed = np.pad(_median3_zero_pad(y), half, mode="reflect")[idx]
    coeffs = y_seed @ np.linalg.pinv(V).T
    y_hat = coeffs @ V.T

    for _ in range(n_iter):
        resid = np.abs(Y - y_hat)
        mad = np.median(resid, axis=1, keepdims=True) * MAD_SCALE
        thresh = np.maximum(delta, MAD_THRESHOLD_MULT * mad)
        w = np.where(resid <= thresh, 1.0, thresh / np.maximum(resid, 1e-8))

        VT_w = V.T[None, :, :] * w[:, None, :]
        A = np.einsum("niw,wj->nij", VT_w, V)
        b = np.einsum("niw,nw->ni", VT_w, Y)
        coeffs = np.linalg.solve(A, b[..., None])[..., 0]
        y_hat = coeffs @ V.T

    return coeffs[:, 0]


def adaptive_spline_s(y, tv_ref, grid_f, params):
    """Spline smoothness s for one curve, from its noise level below the crossover.

    The spline's s is a residual budget for the whole curve, so a noisy curve
    (e.g. ARI) spends a fixed budget on broad noise and keeps its narrow ripple.
    sigma is the RMS of the TV residual y - tv_ref below xover_hz (where the
    spline is the active regime); s = s_factor * N * sigma^2, clipped to
    [s_min, s_max]. y is the canonical-grid curve and N its length.
    """
    low = grid_f < params.xover_hz
    sigma2 = float(np.mean((y - tv_ref)[low] ** 2)) if np.any(low) else 0.0
    return float(np.clip(params.s_factor * len(y) * sigma2, params.s_min, params.s_max))


def smooth_option4(freqs, mag_db, params=None, canonical_ppo=CANONICAL_PPO,
                   s=None, return_s=False):
    """Option 4 adaptive hybrid smoother (faithful port of the reference).

    s: explicit spline smoothness. With s=None (the app's mode) it is chosen
    per curve by adaptive_spline_s(); an explicit value reproduces the
    reference algorithm exactly (the reference-parity tests use this). With
    return_s=True the result is (smoothed, s_used).

    Resamples to a 1/48-octave log grid (PCHIP, after pre-binning anything
    denser than 1/96 octave), runs the low/mid TV-weighted spline and the HF
    Huber cubic, crossfades them, and interpolates back to freqs. DC (0 Hz)
    is preserved. Two deliberate departures from the reference: the HF spline
    weight boost starts at xover_hz rather than a hard-coded 7500 Hz, and
    inputs denser than 1/96 octave (where the reference raised a shape error)
    are returned on their original bins.
    """
    from scipy.interpolate import PchipInterpolator, UnivariateSpline

    p = params if params is not None else HybridParams()
    freqs = np.asarray(freqs, dtype=float)
    mag_db = np.asarray(mag_db, dtype=float)

    valid = freqs > 0.0
    f_in, m_in = freqs[valid], mag_db[valid]

    # 1. Pre-bin dense FFT bins, resample onto the canonical grid
    log_in = np.log2(f_in)
    f_in, m_in = _prebin_dense_fft(f_in, m_in, max_ppo=canonical_ppo * 2.0)
    log_f = np.log2(f_in)
    grid_log_f = np.arange(log_f[0], log_f[-1] + 1e-9, 1.0 / canonical_ppo)
    grid_f = 2.0 ** grid_log_f
    y = PchipInterpolator(log_f, m_in)(grid_log_f)

    # 2. Low/mid regime: TV-residual-weighted smoothing spline
    x_bins = np.arange(len(grid_f), dtype=float)
    tv_ref = _tv_denoise_1d(y, p.tv_weight, eps=1e-6, max_num_iter=1000)
    s_used = float(s) if s is not None else adaptive_spline_s(y, tv_ref, grid_f, p)
    resid = np.abs(y - tv_ref)
    weights = (1.0 / (1.0 + (resid / RESIDUAL_SCALE_DB) ** 2)) * np.where(
        grid_f > p.xover_hz, HF_WEIGHT_BOOST, 1.0)
    with warnings.catch_warnings():
        # FITPACK warns when s is smaller than the data can reach (a nearly
        # noise-free curve interpolates); the interpolating fit is the intent.
        warnings.simplefilter("ignore", UserWarning)
        tv_spline = UnivariateSpline(x_bins, y, w=weights, s=s_used)(x_bins)

    # 3. High-frequency regime: Huber IRLS local cubic regression
    huber_hf = huber_local_cubic_1d(
        y, window_length=p.huber_window, delta=p.huber_delta, n_iter=p.huber_iter)

    # 4. Logistic log-frequency crossover, then back to the input grid
    alpha = 1.0 / (1.0 + np.exp((grid_log_f - np.log2(p.xover_hz)) / p.xover_width_oct))
    out_grid = alpha * tv_spline + (1.0 - alpha) * huber_hf

    out = mag_db.copy()
    # Evaluate at the ORIGINAL bins (log_in, not the pre-binned log_f): the
    # reference assigned the pre-binned length here and crashed on dense input.
    # Identical whenever pre-binning is a no-op.
    out[valid] = PchipInterpolator(grid_log_f, out_grid)(log_in)
    return (out, s_used) if return_s else out


# -- App entry point ----------------------------------------------------------

def hybrid_smooth(freqs, mag_db, params, out_freqs, mean_mode="power",
                  window="gaussian", f_max=BAND_LIMIT_HZ, return_s=False):
    """Hybrid smoothing of a raw spectrum, evaluated at out_freqs.

    Stage 0 smooths to the canonical 1/48-octave grid with the app's fixed
    smoother (mean_mode / window), Option 4 runs there, and the result is
    PCHIP-interpolated in log-frequency to out_freqs (identity for the default
    1/48 output grid; points beyond the canonical grid hold its edge value
    instead of extrapolating).

    The curve is processed relative to its level at 1 kHz and shifted back, so
    the result does not depend on absolute level (the reference was validated
    on curves normalised to 0 dB at 1 kHz, and a 3-point median with zero
    padding would otherwise make the top-edge fit level dependent).

    The spline smoothness s is chosen for this curve alone (adaptive_spline_s);
    with return_s=True the result is (smoothed, s_used).
    """
    from scipy.interpolate import PchipInterpolator

    out_freqs = np.asarray(out_freqs, dtype=float)
    if np.any(out_freqs <= 0):
        raise ValueError("out_freqs must be positive")
    canon = generate_fractional_octave_frequencies(20.0, f_max, fraction=int(CANONICAL_PPO))
    y0 = fixed_smooth(freqs, mag_db, CANONICAL_PPO, canon, mean_mode=mean_mode,
                      window=window, f_max=f_max)
    ref = y0[int(np.argmin(np.abs(canon - LEVEL_REF_HZ)))]
    y, s_used = smooth_option4(canon, y0 - ref, params, return_s=True)
    y = y + ref

    log_c = np.log2(canon)
    x = np.clip(np.log2(out_freqs), log_c[0], log_c[-1])
    out = PchipInterpolator(log_c, y)(x)
    return (out, s_used) if return_s else out
