"""Regenerate tests/data/hybrid_reference.npz, the golden fixture for the Hybrid smoother.

Maintainer tool, not part of the shipped app. It needs scikit-image, which the app
deliberately does not depend on, so run it from a throwaway virtual environment:

    python -m venv hybridvenv
    hybridvenv/Scripts/pip install numpy scipy scikit-image
    hybridvenv/Scripts/python tools/build_hybrid_reference.py --curves-dir <dir>

<dir> holds SOFA Buddy 1/48-octave DFHRTF text outputs (freq<TAB>dB per line). Pick
curves by file name in CURVE_FILES below; the files are read, never modified.

The fixture holds
  * tv_*   : skimage.restoration.denoise_tv_chambolle inputs/outputs, to check the
             numpy port core.hybrid_smoothing._tv_denoise_1d against;
  * ref_*  : the Option 4 reference algorithm (verbatim from its author, with ONE
             deliberate change: the high-frequency spline weight boost starts at
             xover_hz instead of a hard-coded 7500 Hz) run on real and synthetic
             curves, to check smooth_option4;
  * e2e_*  : a raw FFT-like spectrum plus the reference result for the app's own
             stage-0 output (quantised to 0.001 dB like the text files), to check
             hybrid_smooth end to end;
  * versions of numpy / scipy / scikit-image that produced it.
"""
import argparse
import os
import sys

import numpy as np
import scipy
import skimage
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.signal import medfilt
from skimage.restoration import denoise_tv_chambolle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from core.smoothing import fixed_smooth, generate_fractional_octave_frequencies  # noqa: E402

CURVE_FILES = [
    "HUTUBS-meas pp001 L 1-48.txt",   # ~14.69 kHz single-bin glitch
    "HUTUBS-meas pp003 R 1-48.txt",   # deep notch
    "HRIR_ITA1_ECEbl L 1-48.txt",
    "HRIR_ITA4_ECEbl R 1-48.txt",
    "HRIR_EK06AS04_ECEbl L 1-48.txt",
    "SS2 AKO536081622 L Variable.txt",
]


# ---- Option 4 reference (verbatim, except the xover_hz weight boost) --------

def _prebin_dense_fft(freqs, mag_db, max_ppo=96.0):
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
    if window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer.")

    half = window_length // 2
    n = len(y)
    y_pad = np.pad(y, half, mode="reflect")
    idx = np.arange(n)[:, None] + np.arange(window_length)[None, :]
    Y = y_pad[idx]  # (n, W)

    t = np.arange(-half, half + 1, dtype=float)
    V = np.vstack([np.ones_like(t), t, t**2, t**3]).T  # (W, 4)

    y_seed = np.pad(medfilt(y, kernel_size=3), half, mode="reflect")[idx]
    coeffs = y_seed @ np.linalg.pinv(V).T
    y_hat = coeffs @ V.T

    for _ in range(n_iter):
        resid = np.abs(Y - y_hat)
        mad = np.median(resid, axis=1, keepdims=True) * 1.4826
        thresh = np.maximum(delta, 1.8 * mad)
        w = np.where(resid <= thresh, 1.0, thresh / np.maximum(resid, 1e-8))

        VT_w = V.T[None, :, :] * w[:, None, :]
        A = np.einsum("niw,wj->nij", VT_w, V)
        b = np.einsum("niw,nw->ni", VT_w, Y)
        coeffs = np.linalg.solve(A, b[..., None])[..., 0]
        y_hat = coeffs @ V.T

    return coeffs[:, 0]


def smooth_option4_huber_hybrid(freqs, mag_db, xover_hz=7350.0, xover_width_oct=0.08,
                                canonical_ppo=48.0, tv_weight=0.65, spline_smoothness=6.5,
                                huber_window=9, huber_delta=0.12, huber_iter=6):
    freqs = np.asarray(freqs, dtype=float)
    mag_db = np.asarray(mag_db, dtype=float)

    valid = freqs > 0.0
    f_in, m_in = freqs[valid], mag_db[valid]

    f_in, m_in = _prebin_dense_fft(f_in, m_in, max_ppo=canonical_ppo * 2.0)
    log_f = np.log2(f_in)
    grid_log_f = np.arange(log_f[0], log_f[-1] + 1e-9, 1.0 / canonical_ppo)
    grid_f = 2.0 ** grid_log_f
    y = PchipInterpolator(log_f, m_in)(grid_log_f)

    x_bins = np.arange(len(grid_f), dtype=float)
    tv_ref = denoise_tv_chambolle(y, weight=tv_weight, eps=1e-6, max_num_iter=1000)
    resid = np.abs(y - tv_ref)
    # DELIBERATE CHANGE: the reference hard-coded 7500.0 here; the weight boost
    # belongs at the crossover.
    weights = (1.0 / (1.0 + (resid / 0.18) ** 2)) * np.where(grid_f > xover_hz, 2.2, 1.0)
    tv_spline = UnivariateSpline(x_bins, y, w=weights, s=spline_smoothness)(x_bins)

    huber_hf = huber_local_cubic_1d(y, window_length=huber_window, delta=huber_delta,
                                    n_iter=huber_iter)

    alpha = 1.0 / (1.0 + np.exp((grid_log_f - np.log2(xover_hz)) / xover_width_oct))
    out_grid = alpha * tv_spline + (1.0 - alpha) * huber_hf

    out = mag_db.copy()
    out[valid] = PchipInterpolator(grid_log_f, out_grid)(log_f)
    return out


# ---- fixture assembly -------------------------------------------------------

def load_curve(path):
    d = np.loadtxt(path, delimiter="\t")
    return d[:, 0], d[:, 1]


def synthetic_curves(grid):
    rng = np.random.default_rng(20261127)
    lf = np.log2(grid / 1000.0)
    base = 3.0 * np.exp(-((np.log2(grid / 3000.0)) / 0.6) ** 2) - 2.0 * lf * (lf > 0)
    noisy = base + rng.normal(0, 0.15, grid.size)
    noisy += -3.0 * np.exp(-((np.log2(grid / 6500.0)) / 0.015) ** 2)    # narrow 6.5 kHz dip
    noisy += 4.0 * (np.abs(grid - 14690.0) < 60)                         # 1-bin spike
    noisy += -12.0 * np.exp(-((np.log2(grid / 11000.0)) / 0.03) ** 2)    # real HF notch
    noisy -= noisy[np.argmin(np.abs(grid - 1000.0))]
    out = {"syn_1_48": (grid, noisy)}
    g24 = generate_fractional_octave_frequencies(20.0, 20000.0, 24)      # sparser grid
    out["syn_1_24"] = (g24, np.interp(np.log2(g24), np.log2(grid), noisy))
    return out


def tv_cases():
    rng = np.random.default_rng(7)
    cases = []
    n = 479
    walk = np.cumsum(rng.normal(0, 0.05, n)) + rng.normal(0, 0.2, n)
    cases.append((walk, 0.65, 1e-6, 1000))
    cases.append((walk, 0.05, 1e-6, 1000))
    cases.append((walk, 5.0, 1e-6, 1000))
    cases.append((walk, 0.65, 2e-4, 200))                # skimage defaults
    cases.append((walk, 0.65, 1e-6, 25))                 # stops on max_num_iter
    step = np.where(np.arange(200) < 100, 0.0, 3.0) + rng.normal(0, 0.3, 200)
    cases.append((step, 1.0, 1e-6, 1000))
    cases.append((np.full(50, 2.5), 0.65, 1e-6, 1000))   # constant
    cases.append((rng.normal(0, 1, 5), 0.65, 1e-6, 1000))
    cases.append((rng.normal(0, 1, 2), 0.65, 1e-6, 1000))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-dir", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "data", "hybrid_reference.npz"))
    a = ap.parse_args()

    fx = {
        "versions": np.array([f"numpy {np.__version__}", f"scipy {scipy.__version__}",
                              f"scikit-image {skimage.__version__}"]),
    }

    for i, (y, w, eps, it) in enumerate(tv_cases()):
        fx[f"tv_{i}_in"] = y
        fx[f"tv_{i}_params"] = np.array([w, eps, it], dtype=float)
        fx[f"tv_{i}_out"] = denoise_tv_chambolle(y, weight=w, eps=eps, max_num_iter=it)

    curves = {}
    for name in CURVE_FILES:
        f, m = load_curve(os.path.join(a.curves_dir, name))
        curves[os.path.splitext(name)[0]] = (f, m)
    grid = curves[os.path.splitext(CURVE_FILES[0])[0]][0]
    curves.update(synthetic_curves(grid))
    names = []
    for j, (name, (f, m)) in enumerate(curves.items()):
        names.append(name)
        fx[f"ref_{j}_freqs"] = f
        fx[f"ref_{j}_in"] = m
        fx[f"ref_{j}_out"] = smooth_option4_huber_hybrid(f, m)
    fx["ref_names"] = np.array(names)

    # non-default parameters on one curve
    f, m = curves[names[0]]
    kw = dict(xover_hz=6000.0, xover_width_oct=0.15, tv_weight=1.2, spline_smoothness=4.0,
              huber_window=11, huber_delta=0.2, huber_iter=4)
    fx["ref_custom_freqs"], fx["ref_custom_in"] = f, m
    fx["ref_custom_out"] = smooth_option4_huber_hybrid(f, m, **kw)
    fx["ref_custom_params"] = np.array([kw["xover_hz"], kw["xover_width_oct"], kw["tv_weight"],
                                        kw["spline_smoothness"], kw["huber_window"],
                                        kw["huber_delta"], kw["huber_iter"]], dtype=float)

    # end to end: raw FFT-like spectrum -> app stage 0 -> reference
    rng = np.random.default_rng(11)
    raw_f = np.fft.rfftfreq(16384, 1 / 48000.0)[1:]
    raw_m = np.interp(np.log2(raw_f), np.log2(f), m) + rng.normal(0, 0.3, raw_f.size) + 7.0
    canon = generate_fractional_octave_frequencies(20.0, 20000.0, 48)
    stage0 = fixed_smooth(raw_f, raw_m, 48, canon, mean_mode="power", window="gaussian")
    stage0 -= stage0[np.argmin(np.abs(canon - 1000.0))]
    txt = np.round(stage0, 3)                       # what a "1-48" text file holds
    fx["e2e_raw_freqs"], fx["e2e_raw_mags"] = raw_f, raw_m
    fx["e2e_canon_freqs"] = canon
    fx["e2e_ref_out"] = smooth_option4_huber_hybrid(canon, txt)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("wrote", a.out, os.path.getsize(a.out), "bytes;", "; ".join(fx["versions"]))


if __name__ == "__main__":
    main()
