"""Diagnostic harness for the LF-splice mismatch on measured SOFA files.

THROWAWAY diagnostics (gitignored). Attributes the ~1-2 dB below-crossover
magnitude delta and the sub-1 kHz phase delta on MEASURED files (vs SIMULATED,
which match perfectly) to the mechanisms catalogued in
~/.claude/plans/pick-up-the-lf-gentle-pony.md.

Runs master_sofa() in-process with --debug-dump (reuses the argparse.Namespace
pattern from test_grid_norm.py), then:

  * Magnitude: pre-crop |H_rep| DF vs scaled_lf_target (should be ~0 by
    construction below f_low) and post-crop output DF vs target. The difference
    IS the crop/fade contribution (mechanisms 3/4).
  * Phase: compares the pipeline's single-point DC-unwrap delay estimate against
    an unwrap-IMMUNE group-delay (cross-spectrum) fit over the blend band. A
    ~fs/crossover (=128 @ 48k/1000) sample gap on measured files but not
    simulated is the smoking gun for the DC-anchored unwrap slip (mechanism 1).

Usage:
    python tests/diag_lf_splice.py                 # full report on all fixtures
    python tests/diag_lf_splice.py E1|E3|E4        # single experiment
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import sofar as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.scripts.generate_sofa import master_sofa, PROCESSING_LENGTH  # noqa: E402
from core.geometry_weights import calculate_geometric_weights  # noqa: E402
from core.sofa_io import read_sofa_robust  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

FIXTURES = {
    "simulated_VARLOG (control)": os.path.join(HERE, "data", "sofa", "simulated_VARLOG.sofa"),
    "measured_HUTUBS (ref)":      os.path.join(HERE, "data", "sofa", "measured_HUTUBS.sofa"),
    "measured_HUTUBS_2 (ref)":    os.path.join(HERE, "data", "sofa", "measured_HUTUBS_2.sofa"),
    "measured_ARI":               os.path.join(HERE, "data", "sofa", "measured_ARI.sofa"),
    "measured_BILI":              os.path.join(HERE, "data", "sofa", "measured_BILI.sofa"),
    "measured_LISTEN":            os.path.join(HERE, "data", "sofa", "measured_LISTEN.sofa"),
}


def _args(**overrides):
    base = dict(
        harmonize_radius=False, target_radius=1.5,
        lf_extend=True, lf_crossover=1000.0,  # live config default (core/default_settings.json)
        weighting_mode="auto", hrir_length=512, dfeq=False,
        processing_length=PROCESSING_LENGTH, norm_target_db=-1.0,
        onset_buffer=32, onset_threshold=0.05,
        blend_low_oct=0.50, blend_high_oct=0.75,
        ild_fix=False, debug_dump=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def master_with_dump(input_path, **arg_overrides):
    """Master input_path in-process, returning (output_sofa, dump_dict)."""
    raw = read_sofa_robust(input_path)
    tmpdir = tempfile.mkdtemp(prefix="diag_lf_")
    out_sofa_path = os.path.join(tmpdir, "out.sofa")
    npz_path = os.path.join(tmpdir, "dump.npz")
    args = _args(debug_dump=npz_path, **arg_overrides)
    # Silence the pipeline's stdout chatter for a clean report.
    _stdout = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            master_sofa(raw, 48000, False, out_sofa_path, args)
        finally:
            sys.stdout = _stdout
    out = sf.read_sofa(out_sofa_path)
    dump = dict(np.load(npz_path))
    return out, dump


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def df_magnitude(hrirs, source_pos, n_fft, weighting_mode="auto"):
    """Geometry-weighted diffuse-field magnitude, zero-padded to n_fft so bins
    align with the pipeline's processing grid. Returns (R, n_bins)."""
    hrirs = np.asarray(hrirs, dtype=float)
    M, R, N = hrirs.shape
    if N < n_fft:
        hrirs = np.pad(hrirs, ((0, 0), (0, 0), (0, n_fft - N)))
    else:
        hrirs = hrirs[:, :, :n_fft]
    H = fft.rfft(hrirs, n=n_fft, axis=-1)
    w = calculate_geometric_weights(source_pos, advanced_dfeq=True,
                                    weighting_mode=weighting_mode)
    w = w[:, None, None]
    df_power = np.sum((np.abs(H) ** 2) * w, axis=0)
    return np.sqrt(df_power)  # (R, n_bins)


def df_magnitude_from_H(H, source_pos, weighting_mode="auto"):
    """DF magnitude directly from a complex spectrum array (M, R, bins)."""
    w = calculate_geometric_weights(source_pos, advanced_dfeq=True,
                                    weighting_mode=weighting_mode)
    w = w[:, None, None]
    return np.sqrt(np.sum((np.abs(H) ** 2) * w, axis=0))


def anchored_db(mag, idx_xover):
    """20log10(mag) anchored to 0 dB at the crossover (±2 bins) so an overall
    normalization gain cancels."""
    ref = np.mean(mag[..., max(0, idx_xover - 2):idx_xover + 3], axis=-1, keepdims=True)
    return 20 * np.log10((mag + 1e-12) / (ref + 1e-12))


def group_delay_fit(H, freqs, band):
    """Unwrap-IMMUNE per-position delay (seconds) from the cross-spectrum slope
    of the excess phase. Excess phase = angle(H) - minphase(|H|); consecutive-bin
    angle differences are wrapped to (-pi, pi], so a global 2*pi slip cancels.

    Returns delay_sec array of shape H.shape[:-1].
    """
    mag = np.abs(H)
    log_mag = np.log(mag + 1e-12)
    full = np.concatenate([log_mag, log_mag[..., -2:0:-1]], axis=-1)
    min_phase = -np.imag(signal.hilbert(full, axis=-1))[..., :H.shape[-1]]
    excess = (H / (mag + 1e-12)) * np.exp(-1j * min_phase)  # unit-modulus excess
    bandmask = (freqs >= band[0]) & (freqs <= band[1])
    idx = np.where(bandmask)[0]
    # consecutive-bin phase increments (immune to global wrap)
    dphi = np.angle(excess[..., idx[1:]] * np.conj(excess[..., idx[:-1]]))
    df = np.diff(freqs[idx])
    # magnitude-weight each increment by the smaller of the two adjacent |H|
    w = np.minimum(mag[..., idx[1:]], mag[..., idx[:-1]])
    # delay tau from dphi = -2*pi*df*tau  ->  weighted LSQ through origin
    num = np.sum(w * (-dphi) * (2 * np.pi * df), axis=-1)
    den = np.sum(w * (2 * np.pi * df) ** 2, axis=-1)
    return num / (den + 1e-30)


def summ(x):
    x = np.asarray(x).ravel()
    return f"mean={x.mean():7.1f}  median={np.median(x):7.1f}  std={x.std():6.1f}  [{x.min():.1f},{x.max():.1f}]"


# --------------------------------------------------------------------------- #
# Per-fixture analysis
# --------------------------------------------------------------------------- #
def analyze(label, path, hrir_length=512, quiet=False):
    out, d = master_with_dump(path, hrir_length=hrir_length)
    freqs = d["freqs"]
    fs = float(d["target_fs"])
    idx_xover = int(d["idx_xover"])
    xf = float(d["crossover_f"])
    f_low = xf / (2 ** 0.50)          # blend weight exactly 0 below here
    sp = np.asarray(out.SourcePosition, dtype=float)

    exact = (freqs > 20) & (freqs <= f_low)     # exact-match band
    blend = (freqs > f_low) & (freqs < xf * 2 ** 0.75)

    # --- magnitude: target is scaled_lf_target (per-ear), anchored at crossover
    tgt = d["scaled_lf_target"]                 # (R, bins)
    tgt_db = anchored_db(tgt, idx_xover)

    # pre-crop DF from H_rep (should equal target below f_low by construction)
    df_pre = df_magnitude_from_H(d["H_rep"], sp)
    pre_db = anchored_db(df_pre, idx_xover)

    # post-crop DF from the actual written output IRs
    df_post = df_magnitude(out.Data_IR, sp, n_fft=len(freqs) * 2 - 2)
    post_db = anchored_db(df_post, idx_xover)

    d_pre = pre_db - tgt_db
    d_post = post_db - tgt_db

    # Anchor-immune SHAPE error over the exact-match band: remove the per-ear
    # mean offset (the crossover-anchor artifact and any residual level) and
    # report the RMS ripple that remains. Pre-crop this is ~0 by construction
    # (|H_rep| == scaled_lf_target below f_low); post-crop it is pure crop damage.
    def shape_rms(delta):
        seg = delta[:, exact]
        return np.sqrt(((seg - seg.mean(axis=-1, keepdims=True)) ** 2).mean())
    shape_pre = shape_rms(d_pre)
    shape_post = shape_rms(d_post)

    # --- phase: pipeline single-point vs unwrap-immune group-delay
    delay_pipeline = d["delay_sec_arr"] * fs            # samples (M, R)
    delay_robust = group_delay_fit(d["H_raw"], freqs, band=(max(150, f_low), xf * 2)) * fs
    slip = fs / xf

    if not quiet:
        print(f"\n=== {label}   (hrir_length={hrir_length}) ===")
        print(f"  crossover={xf:.0f}Hz  f_low={f_low:.0f}Hz  one-2pi-slip={slip:.0f} samples")
        print(f"  MAG exact-band ({freqs[exact].min():.0f}-{f_low:.0f}Hz) dB delta vs target:")
        print(f"      pre-crop |H_rep| : maxabs={np.abs(d_pre[:, exact]).max():.3f}  rms={np.sqrt((d_pre[:, exact]**2).mean()):.3f}")
        print(f"      post-crop output : maxabs={np.abs(d_post[:, exact]).max():.3f}  rms={np.sqrt((d_post[:, exact]**2).mean()):.3f}")
        print(f"      crop contribution: maxabs={np.abs(d_post[:, exact]-d_pre[:, exact]).max():.3f}")
        print(f"      SHAPE rms (anchor-immune): pre-crop={shape_pre:.4f}  post-crop={shape_post:.4f}")
        print(f"  PHASE delay estimate (samples):")
        print(f"      pipeline (DC-unwrap, single-pt): {summ(delay_pipeline)}")
        print(f"      robust  (group-delay, immune)  : {summ(delay_robust)}")
        gap = delay_pipeline - delay_robust
        print(f"      gap pipeline-robust            : {summ(gap)}   (~{slip:.0f} => one 2pi slip)")

    return dict(label=label, d_pre=d_pre, d_post=d_post, exact=exact, blend=blend,
                delay_pipeline=delay_pipeline, delay_robust=delay_robust, slip=slip,
                freqs=freqs, fs=fs, xf=xf)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def experiment_full():
    print("#" * 78)
    print("# FULL REPORT — per-fixture magnitude + phase attribution (hrir_length=512)")
    print("#" * 78)
    for label, path in FIXTURES.items():
        if os.path.exists(path):
            analyze(label, path, hrir_length=512)
        else:
            print(f"\n[skip] missing fixture: {path}")


def experiment_E1():
    print("#" * 78)
    print("# E1 — crop vs splice: does the magnitude delta vanish at hrir_length=2048?")
    print("#" * 78)
    for label, path in FIXTURES.items():
        if not os.path.exists(path):
            continue
        r512 = analyze(label, path, hrir_length=512, quiet=True)
        r2048 = analyze(label, path, hrir_length=2048, quiet=True)
        e = r512["exact"]
        print(f"\n{label}")
        print(f"  post-crop exact-band maxabs dB delta @512 : {np.abs(r512['d_post'][:, e]).max():.3f}")
        print(f"  post-crop exact-band maxabs dB delta @2048: {np.abs(r2048['d_post'][:, e]).max():.3f}")
        print(f"  -> crop-attributable magnitude error      : "
              f"{np.abs(r512['d_post'][:, e]).max() - np.abs(r2048['d_post'][:, e]).max():+.3f} dB")


def experiment_E3():
    print("#" * 78)
    print("# E3 — unwrap-slip check: pipeline delay estimate vs unwrap-immune fit")
    print("#" * 78)
    for label, path in FIXTURES.items():
        if not os.path.exists(path):
            continue
        r = analyze(label, path, hrir_length=512, quiet=True)
        gap = (r["delay_pipeline"] - r["delay_robust"]).ravel()
        slip = r["slip"]
        n_slipped = np.sum(np.abs(gap) > slip * 0.5)
        print(f"\n{label}")
        print(f"  gap (pipeline-robust) samples: {summ(gap)}")
        print(f"  positions with |gap| > {slip*0.5:.0f} (=half a slip): "
              f"{n_slipped}/{gap.size}  ({100*n_slipped/gap.size:.0f}%)")
        print(f"  gap / slip ratio: mean={gap.mean()/slip:+.2f} cycles")


def experiment_E4():
    print("#" * 78)
    print("# E4 — delay injection on the SIMULATED control: does a clean file break")
    print("#      as we roll its IRs later (pre-impulse headroom / crop)?")
    print("#" * 78)
    ctrl = FIXTURES["simulated_VARLOG (control)"]
    if not os.path.exists(ctrl):
        print("[skip] control fixture missing")
        return
    raw0 = read_sofa_robust(ctrl)
    for k in (0, 8, 16, 32, 48, 64):
        raw = raw0.copy()
        raw.Data_IR = np.roll(np.asarray(raw0.Data_IR, dtype=float), k, axis=-1)
        tmpdir = tempfile.mkdtemp(prefix="diag_e4_")
        npz = os.path.join(tmpdir, "d.npz")
        args = _args(debug_dump=npz)
        _o = sys.stdout
        with open(os.devnull, "w") as dn:
            sys.stdout = dn
            try:
                master_sofa(raw, 48000, False, os.path.join(tmpdir, "o.sofa"), args)
            finally:
                sys.stdout = _o
        out = sf.read_sofa(os.path.join(tmpdir, "o.sofa"))
        d = dict(np.load(npz))
        freqs = d["freqs"]; idx_xover = int(d["idx_xover"]); xf = float(d["crossover_f"])
        f_low = xf / (2 ** 0.50)
        exact = (freqs > 20) & (freqs <= f_low)
        sp = np.asarray(out.SourcePosition, dtype=float)
        tgt_db = anchored_db(d["scaled_lf_target"], idx_xover)
        post_db = anchored_db(df_magnitude(out.Data_IR, sp, n_fft=len(freqs) * 2 - 2), idx_xover)
        dd = np.abs(post_db - tgt_db)[:, exact].max()
        print(f"  roll k={k:2d} samples -> exact-band maxabs dB delta = {dd:.3f}")


def _reconstruct_crop_shape(d, delay_mode):
    """Rebuild the post-crop diffuse-field magnitude from a dump under the OLD
    (single-point DC-unwrap) or NEW (group-delay) delay estimate — no re-master.
    Returns the anchor-immune below-f_low SHAPE rms (dB). DFEQ assumed off."""
    freqs = d["freqs"]; bw = d["blend_weights"]; xf = float(d["crossover_f"])
    idx_xover = int(d["idx_xover"]); fs = float(d["target_fs"])
    H_raw = d["H_raw"]; excess_phase = d["excess_phase"]; tgt = d["scaled_lf_target"]
    M, R, B = H_raw.shape
    mag = np.abs(H_raw)
    full = np.concatenate([np.log(mag + 1e-12), np.log(mag + 1e-12)[..., -2:0:-1]], axis=-1)
    min_phase = -np.imag(signal.hilbert(full, axis=-1))[..., :B]
    H_raw_min = mag * np.exp(1j * min_phase)
    Hsyn = np.zeros_like(H_raw)
    for r in range(R):
        lb = np.log(tgt[r] + 1e-12)
        mpb = -np.imag(signal.hilbert(np.concatenate([lb, lb[-2:0:-1]])))[:B]
        Hsyn[:, r, :] = tgt[r] * np.exp(1j * mpb)
    H_rep_min = Hsyn * (1.0 - bw) + H_raw_min * bw

    if delay_mode == "old":
        delay = -np.mean(excess_phase[..., max(0, idx_xover-2):idx_xover+3], axis=-1) / (2 * np.pi * xf)
    else:
        excess_c = (H_raw / (mag + 1e-12)) * np.exp(-1j * min_phase)
        f_lo = xf / (2 ** 0.50); idx = np.where((freqs >= f_lo) & (freqs <= xf * 2.0))[0]
        dphi = np.angle(excess_c[..., idx[1:]] * np.conj(excess_c[..., idx[:-1]]))
        df = np.diff(freqs[idx]); w = np.minimum(mag[..., idx[1:]], mag[..., idx[:-1]])
        delay = np.sum(w * (-dphi) * (2*np.pi*df), axis=-1) / (np.sum(w * (2*np.pi*df)**2, axis=-1) + 1e-30)
        delay = np.clip(delay, 0.0, (int(d["onset_buffer"]) + 128) / fs)
    lin = -2 * np.pi * freqs[None, None, :] * delay[..., None]
    blended = lin * (1.0 - bw) + excess_phase * bw
    H_rep = H_rep_min * np.exp(1j * blended)

    n_fft = (B - 1) * 2
    hrirs = fft.irfft(H_rep, n=n_fft, axis=-1)[:, :, :512]
    hrirs[:, :, :8] *= np.hanning(16)[:8]
    hrirs[:, :, -32:] *= np.hanning(64)[32:]
    hp = np.pad(hrirs, ((0, 0), (0, 0), (0, n_fft - 512)))
    H = fft.rfft(hp, n=n_fft, axis=-1)
    # uniform DF weights are fine for a shape-only comparison
    df_mag = np.sqrt(np.mean(np.abs(H) ** 2, axis=0))
    f_lo = xf / (2 ** 0.50); band = (freqs > 20) & (freqs <= f_lo)
    delta = (20*np.log10(df_mag + 1e-12) - 20*np.log10(tgt + 1e-12))[:, band]
    shape = delta - delta.mean(axis=-1, keepdims=True)
    return np.sqrt((shape ** 2).mean())


def experiment_prepost():
    print("#" * 78)
    print("# PREPOST — did the delay fix change below-crossover magnitude SHAPE?")
    print("#          (reconstructs OLD vs NEW pipeline from one dump; DFEQ off)")
    print("#" * 78)
    for label, path in FIXTURES.items():
        if not os.path.exists(path):
            continue
        _o = sys.stdout
        with open(os.devnull, "w") as dn:
            sys.stdout = dn
            try:
                _, d = master_with_dump(path, hrir_length=512)
            finally:
                sys.stdout = _o
        old = _reconstruct_crop_shape(d, "old")
        new = _reconstruct_crop_shape(d, "new")
        print(f"  {label:28s}  OLD shape rms={old:.4f}  NEW shape rms={new:.4f}  "
              f"delta={new-old:+.4f} dB")


if __name__ == "__main__":
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "FULL"
    {"FULL": experiment_full, "E1": experiment_E1, "E3": experiment_E3,
     "E4": experiment_E4, "PREPOST": experiment_prepost}.get(which, experiment_full)()
