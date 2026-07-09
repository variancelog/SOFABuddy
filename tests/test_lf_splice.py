"""Headless regression tests for the LF-splice delay estimate (no Qt required).

Guards the fix for the DC-anchored-unwrap slip: on MEASURED HRTFs the lowest FFT
bins are near-zero-magnitude noise, so `np.unwrap(np.angle(H_raw))` (anchored at
DC) could commit a spurious 2*pi step. A single-point read of the excess phase at
the crossover then inflated the extrapolated LF delay by k*(fs/crossover) samples
(~128 @ 48k/1000), corrupting the sub-1 kHz phase of the spliced low end. The fix
estimates the delay from an unwrap-immune group-delay slope.

Property under test: the per-position LF delay the pipeline applies must be
physically bounded and tightly clustered (no per-position ~128-sample slips),
even when the LF phase is deliberately chaotic — and it must equal an independent
unwrap-immune group-delay computed from the same spectra.

These tests build synthetic fixtures (mirroring `_make_sofa` in test_grid_norm.py)
so no large binaries are needed. tests/ is gitignored -> `git add -f`.
"""

import argparse

import numpy as np
import pytest
import scipy.signal as signal
import scipy.fft as fft
import sofar as sf

from core.scripts.generate_sofa import master_sofa, PROCESSING_LENGTH


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _fibonacci_sphere(n):
    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    return az, el


def _make_measured_like_sofa(n_points=120, fs=48000, n=256, lf_noise=True, seed=0):
    """Synthetic SimpleFreeFieldHRIR whose LF content is near-zero with chaotic
    phase (like a real measured HRTF), plus a per-position bulk delay + ITD.

    A single-pole broadband burst placed at a known onset gives a well-behaved
    mid/high band; when lf_noise is True the sub-~120 Hz band is replaced by tiny
    random-phase content — the exact condition that baited the DC-unwrap slip.
    """
    rng = np.random.default_rng(seed)
    az, el = _fibonacci_sphere(n_points)
    M = len(az)
    freqs = fft.rfftfreq(n, 1.0 / fs)
    lf = freqs < 120.0

    tail = 0.5 ** np.arange(64)
    burst = np.zeros(64)
    burst[0] = 1.0
    burst += 0.4 * tail

    onset = 50
    ir = np.zeros((M, 2, n), dtype=float)
    for m in range(M):
        # small per-position ITD so delays are not identical
        itd = int(round(6 * np.sin(np.radians(az[m])) * np.cos(np.radians(el[m]))))
        for r, lvl in ((0, 0.30), (1, 0.28)):
            o = onset + (itd if r == 0 else -itd)
            ir[m, r, o:o + 64] = lvl * burst
        if lf_noise:
            H = fft.rfft(ir[m], axis=-1)
            # tiny magnitude, random phase in the LF band -> unwrap bait
            amp = 1e-4 * np.max(np.abs(H))
            noise = amp * np.exp(1j * rng.uniform(-np.pi, np.pi, size=(2, lf.sum())))
            H[:, lf] = noise
            ir[m] = fft.irfft(H, n=n, axis=-1)

    s = sf.Sofa("SimpleFreeFieldHRIR")
    s.Data_IR = ir
    s.Data_SamplingRate = float(fs)
    s.SourcePosition = np.column_stack((az, el, np.full(M, 1.5)))
    return s


def _args(**overrides):
    base = dict(
        harmonize_radius=False, target_radius=1.5,
        lf_extend=True, lf_crossover=1000.0,
        weighting_mode="auto", hrir_length=512, dfeq=False,
        processing_length=PROCESSING_LENGTH, norm_target_db=-1.0,
        onset_buffer=32, onset_threshold=0.05,
        blend_low_oct=0.50, blend_high_oct=0.75,
        ild_fix=False, debug_dump=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _group_delay_ref(H_raw, freqs, band):
    """Independent unwrap-immune per-position delay (samples-agnostic; seconds).
    Mirrors the pipeline's estimator but recomputed from scratch in the test."""
    mag = np.abs(H_raw)
    log_mag = np.log(mag + 1e-12)
    full = np.concatenate([log_mag, log_mag[..., -2:0:-1]], axis=-1)
    min_phase = -np.imag(signal.hilbert(full, axis=-1))[..., :H_raw.shape[-1]]
    excess = (H_raw / (mag + 1e-12)) * np.exp(-1j * min_phase)
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    dphi = np.angle(excess[..., idx[1:]] * np.conj(excess[..., idx[:-1]]))
    df = np.diff(freqs[idx])
    w = np.minimum(mag[..., idx[1:]], mag[..., idx[:-1]])
    num = np.sum(w * (-dphi) * (2 * np.pi * df), axis=-1)
    den = np.sum(w * (2 * np.pi * df) ** 2, axis=-1)
    return num / (den + 1e-30)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestLFDelayUnwrapImmunity:
    def test_delay_bounded_and_clustered_on_noisy_lf(self, tmp_path):
        """With chaotic near-zero LF phase, the applied per-position delay must
        stay physically bounded and tightly clustered — no ~128-sample slips.
        The pre-fix single-point estimate produced bimodal delays here."""
        raw = _make_measured_like_sofa(lf_noise=True)
        npz = tmp_path / "dump.npz"
        master_sofa(raw, 48000, False, str(tmp_path / "o.sofa"),
                    _args(debug_dump=str(npz)))
        d = dict(np.load(npz))
        fs = float(d["target_fs"]); ob = int(d["onset_buffer"])
        delay = d["delay_sec_arr"] * fs                    # samples
        slip = fs / float(d["crossover_f"])               # 128 @ 48k/1000

        assert delay.min() >= 0.0
        assert delay.max() <= ob + 128 + 1                 # within the clamp
        # No position is a full 2*pi slip away from the population median.
        spread = np.abs(delay - np.median(delay))
        assert spread.max() < slip * 0.5, (
            f"a position slipped ~{spread.max():.0f} samples (half-slip={slip*0.5:.0f})")
        assert delay.std() < 25.0

    def test_pipeline_delay_matches_independent_group_delay(self, tmp_path):
        """The delay the pipeline actually applies equals an unwrap-immune
        group-delay recomputed independently from the dumped spectra."""
        raw = _make_measured_like_sofa(lf_noise=True)
        npz = tmp_path / "dump.npz"
        master_sofa(raw, 48000, False, str(tmp_path / "o.sofa"),
                    _args(debug_dump=str(npz)))
        d = dict(np.load(npz))
        freqs = d["freqs"]; xf = float(d["crossover_f"])
        f_lo = xf / (2 ** 0.50)
        ref = _group_delay_ref(d["H_raw"], freqs, band=(f_lo, xf * 2.0))
        applied = d["delay_sec_arr"]
        # Agree to a fraction of a sample (both are the same estimator).
        assert np.max(np.abs(applied - ref)) * float(d["target_fs"]) < 1.0


class TestLFSpliceMagnitude:
    def test_below_crossover_shape_tracks_target(self, tmp_path):
        """The spliced LF diffuse-field magnitude tracks the scaled reference in
        SHAPE below the exact-match edge (anchor/level offset removed)."""
        raw = _make_measured_like_sofa(lf_noise=True)
        npz = tmp_path / "dump.npz"
        out_path = tmp_path / "o.sofa"
        master_sofa(raw, 48000, False, str(out_path), _args(debug_dump=str(npz)))
        d = dict(np.load(npz))
        freqs = d["freqs"]; xf = float(d["crossover_f"])
        f_lo = xf / (2 ** 0.50)
        band = (freqs > 20) & (freqs <= f_lo)

        from core.geometry_weights import calculate_geometric_weights
        out = sf.read_sofa(str(out_path))
        sp = np.asarray(out.SourcePosition, dtype=float)
        ir = np.asarray(out.Data_IR, dtype=float)
        n_fft = (len(freqs) - 1) * 2
        irp = np.pad(ir, ((0, 0), (0, 0), (0, max(0, n_fft - ir.shape[-1]))))[:, :, :n_fft]
        H = fft.rfft(irp, n=n_fft, axis=-1)
        w = calculate_geometric_weights(sp, advanced_dfeq=True, weighting_mode="auto")[:, None, None]
        df_mag = np.sqrt(np.sum(np.abs(H) ** 2 * w, axis=0))
        tgt = d["scaled_lf_target"]

        out_db = 20 * np.log10(df_mag + 1e-12)
        tgt_db = 20 * np.log10(tgt + 1e-12)
        delta = (out_db - tgt_db)[:, band]
        shape = delta - delta.mean(axis=-1, keepdims=True)   # anchor-immune
        shape_rms = np.sqrt((shape ** 2).mean())
        assert shape_rms < 0.35, f"LF magnitude shape rms {shape_rms:.3f} dB too high"


class TestLFSpliceCleanControl:
    def test_clean_file_delay_is_stable(self, tmp_path):
        """A clean (no-LF-noise) file yields a well-behaved delay too — the fix
        must not perturb the well-conditioned case."""
        raw = _make_measured_like_sofa(lf_noise=False)
        npz = tmp_path / "dump.npz"
        master_sofa(raw, 48000, False, str(tmp_path / "o.sofa"),
                    _args(debug_dump=str(npz)))
        d = dict(np.load(npz))
        fs = float(d["target_fs"])
        delay = d["delay_sec_arr"] * fs
        assert delay.min() >= 0.0
        assert delay.std() < 15.0


class TestLFCrossoverDefault:
    """Guards the crossover deduced by the 2026-07-07 HUTUBS-match investigation.

    Real measured HRTFs (ARI/BILI/LISTEN) deviate from the smooth diffuse-field
    target well above 400 Hz, and Brinkmann's HUTUBS references are subject-
    independent up to ~1.2 kHz. Splicing at the old 400 Hz put the blend seam
    inside the misbehaving measured band, whose min-phase reconstruction rings
    past a 256-sample crop. Raising the crossover to ~1 kHz moved the seam into
    the reliable, subject-specific band and made the measured files clean at 256
    on their own. This guards that the shipped default stays in that band — a
    revert to ~400 would reintroduce the 256-sample LF ripple.
    """

    def test_shipped_crossover_is_in_deduced_band(self):
        import core.config as C
        defaults = C.load_defaults()
        lo = 700.0
        for key in ("sofa.xover", "dfhrtf.xover"):
            section, name = key.split(".")
            val = defaults[section][name]
            hi = C.SCHEMA[key]["max"]
            assert lo <= val <= hi, (
                f"{key}={val} is outside the deduced LF-splice band "
                f"[{lo}, {hi}]; a low crossover reintroduces the 256-sample ring")

    def test_measured_like_fixture_256_512_equivalent_at_default(self, tmp_path):
        """At the shipped crossover, a rolled-off measured-like file's spliced
        diffuse-field magnitude is length-stable: 256 vs 512 samples agree in the
        LF band. This is the 256<->512 acceptance bar in miniature."""
        import core.config as C
        from core.geometry_weights import calculate_geometric_weights
        xover = float(C.load_defaults()["sofa"]["xover"])
        raw = _make_measured_like_sofa(lf_noise=True)

        def df_db(hrir_length):
            npz = tmp_path / f"d{hrir_length}.npz"
            out_path = tmp_path / f"o{hrir_length}.sofa"
            master_sofa(raw, 48000, False, str(out_path),
                        _args(lf_crossover=xover, hrir_length=hrir_length,
                              debug_dump=str(npz)))
            d = dict(np.load(npz)); freqs = d["freqs"]
            out = sf.read_sofa(str(out_path))
            sp = np.asarray(out.SourcePosition, dtype=float)
            ir = np.asarray(out.Data_IR, dtype=float)
            n_fft = (len(freqs) - 1) * 2
            irp = np.pad(ir, ((0, 0), (0, 0), (0, max(0, n_fft - ir.shape[-1]))))[:, :, :n_fft]
            H = fft.rfft(irp, n=n_fft, axis=-1)
            w = calculate_geometric_weights(sp, advanced_dfeq=True, weighting_mode="auto")[:, None, None]
            mag = np.sqrt(np.sum(np.abs(H) ** 2 * w, axis=0)).mean(0)
            return freqs, 20 * np.log10(mag + 1e-12)

        freqs, db256 = df_db(256)
        _, db512 = df_db(512)
        band = (freqs > 25) & (freqs <= xover * 2 ** 0.75)
        delta = (db256 - db512)[band]
        delta = delta - delta.mean()                       # level-immune
        rms = float(np.sqrt((delta ** 2).mean()))
        assert rms < 1.0, f"256 vs 512 LF diff {rms:.3f} dB exceeds 1.0 dB bar"
