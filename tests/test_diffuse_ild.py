"""
tests/test_diffuse_ild.py
Pytest guard for the diffuse-field ILD audit engine (core/ild_audit.py).

Test battery
------------
1. Synthetic recovery    — inject a known ILD via the `ild_sofa` fixture; assert
   ild_estimate_db recovers it within 0.15 dB.

2. Synthetic invariants  — round-trip an injected ILD through
   diffuse_field_ild() -> apply_ild_fix() and pin the four properties the
   correction promises: residual ILD ~0, gain_L * gain_R == 1, float32
   representability survives, and ITD is untouched.

3. LF-vs-HF decision     — an HF-only imbalance must NOT be called a
   calibration offset, even when the broadband ILD is large.

4. Healthy-file guard    — all non-offender fixtures in tests/data/sofa/ must
   have abs(lf_ild_db) <= LF_TOL and abs(broadband_ild_db) <= BROADBAND_TOL.

5. Known-offender guard  — fixtures with genuine LF-ILD (KNOWN_OFFENDERS) must
   be flagged (abs(lf_ild_db) > LF_TOL).

Tests 1-3 are fully synthetic and always run. Tests 4-5 exercise the real
downloaded SOFA corpus, which is gitignored for size — they are marked
`needs_sofa_fixtures` and skip loudly when tests/data/sofa/ holds no *.sofa
files. See tests/data/sofa/README.md.
"""

import glob
import os

import numpy as np
import pytest

from core.ild_audit import (
    apply_ild_fix,
    diffuse_field_ild,
    LF_TOL,
    BROADBAND_TOL,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOFA_DIR = os.path.join(os.path.dirname(__file__), "data", "sofa")

# Fixtures with genuine LF-ILD that exceeds LF_TOL — these are known offenders,
# not calibration-clean files, so they're excluded from the healthy-file guard
# and instead asserted to be *flagged* by test_known_offenders_are_flagged below.
KNOWN_OFFENDERS = {"measured_BILI.sofa", "measured_LISTEN.sofa"}


def _sofa_fixture_files():
    """Every *.sofa file in tests/data/sofa/.

    Globs for the extension specifically: a non-empty directory (e.g. one that
    only holds README.md) is NOT fixture coverage.
    """
    return sorted(glob.glob(os.path.join(SOFA_DIR, "*.sofa")))


_have_sofa_fixtures = bool(_sofa_fixture_files())

_no_fixtures_reason = (
    "no *.sofa files in tests/data/sofa/ — real-corpus regression tests need "
    "the (gitignored) HRTF database files; see tests/data/sofa/README.md"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median_itd_samples(ir):
    """Median cross-correlation lag (samples) between L and R across positions.

    Positive = L is delayed relative to R.
    """
    n = ir.shape[-1]
    lags = [
        int(np.argmax(np.correlate(ir[m, 0], ir[m, 1], mode="full"))) - (n - 1)
        for m in range(ir.shape[0])
    ]
    return float(np.median(lags))


def _planted_itd_sofa(ild_sofa, ild_db, itd_samples):
    """A known-ILD SOFA whose L/R channels are the SAME signal, shifted apart.

    `ild_sofa`'s default HRIRs are independent noise per ear, so an L-vs-R
    cross-correlation lag is meaningless there. This rebuilds Data_IR from a
    single shared source signal, delayed by +itd_samples/2 in L and
    -itd_samples/2 in R, with the R channel attenuated by `ild_db`. Grid,
    samplerate and convention all come from the fixture.
    """
    s = ild_sofa(ild_db)
    src = np.asarray(s.Data_IR)[:, 0, :].copy()
    half = itd_samples // 2
    ir = np.empty_like(np.asarray(s.Data_IR))
    ir[:, 0, :] = np.roll(src, half, axis=-1)
    ir[:, 1, :] = np.roll(src, -half, axis=-1) * 10.0 ** (-ild_db / 20.0)
    s.Data_IR = ir
    return s


# ---------------------------------------------------------------------------
# Test 1 — synthetic unit: known ILD is recovered within tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("injected_db", [0.0, 2.0, -3.5, 5.0])
def test_synthetic_ild_recovery(ild_sofa, injected_db):
    """ild_estimate_db should recover an injected ILD within 0.15 dB.

    The estimation band (ESTIMATE_BAND, 300-6 kHz) is what apply_ild_fix relies
    on, so that is the quantity checked.
    """
    r = diffuse_field_ild(ild_sofa(injected_db))

    assert abs(r["ild_estimate_db"] - injected_db) < 0.15, (
        f"Injected {injected_db:.1f} dB; recovered {r['ild_estimate_db']:.3f} dB"
    )


# ---------------------------------------------------------------------------
# Test 2 — round trip: measure -> correct -> re-measure leaves ~zero ILD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("injected_db", [3.7, -3.7])
def test_ild_fix_round_trip_residual_is_negligible(ild_sofa, injected_db):
    """diffuse_field_ild -> apply_ild_fix -> diffuse_field_ild must land on ~0 dB.

    With synthetic data there is no measurement noise, so the correction is
    exact to numerical precision; anything above 1e-9 dB means the estimate and
    the applied gain have drifted out of agreement (e.g. a factor-of-2 or sign
    error in the split).
    """
    s = ild_sofa(injected_db)
    before = diffuse_field_ild(s)
    assert abs(before["ild_estimate_db"] - injected_db) < 0.15

    apply_ild_fix(s, before["ild_estimate_db"])
    after = diffuse_field_ild(s)

    assert abs(after["ild_estimate_db"]) < 1e-9, (
        f"Residual ILD after correction: {after['ild_estimate_db']:.3e} dB "
        f"(injected {injected_db} dB)"
    )


def test_ild_fix_gain_product_is_unity(ild_sofa):
    """The split correction must satisfy gain_L * gain_R == 1.

    This is the documented energy-neutrality invariant of apply_ild_fix: L is
    attenuated by X/2 dB and R boosted by X/2 dB, so the geometric mean of the
    two channel amplitudes is preserved. Measured through the public API by
    comparing per-channel sample ratios before and after.
    """
    s = ild_sofa(3.7)
    before = np.asarray(s.Data_IR).copy()

    apply_ild_fix(s, 3.7)
    after = np.asarray(s.Data_IR)

    gain_L = after[:, 0, :] / before[:, 0, :]
    gain_R = after[:, 1, :] / before[:, 1, :]

    # Each channel gain must be a single scalar (no frequency/direction shaping).
    np.testing.assert_allclose(gain_L, gain_L.flat[0], rtol=1e-12)
    np.testing.assert_allclose(gain_R, gain_R.flat[0], rtol=1e-12)

    assert gain_L.flat[0] * gain_R.flat[0] == pytest.approx(1.0, abs=1e-12), (
        f"gain_L * gain_R = {gain_L.flat[0] * gain_R.flat[0]!r}, expected 1.0"
    )
    # And the correction went the right way: positive ILD (L louder) attenuates L.
    assert gain_L.flat[0] < 1.0 < gain_R.flat[0]


def test_ild_fix_preserves_float32_representability(ild_sofa):
    """float32-precision source data must stay float32-representable.

    Real measured HRTFs are stored at float32 precision. Multiplying by a
    float64 gain fills the lower mantissa bits, destroying the compressibility
    of the written file and roughly doubling its size — a silent failure mode
    with no functional symptom. apply_ild_fix detects float32-precision input
    and rounds back; this pins that behavior.
    """
    s = ild_sofa(3.7)
    ir_f32 = np.asarray(s.Data_IR).astype(np.float32).astype(np.float64)
    s.Data_IR = ir_f32

    apply_ild_fix(s, 3.7)
    out = np.asarray(s.Data_IR)

    assert np.array_equal(out, out.astype(np.float32).astype(np.float64)), (
        "apply_ild_fix upcast float32-precision data to true float64 — the "
        "written SOFA would roughly double in size."
    )


def test_ild_fix_preserves_itd(ild_sofa):
    """A planted +/-6-sample ITD must survive the ILD correction untouched.

    apply_ild_fix applies a frequency-flat scalar per channel, so inter-aural
    *timing* is by construction unaffected. This guards against a future
    "improvement" that reaches for a filter and silently smears the ITD — the
    single most perceptually important cue in an HRTF.
    """
    s = _planted_itd_sofa(ild_sofa, ild_db=3.7, itd_samples=6)

    before_itd = _median_itd_samples(np.asarray(s.Data_IR).copy())
    assert before_itd == 6.0, f"planted ITD not measurable: got {before_itd}"

    r = diffuse_field_ild(s)
    assert abs(r["ild_estimate_db"] - 3.7) < 0.15

    apply_ild_fix(s, r["ild_estimate_db"])
    after_itd = _median_itd_samples(np.asarray(s.Data_IR))

    assert after_itd == before_itd, (
        f"ITD changed from {before_itd} to {after_itd} samples across the "
        f"ILD correction"
    )


# ---------------------------------------------------------------------------
# Test 3 — the LF-vs-HF decision boundary
# ---------------------------------------------------------------------------

def test_hf_only_ild_is_not_flagged_as_calibration_offset(ild_sofa):
    """A large HF-only ILD must read as 'acoustic_or_balanced', not an offset.

    This is the whole premise of the audit: long wavelengths diffract around the
    head symmetrically, so a genuine acoustic ILD vanishes below ~500 Hz while a
    per-channel gain error does not. An 8 dB imbalance confined to 6-16 kHz is
    the head/pinna kind — it must be kept, not "corrected" — even though the
    broadband ILD it produces exceeds BROADBAND_TOL.
    """
    r = diffuse_field_ild(ild_sofa(8.0, band=(6000, 16000)))

    assert r["band_ild_db"][(6000, 16000)] > 4.0, (
        "premise failed: HF injection did not produce a large HF-band ILD"
    )
    assert abs(r["broadband_ild_db"]) > BROADBAND_TOL, (
        "premise failed: the HF-only injection should still move the "
        "broadband figure past BROADBAND_TOL — that is what makes this a "
        "meaningful negative case"
    )
    assert abs(r["lf_ild_db"]) <= LF_TOL, (
        f"lf_ild_db = {r['lf_ild_db']:.3f} dB — an HF-only injection must "
        f"leave the 100-500 Hz band clean"
    )
    assert r["verdict"] == "acoustic_or_balanced", (
        f"HF-only ILD wrongly verdicted '{r['verdict']}' — the LF band, not "
        f"the broadband figure, must drive the calibration-offset call"
    )


def test_lf_only_ild_is_flagged_as_calibration_offset(ild_sofa):
    """The positive half of the same boundary: an LF-band ILD IS flagged.

    Paired with the HF case above, this pins both sides of the decision rule
    rather than only the direction that happens to pass today.
    """
    r = diffuse_field_ild(ild_sofa(4.0, band=(100, 500)))

    assert abs(r["lf_ild_db"]) > LF_TOL
    assert r["verdict"] == "calibration_offset"


# ---------------------------------------------------------------------------
# Test 4 — healthy-file guard: all fixtures except the known offenders
# ---------------------------------------------------------------------------

def _healthy_sofa_files():
    """Return (name, path) for every .sofa fixture that is NOT a known offender."""
    return [
        (os.path.basename(p), p)
        for p in _sofa_fixture_files()
        if os.path.basename(p) not in KNOWN_OFFENDERS
        and not p.endswith("_ildfix.sofa")
    ]


@pytest.mark.needs_sofa_fixtures
@pytest.mark.skipif(not _have_sofa_fixtures, reason=_no_fixtures_reason)
@pytest.mark.parametrize("fname,fpath", _healthy_sofa_files())
def test_healthy_sofa_fixtures(fname, fpath):
    """Healthy fixtures should have small LF-ILD and small broadband ILD."""
    sofar = pytest.importorskip("sofar", reason="sofar not installed")

    sofa = sofar.read_sofa(fpath)
    r = diffuse_field_ild(sofa)

    assert abs(r["lf_ild_db"]) <= LF_TOL, (
        f"{fname}: lf_ild_db = {r['lf_ild_db']:.2f} dB "
        f"exceeds tolerance {LF_TOL} dB — possible calibration offset?"
    )
    assert abs(r["broadband_ild_db"]) <= BROADBAND_TOL, (
        f"{fname}: broadband_ild_db = {r['broadband_ild_db']:.2f} dB "
        f"exceeds tolerance {BROADBAND_TOL} dB."
    )


# ---------------------------------------------------------------------------
# Test 5 — known-offender guard: fixtures with genuine LF-ILD are flagged
# ---------------------------------------------------------------------------

def _known_offender_files():
    """Return (name, path) for known-offender fixtures present on disk."""
    return [
        (os.path.basename(p), p)
        for p in _sofa_fixture_files()
        if os.path.basename(p) in KNOWN_OFFENDERS
    ]


@pytest.mark.needs_sofa_fixtures
@pytest.mark.skipif(not _have_sofa_fixtures, reason=_no_fixtures_reason)
@pytest.mark.parametrize("fname,fpath", _known_offender_files())
def test_known_offenders_are_flagged(fname, fpath):
    """Known-offender fixtures (genuine LF-ILD) must exceed LF_TOL — this is a
    positive detection, not a defect: it confirms the audit engine correctly
    flags real calibration/measurement imbalance rather than being loosened
    to call these fixtures 'healthy'."""
    sofar = pytest.importorskip("sofar", reason="sofar not installed")

    sofa = sofar.read_sofa(fpath)
    r = diffuse_field_ild(sofa)

    assert abs(r["lf_ild_db"]) > LF_TOL, (
        f"{fname}: expected |lf_ild_db| > {LF_TOL} dB (known offender), "
        f"got {r['lf_ild_db']:.2f} dB — has this fixture been fixed upstream?"
    )
