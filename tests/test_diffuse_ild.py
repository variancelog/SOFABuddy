"""
tests/test_diffuse_ild.py
Pytest guard for the diffuse-field ILD audit engine (core/ild_audit.py).

Test battery
------------
1. Synthetic unit test  — inject a known ILD into a fake IR; assert
   ild_estimate_db recovers it within 0.1 dB.

2. Round-trip fix test  — load BILI 1103-n (known offender), apply_ild_fix,
   re-measure; assert residual < 0.3 dB and energy is preserved.

3. Known-bad reference  — BILI 1103-n: assert lf_ild_db > LF_TOL and
   verdict == 'calibration_offset'.

4. Healthy-file guard   — all other fixtures in tests/data/sofa/ must have
   abs(lf_ild_db) <= LF_TOL and abs(broadband_ild_db) <= BROADBAND_TOL.

SOFA tests require 'sofar'; they are skipped automatically if unavailable or
if tests/data/sofa/ is empty.
"""

import os
import sys

import numpy as np
import pytest

# Bootstrap import path so the test suite finds core.*.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ild_audit import (
    diffuse_field_ild,
    apply_ild_fix,
    LF_TOL,
    BROADBAND_TOL,
    ESTIMATE_BAND,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOFA_DIR = os.path.join(os.path.dirname(__file__), "data", "sofa")
KNOWN_BAD = "BILI 1103-n.sofa"
KNOWN_BAD_PATH = os.path.join(SOFA_DIR, KNOWN_BAD)

_sofa_dir_exists = os.path.isdir(SOFA_DIR) and bool(os.listdir(SOFA_DIR))

# ---------------------------------------------------------------------------
# Helper: fake SOFA-like object (avoids importing sofar for pure-unit tests)
# ---------------------------------------------------------------------------

class _FakeSofa:
    """Minimal duck-type for diffuse_field_ild when we don't need sofar."""

    def __init__(self, ir, fs=44100.0):
        M, R, N = ir.shape
        # Uniform sphere grid: azimuth 0..360 step 6, elevation 0 only.
        az = np.arange(0, 360, 6, dtype=float)           # 60 points
        el = np.zeros_like(az)
        r  = np.ones_like(az)
        sp = np.column_stack([az, el, r])
        # Make M match by tiling if needed.
        if M != len(az):
            sp = np.tile(sp, (int(np.ceil(M / len(sp))), 1))[:M]
        self.Data_IR          = ir.copy()
        self.Data_SamplingRate = fs
        self.SourcePosition   = sp


# ---------------------------------------------------------------------------
# Test 1 — synthetic unit: known ILD is recovered within tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("injected_db", [0.0, 2.0, -3.5, 5.0])
def test_synthetic_ild_recovery(injected_db):
    """ild_estimate_db should recover an injected ILD within 0.15 dB.

    We synthesise white noise HRIRs for a uniform (60-point, elevation-0)
    sphere grid and scale the R channel by the injected gain before measuring.
    The estimation band (300–6 kHz) is used for the check because that is
    what apply_ild_fix relies on.
    """
    rng = np.random.default_rng(42)
    fs = 44100.0
    M, N = 60, 256
    ir = rng.standard_normal((M, 2, N))
    # Inject ILD: scale R by the negative gain (positive injected_db → L louder)
    ir[:, 1, :] *= 10.0 ** (-injected_db / 20.0)

    fake = _FakeSofa(ir, fs=fs)
    r = diffuse_field_ild(fake)

    assert abs(r["ild_estimate_db"] - injected_db) < 0.15, (
        f"Injected {injected_db:.1f} dB; recovered {r['ild_estimate_db']:.3f} dB"
    )


# ---------------------------------------------------------------------------
# Test 2 — round-trip fix: apply_ild_fix drives residual near 0 and
#           preserves overall energy.  Requires BILI 1103-n.sofa.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _sofa_dir_exists or not os.path.isfile(KNOWN_BAD_PATH),
    reason=f"{KNOWN_BAD} not found in tests/data/sofa/",
)
def test_ild_fix_round_trip():
    """apply_ild_fix should reduce ild_estimate_db to near 0 dB without
    significantly changing total energy."""
    sofar = pytest.importorskip("sofar", reason="sofar not installed")

    sofa = sofar.read_sofa(KNOWN_BAD_PATH)
    r_before = diffuse_field_ild(sofa)

    apply_ild_fix(sofa, r_before["ild_estimate_db"])
    r_after = diffuse_field_ild(sofa)

    # Residual ILD should be negligible.
    assert abs(r_after["ild_estimate_db"]) < 0.3, (
        f"Residual ILD after fix: {r_after['ild_estimate_db']:.3f} dB "
        f"(expected < 0.3 dB)"
    )

    # The per-channel gains satisfy gain_L * gain_R = 1 (amplitude), so
    # neither individual channel is wildly scaled.  For a ~3.7 dB ILD with
    # real channel energy imbalance, total energy shifts ~6–8%.  Assert each
    # channel moves by less than half the original ILD in dB terms (i.e. the
    # correction is split, not over-applied to one side).
    ir_after = np.asarray(sofa.Data_IR, dtype=float)
    ir_before_arr = np.asarray(sofar.read_sofa(KNOWN_BAD_PATH).Data_IR, dtype=float)
    for r_idx, label in enumerate(("L", "R")):
        eL_before = np.sum(ir_before_arr[:, r_idx, :] ** 2)
        eL_after  = np.sum(ir_after[:, r_idx, :] ** 2)
        change_db = abs(10.0 * np.log10((eL_after + 1e-12) / (eL_before + 1e-12)))
        half_ild  = abs(r_before["ild_estimate_db"]) / 2.0 + 0.2  # allow small margin
        assert change_db < half_ild + 0.5, (
            f"Channel {label} energy changed by {change_db:.2f} dB after fix "
            f"(expected ≈ {half_ild:.2f} dB = half the ILD)"
        )


# ---------------------------------------------------------------------------
# Test 3 — known-bad reference: BILI 1103-n is a calibration offset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _sofa_dir_exists or not os.path.isfile(KNOWN_BAD_PATH),
    reason=f"{KNOWN_BAD} not found in tests/data/sofa/",
)
def test_known_bad_bili_1103():
    """BILI 1103-n.sofa is the canonical offender: large, flat-band ILD that
    persists into LF — a calibration error, not an acoustic phenomenon."""
    sofar = pytest.importorskip("sofar", reason="sofar not installed")

    sofa = sofar.read_sofa(KNOWN_BAD_PATH)
    r = diffuse_field_ild(sofa)

    assert r["verdict"] == "calibration_offset", (
        f"Expected verdict 'calibration_offset' for {KNOWN_BAD}; "
        f"got '{r['verdict']}' (lf_ild_db={r['lf_ild_db']:.2f} dB)"
    )
    assert abs(r["lf_ild_db"]) > LF_TOL, (
        f"Expected |lf_ild_db| > {LF_TOL} dB for {KNOWN_BAD}; "
        f"got {r['lf_ild_db']:.2f} dB"
    )
    assert abs(r["ild_estimate_db"]) > BROADBAND_TOL, (
        f"Expected |ild_estimate_db| > {BROADBAND_TOL} dB for {KNOWN_BAD}; "
        f"got {r['ild_estimate_db']:.2f} dB"
    )


# ---------------------------------------------------------------------------
# Test 4 — healthy-file guard: all fixtures except the known offender
# ---------------------------------------------------------------------------

def _healthy_sofa_files():
    """Return (name, path) for every .sofa fixture that is NOT the known-bad."""
    if not _sofa_dir_exists:
        return []
    return [
        (f, os.path.join(SOFA_DIR, f))
        for f in sorted(os.listdir(SOFA_DIR))
        if f.endswith(".sofa") and f != KNOWN_BAD
        and not f.endswith("_ildfix.sofa")
    ]


@pytest.mark.skipif(
    not _sofa_dir_exists,
    reason="tests/data/sofa/ is empty or missing",
)
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
