"""Hybrid smoothing (core/hybrid_smoothing.py).

Parity tests compare against tests/data/hybrid_reference.npz, produced by the
reference Option 4 code and scikit-image (see tools/build_hybrid_reference.py).
The behaviour tests pin what the algorithm is for: reject single-bin glitches,
keep real narrow notches above the crossover, bridge narrow dips below it.
"""

import numpy as np
import pytest
from pathlib import Path

import core.config as C
from core.hybrid_smoothing import (
    HybridParams, _median3_zero_pad, _prebin_dense_fft, _tv_denoise_1d,
    adaptive_spline_s, huber_local_cubic_1d, hybrid_smooth, smooth_option4,
)
from core.smoothing import fixed_smooth, generate_fractional_octave_frequencies

FIXTURE = Path(__file__).parent / "data" / "hybrid_reference.npz"
GRID = generate_fractional_octave_frequencies(20.0, 20000.0, 48)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _bin(f):
    return int(np.argmin(np.abs(GRID - f)))


def _curve(seed=0, noise=0.1):
    """A smooth HRTF-like 1/48-octave curve (0 dB at 1 kHz) plus a little noise."""
    rng = np.random.default_rng(seed)
    base = 3.0 * np.exp(-(np.log2(GRID / 3000.0) / 0.6) ** 2)
    y = base + rng.normal(0.0, noise, GRID.size)
    return y - y[_bin(1000.0)]


# --------------------------------------------------------------------------- #
# Parity with scikit-image and the reference implementation
# --------------------------------------------------------------------------- #
def test_fixture_records_the_versions_that_made_it(fx):
    assert [str(v).split()[0] for v in fx["versions"]] == ["numpy", "scipy", "scikit-image"]


@pytest.mark.parametrize("i", range(9))
def test_tv_port_matches_skimage(fx, i):
    w, eps, it = fx[f"tv_{i}_params"]
    got = _tv_denoise_1d(fx[f"tv_{i}_in"], w, eps, int(it))
    # Same arithmetic as skimage on float64: exact in practice; allow rounding.
    np.testing.assert_allclose(got, fx[f"tv_{i}_out"], rtol=0, atol=1e-12)


def test_tv_port_does_not_modify_its_input(fx):
    y = fx["tv_0_in"].copy()
    _tv_denoise_1d(y, 0.65)
    np.testing.assert_array_equal(y, fx["tv_0_in"])


def test_tv_denoise_reduces_total_variation():
    y = _curve(1, noise=0.3)
    out = _tv_denoise_1d(y, 0.65)
    assert np.abs(np.diff(out)).sum() < np.abs(np.diff(y)).sum()


def test_median3_matches_scipy_medfilt():
    from scipy.signal import medfilt
    y = np.random.default_rng(3).normal(5.0, 2.0, 101)
    np.testing.assert_array_equal(_median3_zero_pad(y), medfilt(y, 3))


def test_port_matches_reference_on_every_fixture_curve(fx):
    # Identical arrays through identical arithmetic; ~1e-14 in practice. Allow
    # 1e-6 dB, and a few 1e-4 across scipy versions (FITPACK knot selection).
    names = [str(n) for n in fx["ref_names"]]
    assert any("pp001" in n for n in names) and any("syn" in n for n in names)
    for j, name in enumerate(names):
        # explicit s: the fixed-s path IS the reference algorithm
        got = smooth_option4(fx[f"ref_{j}_freqs"], fx[f"ref_{j}_in"], s=6.5)
        np.testing.assert_allclose(got, fx[f"ref_{j}_out"], rtol=0, atol=1e-6, err_msg=name)


def test_port_matches_reference_with_non_default_parameters(fx):
    xo, wd, tv, sp, win, dl, it = fx["ref_custom_params"]
    p = HybridParams(xover_hz=xo, xover_width_oct=wd, tv_weight=tv,
                     huber_window=int(win), huber_delta=dl, huber_iter=int(it))
    got = smooth_option4(fx["ref_custom_freqs"], fx["ref_custom_in"], p, s=sp)
    np.testing.assert_allclose(got, fx["ref_custom_out"], rtol=0, atol=1e-6)


def test_app_entry_point_matches_reference_end_to_end(fx):
    """hybrid_smooth() on a raw FFT-like spectrum vs the reference run on the app's
    own stage-0 text output (0 dB at 1 kHz, quantised to 0.001 dB like the 1-48
    files). The app subtracts its level reference instead of rounding, hence the
    looser tolerance (0.0484 dB measured: FITPACK knot selection is discrete, so
    sub-millidB input differences move the spline a few hundredths of a dB)."""
    # s_min == s_max pins the adaptive s to the reference's 6.5.
    pinned = HybridParams(s_min=6.5, s_max=6.5)
    out = hybrid_smooth(fx["e2e_raw_freqs"], fx["e2e_raw_mags"], pinned, GRID)
    out = out - out[_bin(1000.0)]
    assert out.shape == GRID.shape
    np.testing.assert_allclose(out, fx["e2e_ref_out"], rtol=0, atol=0.06)


# --------------------------------------------------------------------------- #
# What the smoother is for
# --------------------------------------------------------------------------- #
def test_one_bin_spike_above_crossover_is_rejected():
    y = _curve()
    k = _bin(14690.0)
    spiked = y.copy()
    spiked[k] += 6.0
    out = smooth_option4(GRID, spiked)
    assert abs(out[k] - smooth_option4(GRID, y)[k]) < 0.5
    assert out[k] < y[k] + 1.0


def test_multibin_notch_above_crossover_keeps_depth_and_position():
    y = _curve()
    k = _bin(11000.0)
    depth = 15.0
    # 7 bins wide (~1/7 octave): a real pinna notch; a 1-2 bin one is rejected
    # as a glitch by design (see the spike test)
    shape = np.cos(np.linspace(-np.pi / 2, np.pi / 2, 7)) ** 2
    notched = y.copy()
    notched[k - 3:k + 4] -= depth * shape
    out = smooth_option4(GRID, notched)
    assert np.argmin(out[k - 8:k + 9]) + k - 8 == k
    kept = y[k] - out[k]
    assert kept > 0.6 * depth
    # and it keeps more of the notch than plain 1/12-octave smoothing does
    assert kept > y[k] - fixed_smooth(GRID, notched, 12, GRID)[k]


def test_narrow_dip_below_crossover_is_bridged():
    # Noise-free base: the spline's smoothing factor is a budget for the whole
    # curve, so heavy noise elsewhere leaves it less room to bridge the dip.
    y = _curve(noise=0.0)
    k = _bin(6500.0)
    shape = np.cos(np.linspace(-np.pi / 2, np.pi / 2, 5)) ** 2        # 5 bins, 3 dB
    dipped = y.copy()
    dipped[k - 2:k + 3] -= 3.0 * shape
    out = smooth_option4(GRID, dipped)
    assert (y[k] - out[k]) < 0.8            # most of the 3 dB dip is gone
    # the same dip above the crossover is kept (it could be a real pinna feature)
    k2 = _bin(11000.0)
    moved = y.copy()
    moved[k2 - 2:k2 + 3] -= 3.0 * shape
    assert (y[k2] - smooth_option4(GRID, moved)[k2]) > 2.0 * (y[k] - out[k])


def test_crossover_is_continuous():
    out = smooth_option4(GRID, _curve(2, noise=0.2))
    k = _bin(7350.0)
    steps = np.abs(np.diff(out[k - 15:k + 15]))
    assert steps.max() < 0.3


def test_smooth_curve_is_nearly_unchanged():
    y = _curve(noise=0.0)
    out = smooth_option4(GRID, y)
    # not exactly: the spline's smoothing budget flattens the top of the bump a little
    assert np.abs(out - y).max() < 0.5


def test_xover_hz_moves_the_weight_boost():
    """The HF spline weight boost follows xover_hz (not a hard-coded 7500 Hz)."""
    y = _curve(4, noise=0.25)
    a = smooth_option4(GRID, y, HybridParams(xover_hz=5600.0))
    b = smooth_option4(GRID, y, HybridParams(xover_hz=8800.0))
    assert not np.allclose(a, b, atol=1e-3)


def test_dc_bin_is_preserved():
    f = np.concatenate(([0.0], GRID))
    y = np.concatenate(([-40.0], _curve()))
    out = smooth_option4(f, y)
    assert out[0] == -40.0 and out.shape == y.shape


def test_dense_input_is_prebinned_and_returned_on_its_own_bins():
    lin = np.arange(100.0, 19000.0, 46.875)
    y = np.interp(np.log2(lin), np.log2(GRID), _curve(5))
    out = smooth_option4(lin, y)
    assert out.shape == lin.shape and np.all(np.isfinite(out))
    f_b, m_b = _prebin_dense_fft(lin, y, max_ppo=96.0)
    assert f_b.size < lin.size and f_b.size == m_b.size


def test_prebin_leaves_a_48th_octave_grid_alone():
    f, m = _prebin_dense_fft(GRID, _curve(), max_ppo=96.0)
    assert f is GRID


def test_huber_window_must_be_odd():
    with pytest.raises(ValueError):
        huber_local_cubic_1d(np.zeros(50), window_length=8)


# --------------------------------------------------------------------------- #
# hybrid_smooth(): the two-stage app entry point
# --------------------------------------------------------------------------- #
def _raw(seed=0):
    f = np.fft.rfftfreq(16384, 1 / 48000.0)[1:]
    rng = np.random.default_rng(seed)
    return f, np.interp(np.log2(f), np.log2(GRID), _curve(seed)) + rng.normal(0, 0.3, f.size)


def test_hybrid_smooth_is_level_invariant():
    f, m = _raw()
    base = hybrid_smooth(f, m, HybridParams(), GRID)
    for c in (-30.0, 12.5, 80.0):
        np.testing.assert_allclose(hybrid_smooth(f, m + c, HybridParams(), GRID),
                                   base + c, rtol=0, atol=1e-6)


def test_hybrid_smooth_is_invariant_to_input_grid_density():
    """Stage 0 puts everything on the canonical grid first, so a denser FFT of the
    same underlying spectrum gives (nearly) the same curve."""
    spec = lambda f: np.interp(np.log2(f), np.log2(GRID), _curve(6, noise=0.0))
    f1 = np.fft.rfftfreq(16384, 1 / 48000.0)[1:]
    f2 = np.fft.rfftfreq(65536, 1 / 48000.0)[1:]
    a = hybrid_smooth(f1, spec(f1), HybridParams(), GRID)
    b = hybrid_smooth(f2, spec(f2), HybridParams(), GRID)
    assert np.abs(a - b).max() < 0.05


def test_hybrid_smooth_stage_zero_uses_the_requested_window_and_mean_mode():
    f, m = _raw()
    g = hybrid_smooth(f, m, HybridParams(), GRID, window="gaussian")
    fl = hybrid_smooth(f, m, HybridParams(), GRID, window="flat")
    db = hybrid_smooth(f, m, HybridParams(), GRID, mean_mode="db")
    assert not np.allclose(g, fl, atol=1e-6)
    assert not np.allclose(g, db, atol=1e-6)


def test_hybrid_smooth_removes_noise():
    f, m = _raw()
    # s pinned: this synthetic curve is noisy enough that the adaptive s hits
    # s_max, which (by design) also flattens its genuine 3 dB bump.
    out = hybrid_smooth(f, m, HybridParams(s_min=6.5, s_max=6.5), GRID)
    stage0 = fixed_smooth(f, m, 48, GRID)
    clean = _curve(0, noise=0.0)

    def rms(y):
        return float(np.sqrt(np.mean((y - y[_bin(1000.0)] - clean) ** 2)))

    assert rms(out) < rms(stage0)


def test_hybrid_smooth_other_output_grids():
    f, m = _raw()
    full = hybrid_smooth(f, m, HybridParams(), GRID)
    out24 = generate_fractional_octave_frequencies(20.0, 20000.0, 24)
    sub = hybrid_smooth(f, m, HybridParams(), out24)
    assert sub.shape == out24.shape
    ref = np.interp(np.log2(out24[:-2]), np.log2(GRID), full)
    assert np.abs(sub[:-2] - ref).max() < 0.1


def test_hybrid_smooth_does_not_extrapolate_past_the_canonical_grid():
    f, m = _raw()
    out83 = generate_fractional_octave_frequencies(20.0, 20000.0, 83)
    assert out83[-1] > GRID[-1]
    out = hybrid_smooth(f, m, HybridParams(), out83)
    full = hybrid_smooth(f, m, HybridParams(), GRID)
    assert np.all(np.isfinite(out))
    assert abs(out[-1] - full[-1]) < 1e-9       # held at the edge value


def test_hybrid_smooth_rejects_nonpositive_out_freqs():
    f, m = _raw()
    with pytest.raises(ValueError):
        hybrid_smooth(f, m, HybridParams(), np.array([0.0, 100.0]))


def test_hybrid_smooth_stage_zero_is_the_fixed_48th_octave_smoother():
    """Hybrid output must stay close (within 1 dB) to the stage-0 input, which is
    what fixed_smooth gives at 1/48 octave, i.e. the user's 1-48 curve."""
    f, m = _raw()
    stage0 = fixed_smooth(f, m, 48, GRID, mean_mode="power", window="gaussian")
    out = hybrid_smooth(f, m, HybridParams(), GRID)
    # Hybrid only removes ripple: it stays close to stage 0 everywhere.
    assert np.abs(out - stage0).max() < 1.0


# --------------------------------------------------------------------------- #
# Parameters and settings
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [
    {"xover_hz": 5000.0}, {"xover_hz": 9500.0}, {"xover_width_oct": 0.0},
    {"xover_width_oct": 0.9}, {"tv_weight": 0.0}, {"tv_weight": 9.0},
    {"s_factor": 0.1}, {"s_factor": 20.0}, {"s_min": 0.1}, {"s_max": 40.0},
    {"s_min": 10.0, "s_max": 5.0}, {"s_factor": float("inf")},
    {"huber_window": 8}, {"huber_window": 3}, {"huber_window": 17}, {"huber_window": 9.0},
    {"huber_delta": 0.0}, {"huber_delta": 1.0}, {"huber_iter": 0}, {"huber_iter": 99},
    {"huber_iter": 2.5}, {"xover_hz": float("nan")}, {"tv_weight": True},
])
def test_bad_params_raise(kw):
    with pytest.raises(ValueError):
        HybridParams(**kw)


def test_default_params_construct():
    p = HybridParams()
    assert (p.xover_hz, p.huber_window, p.huber_iter) == (7350.0, 9, 6)


_HYBRID_SETTINGS = [
    ("hybrid_xover_hz", 7000.0, 100.0),
    ("hybrid_xover_width_oct", 0.1, 5.0),
    ("hybrid_tv_weight", 1.0, 99.0),
    ("hybrid_s_factor", 5.0, 0.1),
    ("hybrid_s_min", 6.0, 0.1),
    ("hybrid_s_max", 20.0, 99.0),
    ("hybrid_huber_window", 11, 4),
    ("hybrid_huber_delta", 0.2, 3.0),
    ("hybrid_huber_iter", 10, 500),
]


@pytest.mark.parametrize("key, good, bad", _HYBRID_SETTINGS)
def test_config_validation_per_hybrid_key(key, good, bad):
    defaults = C.load_defaults()
    default = defaults["dfhrtf"]["advanced"][key]

    cfg = C.load_defaults()
    cfg["dfhrtf"]["advanced"][key] = good
    out, warns = C.validate(cfg)
    assert out["dfhrtf"]["advanced"][key] == good and not warns

    cfg = C.load_defaults()
    cfg["dfhrtf"]["advanced"][key] = bad
    out, warns = C.validate(cfg)
    got = out["dfhrtf"]["advanced"][key]
    spec = C.SCHEMA[f"dfhrtf.advanced.{key}"]
    if "choices" in spec:
        assert got == default                      # invalid choice reverts to the default
    else:
        assert got == (spec["max"] if bad > spec["max"] else spec["min"])   # clamped
    assert got != bad and any(key in w for w in warns)


def test_missing_hybrid_keys_are_filled_from_defaults(tmp_path):
    import json
    user = tmp_path / "user_settings.json"
    user.write_text(json.dumps({"schema_version": 2, "dfhrtf": {"advanced": {"n_fft": 8192}}}))
    cfg, _ = C.load_merged(user)
    assert cfg["dfhrtf"]["advanced"]["hybrid_huber_window"] == 9
    assert cfg["dfhrtf"]["advanced"]["hybrid_xover_hz"] == 7350.0
    assert cfg["dfhrtf"]["advanced"]["n_fft"] == 8192


def test_hybrid_is_a_valid_smoothing_choice():
    cfg = C.load_defaults()
    cfg["dfhrtf"]["smoothing"] = "Hybrid"
    out, warns = C.validate(cfg)
    assert out["dfhrtf"]["smoothing"] == "Hybrid" and not warns


def test_schema_describes_every_hybrid_key():
    import json
    schema = json.loads((Path(C.__file__).parent / "default_settings.schema.json").read_text("utf-8"))
    adv = schema["properties"]["dfhrtf"]["properties"]["advanced"]["properties"]
    for key, _, _ in _HYBRID_SETTINGS:
        assert key in adv and adv[key].get("description")


def test_hrtf_explorer_maps_hybrid_choice():
    pytest.importorskip("customtkinter")
    pytest.importorskip("tkinterdnd2")
    # hrtf_explorer.py is an untracked standalone tool; skip where it isn't present.
    if not (Path(__file__).resolve().parents[1] / "hrtf_explorer.py").exists():
        pytest.skip("hrtf_explorer.py not present")
    import hrtf_explorer as hx
    assert "Hybrid" in hx.SMOOTHING_CHOICES
    assert hx.smoothing_choice_to_mode("Hybrid") == "hybrid"
    assert hx.smoothing_choice_to_mode("Variable") == "variable"
    assert hx.smoothing_choice_to_mode("1/24") == 24
    fn, label = hx.make_smoother("hybrid")
    assert label == "Hybrid Smoothing"
    f, m = _raw()
    assert fn(f, m, GRID).shape == GRID.shape


# --------------------------------------------------------------------------- #
# Noise-adaptive spline smoothness (s chosen per curve)
# --------------------------------------------------------------------------- #
def _fixture_curves(fx):
    return [(str(n), fx[f"ref_{j}_freqs"], fx[f"ref_{j}_in"])
            for j, n in enumerate(fx["ref_names"])]


def _s_of(f, y, p=None):
    return smooth_option4(f, y, p, return_s=True)[1]


def test_adaptive_s_follows_the_rule_on_fixture_curves(fx):
    p = HybridParams()
    for name, f, y in _fixture_curves(fx):
        out, s = smooth_option4(f, y, p, return_s=True)
        assert p.s_min <= s <= p.s_max, name
        # recompute from the documented definition on the canonical grid
        from scipy.interpolate import PchipInterpolator
        lf = np.log2(f[f > 0])
        g = np.arange(lf[0], lf[-1] + 1e-9, 1 / 48.0)
        yy = PchipInterpolator(lf, y[f > 0])(g)
        tv = _tv_denoise_1d(yy, p.tv_weight)
        sigma2 = np.mean((yy - tv)[2.0 ** g < p.xover_hz] ** 2)
        want = float(np.clip(p.s_factor * len(yy) * sigma2, p.s_min, p.s_max))
        assert s == pytest.approx(want, rel=1e-9), name


def test_adaptive_s_grows_with_noise_and_clamps_at_both_ends():
    quiet, noisy, very = (_curve(1, noise=n) for n in (0.0, 0.08, 0.3))
    assert _s_of(GRID, quiet) == 4.0                       # s_min
    assert _s_of(GRID, very) == 12.0                       # s_max
    mid = _s_of(GRID, noisy)
    assert 4.0 < mid < 12.0
    assert _s_of(GRID, noisy, HybridParams(s_min=5.0, s_max=5.0)) == 5.0


def test_s_factor_changes_s_and_the_result():
    y = _curve(2, noise=0.08)
    wide = dict(s_min=0.5, s_max=30.0)      # keep both factors out of the clamps
    a, sa = smooth_option4(GRID, y, HybridParams(s_factor=2.0, **wide), return_s=True)
    b, sb = smooth_option4(GRID, y, HybridParams(s_factor=4.0, **wide), return_s=True)
    assert sb == pytest.approx(2.0 * sa) and not np.allclose(a, b, atol=1e-3)


def test_explicit_s_overrides_the_adaptive_rule():
    y = _curve(2, noise=0.25)
    out, s = smooth_option4(GRID, y, s=7.25, return_s=True)
    assert s == 7.25
    np.testing.assert_array_equal(out, smooth_option4(GRID, y, s=7.25))


def test_each_curve_gets_its_own_s():
    f = np.fft.rfftfreq(16384, 1 / 48000.0)[1:]
    base = np.interp(np.log2(f), np.log2(GRID), _curve(0, noise=0.0))
    rng = np.random.default_rng(9)
    _, s_quiet = hybrid_smooth(f, base, HybridParams(), GRID, return_s=True)
    _, s_loud = hybrid_smooth(f, base + rng.normal(0, 1.0, f.size), HybridParams(), GRID,
                              return_s=True)
    assert s_quiet == 4.0 and s_loud > s_quiet


def test_hybrid_smooth_return_s_matches_plain_call_and_is_level_invariant():
    f, m = _raw()
    out, s = hybrid_smooth(f, m, HybridParams(), GRID, return_s=True)
    np.testing.assert_array_equal(out, hybrid_smooth(f, m, HybridParams(), GRID))
    _, s_shift = hybrid_smooth(f, m + 25.0, HybridParams(), GRID, return_s=True)
    assert s_shift == pytest.approx(s, rel=1e-9)


def test_adaptive_s_helper_ignores_noise_above_the_crossover():
    y = _curve(3, noise=0.0)
    tv = np.zeros_like(y)
    y_hf = y.copy()
    y_hf[GRID > 9000.0] += 5.0
    p = HybridParams(s_min=0.5, s_max=30.0)
    assert adaptive_spline_s(y_hf, tv, GRID, p) == adaptive_spline_s(y, tv, GRID, p)


def test_swapped_s_bounds_are_repaired_by_config_validation():
    cfg = C.load_defaults()
    cfg["dfhrtf"]["advanced"]["hybrid_s_min"] = 20.0
    cfg["dfhrtf"]["advanced"]["hybrid_s_max"] = 6.0
    out, warns = C.validate(cfg)
    adv, dflt = out["dfhrtf"]["advanced"], C.load_defaults()["dfhrtf"]["advanced"]
    assert (adv["hybrid_s_min"], adv["hybrid_s_max"]) == (dflt["hybrid_s_min"], dflt["hybrid_s_max"])
    assert any("hybrid_s_min" in w for w in warns)


def test_retired_spline_smoothness_key_is_dropped_quietly():
    cfg = C.load_defaults()
    cfg["dfhrtf"]["advanced"]["hybrid_spline_smoothness"] = 9.0
    out, warns = C.validate(cfg)
    assert "hybrid_spline_smoothness" not in out["dfhrtf"]["advanced"] and not warns
    assert C.SCHEMA_VERSION == 2        # the key never shipped: no migration


def test_swapped_s_bounds_fail_the_job_cleanly():
    from core.scripts.generate_dfhrtf import build_parser, make_smoother
    args = build_parser().parse_args(["--smoothing", "hybrid", "--hybrid-s-min", "20",
                                      "--hybrid-s-max", "6"])
    with pytest.raises(ValueError, match="s_min"):
        make_smoother(args)
