"""Headless tests for core.smoothing (no Qt required)."""

import numpy as np
import pytest

from core.smoothing import (
    BAND_LIMIT_HZ,
    MEAN_MODES,
    WINDOWS,
    SmoothingParams,
    fixed_smooth,
    format_profile,
    parse_profile,
    profile_from_dicts,
    profile_to_dicts,
    smooth_spectrum,
    smoothing_profile,
    variable_smooth,
)

# Gaussian is cut at 5 sigma (tail mass ~6e-7), so it matches exact identities to
# ~1e-5 dB rather than to rounding error; the flat window is exact.
ATOL = {"flat": 1e-9, "gaussian": 1e-5}


def _log_freqs(start=20.0, stop=20000.0, num=600):
    return np.logspace(np.log10(start), np.log10(stop), num)


def _p(*pts):
    """SmoothingParams from (freq, n[, gamma]) tuples."""
    return SmoothingParams(points=tuple((p + (1.0,))[:3] if len(p) == 2 else p for p in pts))


# The user's starting profile (also the factory default).
USER_PROFILE = _p((350.0, 2.0), (6000.0, 12.0, 0.88), (14000.0, 12.0), (17500.0, 4.0, 1.6))


class TestSmoothingParams:
    def test_defaults_match_factory_settings(self):
        """The dataclass defaults must equal the `dfhrtf.advanced.
        variable_smoothing_profile`, `smoothing_window` and `smoothing_mean_mode`
        values in core/default_settings.json.

        Reads the JSON rather than restating literals, so the assertion cannot
        become a tautology when the factory settings change.  (The argparse and
        dfhrtf_processor arms of the same contract are covered by
        tests/test_defaults_parity.py::test_variable_smoothing_four_way_sync.)
        """
        import core.config as C

        adv = C.load_defaults()["dfhrtf"]["advanced"]
        p = SmoothingParams()
        assert p.points == profile_from_dicts(adv["variable_smoothing_profile"])
        assert p.window == adv["smoothing_window"]
        assert p.mean_mode == adv["smoothing_mean_mode"]

    def test_is_frozen_and_hashable(self):
        p = SmoothingParams()
        assert hash(p) == hash(SmoothingParams())
        with pytest.raises(Exception):
            p.window = "flat"

    def test_window_choices(self):
        assert WINDOWS == ("gaussian", "flat")
        with pytest.raises(ValueError, match="window"):
            SmoothingParams(window="triangular")


def _old_two_point_profile(freq, freq_low, freq_hi, n_low, n_hi, gamma):
    """The pre-profile (v0.7.0) Variable width maths, frozen as a reference."""
    freq = np.asarray(freq, dtype=float)
    width_low, width_hi = 1.0 / n_low, 1.0 / n_hi
    lo = np.log2(max(freq_low, 1e-9))
    hi = np.log2(max(freq_hi, 1e-9))
    t = (np.log2(np.clip(freq, 1e-9, None)) - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    return width_low + (width_hi - width_low) * t ** max(gamma, 1e-6)


class TestSmoothingProfile:
    def test_anchors(self):
        p = _p((350.0, 1.0), (3000.0, 6.0))
        freqs = np.array([20.0, 350.0, 3000.0, 20000.0])
        widths = smoothing_profile(freqs, p)
        assert widths[0] == pytest.approx(1.0)      # below first point -> 1/n
        assert widths[1] == pytest.approx(1.0)      # at first point
        assert widths[2] == pytest.approx(1.0 / 6)  # at last point
        assert widths[3] == pytest.approx(1.0 / 6)  # above last point

    def test_monotonic_between_anchors(self):
        freqs = _log_freqs(350.0, 6000.0, 100)
        widths = smoothing_profile(freqs, SmoothingParams())
        assert np.all(np.diff(widths) <= 1e-12)

    def test_gamma_shifts_transition(self):
        # Smaller gamma pulls the transition toward the first point (narrower
        # widths at the geometric midpoint); larger gamma pushes it toward the last.
        mid = np.array([np.sqrt(350.0 * 3000.0)])
        w_low_gamma = smoothing_profile(mid, _p((350.0, 2.0), (3000.0, 12.0, 0.4)))
        w_high_gamma = smoothing_profile(mid, _p((350.0, 2.0), (3000.0, 12.0, 2.5)))
        assert w_low_gamma[0] < w_high_gamma[0]

    def test_two_points_reproduce_old_variable_bit_for_bit(self):
        freqs = _log_freqs(5.0, 24000.0, 2000)
        for fl, fh, nl, nh, g in [(350.0, 6000.0, 2.0, 12.0, 0.88),
                                  (100.0, 10000.0, 3.0, 12.0, 1.0),
                                  (500.0, 2000.0, 1.0, 24.0, 2.5)]:
            new = smoothing_profile(freqs, _p((fl, nl), (fh, nh, g)))
            assert np.array_equal(new, _old_two_point_profile(freqs, fl, fh, nl, nh, g))

    def test_one_point_is_constant_width(self):
        w = smoothing_profile(_log_freqs(), _p((1000.0, 6.0)))
        assert np.all(w == 1.0 / 6)

    def test_hold_has_constant_width_and_gamma_is_irrelevant(self):
        f = _log_freqs(6000.0, 14000.0, 50)
        for g in (0.2, 1.0, 5.0):
            w = smoothing_profile(f, _p((350.0, 2.0), (6000.0, 12.0), (14000.0, 12.0, g)))
            np.testing.assert_allclose(w, 1.0 / 12, rtol=0, atol=1e-15)

    def test_user_profile_widths(self):
        f = np.array([6000.0, 10000.0, 14000.0, 17500.0, 19000.0, 20000.0])
        w = smoothing_profile(f, USER_PROFILE)
        np.testing.assert_allclose(w[:3], 1.0 / 12)
        np.testing.assert_allclose(w[3:], 1.0 / 4)

    def test_gamma_warp_at_mid_segment(self):
        # Midpoint (in log2 f) of the 14 kHz -> 17.5 kHz ramp with gamma 1.6.
        mid = np.sqrt(14000.0 * 17500.0)
        got = smoothing_profile(np.array([mid]), USER_PROFILE)[0]
        expected = 1 / 12 + (1 / 4 - 1 / 12) * 0.5 ** 1.6
        assert got == pytest.approx(expected, rel=1e-12)
        assert got < 0.5 * (1 / 12 + 1 / 4)     # gamma > 1 lags the linear midpoint

    def test_width_continuous_at_every_breakpoint(self):
        for p in (USER_PROFILE, _p((100.0, 3.0), (400.0, 6.0, 0.5), (900.0, 2.0, 3.0), (5000.0, 20.0))):
            for freq, n, _g in p.points:
                eps = freq * 1e-12
                w = smoothing_profile(np.array([freq - eps, freq, freq + eps]), p)
                assert w[1] == pytest.approx(1.0 / n, rel=1e-12)
                np.testing.assert_allclose(w, 1.0 / n, rtol=1e-4)


class TestProfileValidation:
    @pytest.mark.parametrize("points, match", [
        ((), "at least one"),
        (((500.0, 2.0, 1.0), (500.0, 4.0, 1.0)), "strictly increasing"),
        (((500.0, 2.0, 1.0), (400.0, 4.0, 1.0)), "strictly increasing"),
        (((10.0, 2.0, 1.0),), "freq"),
        (((30000.0, 2.0, 1.0),), "freq"),
        (((500.0, 0.5, 1.0),), "n "),
        (((500.0, 49.0, 1.0),), "n "),
        (((500.0, 2.0, 1.0), (900.0, 2.0, 0.05)), "gamma"),
        (((500.0, 2.0, 1.0), (900.0, 2.0, 11.0)), "gamma"),
        (((500.0, 2.0, 0.9),), "first"),
    ])
    def test_params_reject(self, points, match):
        with pytest.raises(ValueError, match=match):
            SmoothingParams(points=points)

    def test_dicts_roundtrip(self):
        d = profile_to_dicts(USER_PROFILE.points)
        assert d[0] == {"freq": 350.0, "n": 2.0}              # no gamma on the first point
        assert d[2] == {"freq": 14000.0, "n": 12.0}           # gamma 1.0 omitted
        assert profile_from_dicts(d) == USER_PROFILE.points

    @pytest.mark.parametrize("items", [
        [],
        "350:2",
        [{"freq": 350.0}],
        [{"n": 2.0}],
        [{"freq": 350.0, "n": 2.0, "colour": "red"}],
        [{"freq": 350.0, "n": 2.0, "gamma": 0.9}],
        [{"freq": 900.0, "n": 2.0}, {"freq": 400.0, "n": 3.0}],
        [{"freq": "abc", "n": 2.0}],
        [{"freq": None, "n": 2.0}],
        [{"freq": "350", "n": 2.0}],
        [{"freq": 350.0, "n": True}],
        [{"freq": 350.0, "n": 2.0}, {"freq": 900.0, "n": 3.0, "gamma": "1"}],
        [350.0],
    ])
    def test_dicts_reject(self, items):
        with pytest.raises(ValueError):
            profile_from_dicts(items)

    def test_compact_string_roundtrip(self):
        s = "350:2,6000:12:0.88,14000:12,17500:4:1.6"
        assert parse_profile(s) == USER_PROFILE.points
        assert format_profile(USER_PROFILE.points) == s

    def test_compact_string_float_values(self):
        pts = parse_profile("350.5:2.5,1000:12:1.25")
        assert parse_profile(format_profile(pts)) == pts

    @pytest.mark.parametrize("s", ["", "350", "350:2:1:9", "a:b", "350:2,", "350:2:0.9,900:3",
                                   "900:2,400:3", "350:0,900:3"])
    def test_compact_string_rejects(self, s):
        with pytest.raises(ValueError):
            parse_profile(s)


class TestVariableSmooth:
    def test_flat_input_stays_flat_db(self):
        freqs = _log_freqs()
        mags = np.full_like(freqs, -3.0)
        out = variable_smooth(freqs, mags, SmoothingParams())
        np.testing.assert_allclose(out, mags, atol=1e-6)

    def test_flat_input_stays_flat_cubic(self):
        freqs = _log_freqs()
        mags = np.full_like(freqs, -3.0)
        out = variable_smooth(freqs, mags, SmoothingParams(mean_mode="cubic"))
        np.testing.assert_allclose(out, mags, atol=1e-6)

    def test_reduces_ripple(self):
        freqs = _log_freqs()
        ripple = 3.0 * np.sin(40.0 * np.log2(freqs))
        out = variable_smooth(freqs, ripple, SmoothingParams())
        assert np.std(out) < np.std(ripple)

    def test_cubic_mean_above_db_mean(self):
        # On symmetric dB ripple, the cubic (peak-weighted) mean sits above
        # the dB (null-favoring) mean.
        freqs = _log_freqs()
        ripple = 6.0 * np.sin(40.0 * np.log2(freqs))
        out_db = variable_smooth(freqs, ripple, SmoothingParams(mean_mode="db"))
        out_cubic = variable_smooth(freqs, ripple, SmoothingParams(mean_mode="cubic"))
        assert np.mean(out_cubic) > np.mean(out_db)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            variable_smooth(np.array([100.0, 200.0]), np.array([0.0]), SmoothingParams())

    def test_empty_input(self):
        out = variable_smooth(np.array([]), np.array([]), SmoothingParams())
        assert out.size == 0

    def test_rfftfreq_grid_with_dc_guard(self):
        # Same pattern generate_dfhrtf.py uses: rfftfreq includes the 0 Hz bin,
        # which is masked out and passed through unchanged.
        freqs = np.fft.rfftfreq(16384, d=1 / 48000.0)
        rng = np.random.default_rng(42)
        mags = rng.normal(0.0, 3.0, freqs.size)
        pos = freqs > 0

        out = mags.copy()
        out[pos] = variable_smooth(freqs[pos], mags[pos], SmoothingParams())

        assert np.all(np.isfinite(out))
        assert out[0] == mags[0]
        assert np.std(out[pos]) < np.std(mags[pos])


# --------------------------------------------------------------------------- #
# The unified smoother (both windows)
# --------------------------------------------------------------------------- #

FS = 44100.0
FFT_FREQS = np.fft.rfftfreq(16384, d=1 / FS)[1:]          # positive bins, Nyquist 22.05 kHz
OUT = np.geomspace(20.0, 20000.0, 700)
DEFAULT = SmoothingParams()


def _realistic_spectrum(seed=7, freqs=FFT_FREQS):
    """Random-walk-ish dB spectrum with notches, resembling a measured HRTF."""
    rng = np.random.default_rng(seed)
    x = np.log2(freqs)
    smooth = 6 * np.sin(1.3 * x) + 3 * np.sin(4.1 * x + 1)
    return smooth + rng.normal(0.0, 2.5, freqs.size) - 12 * (freqs > 12000)


class TestUnifiedSmoother:
    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("mode", MEAN_MODES)
    @pytest.mark.parametrize("n", [3, 12, 48])
    def test_one_point_profile_equals_fixed(self, n, mode, window):
        y = _realistic_spectrum()
        params = SmoothingParams(points=((1000.0, float(n), 1.0),), mean_mode=mode, window=window)
        np.testing.assert_allclose(
            fixed_smooth(FFT_FREQS, y, n, OUT, mean_mode=mode, window=window),
            variable_smooth(FFT_FREQS, y, params, OUT), rtol=0, atol=1e-12)

    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("mode", MEAN_MODES)
    def test_two_points_with_equal_n_equal_fixed(self, mode, window):
        y = _realistic_spectrum()
        params = SmoothingParams(points=((350.0, 12.0, 1.0), (6000.0, 12.0, 0.88)),
                                 mean_mode=mode, window=window)
        np.testing.assert_allclose(
            fixed_smooth(FFT_FREQS, y, 12, OUT, mean_mode=mode, window=window),
            variable_smooth(FFT_FREQS, y, params, OUT), rtol=0, atol=1e-12)

    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("mode", MEAN_MODES)
    def test_default_variable_equals_fixed_12_on_the_hold(self, mode, window):
        y = _realistic_spectrum()
        hold = OUT[(OUT >= 6000.0) & (OUT <= 14000.0)]
        params = SmoothingParams(mean_mode=mode, window=window)
        np.testing.assert_allclose(
            variable_smooth(FFT_FREQS, y, params, hold),
            fixed_smooth(FFT_FREQS, y, 12, hold, mean_mode=mode, window=window),
            rtol=0, atol=1e-12)

    def test_default_window_is_gaussian(self):
        y = _realistic_spectrum()
        np.testing.assert_array_equal(
            fixed_smooth(FFT_FREQS, y, 6, OUT),
            fixed_smooth(FFT_FREQS, y, 6, OUT, window="gaussian"))

    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("mode", MEAN_MODES)
    @pytest.mark.parametrize("smoother", ["fixed12", "variable"])
    def test_anti_alias_cliff_does_not_leak_into_band(self, smoother, mode, window):
        y = np.where(FFT_FREQS <= BAND_LIMIT_HZ, 0.0, -60.0)
        if smoother == "fixed12":
            out = fixed_smooth(FFT_FREQS, y, 12, OUT, mean_mode=mode, window=window)
        else:
            out = variable_smooth(FFT_FREQS, y, SmoothingParams(mean_mode=mode, window=window), OUT)
        np.testing.assert_allclose(out, 0.0, atol=ATOL[window])

    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("mode", MEAN_MODES)
    def test_log_symmetric_bump_stays_symmetric(self, mode, window):
        f0 = 2000.0
        x = np.log2(FFT_FREQS / f0)
        y = 10.0 * np.exp(-0.5 * (x / 0.15) ** 2)         # symmetric in log-frequency
        offs = np.linspace(0.02, 1.5, 40)
        up = fixed_smooth(FFT_FREQS, y, 6, f0 * 2.0 ** offs, mean_mode=mode, window=window)
        down = fixed_smooth(FFT_FREQS, y, 6, f0 * 2.0 ** -offs, mean_mode=mode, window=window)
        np.testing.assert_allclose(up, down, atol=2e-3)   # bin sampling of the bump only

    def test_narrow_peak_at_the_edge_is_attenuated_not_passed_through(self):
        y = np.zeros(FFT_FREQS.size)
        y[np.argmin(np.abs(FFT_FREQS - 19700.0))] = 12.0
        raw = np.interp(19700.0, FFT_FREQS, y)
        out = fixed_smooth(FFT_FREQS, y, 12, np.array([19700.0]))[0]
        assert raw - out > 6.0

    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("mode", MEAN_MODES)
    def test_window_narrower_than_bin_spacing_is_linear_interpolation(self, mode, window):
        from core.smoothing import _MODES
        to_domain, to_db = _MODES[mode]
        f = np.array([100.0, 200.0, 400.0, 800.0])
        y = np.array([0.0, 6.0, -3.0, 2.0])
        q = np.array([130.0, 250.0, 600.0])
        out = smooth_spectrum(f, y, q, 1e-4, mean_mode=mode, window=window)
        expected = to_db(np.interp(np.log2(q), np.log2(f), to_domain(y)))
        np.testing.assert_allclose(out, expected, atol=1e-6)

    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("mode", MEAN_MODES)
    def test_constant_input_returns_constant(self, mode, window):
        y = np.full(FFT_FREQS.size, -7.5)
        np.testing.assert_allclose(
            fixed_smooth(FFT_FREQS, y, 6, OUT, mean_mode=mode, window=window), -7.5,
            atol=ATOL[window])

    def test_mean_mode_ordering(self):
        y = _realistic_spectrum()
        db, power, cubic = (fixed_smooth(FFT_FREQS, y, 3, OUT, mean_mode=m) for m in ("db", "power", "cubic"))
        assert np.all(power >= db - 1e-5)
        assert np.all(cubic >= power - 1e-5)
        assert np.max(power - db) > 0.1

    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("mode", MEAN_MODES)
    @pytest.mark.parametrize("width", [1 / 48, 1 / 12, 1 / 3, 1.0])
    def test_matches_slow_brute_force_reference(self, mode, width, window):
        from core.smoothing import _MODES
        to_domain, to_db = _MODES[mode]
        y = _realistic_spectrum(seed=11)
        band = FFT_FREQS <= BAND_LIMIT_HZ
        # Dense log-uniform resampling of the same piecewise-linear spectrum.
        x_dense = np.linspace(np.log2(FFT_FREQS[0]), np.log2(FFT_FREQS[band][-1]), 400_000)
        v_dense = np.interp(x_dense, np.log2(FFT_FREQS[band]), to_domain(y[band]))
        q = OUT[::25]
        ref = np.empty(q.size)
        for k, fq in enumerate(q):
            xc = np.log2(fq)
            if window == "flat":
                sel = (x_dense >= xc - width / 2) & (x_dense <= xc + width / 2)
                ref[k] = to_db(v_dense[sel].mean()) if sel.any() else np.nan
            else:       # Gaussian of sigma w/sqrt(12), renormalised over the band
                wt = np.exp(-0.5 * ((x_dense - xc) / (width / np.sqrt(12.0))) ** 2)
                ref[k] = to_db(np.sum(wt * v_dense) / np.sum(wt))
        out = smooth_spectrum(FFT_FREQS, y, q, width, mean_mode=mode, window=window)
        assert np.nanmax(np.abs(out - ref)) < 0.01

    def test_unknown_mean_mode_raises(self):
        with pytest.raises(ValueError, match="mean_mode"):
            smooth_spectrum(FFT_FREQS, np.zeros(FFT_FREQS.size), OUT, 1 / 12, mean_mode="mag")
        with pytest.raises(ValueError, match="mean_mode"):
            variable_smooth(FFT_FREQS, np.zeros(FFT_FREQS.size), SmoothingParams(mean_mode="dB"))

    def test_unknown_window_raises(self):
        with pytest.raises(ValueError, match="window"):
            smooth_spectrum(FFT_FREQS, np.zeros(FFT_FREQS.size), OUT, 1 / 12, window="triangular")

    def test_invalid_inputs_raise(self):
        y = np.zeros(FFT_FREQS.size)
        with pytest.raises(ValueError):
            smooth_spectrum(FFT_FREQS[::-1], y, OUT, 1 / 12)          # not increasing
        with pytest.raises(ValueError):
            smooth_spectrum(FFT_FREQS, y, OUT, 0.0)                   # zero width
        with pytest.raises(ValueError):
            smooth_spectrum(FFT_FREQS, y, [0.0, 100.0], 1 / 12)       # non-positive output freq
        with pytest.raises(ValueError):
            smooth_spectrum(FFT_FREQS, y, OUT, 1 / 12, f_max=3.0)     # < 2 bins in band

    def test_dc_bin_is_ignored(self):
        freqs = np.fft.rfftfreq(16384, d=1 / FS)
        y = _realistic_spectrum(freqs=freqs[1:])
        with_dc = np.concatenate([[999.0], y])
        for window in WINDOWS:
            np.testing.assert_array_equal(fixed_smooth(freqs, with_dc, 12, OUT, window=window),
                                          fixed_smooth(freqs[1:], y, 12, OUT, window=window))


# --------------------------------------------------------------------------- #
# Window-specific behaviour
# --------------------------------------------------------------------------- #

def _frozen_box_smooth(freq, mag_db, out_freq, width_oct, mean_mode="power", f_max=BAND_LIMIT_HZ):
    """The flat smoother exactly as it was before the window option existed
    (copied from the 2026-11-26 redesign). `window="flat"` must reproduce it
    bit for bit."""
    from core.smoothing import _MODES
    freq = np.asarray(freq, dtype=float)
    keep = (freq > 0) & (freq <= f_max)
    out_freq = np.asarray(out_freq, dtype=float)
    width = np.broadcast_to(np.asarray(width_oct, dtype=float), out_freq.shape)
    to_domain, to_db = _MODES[mean_mode]
    x = np.log2(freq[keep])
    v = to_domain(np.asarray(mag_db, dtype=float)[keep])
    dx = np.diff(x)
    slope = np.diff(v) / dx
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dx)])

    def integral_to(xq):
        xq = np.clip(xq, x[0], x[-1])
        i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
        d = xq - x[i]
        return cum[i] + v[i] * d + 0.5 * slope[i] * d * d

    xc = np.log2(out_freq)
    a = np.maximum(xc - width / 2.0, x[0])
    b = np.minimum(xc + width / 2.0, x[-1])
    span = b - a
    ok = span > 1e-12
    mean = np.where(ok, (integral_to(b) - integral_to(a)) / np.where(ok, span, 1.0),
                    np.interp(xc, x, v))
    return to_db(mean)


class TestFlatWindowUnchanged:
    @pytest.mark.parametrize("mode", MEAN_MODES)
    @pytest.mark.parametrize("width", [1 / 48, 1 / 12, 1 / 3, 1.0])
    def test_flat_is_bit_identical_to_the_old_smoother(self, mode, width):
        y = _realistic_spectrum(seed=5)
        np.testing.assert_array_equal(
            smooth_spectrum(FFT_FREQS, y, OUT, width, mean_mode=mode, window="flat"),
            _frozen_box_smooth(FFT_FREQS, y, OUT, width, mode))

    def test_flat_variable_with_two_points_is_bit_identical_to_old_variable(self):
        y = _realistic_spectrum(seed=6)
        old_w = _old_two_point_profile(OUT, 350.0, 6000.0, 2.0, 12.0, 0.88)
        params = SmoothingParams(points=((350.0, 2.0, 1.0), (6000.0, 12.0, 0.88)), window="flat")
        np.testing.assert_array_equal(
            variable_smooth(FFT_FREQS, y, params, OUT),
            _frozen_box_smooth(FFT_FREQS, y, OUT, old_w, "power"))


class TestGaussianWindow:
    """Regression checks ported from frequency_smoother's gaussian_comparison.py."""

    FS48 = 48000.0
    F512 = np.fft.rfftfreq(512, 1 / FS48)[1:]          # linear FFT bins, like DFHRTF
    PTS = np.array([4000.0, 10000.0, 16000.0])

    def test_minus_6db_per_octave_tilt_is_unchanged_in_band(self):
        tilt = -6.0 * np.log2(self.F512 / 1000.0)
        truth = -6.0 * np.log2(self.PTS / 1000.0)
        out = smooth_spectrum(self.F512, tilt, self.PTS, 1 / 3, mean_mode="db", window="gaussian")
        np.testing.assert_allclose(out, truth, atol=0.01)       # old Gaussian: -0.08 dB bias

    def test_tilt_near_the_band_edge_stays_close(self):
        tilt = -6.0 * np.log2(self.F512 / 1000.0)
        pts = np.array([19000.0, 20000.0])
        out = smooth_spectrum(self.F512, tilt, pts, 1 / 3, mean_mode="db", window="gaussian")
        # one-sided window at the edge: bounded, not a roll-off (old Gaussian: ~3-6 dB droop)
        assert np.all(np.abs(out - (-6.0 * np.log2(pts / 1000.0))) < 0.5)

    @pytest.mark.parametrize("mode", MEAN_MODES)
    def test_rolloff_above_20k_does_not_touch_values_below_it(self, mode):
        roll = np.where(self.F512 > 20000.0, -60.0 * ((self.F512 - 20000.0) / 4000.0) ** 2, 0.0)
        pts = np.array([4000.0, 10000.0, 16000.0, 19000.0, 20000.0])
        out = smooth_spectrum(self.F512, roll, pts, 1 / 3, mean_mode=mode, window="gaussian")
        np.testing.assert_allclose(out, 0.0, atol=1e-5)         # old Gaussian: -3.1 dB @ 19 kHz

    def test_same_variance_as_flat_on_a_slow_ripple(self):
        # Ripple period P >> w: both windows attenuate by ~1 - (2*pi*sigma/P)^2/2
        # with sigma^2 = w^2/12, so the two must agree closely.
        x = np.log2(FFT_FREQS)
        P, w = 0.5, 1 / 12
        y = np.sin(2 * np.pi * x / P)
        q = np.geomspace(500.0, 5000.0, 400)
        gains = {}
        for window in WINDOWS:
            out = smooth_spectrum(FFT_FREQS, y, q, w, mean_mode="db", window=window)
            ref = np.sin(2 * np.pi * np.log2(q) / P)
            gains[window] = abs(np.dot(out, ref) / np.dot(ref, ref))
        assert gains["gaussian"] == pytest.approx(gains["flat"], abs=0.005)
        assert gains["gaussian"] == pytest.approx(np.exp(-(np.pi * w / P) ** 2 / 6), abs=0.005)
        assert gains["gaussian"] < 0.99          # and it really did smooth

    def test_ripple_between_w_over_2_and_w_is_not_inverted(self):
        # w = 1/6 oct, ripple period 1/6 oct: flat gain is sinc(1) = 0 (and
        # negative just below), Gaussian gain is exp(-(pi*w/P)^2/6) ~ 0.19.
        x = np.log2(FFT_FREQS)
        w, P = 1 / 6, 1 / 8                      # w/P = 1.33: flat gain < 0
        y = np.sin(2 * np.pi * x / P)
        q = np.geomspace(500.0, 5000.0, 600)
        ref = np.sin(2 * np.pi * np.log2(q) / P)
        flat = smooth_spectrum(FFT_FREQS, y, q, w, mean_mode="db", window="flat")
        gauss = smooth_spectrum(FFT_FREQS, y, q, w, mean_mode="db", window="gaussian")
        assert np.dot(flat, ref) < 0 < np.dot(gauss, ref)
