"""Headless unit tests for the pure helper functions in the backend scripts.

Covers the small, side-effect-free maths helpers scattered across
``core/scripts/generate_sofa.py``, ``core/scripts/generate_dfhrtf.py`` and the
grid helper in ``core/averaging.py``.  Everything here runs on synthetic numpy
inputs: no SOFA files, no subprocesses, no Qt.
"""

import numpy as np
import pytest

from core.averaging import _fractional_octave_grid
from core.scripts.generate_sofa import get_blend_weights
from core.scripts.generate_dfhrtf import (
    apply_lf_extension,
    apply_smoothing,
    apply_spectral_tilt,
    find_closest_grid_point,
    generate_fractional_octave_frequencies,
    make_smoother,
)


def _log_freqs(start=10.0, stop=20000.0, num=800):
    return np.logspace(np.log10(start), np.log10(stop), num)


class TestGetBlendWeights:
    """`get_blend_weights` is the SOFA-side LF-splice crossfade: 0 (all
    synthetic LF) below the band, 1 (all measured) above it."""

    XOVER = 1000.0
    LOW_OCT = 0.50
    HIGH_OCT = 0.75

    def _band(self):
        return (self.XOVER / 2 ** self.LOW_OCT, self.XOVER * 2 ** self.HIGH_OCT)

    def test_complementary_weights_sum_to_one_pointwise(self):
        freqs = _log_freqs()
        w = get_blend_weights(freqs, self.XOVER, self.LOW_OCT, self.HIGH_OCT)
        # The splice blends `w * measured + (1 - w) * synthetic`; the pair must
        # be a partition of unity at every frequency, with no bin outside [0, 1]
        # (an out-of-range weight would boost or invert one of the two sources).
        assert np.all(w >= 0.0) and np.all(w <= 1.0)
        np.testing.assert_allclose(w + (1.0 - w), 1.0, rtol=0, atol=0)

    def test_hard_zero_and_one_outside_transition_band(self):
        f_low, f_high = self._band()
        freqs = _log_freqs()
        w = get_blend_weights(freqs, self.XOVER, self.LOW_OCT, self.HIGH_OCT)

        below = freqs <= f_low
        above = freqs >= f_high
        assert below.any() and above.any()
        # Exact 0/1, not merely "close": outside the band the crossfade must not
        # contaminate either source with a residual sliver of the other.
        assert np.all(w[below] == 0.0)
        assert np.all(w[above] == 1.0)

    def test_transition_is_monotonic_and_half_at_band_center(self):
        f_low, f_high = self._band()
        freqs = _log_freqs()
        w = get_blend_weights(freqs, self.XOVER, self.LOW_OCT, self.HIGH_OCT)
        assert np.all(np.diff(w) >= -1e-15)

        mid = np.array([np.sqrt(f_low * f_high)])  # geometric (log) center
        w_mid = get_blend_weights(mid, self.XOVER, self.LOW_OCT, self.HIGH_OCT)
        assert w_mid[0] == pytest.approx(0.5, abs=1e-12)


class TestApplySpectralTilt:
    """Tilt pivots at `ref_freq` (default 1 kHz) and is `slope` dB per octave."""

    def test_zero_slope_is_exact_identity(self):
        freqs = _log_freqs()
        mags = np.sin(np.log2(freqs)) * 3.0
        out = apply_spectral_tilt(freqs, mags, 0.0)
        np.testing.assert_array_equal(out, mags)

    def test_one_db_per_octave_at_anchor_and_octaves_out(self):
        freqs = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0])
        mags = np.zeros_like(freqs)
        out = apply_spectral_tilt(freqs, mags, -1.0, ref_freq=1000.0)
        # -1 dB/oct pivoting at 1 kHz: +2/+1/0/-1/-2 dB going up in octaves.
        np.testing.assert_allclose(out, [2.0, 1.0, 0.0, -1.0, -2.0], atol=1e-12)

    def test_tilt_is_additive_offset_not_a_rescale(self):
        freqs = _log_freqs()
        mags = np.linspace(-6.0, 6.0, freqs.size)
        out = apply_spectral_tilt(freqs, mags, -0.5)
        flat = apply_spectral_tilt(freqs, np.zeros_like(mags), -0.5)
        np.testing.assert_allclose(out - mags, flat, atol=1e-12)


class TestMakeSmoother:
    """`make_smoother(args)` dispatches on `args.smoothing`, returning
    `(smooth_fn, label)`."""

    def test_variable_mode_dispatch(self, dfhrtf_args):
        from core.smoothing import SmoothingParams, variable_smooth

        args = dfhrtf_args(smoothing="variable")
        smooth_fn, label = make_smoother(args)
        assert callable(smooth_fn)
        assert label == "Variable Smoothing"

        freqs = _log_freqs(20.0, 20000.0, 400)
        rng = np.random.default_rng(0)
        mags = rng.normal(0.0, 2.0, freqs.size)

        params = SmoothingParams(
            freq_low=args.var_smooth_freq_low,
            freq_hi=args.var_smooth_freq_hi,
            n_low=args.var_smooth_n_low,
            n_hi=args.var_smooth_n_hi,
            gamma=args.var_smooth_gamma,
            mean_mode=args.var_smooth_mean_mode,
        )
        np.testing.assert_allclose(
            smooth_fn(freqs, mags), variable_smooth(freqs, mags, params), atol=1e-12
        )

    def test_variable_mode_is_case_insensitive(self, dfhrtf_args):
        _, label = make_smoother(dfhrtf_args(smoothing="Variable"))
        assert label == "Variable Smoothing"

    @pytest.mark.parametrize("fraction", [3, 12, 48])
    def test_fixed_octave_dispatch(self, dfhrtf_args, fraction):
        args = dfhrtf_args(smoothing=str(fraction))
        smooth_fn, label = make_smoother(args)
        assert label == f"1/{fraction} Octave Smoothing"

        freqs = _log_freqs(20.0, 20000.0, 400)
        rng = np.random.default_rng(1)
        mags = rng.normal(0.0, 2.0, freqs.size)
        np.testing.assert_allclose(
            smooth_fn(freqs, mags), apply_smoothing(freqs, mags, fraction=fraction),
            atol=1e-12,
        )

    def test_variable_and_fixed_produce_different_curves(self, dfhrtf_args):
        freqs = _log_freqs(20.0, 20000.0, 400)
        rng = np.random.default_rng(2)
        mags = rng.normal(0.0, 2.0, freqs.size)
        var_fn, _ = make_smoother(dfhrtf_args(smoothing="variable"))
        fix_fn, _ = make_smoother(dfhrtf_args(smoothing="3"))
        assert not np.allclose(var_fn(freqs, mags), fix_fn(freqs, mags), atol=1e-6)


class TestApplyLfExtension:
    """The reference LF curve is level-matched to the measured curve at the
    crossover before the cosine crossfade, so the splice must be seamless
    there."""

    XOVER = 1000.0

    def _fixtures(self, val_level=0.0):
        target_freqs = _log_freqs(10.0, 20000.0, 1000)
        val_db = np.full_like(target_freqs, val_level)
        # Reference sits at a deliberately wrong absolute level and slopes, so
        # an un-matched splice would leave an obvious step at the crossover.
        ref_freqs = _log_freqs(5.0, 20000.0, 300)
        ref_db = -20.0 + 4.0 * np.log2(ref_freqs / 1000.0)
        return target_freqs, val_db, ref_freqs, ref_db

    def test_output_equals_measured_level_at_crossover(self):
        target_freqs, val_db, ref_freqs, ref_db = self._fixtures(val_level=-3.0)
        out = apply_lf_extension(target_freqs, val_db, ref_freqs, ref_db, self.XOVER)
        xover_idx = int(np.abs(target_freqs - self.XOVER).argmin())
        assert out[xover_idx] == pytest.approx(val_db[xover_idx], abs=1e-6)

    def test_no_discontinuity_across_the_splice(self):
        target_freqs, val_db, ref_freqs, ref_db = self._fixtures()
        out = apply_lf_extension(target_freqs, val_db, ref_freqs, ref_db, self.XOVER)
        # A level-match failure shows up as a single large bin-to-bin jump.
        assert np.max(np.abs(np.diff(out))) < 1.0

    def test_far_above_crossover_is_untouched_measured_data(self):
        target_freqs, val_db, ref_freqs, ref_db = self._fixtures(val_level=-3.0)
        out = apply_lf_extension(target_freqs, val_db, ref_freqs, ref_db, self.XOVER)
        f_high = self.XOVER * 2 ** 1.0  # default blend_high_oct
        above = target_freqs >= f_high
        assert above.any()
        np.testing.assert_allclose(out[above], val_db[above], atol=1e-6)

    def test_far_below_crossover_follows_level_matched_reference(self):
        target_freqs, val_db, ref_freqs, ref_db = self._fixtures(val_level=-3.0)
        out = apply_lf_extension(target_freqs, val_db, ref_freqs, ref_db, self.XOVER)
        f_low = self.XOVER / 2 ** 0.50  # default blend_low_oct
        below = target_freqs <= f_low
        interp_ref = np.interp(target_freqs, ref_freqs, ref_db)
        xover_idx = int(np.abs(target_freqs - self.XOVER).argmin())
        offset = val_db[xover_idx] - interp_ref[xover_idx]
        np.testing.assert_allclose(out[below], (interp_ref + offset)[below], atol=1e-6)


class TestFindClosestGridPoint:
    """Matching is done in Cartesian space, so it must wrap around azimuth
    0/360 rather than comparing raw degree values linearly."""

    def test_wraps_across_the_azimuth_seam(self):
        # A naive |az_query - az_grid| would score 355 deg (4 deg away) as 4
        # and 1 deg (2 deg away) as 358, picking the wrong point.
        source_pos = np.array([[1.0, 0.0], [355.0, 0.0], [180.0, 0.0], [90.0, 0.0]])
        idx = find_closest_grid_point(source_pos, 359.0, 0.0)
        assert idx == 0

    def test_wraps_for_a_query_just_above_zero(self):
        source_pos = np.array([[358.0, 0.0], [5.0, 0.0], [180.0, 0.0]])
        idx = find_closest_grid_point(source_pos, 359.5, 0.0)
        assert idx == 0

    def test_negative_and_positive_azimuth_conventions_agree(self):
        source_pos = np.array([[30.0, 0.0], [330.0, 0.0], [180.0, 0.0]])
        # -30 deg is the same direction as 330 deg.
        assert find_closest_grid_point(source_pos, -30.0, 0.0) == 1
        assert find_closest_grid_point(source_pos, 330.0, 0.0) == 1

    def test_exact_hit_and_elevation_discrimination(self):
        source_pos = np.array([[0.0, 0.0], [0.0, 45.0], [0.0, -45.0]])
        assert find_closest_grid_point(source_pos, 0.0, 40.0) == 1
        assert find_closest_grid_point(source_pos, 0.0, -40.0) == 2
        assert find_closest_grid_point(source_pos, 0.0, 0.0) == 0


class TestFractionalOctaveGrids:
    """`generate_dfhrtf.generate_fractional_octave_frequencies` and
    `core.averaging._fractional_octave_grid` are documented as producing the
    same axis (the averaging output must land on the same frequency grid as the
    per-SOFA text files it averages)."""

    def test_shipped_pipeline_range_agrees(self):
        # The range the app actually uses: 20 Hz -> 20 kHz at the default
        # 1/48 octave.  Here the two implementations do agree.
        a = generate_fractional_octave_frequencies(20, 20000, fraction=48)
        b = _fractional_octave_grid(20, 20000, 48)
        assert a.shape == b.shape
        np.testing.assert_array_equal(a, b)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Real shipping discrepancy: _fractional_octave_grid() loops while "
            "f <= max_freq + 1e-6 but generate_fractional_octave_frequencies() "
            "loops while f <= end_freq with no tolerance. When end_freq is an "
            "exact 1/N-octave multiple of start_freq (here 100 -> 3200 = 5 "
            "octaves at 1/6 oct), the iterated multiplication overshoots the "
            "endpoint by ~4e-12, so the averaging grid keeps a final 3200 Hz "
            "point (31 points) that the dfhrtf grid drops (30 points, ending "
            "at 2850.88 Hz) -- the two axes differ by one point at the top."
        ),
    )
    def test_fractional_octave_grids_agree(self):
        a = generate_fractional_octave_frequencies(100, 3200, fraction=6)
        b = _fractional_octave_grid(100, 3200, 6)
        assert a.shape == b.shape, (a.shape, b.shape, a[-1], b[-1])
        np.testing.assert_allclose(a, b, rtol=1e-12)

    @pytest.mark.parametrize("fraction", [2, 6, 12, 24, 48])
    def test_grids_agree_on_a_non_boundary_range(self, fraction):
        # When the endpoint is not itself a grid point, the 1e-6 tolerance is
        # irrelevant and the two implementations match exactly.
        a = generate_fractional_octave_frequencies(20, 20000, fraction=fraction)
        b = _fractional_octave_grid(20, 20000, fraction)
        np.testing.assert_array_equal(a, b)

    def test_spacing_is_one_nth_octave(self):
        f = generate_fractional_octave_frequencies(20, 20000, fraction=12)
        ratios = f[1:] / f[:-1]
        np.testing.assert_allclose(ratios, 2 ** (1 / 12), rtol=1e-12)
        assert f[0] == 20
        assert f[-1] <= 20000
