"""Headless tests for core.smoothing (no Qt required)."""

import numpy as np
import pytest

from core.smoothing import SmoothingParams, smoothing_profile, variable_smooth


def _log_freqs(start=20.0, stop=20000.0, num=600):
    return np.logspace(np.log10(start), np.log10(stop), num)


class TestSmoothingParams:
    def test_defaults_match_factory_settings(self):
        """The dataclass defaults must equal the `dfhrtf.advanced.
        variable_smoothing_*` values in core/default_settings.json.

        Reads the JSON rather than restating literals, so the assertion cannot
        become a tautology when the factory settings change.  (The argparse and
        dfhrtf_processor arms of the same contract are covered by
        tests/test_defaults_parity.py::test_variable_smoothing_four_way_sync.)
        """
        import core.config as C

        adv = C.load_defaults()["dfhrtf"]["advanced"]
        p = SmoothingParams()
        assert p.freq_low  == adv["variable_smoothing_freq_low"]
        assert p.freq_hi   == adv["variable_smoothing_freq_hi"]
        assert p.n_low     == adv["variable_smoothing_n_low"]
        assert p.n_hi      == adv["variable_smoothing_n_hi"]
        assert p.gamma     == adv["variable_smoothing_gamma"]
        assert p.mean_mode == adv["variable_smoothing_mean_mode"]


class TestSmoothingProfile:
    def test_anchors(self):
        p = SmoothingParams(freq_low=350.0, freq_hi=3000.0, n_low=1.0, n_hi=6.0, gamma=1.0)
        freqs = np.array([20.0, 350.0, 3000.0, 20000.0])
        widths = smoothing_profile(freqs, p)
        assert widths[0] == pytest.approx(1.0)      # below freq_low -> 1/n_low
        assert widths[1] == pytest.approx(1.0)      # at freq_low
        assert widths[2] == pytest.approx(1.0 / 6)  # at freq_hi
        assert widths[3] == pytest.approx(1.0 / 6)  # above freq_hi

    def test_monotonic_between_anchors(self):
        p = SmoothingParams()
        freqs = _log_freqs(350.0, 3000.0, 100)
        widths = smoothing_profile(freqs, p)
        assert np.all(np.diff(widths) <= 1e-12)

    def test_gamma_shifts_transition(self):
        # Smaller gamma pulls the transition toward freq_low (narrower widths
        # at the geometric midpoint); larger gamma pushes it toward freq_hi.
        mid = np.array([np.sqrt(350.0 * 3000.0)])
        w_low_gamma = smoothing_profile(mid, SmoothingParams(gamma=0.4))
        w_high_gamma = smoothing_profile(mid, SmoothingParams(gamma=2.5))
        assert w_low_gamma[0] < w_high_gamma[0]

    def test_degenerate_band(self):
        p = SmoothingParams(freq_low=1000.0, freq_hi=1000.0, n_low=1.0, n_hi=6.0)
        freqs = np.array([500.0, 2000.0])
        widths = smoothing_profile(freqs, p)
        assert widths[0] == pytest.approx(1.0)
        assert widths[1] == pytest.approx(1.0 / 6)


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
        out_db = variable_smooth(freqs, ripple, SmoothingParams())
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
