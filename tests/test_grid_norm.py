"""Headless tests for radius harmonization / grid normalization (no Qt required).

Guards the fix for the bug where distance normalization encoded an absolute
time-of-arrival delay into the HRIR, pushing the impulse onset past short output
windows and destroying the frequency response below 512 samples.

Core requirement under test: changing the grid radius must not alter the
frequency response of the output — it may only change the overall level and the
declared SourcePosition radius.
"""

import numpy as np
import pytest
import scipy.fft as fft

import sofar as sf

from core.scripts.generate_sofa import master_sofa


@pytest.fixture
def gn_args(sofa_args):
    """generate_sofa args with this module's non-default baseline
    (target_radius 1.5, hrir_length 256); everything else is the real
    argparse default, via the shared `sofa_args` fixture in conftest.py."""
    def _make(**overrides):
        base = dict(target_radius=1.5, hrir_length=256, dfeq=False)
        base.update(overrides)
        return sofa_args(**base)
    return _make


def _read_ir(path):
    s = sf.read_sofa(str(path))
    return np.asarray(s.Data_IR, dtype=float), np.asarray(s.SourcePosition, dtype=float)


def _onset_index(ir, threshold_ratio=0.05):
    env = np.max(np.abs(ir), axis=(0, 1))
    peak = np.max(env)
    return int(np.argmax(env > threshold_ratio * peak))


class TestGridNormFrequencyResponse:
    def test_response_unchanged_by_harmonization(self, tmp_path, synthetic_sofa, gn_args):
        """grid-norm vs grid-raw at 256 samples must be identical up to a flat
        gain — this is the property the original bug violated."""
        raw = synthetic_sofa(radius_m=2.0)

        p_raw = tmp_path / "raw.sofa"
        p_norm = tmp_path / "norm.sofa"
        master_sofa(raw, 48000, False, str(p_raw), gn_args(harmonize_radius=False))
        master_sofa(raw, 48000, False, str(p_norm),
                    gn_args(harmonize_radius=True, target_radius=1.5))

        ir_raw, _ = _read_ir(p_raw)
        ir_norm, _ = _read_ir(p_norm)

        # Onset lands at the same place (~onset_buffer) in both.
        assert abs(_onset_index(ir_raw) - _onset_index(ir_norm)) <= 1

        # Magnitude ratio is flat across the band: |H_norm| / |H_raw| ~ const.
        H_raw = np.abs(fft.rfft(ir_raw, axis=-1))
        H_norm = np.abs(fft.rfft(ir_norm, axis=-1))
        freqs = fft.rfftfreq(ir_raw.shape[-1], 1.0 / 48000)
        band = (freqs >= 100) & (freqs <= 12000)

        ratio = H_norm[:, :, band] / H_raw[:, :, band]
        # Every ratio agrees with the mean ratio to a tiny relative tolerance.
        assert np.allclose(ratio, ratio.mean(), rtol=1e-4, atol=1e-6)

    def test_response_unchanged_across_lengths(self, tmp_path, synthetic_sofa, gn_args):
        """The normalized response at 256 matches the normalized response at 512
        over the shared band (short windows no longer degrade it)."""
        raw = synthetic_sofa(radius_m=2.0)

        results = {}
        for length in (256, 512):
            out = tmp_path / f"norm_{length}.sofa"
            master_sofa(raw, 48000, False, str(out),
                        gn_args(harmonize_radius=True, target_radius=1.5,
                              hrir_length=length))
            ir, _ = _read_ir(out)
            H = np.abs(fft.rfft(ir[:, :, :256], axis=-1))
            # Peak-normalize each to compare shape, not level.
            results[length] = H / H.max()

        freqs = fft.rfftfreq(256, 1.0 / 48000)
        band = (freqs >= 200) & (freqs <= 10000)
        diff_db = 20 * np.log10(
            (results[256][:, :, band] + 1e-9) / (results[512][:, :, band] + 1e-9)
        )
        assert np.max(np.abs(diff_db)) < 1.0


class TestGridNormMetadata:
    def test_source_radius_rewritten_to_target(self, tmp_path, synthetic_sofa, gn_args):
        raw = synthetic_sofa(radius_m=2.0)
        out = tmp_path / "norm.sofa"
        master_sofa(raw, 48000, False, str(out),
                    gn_args(harmonize_radius=True, target_radius=1.5))
        _, sp = _read_ir(out)
        assert np.allclose(sp[:, 2], 1.5, atol=1e-4)

    def test_source_radius_untouched_when_disabled(self, tmp_path, synthetic_sofa, gn_args):
        raw = synthetic_sofa(radius_m=2.0)
        out = tmp_path / "raw.sofa"
        master_sofa(raw, 48000, False, str(out), gn_args(harmonize_radius=False))
        _, sp = _read_ir(out)
        assert np.allclose(sp[:, 2], 2.0, atol=1e-4)


class TestGridNormClipGuard:
    def test_high_gain_does_not_clip(self, tmp_path, synthetic_sofa, gn_args):
        """orig 2.0 m -> target 0.5 m is a x4 gain; the guard must scale the
        output back under full scale."""
        raw = synthetic_sofa(radius_m=2.0)
        out = tmp_path / "norm.sofa"
        master_sofa(raw, 48000, False, str(out),
                    gn_args(harmonize_radius=True, target_radius=0.5))
        ir, _ = _read_ir(out)
        assert np.max(np.abs(ir)) <= 1.0
