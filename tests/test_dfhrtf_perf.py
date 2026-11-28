"""Headless tests for the DFHRTF speed-ups: skipped work and equivalence of the
cheaper code paths (no Qt, no real SOFA files, no subprocesses)."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core import dfhrtf_processor
from core.scripts import generate_dfhrtf as gd
from tests.test_processors import Recorder, make_config, spies, touch_inputs  # noqa: F401 (spies is a fixture)


def _fake_sofa(n_meas=24, n_samples=128, seed=0):
    rng = np.random.default_rng(seed)
    az = np.linspace(0, 360, n_meas, endpoint=False)
    el = np.where(np.arange(n_meas) % 2 == 0, 0.0, 30.0)
    pos = np.column_stack([az, el, np.full(n_meas, 1.2)])
    ir = rng.standard_normal((n_meas, 2, n_samples)) * np.exp(-np.arange(n_samples) / 20.0)
    return SimpleNamespace(Data_IR=ir, Data_SamplingRate=48000.0, SourcePosition=pos)


def _args(tmp_path, **over):
    ns = argparse_ns()
    ns.input = str(tmp_path / "x.sofa")
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def argparse_ns():
    return gd.build_parser().parse_args([])


@pytest.fixture
def fake_input(tmp_path, monkeypatch):
    (tmp_path / "x.sofa").write_bytes(b"")
    sofa = _fake_sofa()
    monkeypatch.setattr(gd, "read_sofa_robust", lambda p: sofa)
    return sofa


class TestLazyAverage:
    def test_average_not_computed_without_text_out_avg(self, tmp_path, monkeypatch, fake_input):
        # A None entry makes `from core.averaging import ...` raise ImportError.
        monkeypatch.setitem(sys.modules, "core.averaging", None)
        args = _args(tmp_path, avg_algo="Soft-DTW",
                     text_out_l=str(tmp_path / "L.txt"),
                     text_out_r=str(tmp_path / "R.txt"))
        gd.run_dfhrtf(args)
        assert (tmp_path / "L.txt").exists() and (tmp_path / "R.txt").exists()

    def test_average_is_computed_when_requested(self, tmp_path, monkeypatch, fake_input):
        monkeypatch.setitem(sys.modules, "core.averaging", None)
        args = _args(tmp_path, avg_algo="Soft-DTW", text_out_avg=str(tmp_path / "A.txt"))
        with pytest.raises(ImportError):
            gd.run_dfhrtf(args)

    def test_grid_only_job_skips_spectral_work(self, tmp_path, monkeypatch, fake_input):
        def boom(*a, **k):
            raise AssertionError("spectral work ran for a grid-only job")
        monkeypatch.setattr(gd, "diffuse_field_db", boom)
        monkeypatch.setattr(gd, "make_smoother", boom)
        args = _args(tmp_path, graph_weighted_out=str(tmp_path / "grid.png"))
        gd.run_dfhrtf(args)
        assert (tmp_path / "grid.png").exists()


class TestDiffuseFieldDb:
    def test_matches_single_pass_reference_bit_for_bit(self, monkeypatch):
        sofa = _fake_sofa(n_meas=37)
        w = np.random.default_rng(1).random(37)
        w[[0, 5, 36]] = 0.0
        n_fft = 512
        ref_power = np.abs(np.fft.rfft(sofa.Data_IR, n=n_fft, axis=2)) ** 2
        ref = 20 * np.log10(np.sqrt(np.sum(ref_power * w[:, None, None], axis=0)) + 1e-12)
        # force several tiny blocks so block boundaries are exercised
        monkeypatch.setattr(gd, "_FFT_BLOCK_BYTES", 2 * 257 * 16 * 4)
        assert np.array_equal(gd.diffuse_field_db_direct(sofa.Data_IR, w, n_fft), ref)


class TestFastDiffuseField:
    """Autocorrelation-identity path vs the direct reference (not bit-identical)."""

    @staticmethod
    def _assert_close(ir, w, n_fft, fs=48000.0):
        direct = gd.diffuse_field_db_direct(ir, w, n_fft)
        fast = gd.diffuse_field_db_fast(ir, w, n_fft)
        keep = direct > direct.max() - 120.0
        assert np.abs(direct - fast)[keep].max() <= 1e-6

    @pytest.mark.parametrize("n_samples", [64, 256, 1000, 2048])
    def test_synthetic_matches_direct(self, n_samples):
        sofa = _fake_sofa(n_meas=40, n_samples=n_samples, seed=n_samples)
        w = np.random.default_rng(3).random(40)
        w[[2, 9]] = 0.0
        self._assert_close(sofa.Data_IR, w, 16384)

    def test_auto_selects_fast_only_when_valid(self):
        assert gd.fast_path_length(256, 16384) is not None
        assert gd.fast_path_length(8192, 16384) is None      # IR as long as n_fft/2
        assert gd.fast_path_length(20000, 16384) is None     # truncated by rfft: stay exact
        sofa = _fake_sofa(n_samples=8192)
        w = np.ones(24)
        assert np.array_equal(gd.diffuse_field_db(sofa.Data_IR, w, 16384),
                              gd.diffuse_field_db_direct(sofa.Data_IR, w, 16384))

    @pytest.mark.needs_sofa_fixtures
    def test_every_fixture_matches_direct(self):
        import glob
        from core.geometry_weights import calculate_geometric_weights
        from core.sofa_io import read_sofa_robust
        root = os.path.join(os.path.dirname(__file__), "data", "sofa")
        paths = sorted(glob.glob(os.path.join(root, "*.sofa")))
        if not paths:
            pytest.skip("no SOFA fixtures in tests/data/sofa")
        checked = 0
        for p in paths:
            sofa = read_sofa_robust(p)
            if gd.fast_path_length(sofa.Data_IR.shape[2], 16384) is None:
                continue
            w = calculate_geometric_weights(sofa.SourcePosition, weighting_mode="auto")
            self._assert_close(sofa.Data_IR, w, 16384)
            checked += 1
        assert checked


class TestSmoothingEvaluatedOnOutputGrid:
    """Smoothing is evaluated directly at the output frequencies, never at every
    FFT bin first (no needless work), and the result does not depend on which
    other points are requested."""

    FREQS = np.fft.rfftfreq(4096, d=1 / 48000.0)

    def _mags(self):
        return np.random.default_rng(2).standard_normal(self.FREQS.size) * 3

    @pytest.mark.parametrize("window", ["flat", "gaussian"])
    @pytest.mark.parametrize("smoothing", ["variable", "hybrid", "24", "3"])
    def test_subset_evaluation_equals_full(self, smoothing, window):
        args = argparse_ns()
        args.smoothing = smoothing
        args.smoothing_window = window
        fn, _ = gd.make_smoother(args)
        m = self._mags()
        target = gd.generate_fractional_octave_frequencies(20, 20000, fraction=48)
        pick = np.array([0, 5, 100, 250, target.size - 1])
        sub, full = fn(self.FREQS, m, target[pick]), fn(self.FREQS, m, target)[pick]
        if window == "flat":
            assert np.array_equal(sub, full)
        else:       # Gaussian blocks visit different far-tail nodes: equal to ~1e-5 dB
            np.testing.assert_allclose(sub, full, rtol=0, atol=1e-5)

    def test_job_smooths_only_at_target_frequencies(self, tmp_path, monkeypatch, fake_input):
        calls = []
        real = gd.make_smoother

        def spy_make(args):
            fn, label = real(args)

            def wrapped(freqs, mags, out_freqs):
                calls.append(np.asarray(out_freqs).size)
                return fn(freqs, mags, out_freqs)
            return wrapped, label

        monkeypatch.setattr(gd, "make_smoother", spy_make)
        args = _args(tmp_path, text_out_l=str(tmp_path / "L.txt"), text_out_r=str(tmp_path / "R.txt"))
        gd.run_dfhrtf(args)
        target = gd.generate_fractional_octave_frequencies(20, 20000, fraction=args.output_octave_fraction)
        assert calls == [target.size, target.size]      # once per ear, output grid only


class TestProcessorDropsNoOutputJobs:
    def _cfg(self, files):
        cfg = make_config(files, sofa_enabled=False)
        cfg["dfhrtf"].update({"per_sofa_lr": False, "weighted_grid": True,
                              "weighted_grid_scope": "Per Folder"})
        return cfg

    def test_per_folder_grid_sends_only_first_file(self, spies, tmp_path):
        files = touch_inputs(tmp_path, ["a.sofa", "b.sofa", "c.sofa"])
        rec = Recorder()
        assert dfhrtf_processor.process_dfhrtf_exports(self._cfg(files), **rec.kwargs) is True
        jobs = spies["dfhrtf"].jobs
        assert [j["input"] for j in jobs] == [files[0]]
        assert all("graph_weighted_out" in j for j in jobs)
        assert rec.logged("2 file(s) needed no output")

    def test_no_outputs_at_all_runs_nothing_and_reports_100(self, spies, tmp_path):
        files = touch_inputs(tmp_path, ["a.sofa", "b.sofa"])
        cfg = self._cfg(files)
        cfg["dfhrtf"]["weighted_grid"] = False
        rec = Recorder()
        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **rec.kwargs) is True
        assert not spies["dfhrtf"].called
        assert rec.logged("2 file(s) needed no output")
        assert rec.progress and rec.progress[-1][0] == 100

    def test_missing_input_without_outputs_is_still_sent(self, spies, tmp_path):
        # A no-output job is kept when its input is missing, so the backend
        # still reports the missing file instead of it vanishing silently.
        files = touch_inputs(tmp_path, ["a.sofa", "b.sofa", "c.sofa"])
        os.remove(files[1])
        rec = Recorder()
        dfhrtf_processor.process_dfhrtf_exports(self._cfg(files), **rec.kwargs)
        assert [j["input"] for j in spies["dfhrtf"].jobs] == [files[0], files[1]]
        assert rec.logged("1 file(s) needed no output")

    def test_all_files_scope_keeps_every_job(self, spies, tmp_path):
        files = touch_inputs(tmp_path, ["a.sofa", "b.sofa"])
        cfg = self._cfg(files)
        cfg["dfhrtf"]["weighted_grid_scope"] = "All Files"
        rec = Recorder()
        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **rec.kwargs) is True
        assert len(spies["dfhrtf"].jobs) == 2
        assert not rec.logged("needed no output")


def test_default_per_sofa_avg_algo_is_geo_arc_length():
    import core.config as C
    assert C.load_defaults()["dfhrtf"]["per_sofa_avg_algo"] == "Geo-Arc-Length"
