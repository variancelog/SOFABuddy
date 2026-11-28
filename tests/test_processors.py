"""Tests for the two orchestrators: core.sofa_processor.process_sofa_exports()
and core.dfhrtf_processor.process_dfhrtf_exports().

Every test here runs fully in-process. `run_chunks_parallel` is monkeypatched out
in an autouse fixture, so no backend subprocess is ever spawned: the guard
branches all return before reaching it, and the cases that need to observe or
fake chunk execution (job-list construction, the all-fail abort) do so through
the patched seam.
"""

import os

import pytest

from core import dfhrtf_processor, sofa_processor


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class Recorder:
    """Collects the four processor callbacks."""

    def __init__(self):
        self.progress = []
        self.logs = []
        self.errors = []

    def on_progress(self, pct, msg):
        self.progress.append((pct, msg))

    def on_log(self, msg):
        self.logs.append(msg)

    def on_error(self, msg):
        self.errors.append(msg)

    @property
    def kwargs(self):
        return dict(progress_callback=self.on_progress,
                    log_callback=self.on_log,
                    error_callback=self.on_error)

    def logged(self, needle):
        return any(needle in m for m in self.logs)

    def errored(self, needle):
        return any(needle in m for m in self.errors)


class ChunkSpy:
    """Stand-in for run_chunks_parallel: records calls, returns a canned result."""

    def __init__(self, result=([], [], False)):
        self.calls = []
        self.result = result

    def __call__(self, chunk_specs, max_workers, progress_callback, log_callback,
                 stop_checker, log_path, error_callback, env=None):
        self.calls.append(chunk_specs)
        return self.result

    @property
    def called(self):
        return bool(self.calls)

    @property
    def chunks(self):
        assert len(self.calls) == 1, f"expected one call, got {len(self.calls)}"
        return self.calls[0]

    @property
    def jobs(self):
        """Flattened job list in build order."""
        return [j for chunk in self.chunks for j in chunk["jobs"]]


@pytest.fixture(autouse=True)
def spies(monkeypatch):
    """Replace run_chunks_parallel in both processor modules. Autouse, so a test
    that unexpectedly reaches chunk execution still spawns nothing."""
    sofa_spy = ChunkSpy()
    dfhrtf_spy = ChunkSpy()
    monkeypatch.setattr(sofa_processor, "run_chunks_parallel", sofa_spy)
    monkeypatch.setattr(dfhrtf_processor, "run_chunks_parallel", dfhrtf_spy)
    return {"sofa": sofa_spy, "dfhrtf": dfhrtf_spy}


def make_config(files, *, sofa_enabled=True, dfhrtf_enabled=True,
                output=None, samplerates=(48000,), uneq=True, dfeq=False):
    """Minimal-but-complete config dict of the shape ProcessingWorker builds."""
    return {
        "files": [str(f) for f in files],
        "output": output if output is not None else {"default": True, "folder": ""},
        "sofa": {
            "enabled": sofa_enabled,
            "samplerates": list(samplerates),
            "uneq": uneq,
            "dfeq": dfeq,
            "naming": "{filename}_{samplerate}_{eq}",
        },
        "dfhrtf": {
            "enabled": dfhrtf_enabled,
            "per_sofa_lr": True,
            "text_filename": "{filename}_{samplerate}_{LR}",
            "text_extension": "txt",
        },
        "processing": {"max_workers": 1, "chunk_size": 16},
    }


def touch_inputs(tmp_path, names):
    """Create placeholder input paths. The processors never open these — they
    only ever take dirname/basename of them."""
    out = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"")
        out.append(str(p))
    return out


# `proc` parametrization: (entry point, key to flip for "disabled", spy key)
PROCESSORS = [
    pytest.param(
        (sofa_processor.process_sofa_exports, "sofa", "sofa"), id="sofa"
    ),
    pytest.param(
        (dfhrtf_processor.process_dfhrtf_exports, "dfhrtf", "dfhrtf"), id="dfhrtf"
    ),
]


@pytest.fixture(params=PROCESSORS)
def proc(request):
    return request.param


# --------------------------------------------------------------------------- #
# Guard branches shared by both processors
# --------------------------------------------------------------------------- #
class TestSharedGuards:
    def test_disabled_returns_true_without_working(self, proc, spies, tmp_path):
        run, key, spy_key = proc
        files = touch_inputs(tmp_path, ["a.sofa"])
        cfg = make_config(files)
        cfg[key]["enabled"] = False
        rec = Recorder()

        assert run(cfg, **rec.kwargs) is True
        assert not spies[spy_key].called
        assert rec.errors == []
        assert rec.progress == []

    def test_empty_file_list_returns_true(self, proc, spies):
        run, key, spy_key = proc
        cfg = make_config([])
        rec = Recorder()

        assert run(cfg, **rec.kwargs) is True
        assert not spies[spy_key].called
        assert rec.errors == []

    def test_blank_custom_output_folder_returns_false(self, proc, spies, tmp_path):
        run, key, spy_key = proc
        files = touch_inputs(tmp_path, ["a.sofa"])
        cfg = make_config(files, output={"default": False, "folder": "   "})
        rec = Recorder()

        assert run(cfg, **rec.kwargs) is False
        assert not spies[spy_key].called
        assert rec.errored("No custom folder selected")

    def test_stop_checker_fires_immediately(self, proc, spies, tmp_path):
        run, key, spy_key = proc
        files = touch_inputs(tmp_path, ["a.sofa"])
        cfg = make_config(files, sofa_enabled=(key == "sofa"))
        rec = Recorder()

        assert run(cfg, **rec.kwargs, stop_checker=lambda: True) is False
        assert not spies[spy_key].called

    def test_processor_error_is_reported_not_raised(self, proc, spies, tmp_path):
        """The blanket try/except turns an internal failure into (False + error)."""
        run, key, spy_key = proc
        cfg = make_config(touch_inputs(tmp_path, ["a.sofa"]))
        cfg["output"] = "not-a-dict"  # forces an AttributeError inside
        rec = Recorder()

        assert run(cfg, **rec.kwargs) is False
        assert rec.errors


# --------------------------------------------------------------------------- #
# Backend-script resolution
# --------------------------------------------------------------------------- #
class TestBackendScriptGuard:
    def _hide(self, monkeypatch, script_name):
        real_exists = os.path.exists
        monkeypatch.setattr(
            os.path, "exists",
            lambda p: False if str(p).endswith(script_name) else real_exists(p),
        )

    def test_sofa_missing_backend_script_returns_false(self, monkeypatch, spies, tmp_path):
        self._hide(monkeypatch, "generate_sofa.py")
        cfg = make_config(touch_inputs(tmp_path, ["a.sofa"]))
        rec = Recorder()

        assert sofa_processor.process_sofa_exports(cfg, **rec.kwargs) is False
        assert not spies["sofa"].called
        assert rec.errored("Backend script not found")

    def test_dfhrtf_missing_backend_script_returns_false(self, monkeypatch, spies, tmp_path):
        self._hide(monkeypatch, "generate_dfhrtf.py")
        cfg = make_config(touch_inputs(tmp_path, ["a.sofa"]), sofa_enabled=False)
        rec = Recorder()

        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **rec.kwargs) is False
        assert not spies["dfhrtf"].called
        assert rec.errored("Backend script not found")


# --------------------------------------------------------------------------- #
# SOFA job-list construction
# --------------------------------------------------------------------------- #
class TestSofaJobList:
    def test_two_files_two_samplerates_two_eq_yields_eight_contiguous_jobs(
            self, spies, tmp_path):
        files = touch_inputs(tmp_path, ["a.sofa", "b.sofa"])
        cfg = make_config(files, samplerates=(48000, 44100), uneq=True, dfeq=True)
        rec = Recorder()

        assert sofa_processor.process_sofa_exports(cfg, **rec.kwargs) is True

        jobs = spies["sofa"].jobs
        assert len(jobs) == 8

        # Same-input jobs must be adjacent: generate_sofa.py's manifest mode reads
        # each unique input SOFA only once per chunk.
        inputs = [j["input"] for j in jobs]
        assert inputs == [files[0]] * 4 + [files[1]] * 4

        # ...and adjacent *within a single chunk*, which is what the read-once
        # optimization actually depends on.
        for chunk in spies["sofa"].chunks:
            chunk_inputs = [j["input"] for j in chunk["jobs"]]
            for name in set(chunk_inputs):
                first, last = chunk_inputs.index(name), len(chunk_inputs) - 1 - chunk_inputs[::-1].index(name)
                assert chunk_inputs[first:last + 1] == [name] * (last - first + 1)

        # Full SR x EQ cross-product per input, in build order.
        variants = [(j["target_fs"], j["dfeq"]) for j in jobs[:4]]
        assert variants == [(48000, False), (48000, True),
                            (44100, False), (44100, True)]

        # Rendered output names carry the resolved tokens.
        names = [os.path.basename(j["output"]) for j in jobs[:4]]
        assert names == ["a_48000_uneq.sofa", "a_48000_dfeq.sofa",
                         "a_44100_uneq.sofa", "a_44100_dfeq.sofa"]
        # Default output mode writes beside each source file.
        assert all(os.path.dirname(j["output"]) ==
                   os.path.join(str(tmp_path), dfhrtf_processor.SOFA_MASTERED_DIR)
                   for j in jobs)

    def test_no_eq_variants_selected_produces_no_jobs(self, spies, tmp_path):
        cfg = make_config(touch_inputs(tmp_path, ["a.sofa"]), uneq=False, dfeq=False)
        rec = Recorder()

        assert sofa_processor.process_sofa_exports(cfg, **rec.kwargs) is True
        assert not spies["sofa"].called


# --------------------------------------------------------------------------- #
# SOFA all-fail abort
# --------------------------------------------------------------------------- #
class TestSofaAllFailAbort:
    def _run(self, tmp_path, result, n_files=2):
        files = touch_inputs(tmp_path, [f"f{i}.sofa" for i in range(n_files)])
        cfg = make_config(files)
        rec = Recorder()
        spy = ChunkSpy(result=result)
        return files, cfg, rec, spy

    def test_every_job_failed_returns_false(self, monkeypatch, tmp_path):
        files, cfg, rec, spy = self._run(tmp_path, None)
        spy.result = ([(f, "boom") for f in files], [], False)
        monkeypatch.setattr(sofa_processor, "run_chunks_parallel", spy)

        assert sofa_processor.process_sofa_exports(cfg, **rec.kwargs) is False
        assert rec.errored("all tasks failed")
        assert rec.errored("Aborting before DFHRTF analysis")

    def test_partial_failure_still_returns_true(self, monkeypatch, tmp_path):
        files, cfg, rec, spy = self._run(tmp_path, None)
        spy.result = ([(files[0], "boom")], [], False)
        monkeypatch.setattr(sofa_processor, "run_chunks_parallel", spy)

        assert sofa_processor.process_sofa_exports(cfg, **rec.kwargs) is True
        assert rec.errored("1/2 tasks failed")
        assert not rec.errored("Aborting before DFHRTF analysis")

    def test_whole_chunk_failure_counts_toward_the_abort(self, monkeypatch, tmp_path):
        """A chunk that died with no per-file sentinel still accounts for all of
        its jobs, so a single all-encompassing failed chunk aborts too."""
        files, cfg, rec, _ = self._run(tmp_path, None)

        def failing(chunk_specs, *a, **kw):
            return [], [c["desc"] for c in chunk_specs], False

        monkeypatch.setattr(sofa_processor, "run_chunks_parallel", failing)
        assert sofa_processor.process_sofa_exports(cfg, **rec.kwargs) is False
        assert rec.errored("Aborting before DFHRTF analysis")

    def test_stop_during_chunk_run_returns_false_without_abort_error(
            self, monkeypatch, tmp_path):
        files, cfg, rec, spy = self._run(tmp_path, None)
        spy.result = ([], [], True)
        monkeypatch.setattr(sofa_processor, "run_chunks_parallel", spy)

        assert sofa_processor.process_sofa_exports(cfg, **rec.kwargs) is False
        assert not rec.errored("all tasks failed")


# --------------------------------------------------------------------------- #
# DFHRTF input resolution
# --------------------------------------------------------------------------- #
class TestDfhrtfInputResolution:
    def test_missing_mastered_dir_skips_input_and_never_falls_back_to_raw(
            self, spies, tmp_path, synthetic_sofa_file):
        """SOFA mastering enabled + nothing in sofa_mastered/ => the raw input is
        skipped with a log line, not silently analyzed raw."""
        raw = synthetic_sofa_file(tmp_path / "subject01.sofa")
        cfg = make_config([raw], sofa_enabled=True)
        rec = Recorder()

        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **rec.kwargs) is True
        assert rec.logged("Skipping subject01: no mastered SOFA found")
        assert not spies["dfhrtf"].called

    def test_empty_mastered_dir_skips_input(self, spies, tmp_path, synthetic_sofa_file):
        raw = synthetic_sofa_file(tmp_path / "subject01.sofa")
        (tmp_path / dfhrtf_processor.SOFA_MASTERED_DIR).mkdir()
        cfg = make_config([raw], sofa_enabled=True)
        rec = Recorder()

        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **rec.kwargs) is True
        assert rec.logged("Skipping subject01: no mastered SOFA found")
        assert not spies["dfhrtf"].called

    def test_mastered_sofa_is_preferred_over_raw_and_uneq_wins(
            self, spies, tmp_path, synthetic_sofa_file):
        raw = synthetic_sofa_file(tmp_path / "subject01.sofa")
        mastered = tmp_path / dfhrtf_processor.SOFA_MASTERED_DIR
        mastered.mkdir()
        (mastered / "subject01_44100_uneq.sofa").write_bytes(b"")
        (mastered / "subject01_48000_dfeq.sofa").write_bytes(b"")
        cfg = make_config([raw], sofa_enabled=True)
        rec = Recorder()

        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **rec.kwargs) is True
        jobs = spies["dfhrtf"].jobs
        assert [j["input"] for j in jobs] == [str(mastered / "subject01_44100_uneq.sofa")]
        assert raw not in [j["input"] for j in jobs]

    def test_sofa_disabled_uses_raw_inputs(self, spies, tmp_path):
        files = touch_inputs(tmp_path, ["a.sofa", "b.sofa"])
        cfg = make_config(files, sofa_enabled=False)
        rec = Recorder()

        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **rec.kwargs) is True
        jobs = spies["dfhrtf"].jobs
        assert [j["input"] for j in jobs] == files
        # 'raw' sr label lands in the rendered text output names
        assert os.path.basename(jobs[0]["text_out_l"]) == "a_raw_L.txt"
        assert os.path.basename(jobs[0]["text_out_r"]) == "a_raw_R.txt"
        assert all(os.path.dirname(j["text_out_l"]) ==
                   os.path.join(str(tmp_path), dfhrtf_processor.DFHRTF_OUTPUT_DIR)
                   for j in jobs)


    @pytest.mark.parametrize("dropdown,expected", [
        ("Variable", "Variable"), ("Hybrid", "Hybrid"), ("1/3", "1-3"), ("1/6", "1-6"),
        ("1/12", "1-12"), ("1/24", "1-24"), ("1/48", "1-48"),
    ])
    def test_smoothing_token_in_dfhrtf_output_names(self, spies, tmp_path,
                                                    dropdown, expected):
        files = touch_inputs(tmp_path, ["a.sofa"])
        cfg = make_config(files, sofa_enabled=False)
        cfg["dfhrtf"].update({
            "smoothing": dropdown,
            "text_filename": "{filename} {LR} {smoothing}",
            "per_sofa_avg": True,
            "per_sofa_avg_filename": "{filename} avg {smoothing}",
            "dfhrtf_graphs": True,
            "weighted_grid": True,
            "graph_filename": "{graphtype} {filename} {smoothing}",
        })
        rec = Recorder()

        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **rec.kwargs) is True
        job = spies["dfhrtf"].jobs[0]
        names = {k: os.path.basename(job[k]) for k in
                 ("text_out_l", "text_out_r", "text_out_avg",
                  "graph_out", "graph_weighted_out")}
        assert names["text_out_l"] == f"a L {expected}.txt"
        assert names["text_out_r"] == f"a R {expected}.txt"
        assert names["text_out_avg"] == f"a avg {expected}.txt"
        assert names["graph_out"] == f"df a {expected}.png"
        assert names["graph_weighted_out"] == f"weighted a {expected}.png"
        assert not any("{" in n for n in names.values())

    def test_global_average_names_carry_smoothing(self, spies, tmp_path, monkeypatch):
        files = touch_inputs(tmp_path, ["a.sofa"])
        cfg = make_config(files, sofa_enabled=False)
        cfg["dfhrtf"].update({
            "smoothing": "1/12", "avg_all": True, "dfhrtf_graphs": True,
            "avg_filename": "{folder} avg {LR} {smoothing}",
            "avg_graph_filename": "{folder} avg {smoothing}",
        })
        seen = []
        monkeypatch.setattr(dfhrtf_processor, "run_averaging",
                            lambda l, algo, ex, out, *a, **k: seen.append(out) or None)
        rec = Recorder()

        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **rec.kwargs) is True
        assert [os.path.basename(p) for p in seen] == [
            f"{os.path.basename(tmp_path)} avg L 1-12.txt",
            f"{os.path.basename(tmp_path)} avg R 1-12.txt"]


# --------------------------------------------------------------------------- #
# Plotting: no figure leaks (runs inside the worker thread)
# --------------------------------------------------------------------------- #
def test_plot_diffuse_field_closes_its_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from core.plotting import plot_diffuse_field

    plt.close("all")
    freqs = np.logspace(np.log10(20), np.log10(20000), 256)
    val_l = np.sin(np.log10(freqs))
    val_r = val_l * 0.9
    out = tmp_path / "avg.png"

    plot_diffuse_field(freqs, val_l, val_r, str(out), "0.0")

    assert out.exists()
    assert plt.get_fignums() == []


class TestSmoothingProfileHandoff:
    """The settings' profile list reaches the backend as the compact
    --var-smooth-profile string (via the manifest's common section)."""

    def _common(self, spies, tmp_path, monkeypatch, advanced):
        seen = {}
        real = dfhrtf_processor.write_manifest

        def spy(tmp_dir, common, jobs):
            seen.update(common)
            return real(tmp_dir, common, jobs)

        monkeypatch.setattr(dfhrtf_processor, "write_manifest", spy)
        files = touch_inputs(tmp_path, ["a.sofa"])
        cfg = make_config(files, sofa_enabled=False)
        cfg["dfhrtf"]["advanced"] = advanced
        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **Recorder().kwargs) is True
        return seen

    def test_profile_list_becomes_the_compact_string(self, spies, tmp_path, monkeypatch):
        common = self._common(spies, tmp_path, monkeypatch, {
            "variable_smoothing_profile": [
                {"freq": 500.0, "n": 3.0}, {"freq": 5000.0, "n": 12.0, "gamma": 0.7},
                {"freq": 12000.0, "n": 12.0}],
            "smoothing_window": "flat",
        })
        assert common["var_smooth_profile"] == "500:3,5000:12:0.7,12000:12"
        assert common["smoothing_window"] == "flat"

    @pytest.mark.parametrize("dropdown, arg", [
        ("Variable", "variable"), ("Hybrid", "hybrid"), ("1/24", "24"), ("24", "24")])
    def test_smoothing_dropdown_maps_to_the_backend_value(self, spies, tmp_path, monkeypatch,
                                                          dropdown, arg):
        seen = {}
        real = dfhrtf_processor.write_manifest

        def spy(tmp_dir, common, jobs):
            seen.update(common)
            return real(tmp_dir, common, jobs)

        monkeypatch.setattr(dfhrtf_processor, "write_manifest", spy)
        files = touch_inputs(tmp_path, ["a.sofa"])
        cfg = make_config(files, sofa_enabled=False)
        cfg["dfhrtf"]["smoothing"] = dropdown
        assert dfhrtf_processor.process_dfhrtf_exports(cfg, **Recorder().kwargs) is True
        assert seen["smoothing"] == arg

    def test_hybrid_settings_reach_the_backend(self, spies, tmp_path, monkeypatch):
        common = self._common(spies, tmp_path, monkeypatch, {
            "hybrid_xover_hz": 6000.0, "hybrid_xover_width_oct": 0.2, "hybrid_tv_weight": 1.0,
            "hybrid_s_factor": 2.0, "hybrid_s_min": 3.0, "hybrid_s_max": 9.0,
            "hybrid_huber_window": 11,
            "hybrid_huber_delta": 0.2, "hybrid_huber_iter": 3,
        })
        assert (common["hybrid_xover_hz"], common["hybrid_xover_width_oct"],
                common["hybrid_tv_weight"], common["hybrid_s_factor"],
                common["hybrid_s_min"], common["hybrid_s_max"],
                common["hybrid_huber_window"], common["hybrid_huber_delta"],
                common["hybrid_huber_iter"]) == (6000.0, 0.2, 1.0, 2.0, 3.0, 9.0, 11, 0.2, 3)

    def test_missing_hybrid_settings_fall_back_to_the_factory_values(self, spies, tmp_path, monkeypatch):
        from core.hybrid_smoothing import HybridParams
        common = self._common(spies, tmp_path, monkeypatch, {})
        p = HybridParams()
        assert common["hybrid_huber_window"] == p.huber_window
        assert common["hybrid_xover_hz"] == p.xover_hz
        assert common["hybrid_huber_iter"] == p.huber_iter

    def test_missing_settings_fall_back_to_the_factory_values(self, spies, tmp_path, monkeypatch):
        from core.smoothing import SmoothingParams, format_profile
        common = self._common(spies, tmp_path, monkeypatch, {})
        assert common["var_smooth_profile"] == format_profile(SmoothingParams().points)
        assert common["smoothing_window"] == SmoothingParams().window


@pytest.mark.parametrize("value, shown", [
    ("Variable", "Variable"),   # GUI dropdown value
    ("variable", "Variable"),   # backend CLI value
    ("1/24", "1/24"),           # GUI dropdown value
    ("24", "1/24"),             # backend CLI value
    ("Hybrid", "Hybrid"),       # GUI dropdown value
    ("hybrid", "Hybrid"),       # backend CLI value
])
def test_graph_subtitle_names_smoothing_like_the_dropdown(value, shown):
    from core.plotting import graph_subtitle

    assert graph_subtitle(0.0, value) == f"Spectral Tilt: 0.0 dB/oct; Smoothing: {shown}"


def test_graph_subtitle_without_smoothing_is_tilt_only():
    from core.plotting import graph_subtitle

    assert graph_subtitle("-1.0") == "Spectral Tilt: -1.0 dB/oct"


@pytest.mark.parametrize("value, window, shown", [
    ("Variable", "gaussian", "Variable (Gaussian)"),
    ("variable", "flat", "Variable (Flat)"),
    ("1/24", "gaussian", "1/24 (Gaussian)"),
    ("24", "flat", "1/24 (Flat)"),
    ("Hybrid", "gaussian", "Hybrid"),   # the window only shapes Hybrid's first stage
    ("hybrid", "flat", "Hybrid"),
])
def test_graph_subtitle_names_the_window_when_given(value, window, shown):
    from core.plotting import graph_subtitle

    assert graph_subtitle(0.0, value, window) == f"Spectral Tilt: 0.0 dB/oct; Smoothing: {shown}"


def test_graph_subtitle_shows_hybrid_s_per_ear_to_one_decimal():
    from core.plotting import graph_subtitle

    assert (graph_subtitle(0.0, "hybrid", "gaussian", (9.43, 8.96))
            == "Spectral Tilt: 0.0 dB/oct; Smoothing: Hybrid (s L 9.4, R 9.0)")
    assert graph_subtitle(0.0, "Hybrid", None, (4.0, 12.0)).endswith("Hybrid (s L 4.0, R 12.0)")


def test_graph_subtitle_hybrid_without_s_stays_plain():
    """Averaged graphs (no single s) and filenames/{smoothing} keep plain 'Hybrid'."""
    from core.plotting import graph_subtitle, smoothing_display_name

    assert graph_subtitle(0.0, "Hybrid", "gaussian") == "Spectral Tilt: 0.0 dB/oct; Smoothing: Hybrid"
    assert smoothing_display_name("Hybrid") == "Hybrid"


def test_hybrid_s_is_ignored_for_non_hybrid_smoothing():
    from core.plotting import smoothing_display_name

    assert smoothing_display_name("Variable", "gaussian", (9.0, 9.0)) == "Variable (Gaussian)"
