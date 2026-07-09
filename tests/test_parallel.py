"""
tests/test_parallel.py

Headless tests (no Qt required) for the parallel + chunked batch processing
path added to core/subprocess_utils.py, core/scripts/generate_sofa.py, and
core/scripts/generate_dfhrtf.py.

Covers:
  - effective_chunk_size / chunk_jobs clamp math
  - manifest round-trip through each backend script's run_manifest(): mixed
    OK/FAIL jobs coexist, sentinels are emitted, siblings survive a failure
  - the diagnostics-log lock: concurrent run_task() calls against the same
    log_path must not interleave

Run with:
    pytest tests/test_parallel.py
"""

import math
import os
import re
import threading

import numpy as np
import pytest
import sofar as sf

from core.subprocess_utils import (
    effective_chunk_size,
    chunk_jobs,
    write_manifest,
    run_task,
)
from core.scripts.generate_sofa import run_manifest as sofa_run_manifest
from core.scripts.generate_dfhrtf import run_manifest as dfhrtf_run_manifest


# ── 1. effective_chunk_size clamp math ─────────────────────────────────────────

class TestEffectiveChunkSize:
    def test_large_batch_uses_configured_chunk_size(self):
        # 1000 files, 4 workers, cfg 16 -> ceil(1000/4)=250, min(16,250)=16
        n = effective_chunk_size(total_tasks=1000, max_workers=4, chunk_size_cfg=16)
        assert n == 16
        assert math.ceil(1000 / n) == 63

    def test_small_batch_shrinks_chunk_to_fill_all_workers(self):
        # 8 files, 4 workers, cfg 16 -> ceil(8/4)=2, min(16,2)=2 -> 4 chunks
        n = effective_chunk_size(total_tasks=8, max_workers=4, chunk_size_cfg=16)
        assert n == 2
        assert math.ceil(8 / n) == 4

    def test_fewer_tasks_than_workers_yields_one_chunk_per_task(self):
        # 3 files, 4 workers, cfg 16 -> ceil(3/4)=1, min(16,1)=1 -> 3 chunks
        n = effective_chunk_size(total_tasks=3, max_workers=4, chunk_size_cfg=16)
        assert n == 1
        assert math.ceil(3 / n) == 3

    def test_zero_tasks_returns_configured_size(self):
        assert effective_chunk_size(total_tasks=0, max_workers=4, chunk_size_cfg=16) == 16

    def test_zero_or_negative_max_workers_is_clamped_to_one(self):
        n = effective_chunk_size(total_tasks=10, max_workers=0, chunk_size_cfg=16)
        assert n == min(16, 10)

    def test_chunk_size_cfg_below_one_is_clamped_to_one(self):
        n = effective_chunk_size(total_tasks=100, max_workers=4, chunk_size_cfg=0)
        assert n == 1


class TestChunkJobs:
    def test_contiguous_slices_preserve_order(self):
        jobs = list(range(10))
        chunks = chunk_jobs(jobs, 3)
        assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_exact_multiple_has_no_short_final_chunk(self):
        jobs = list(range(8))
        chunks = chunk_jobs(jobs, 4)
        assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7]]

    def test_never_reorders_input(self):
        # Same-input SOFA jobs (SR/EQ variants) rely on build-order adjacency
        # landing in the same chunk — chunking must never sort/shuffle.
        jobs = ["b", "a", "c", "a", "b"]
        chunks = chunk_jobs(jobs, 2)
        assert [x for c in chunks for x in c] == jobs


# ── 2. Manifest round-trip ──────────────────────────────────────────────────────

def _fibonacci_sphere(n):
    """n points ~evenly on the unit sphere; returns (az_deg, el_deg)."""
    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    return az, el


def _make_sofa_file(path, fs=48000, n_points=96, n_taps=64, n_samples=256):
    """Write a tiny synthetic SimpleFreeFieldHRIR SOFA file to *path*."""
    az, el = _fibonacci_sphere(n_points)
    M = len(az)

    tail = 0.5 ** np.arange(n_taps)
    burst = np.zeros(n_taps)
    burst[0] = 1.0
    burst += 0.4 * tail

    onset = 40
    ir = np.zeros((M, 2, n_samples), dtype=float)
    for m in range(M):
        ir[m, 0, onset:onset + n_taps] = 0.30 * burst
        ir[m, 1, onset:onset + n_taps] = 0.28 * burst

    s = sf.Sofa("SimpleFreeFieldHRIR")
    s.Data_IR = ir
    s.Data_SamplingRate = float(fs)
    s.SourcePosition = np.column_stack((az, el, np.full(M, 1.5)))
    sf.write_sofa(str(path), s)
    return str(path)


def _sofa_common(**overrides):
    base = dict(
        processing_length=2048,
        norm_target_db=-1.0,
        onset_buffer=32,
        onset_threshold=0.05,
        blend_low_oct=0.50,
        blend_high_oct=0.75,
        weighting_mode="auto",
        ild_fix=False,
        ild_threshold=1.5,
        ild_low_freq=300.0,
        ild_high_freq=6000.0,
        harmonize_radius=False,
        target_radius=1.5,
        lf_extend=False,
        lf_crossover=1000.0,
        hrir_length=256,
    )
    base.update(overrides)
    return base


def _dfhrtf_common(**overrides):
    base = dict(
        avg_algo="Arithmetic Mean",
        delimiter="\t",
        smoothing="variable",
        tilt=0.0,
        lr_avg_gamma=0.01,
        lf_ext_dfhrtf=False,
        lf_ext_dfhrtf_xover=1000.0,
        n_fft=4096,
        output_octave_fraction=12,
        norm_freq_hz=1000.0,
        lf_blend_low_oct=0.50,
        lf_blend_high_oct=1.0,
        stereo_angle=30.0,
        advanced_dfeq=True,
        weighting_mode="auto",
        grid_scale_mode="auto",
        grid_scale_low=0.001,
        grid_scale_high=0.003,
        preserve_lr_balance=True,
        var_smooth_freq_low=350.0,
        var_smooth_freq_hi=3000.0,
        var_smooth_n_low=1.0,
        var_smooth_n_hi=12.0,
        var_smooth_gamma=0.88,
        var_smooth_mean_mode="db",
    )
    base.update(overrides)
    return base


class TestSofaManifestRoundTrip:
    def test_mixed_ok_and_fail_jobs(self, tmp_path, capsys):
        good_input = _make_sofa_file(tmp_path / "src.sofa")
        missing_input = str(tmp_path / "does_not_exist.sofa")

        out1 = tmp_path / "out1.sofa"
        out2 = tmp_path / "out2.sofa"
        out_bad = tmp_path / "out_bad.sofa"

        jobs = [
            {"input": good_input, "output": str(out1), "target_fs": 48000, "dfeq": False},
            {"input": missing_input, "output": str(out_bad), "target_fs": 48000, "dfeq": False},
            {"input": good_input, "output": str(out2), "target_fs": 48000, "dfeq": True},
        ]
        manifest_path = write_manifest(str(tmp_path), _sofa_common(), jobs)

        exit_code = sofa_run_manifest(manifest_path)
        out = capsys.readouterr().out

        assert exit_code != 0  # one job failed
        assert out.count("[JOB OK] ") == 2
        assert out.count("[JOB FAIL] ") == 1
        assert missing_input in out

        # Siblings of the failed job still completed.
        assert out1.exists()
        assert out2.exists()
        assert not out_bad.exists()


class TestDfhrtfManifestRoundTrip:
    def test_mixed_ok_and_fail_jobs(self, tmp_path, capsys):
        good_input = _make_sofa_file(tmp_path / "src.sofa")
        missing_input = str(tmp_path / "does_not_exist.sofa")

        text_l1, text_r1 = tmp_path / "f1_L.txt", tmp_path / "f1_R.txt"
        text_l2, text_r2 = tmp_path / "f2_L.txt", tmp_path / "f2_R.txt"
        text_l_bad = tmp_path / "bad_L.txt"

        jobs = [
            {"input": good_input, "text_out_l": str(text_l1), "text_out_r": str(text_r1)},
            {"input": missing_input, "text_out_l": str(text_l_bad)},
            {"input": good_input, "text_out_l": str(text_l2), "text_out_r": str(text_r2)},
        ]
        manifest_path = write_manifest(str(tmp_path), _dfhrtf_common(), jobs)

        exit_code = dfhrtf_run_manifest(manifest_path)
        out = capsys.readouterr().out

        assert exit_code != 0
        assert out.count("[JOB OK] ") == 2
        assert out.count("[JOB FAIL] ") == 1
        assert missing_input in out

        assert text_l1.exists() and text_r1.exists()
        assert text_l2.exists() and text_r2.exists()
        assert not text_l_bad.exists()


# ── 3. Diagnostics-log lock ──────────────────────────────────────────────────────

class TestLogLockConcurrency:
    def test_concurrent_run_task_writes_are_not_interleaved(self, tmp_path):
        log_path = str(tmp_path / "run.log")
        # A cheap subprocess that prints many lines then exits nonzero, so
        # run_task() writes a multi-line FAIL block per call. Enough lines
        # give the GIL plenty of chances to interleave thread writes if the
        # log-append block isn't actually locked.
        n_lines = 200
        script = (
            "import sys\n"
            f"for i in range({n_lines}):\n"
            "    print(f'line {i}')\n"
            "sys.exit(1)\n"
        )
        import sys as _sys
        cmd = [_sys.executable, "-c", script]

        results = []

        def _worker(job_id):
            rc, captured = run_task(
                cmd, desc=f"job{job_id}", input_path=f"input{job_id}",
                log_callback=lambda s: None, stop_checker=None, log_path=log_path,
            )
            results.append((job_id, rc, captured))

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(rc == 1 for _, rc, _ in results)

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Exactly two FAIL blocks, each properly delimited and internally
        # sequential (no lines from the other thread's block spliced in).
        blocks = [b for b in content.split("=" * 72) if "FAIL " in b]
        assert len(blocks) == 2

        for block in blocks:
            line_nums = [int(n) for n in re.findall(r"^line (\d+)$", block, re.MULTILINE)]
            assert line_nums == list(range(n_lines)), (
                "Block lines are not a clean, contiguous 0..N-1 sequence — "
                "indicates interleaved writes from the concurrent thread."
            )
