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
  - per-file progress reconciliation in run_chunks_parallel()
  - sofa_io.sanitize_sofa()'s GLOBAL_RoomType repair

Run with:
    pytest tests/test_parallel.py
"""

import math
import os
import re
import sys
import threading

import pytest
import sofar as sf

from core.subprocess_utils import (
    effective_chunk_size,
    chunk_jobs,
    write_manifest,
    run_task,
    run_chunks_parallel,
)
from core.scripts.generate_sofa import run_manifest as sofa_run_manifest
from core.scripts.generate_dfhrtf import run_manifest as dfhrtf_run_manifest
from core.sofa_io import sanitize_sofa


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

# Synthetic SOFA builders and the manifest "common" dicts now live in
# tests/conftest.py (fixtures: synthetic_sofa_file, sofa_common, dfhrtf_common).
# The overrides passed at each call site below are this module's deliberate
# deviations from the shipped argparse defaults: the SOFA baseline this suite
# has always used (target_radius 1.5, hrir_length 256), and DFHRTF
# speed-optimizations (small n_fft, coarse output grid) plus preserve_lr_balance.

_DFHRTF_FAST = dict(n_fft=4096, output_octave_fraction=12, preserve_lr_balance=True)
_SOFA_BASELINE = dict(target_radius=1.5, hrir_length=256)


class TestSofaManifestRoundTrip:
    def test_mixed_ok_and_fail_jobs(self, tmp_path, capsys, synthetic_sofa_file, sofa_common):
        good_input = synthetic_sofa_file(tmp_path / "src.sofa", n_points=96, onset=40)
        missing_input = str(tmp_path / "does_not_exist.sofa")

        out1 = tmp_path / "out1.sofa"
        out2 = tmp_path / "out2.sofa"
        out_bad = tmp_path / "out_bad.sofa"

        jobs = [
            {"input": good_input, "output": str(out1), "target_fs": 48000, "dfeq": False},
            {"input": missing_input, "output": str(out_bad), "target_fs": 48000, "dfeq": False},
            {"input": good_input, "output": str(out2), "target_fs": 48000, "dfeq": True},
        ]
        manifest_path = write_manifest(str(tmp_path), sofa_common(**_SOFA_BASELINE), jobs)

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
    def test_mixed_ok_and_fail_jobs(self, tmp_path, capsys, synthetic_sofa_file, dfhrtf_common):
        good_input = synthetic_sofa_file(tmp_path / "src.sofa", n_points=96, onset=40)
        missing_input = str(tmp_path / "does_not_exist.sofa")

        text_l1, text_r1 = tmp_path / "f1_L.txt", tmp_path / "f1_R.txt"
        text_l2, text_r2 = tmp_path / "f2_L.txt", tmp_path / "f2_R.txt"
        text_l_bad = tmp_path / "bad_L.txt"

        jobs = [
            {"input": good_input, "text_out_l": str(text_l1), "text_out_r": str(text_r1)},
            {"input": missing_input, "text_out_l": str(text_l_bad)},
            {"input": good_input, "text_out_l": str(text_l2), "text_out_r": str(text_r2)},
        ]
        manifest_path = write_manifest(str(tmp_path), dfhrtf_common(**_DFHRTF_FAST), jobs)

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


# ── 4. Per-file progress reconciliation ─────────────────────────────────────────

class TestRunChunksParallelProgress:
    def _chunk_spec(self, desc, n_files, exit_code=0, emit_all_sentinels=True):
        """A chunk whose subprocess prints [JOB OK] for n_files (or none, if
        emit_all_sentinels is False — simulating a crash before any per-file
        sentinel could be printed) then exits with exit_code."""
        n_emit = n_files if emit_all_sentinels else 0
        script_lines = [
            "import sys",
            f"for i in range({n_emit}):",
            "    print(f'[JOB OK] file_{i}')",
            f"sys.exit({exit_code})",
        ]
        script = "\n".join(script_lines)
        return {
            "cmd": [sys.executable, "-c", script],
            "desc": desc,
            "jobs": list(range(n_files)),
            "input_path": desc,
        }

    def test_progress_advances_per_file_not_per_chunk(self, tmp_path):
        chunk_specs = [
            self._chunk_spec("chunk A", 3),
            self._chunk_spec("chunk B", 2),
        ]
        progress_calls = []

        def progress_callback(pct, msg):
            progress_calls.append((pct, msg))

        failed_files, failed_chunks, was_stopped = run_chunks_parallel(
            chunk_specs, max_workers=2,
            progress_callback=progress_callback,
            log_callback=lambda s: None,
            stop_checker=None,
            log_path=str(tmp_path / "run.log"),
            error_callback=lambda s: None,
        )

        assert not was_stopped
        assert not failed_files
        assert not failed_chunks
        # 5 total files across both chunks -> at least 5 distinct progress
        # updates (one per sentinel), not just 2 (one per chunk completion).
        assert len(progress_calls) >= 5
        # Progress must be monotonically non-decreasing and reach 100 at the end.
        pcts = [p for p, _ in progress_calls]
        assert pcts == sorted(pcts)
        assert pcts[-1] == 100

    def test_crashed_chunk_reconciles_progress_to_total(self, tmp_path):
        # One well-behaved chunk, one that "crashes" before emitting any
        # per-file sentinel and exits nonzero — progress should still reach
        # 100% at the end via shortfall reconciliation.
        chunk_specs = [
            self._chunk_spec("good chunk", 3),
            self._chunk_spec("crashed chunk", 4, exit_code=1, emit_all_sentinels=False),
        ]
        progress_calls = []

        failed_files, failed_chunks, was_stopped = run_chunks_parallel(
            chunk_specs, max_workers=2,
            progress_callback=lambda pct, msg: progress_calls.append((pct, msg)),
            log_callback=lambda s: None,
            stop_checker=None,
            log_path=str(tmp_path / "run.log"),
            error_callback=lambda s: None,
        )

        assert not was_stopped
        # No [JOB FAIL] sentinel was printed, only a nonzero exit with no
        # per-file attribution -> the whole chunk is attributed as failed.
        assert failed_chunks == ["crashed chunk"]
        assert not failed_files
        pcts = [p for p, _ in progress_calls]
        assert pcts[-1] == 100


# ── 5. GLOBAL_RoomType sanitize ─────────────────────────────────────────────────

class TestSanitizeRoomType:
    @pytest.mark.parametrize("raw,expected", [
        ("free-field", "free field"),
        ("Free_Field", "free field"),
        ("FREE-FIELD", "free field"),
        ("free field", "free field"),  # already conformant -> untouched
        ("reverberant", "reverberant"),
    ])
    def test_room_type_normalized(self, raw, expected):
        s = sf.Sofa("SimpleFreeFieldHRIR")
        s.GLOBAL_RoomType = raw
        sanitize_sofa(s)
        assert s.GLOBAL_RoomType == expected

    def test_sanitized_room_type_survives_write(self, tmp_path, synthetic_sofa_file):
        # Reproduces the reported failure: a "free-field" RoomType is accepted
        # on read but rejected by sf.write_sofa's convention check unless
        # normalized first.
        src_path = synthetic_sofa_file(tmp_path / "src.sofa", n_points=96, onset=40)
        s2 = sf.read_sofa(src_path)
        s2.GLOBAL_RoomType = "free-field"
        sanitize_sofa(s2)
        out_path = str(tmp_path / "out.sofa")
        sf.write_sofa(out_path, s2)  # must not raise
        assert os.path.exists(out_path)

    def test_unrecognized_room_type_left_alone(self):
        # Don't silently rewrite values we don't recognize as a normalized form
        # of a valid convention value.
        s = sf.Sofa("SimpleFreeFieldHRIR")
        s.GLOBAL_RoomType = "some custom room"
        sanitize_sofa(s)
        assert s.GLOBAL_RoomType == "some custom room"
