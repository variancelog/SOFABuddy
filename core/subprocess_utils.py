"""
core/subprocess_utils.py
Shared helpers for launching subprocesses with full diagnostic capture.
Used by sofa_processor and dfhrtf_processor.

Two execution modes are supported:
  - run_task()            — one subprocess per call; used directly for the
                             sequential path and as the primitive each pooled
                             chunk uses under run_chunks_parallel().
  - run_chunks_parallel()  — bounded ThreadPoolExecutor over multiple chunk
                             subprocesses (each processing several files via
                             a JSON manifest), with per-file progress/failure
                             attribution parsed from [JOB OK]/[JOB FAIL]
                             sentinel lines the backend scripts print in
                             manifest mode.
"""

import concurrent.futures
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime

# Guards appends to the shared on-disk diagnostics log (core/subprocess_utils.py
# is used from multiple pool threads when run_chunks_parallel is active; without
# this lock, concurrent multi-line FAIL blocks could interleave and corrupt the
# log file).
_log_lock = threading.Lock()


def new_run_log(stage: str, output_root: str = None) -> str:
    """
    Return a timestamped log file path.
    Prefers output_root (sits next to the output folder so it's easy to find).
    Falls back to the OS temp dir if output_root is unusable.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sofa_buddy_{stage}_{ts}.log"
    if output_root:
        try:
            os.makedirs(output_root, exist_ok=True)
            candidate = os.path.join(output_root, filename)
            # Touch the file to confirm the path is writable before committing to it
            open(candidate, "a", encoding="utf-8").close()
            return candidate
        except OSError:
            pass
    return os.path.join(tempfile.gettempdir(), filename)


def run_task(cmd, desc, input_path, log_callback, stop_checker, log_path,
             error_callback=None, tail_lines=40, quiet=False, env=None):
    """
    Run a subprocess, stream its output to log_callback and capture it,
    then write a diagnostic entry to log_path.

    quiet: when True, suppress the per-line log_callback(stripped) streaming
    (output is still captured and still written to the on-disk log, and the
    inline FAIL block is still emitted on failure). Used when many chunk
    subprocesses run concurrently, so the console isn't flooded with
    interleaved per-line output from N workers at once.

    env: optional environment dict passed through to Popen. Callers that
    override this should base it on a copy of os.environ (see
    blas_guarded_env()) — a bare partial dict strips PATH/SystemRoot and
    breaks the child on Windows.

    Returns:
        (returncode, captured_lines)  — on normal completion (success or failure)
        (None, captured_lines)        — when stop_checker fired; caller should abort
    """
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags,
        env=env,
    )

    captured = []
    stopped = False

    for line in iter(process.stdout.readline, ""):
        if stop_checker and stop_checker():
            process.terminate()
            process.wait()
            stopped = True
            break
        if line:
            stripped = line.rstrip("\n")
            if not quiet:
                log_callback(stripped)
            captured.append(stripped)

    if not stopped:
        process.stdout.close()
        returncode = process.wait()
    else:
        # Try to drain and close cleanly after terminate
        try:
            process.stdout.close()
        except Exception:
            pass
        return None, captured

    # --- On-disk log entry ---
    try:
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as lf:
                if returncode == 0:
                    lf.write(f"OK   {desc}\n")
                else:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    lf.write(
                        f"\n{'='*72}\n"
                        f"FAIL {desc}\n"
                        f"Time:  {ts}\n"
                        f"Input: {input_path}\n"
                        f"Exit:  {returncode}\n"
                        f"Cmd:   {' '.join(str(c) for c in cmd)}\n"
                        f"--- Output ---\n"
                    )
                    for l in captured:
                        lf.write(l + "\n")
                    lf.write(f"{'='*72}\n")
    except OSError:
        pass  # Never let log I/O break the processing loop

    # --- Inline failure header (on failure) ---
    # Subprocess output was already streamed line-by-line above; only the
    # failure marker and source path are new information here.
    if returncode != 0:
        _err = error_callback if error_callback else log_callback
        _err(f"{'─'*60}")
        _err(f"[FAIL] {desc}  (exit {returncode})")
        _err(f"  Input: {input_path}")
        _err(f"  Full output saved to log file.")

    return returncode, captured


# ── Chunked/parallel execution ─────────────────────────────────────────────────

def blas_guarded_env():
    """
    Return a subprocess environment with BLAS/OpenMP thread counts pinned to 1.

    Each chunk subprocess runs inside a ThreadPoolExecutor slot; without this,
    numpy/BLAS could internally multithread within every worker and
    oversubscribe cores (pool parallelism x BLAS parallelism). Built as a copy
    of the parent environment (not a bare 4-key dict) so PATH/SystemRoot and
    friends survive — a stripped env breaks subprocess launch on Windows.
    """
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    return env


def effective_chunk_size(total_tasks: int, max_workers: int, chunk_size_cfg: int) -> int:
    """
    Clamp the configured chunk size so at least max_workers chunks exist
    whenever there are enough tasks to fill them, while never exceeding
    chunk_size_cfg. Guarantees all worker slots stay busy on small batches
    without losing the import-cost amortization on large ones.

        effective_chunk = min(chunk_size_cfg, max(1, ceil(total_tasks / max_workers)))
    """
    if total_tasks <= 0:
        return max(1, chunk_size_cfg)
    max_workers = max(1, max_workers)
    chunk_size_cfg = max(1, chunk_size_cfg)
    return min(chunk_size_cfg, max(1, math.ceil(total_tasks / max_workers)))


def chunk_jobs(jobs, effective_chunk):
    """
    Split *jobs* into contiguous slices of at most *effective_chunk* items.
    Never sorts or shuffles — callers rely on build-order adjacency (e.g. SOFA
    jobs for one input file's SR/EQ variants) landing in the same chunk.
    """
    effective_chunk = max(1, effective_chunk)
    return [jobs[i:i + effective_chunk] for i in range(0, len(jobs), effective_chunk)]


def write_manifest(tmp_dir, common, jobs_chunk):
    """Write one JSON manifest ({"common": ..., "jobs": [...]}) for a chunk;
    returns its path."""
    fd, path = tempfile.mkstemp(suffix=".json", dir=tmp_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"common": common, "jobs": jobs_chunk}, f)
    return path


def _parse_job_sentinels(captured_lines):
    """
    Extract [JOB OK]/[JOB FAIL] sentinel lines emitted by a backend script's
    manifest mode. Returns (ok_inputs, fail_pairs) where fail_pairs is a list
    of (input_path, error_message).
    """
    ok, fail = [], []
    for line in captured_lines:
        if line.startswith("[JOB OK] "):
            ok.append(line[len("[JOB OK] "):])
        elif line.startswith("[JOB FAIL] "):
            rest = line[len("[JOB FAIL] "):]
            if ": " in rest:
                path, err = rest.split(": ", 1)
            else:
                path, err = rest, "unknown error"
            fail.append((path, err))
    return ok, fail


def run_chunks_parallel(chunk_specs, max_workers, progress_callback, log_callback,
                         stop_checker, log_path, error_callback, env=None):
    """
    Run each chunk in chunk_specs through a bounded ThreadPoolExecutor.

    chunk_specs: list of dicts with keys
      'cmd'         — subprocess argv (backend script invoked with --manifest)
      'desc'        — human-readable label for progress/log lines
      'jobs'        — the list of per-file job dicts this chunk's manifest
                       contains (only its length is used, for progress math)
      'input_path'  — (optional) shown as the "Input:" line if the whole
                       chunk fails before any per-job sentinel is parseable

    Threads are the correct primitive here: run_task() mostly blocks on
    subprocess stdout I/O (GIL released), and the heavy compute happens in
    the child processes, not in this thread pool.

    Progress and failure attribution are parsed from each chunk's captured
    [JOB OK]/[JOB FAIL] sentinel lines once that chunk's subprocess exits, so
    progress advances in file-sized (not chunk-sized) increments and failures
    are reported by filename rather than by opaque chunk index.

    Returns (failed_files, failed_chunks, was_stopped):
      failed_files  — list of (input_path, error_message) from [JOB FAIL] lines
      failed_chunks — desc strings for chunks that exited nonzero without
                      emitting any sentinel at all (e.g. crashed on import,
                      before the per-job try/except in manifest mode could run)
      was_stopped   — True if stop_checker fired during the run
    """
    total_jobs = sum(len(c["jobs"]) for c in chunk_specs)
    completed_jobs = 0
    failed_files = []
    failed_chunks = []
    was_stopped = False
    progress_lock = threading.Lock()

    def _run_one(chunk):
        return run_task(
            chunk["cmd"], chunk["desc"],
            chunk.get("input_path", chunk["desc"]),
            log_callback, stop_checker, log_path,
            error_callback=error_callback, quiet=True, env=env,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(_run_one, chunk): chunk for chunk in chunk_specs}

        for future in concurrent.futures.as_completed(futures):
            chunk = futures[future]
            rc, captured = future.result()

            if rc is None:
                # stop_checker fired inside run_task for this chunk — not a failure.
                was_stopped = True
                continue

            ok_files, fail_pairs = _parse_job_sentinels(captured)
            sentinel_total = len(ok_files) + len(fail_pairs)
            failed_files.extend(fail_pairs)

            with progress_lock:
                completed_jobs += len(chunk["jobs"])
                if sentinel_total == 0 and rc != 0:
                    # Chunk died before printing any sentinel (e.g. crashed on
                    # import or on a corrupt manifest) — attribute to the chunk
                    # since we can't name the individual file.
                    failed_chunks.append(chunk["desc"])

                if total_jobs > 0:
                    pct = int((min(completed_jobs, total_jobs) / total_jobs) * 100)
                    progress_callback(
                        pct,
                        f"{chunk['desc']} ({min(completed_jobs, total_jobs)}/{total_jobs} files)",
                    )

            if fail_pairs:
                log_callback(
                    f"[FAIL] {chunk['desc']} — {len(fail_pairs)}/"
                    f"{sentinel_total or len(chunk['jobs'])} files failed"
                )
            elif ok_files:
                log_callback(f"[OK] {chunk['desc']} ({len(ok_files)} files)")
            elif rc != 0:
                log_callback(f"[FAIL] {chunk['desc']} — chunk exited {rc} with no output")

            if stop_checker and stop_checker():
                was_stopped = True
                break

        if was_stopped:
            # Cancel chunks not yet started. Already-running chunks self-terminate
            # via their own stop_checker poll inside run_task's stdout read loop —
            # that poll only fires when the child emits a line, so Stop can lag by
            # up to one file's compute time (same latency class as the sequential
            # path). Do not "fix" this with a busy-loop.
            executor.shutdown(cancel_futures=True)

    return failed_files, failed_chunks, was_stopped
