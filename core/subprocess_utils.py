"""
core/subprocess_utils.py
Shared helpers for launching subprocesses with full diagnostic capture.
Used by sofa_processor and dfhrtf_processor.
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime


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
             error_callback=None, tail_lines=40):
    """
    Run a subprocess, stream its output to log_callback and capture it,
    then write a diagnostic entry to log_path.

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
