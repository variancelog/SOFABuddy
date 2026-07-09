import os
import shutil
import sys
import tempfile

from core.dfhrtf_processor import SOFA_MASTERED_DIR  # shared constant
from core.subprocess_utils import (
    new_run_log, run_chunks_parallel, chunk_jobs, write_manifest,
    effective_chunk_size, blas_guarded_env,
)

def process_sofa_exports(config, progress_callback, log_callback, error_callback, stop_checker=None):
    """
    Main processing logic for SOFA exports.
    Breaks the UI config into individual per-file jobs (one per file x
    samplerate x EQ variant), groups them into contiguous chunks, and runs
    each chunk as one generate_sofa.py --manifest subprocess through a
    bounded thread pool (core.subprocess_utils.run_chunks_parallel). This
    amortizes interpreter/import startup cost across a chunk while still
    running multiple chunks concurrently.
    """
    try:
        files = config.get('files', [])
        sofa_cfg = config.get('sofa', {})

        if not sofa_cfg.get('enabled'):
            return True # Not an error, just nothing to do for this module

        scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        script_master = os.path.join(scripts_dir, "generate_sofa.py")

        if not os.path.exists(script_master):
            error_callback(f"Backend script not found: {script_master}")
            return False

        # 1. Determine Output Mode
        if not files:
            return True

        if stop_checker and stop_checker():
            return False

        out_cfg = config.get('output', {})
        use_default = out_cfg.get('default', True)
        custom_dir = out_cfg.get('folder', '').strip()
        if not use_default and not custom_dir:
            error_callback("Output Folder: No custom folder selected. Choose a folder or switch to Default.")
            return False

        # Advanced SOFA params (with historically-correct defaults). Built once
        # as the manifest "common" section shared by every job in this run —
        # every attribute master_sofa() reads off `args` (directly or via
        # getattr) must be present here, since manifest mode has no argparse
        # default fallback for values a job doesn't explicitly override.
        sofa_adv = sofa_cfg.get('advanced', {})
        # Weighting: driven by dfhrtf config keys so one setting governs
        # weighting in both the SOFA mastering and DFHRTF analysis stages.
        df_cfg = config.get('dfhrtf', {})

        common = {
            "processing_length": int(sofa_adv.get('processing_length', 2048)),
            "norm_target_db":    float(sofa_adv.get('norm_target_db', -1.0)),
            "onset_buffer":      int(sofa_adv.get('onset_buffer', 32)),
            "onset_threshold":   float(sofa_adv.get('onset_threshold_ratio', 0.05)),
            "blend_low_oct":     float(sofa_adv.get('blend_low_oct', 0.50)),
            "blend_high_oct":    float(sofa_adv.get('blend_high_oct', 0.75)),
            "weighting_mode":    str(df_cfg.get('weighting_mode', 'Auto')).lower(),
            "ild_fix":           bool(sofa_cfg.get('ild_fix', True)),
            "ild_threshold":     float(sofa_cfg.get('ild_threshold', 1.5)),
            "ild_low_freq":      float(sofa_adv.get('ild_low_freq', 300.0)),
            "ild_high_freq":     float(sofa_adv.get('ild_high_freq', 6000.0)),
            "harmonize_radius":  bool(sofa_cfg.get('normalize_distance', False)),
            "target_radius":     float(sofa_cfg.get('distance_m', 1.2)),
            "lf_extend":         bool(sofa_cfg.get('lf_extension', False)),
            "lf_crossover":      float(sofa_cfg.get('xover', 1000.0)),
            "hrir_length":       int(sofa_cfg.get('hrir_length', 512)),
        }

        # 2. Build Job List
        jobs = []  # each: {'input', 'output', 'target_fs', 'dfeq', 'desc'}
        for file_path in files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Per-file output root: default -> each file's own folder; custom -> shared folder
            base_dir = os.path.dirname(file_path) if use_default else custom_dir
            out_root = os.path.join(base_dir, SOFA_MASTERED_DIR)
            os.makedirs(out_root, exist_ok=True)

            for sr in sofa_cfg.get('samplerates', []):
                eq_modes = []
                if sofa_cfg.get('uneq'): eq_modes.append(('uneq', False))
                if sofa_cfg.get('dfeq'): eq_modes.append(('dfeq', True))

                for eq_label, dfeq_on in eq_modes:
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    name_template = sofa_cfg.get('naming', "{filename}_{samplerate}_{eq}")
                    out_filename = name_template.replace("{filename}", base_name)
                    out_filename = out_filename.replace("{folder}", folder_name)
                    out_filename = out_filename.replace("{samplerate}", str(sr))
                    out_filename = out_filename.replace("{eq}", eq_label)

                    if not out_filename.lower().endswith(".sofa"):
                        out_filename += ".sofa"

                    out_path = os.path.join(out_root, out_filename)

                    # SOFA jobs for one input file (its SR x EQ variants) are
                    # appended consecutively here, so contiguous chunking below
                    # keeps them in the same chunk — that's what lets
                    # generate_sofa.py's manifest mode read each unique input
                    # SOFA only once.
                    jobs.append({
                        'input': file_path,
                        'output': out_path,
                        'target_fs': sr,
                        'dfeq': dfeq_on,
                        'desc': f"{base_name} ({sr}Hz, {eq_label})",
                    })

        # 3. Execute Jobs (chunked + parallel)
        total_tasks = len(jobs)
        if total_tasks == 0:
            return True

        log_root = custom_dir if not use_default else os.path.dirname(files[0])
        log_path = new_run_log("sofa", log_root)
        log_callback(f"[Diagnostics log: {log_path}]")

        proc_cfg = config.get('processing', {})
        raw_workers = int(proc_cfg.get('max_workers', 0) or 0)
        max_workers = raw_workers if raw_workers > 0 else max(1, (os.cpu_count() or 2) // 2)
        chunk_size_cfg = int(proc_cfg.get('chunk_size', 16) or 16)
        chunk_size = effective_chunk_size(total_tasks, max_workers, chunk_size_cfg)

        log_callback(
            f"SOFA export: {total_tasks} file(s), {max_workers} worker(s), "
            f"chunk size {chunk_size}."
        )

        tmp_dir = tempfile.mkdtemp(prefix="sofabuddy_manifest_")
        try:
            chunk_specs = []
            for i, job_chunk in enumerate(chunk_jobs(jobs, chunk_size)):
                manifest_jobs = [
                    {'input': j['input'], 'output': j['output'],
                     'target_fs': j['target_fs'], 'dfeq': j['dfeq']}
                    for j in job_chunk
                ]
                manifest_path = write_manifest(tmp_dir, common, manifest_jobs)
                cmd = [sys.executable, "-u", script_master, "--manifest", manifest_path]
                chunk_specs.append({
                    'cmd': cmd,
                    'desc': f"SOFA chunk {i + 1} ({len(job_chunk)} files)",
                    'jobs': job_chunk,
                    'input_path': job_chunk[0]['input'],
                })

            failed_files, failed_chunks, was_stopped = run_chunks_parallel(
                chunk_specs, max_workers,
                progress_callback, log_callback, stop_checker,
                log_path, error_callback, env=blas_guarded_env(),
            )
        finally:
            # A just-terminated child can still hold its manifest file open on
            # Windows; ignore cleanup errors rather than fail the whole run.
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if was_stopped:
            return False

        if failed_files or failed_chunks:
            chunk_job_total = sum(
                len(c['jobs']) for c in chunk_specs if c['desc'] in failed_chunks
            )
            total_failed = len(failed_files) + chunk_job_total
            error_callback(
                f"SOFA export: {total_failed}/{total_tasks} tasks failed "
                f"— see inline blocks above and log: {log_path}"
            )
            for path, err in failed_files:
                error_callback(f"  - {path}: {err}")
            for desc in failed_chunks:
                error_callback(f"  - {desc}: chunk failed before reporting per-file results")

        return True

    except Exception as e:
        error_callback(f"Processor Error: {str(e)}")
        return False
