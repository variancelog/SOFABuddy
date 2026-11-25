import os
import sys
import glob
import shutil
import tempfile
import numpy as np

from core.naming import render_filename
from core.subprocess_utils import (
    new_run_log, run_chunks_parallel, chunk_jobs, write_manifest,
    effective_chunk_size, blas_guarded_env,
)

# Shared folder-name constants (single definition; avoids silent drift between files)
SOFA_MASTERED_DIR = "sofa_mastered"
DFHRTF_OUTPUT_DIR = "DFHRTF"

# In-project imports (Deferred inside functions to speed up startup)

def run_averaging(file_list, algo_name, exclude_outliers, output_path, delimiter,
                  log_callback, stop_checker=None, avg_params=None, n_workers=1):
    """
    Orchestrates the full ingestion -> interpolation -> outlier -> averaging chain.

    avg_params is an optional dict drawn from config['averaging'] and may contain:
      num_points, outlier_threshold_std,
      softdtw_gamma, geo_arc_y_scale
    All keys fall back to the values used historically if absent.

    n_workers caps the Soft-DTW barycenter's internal thread pool (only the
    Soft-DTW algorithm path uses it). This runs in the parent GUI process, so it
    is sized to the same worker count as the chunk pools to avoid oversubscribing
    cores (the parent's BLAS is already pinned to 1 thread at app startup).
    """
    from core.averaging import DataIngestor, CommonAxisInterpolator, OutlierDetector, CurveAverager

    if not file_list:
        return

    # Unpack averaging params with the historically-correct runtime defaults
    p = avg_params or {}
    num_points            = int(p.get('num_points', 512))
    output_octave_fraction = int(p.get('output_octave_fraction', 48))
    threshold_std         = float(p.get('outlier_threshold_std', 1.0))
    softdtw_gamma         = float(p.get('softdtw_gamma_global', 0.03))
    softdtw_max_iter      = int(p.get('softdtw_max_iter', 25))
    softdtw_tol           = float(p.get('softdtw_tol', 0.001))
    geo_arc_y_scale       = float(p.get('geo_arc_y_scale', 0.01))

    try:
        log_callback(f"Averaging {len(file_list)} files using {algo_name}...")

        # 1. Ingest
        dfs = []
        for f in file_list:
            if stop_checker and stop_checker(): return
            dfs.append(DataIngestor.load_csv(f))

        # 2. Interpolate to common axis
        if stop_checker and stop_checker(): return
        interp_data = CommonAxisInterpolator.interpolate_to_common_axis(
            dfs, num_points=num_points, octave_fraction=output_octave_fraction
        )
        common_freqs = interp_data['common_freqs']
        norm_freqs   = interp_data['norm_freqs']
        norm_mags_2d = interp_data['norm_mags_2d']
        mag_min      = interp_data['mag_min']
        mag_max      = interp_data['mag_max']

        # 3. Outlier Detection
        final_mags_2d = norm_mags_2d
        if exclude_outliers:
            if stop_checker and stop_checker(): return
            valid_indices, excluded = OutlierDetector.filter_outliers(
                norm_mags_2d, algo_name,
                threshold_std=threshold_std
            )
            if excluded:
                log_callback(f" -> Excluded {len(excluded)} outliers.")
            final_mags_2d = norm_mags_2d[valid_indices]

        # 4. Average
        if stop_checker and stop_checker(): return
        if algo_name == "Arithmetic Mean":
            avg_norm = CurveAverager.arithmetic_mean(final_mags_2d)
        elif algo_name == "Geo-Arc-Length":
            # Pre-scale data to weight the y-axis in the arc-length distance calculation
            scaled_mags = final_mags_2d * geo_arc_y_scale
            scaled_avg  = CurveAverager.geometric_arc_length_blend(norm_freqs, scaled_mags)
            avg_norm    = scaled_avg / geo_arc_y_scale
        elif algo_name == "Soft-DTW":
            log_callback(" -> Running Soft-DTW Barycenter (this may take a moment)...")
            avg_norm = CurveAverager.soft_dtw_barycenter(
                final_mags_2d,
                gamma=softdtw_gamma,
                max_iter=softdtw_max_iter,
                tol=softdtw_tol,
                n_workers=n_workers
            )
        else:
            avg_norm = CurveAverager.arithmetic_mean(final_mags_2d)

        # 5. Unnormalize
        if stop_checker and stop_checker(): return
        avg_db = CommonAxisInterpolator.unnormalize_magnitude(avg_norm, mag_min, mag_max)

        # 6. Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        is_csv = output_path.lower().endswith(".csv")
        with open(output_path, 'w', newline='') as ffile:
            if is_csv:
                ffile.write(f"Frequency (Hz){delimiter}Magnitude (dB)\n")
            for f, m in zip(common_freqs, avg_db):
                ffile.write(f"{f:.6f}{delimiter}{m:.3f}\n")

        log_callback(f" [+] Saved Global Average: {os.path.basename(output_path)}")
        return common_freqs, avg_db

    except Exception as e:
        log_callback(f" [!] Averaging Error: {str(e)}")

def process_dfhrtf_exports(config, progress_callback, log_callback, error_callback, stop_checker=None):
    """
    Main processing logic for DFHRTF exports.
    Analyzes SOFA files and generates text/graph data. Jobs are built
    sequentially (this is where the groups/seen_grid_dirs bookkeeping decides
    per-job output paths and which file emits the weighted grid), then
    grouped into contiguous chunks and run as generate_dfhrtf.py --manifest
    subprocesses through a bounded thread pool (see
    core.subprocess_utils.run_chunks_parallel).
    """
    try:
        df_cfg = config.get('dfhrtf', {})
        if not df_cfg.get('enabled'):
            return True

        files = config.get('files', [])
        if not files:
            return True

        scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        script_dfhrtf = os.path.join(scripts_dir, "generate_dfhrtf.py")

        if not os.path.exists(script_dfhrtf):
            error_callback(f"Backend script not found: {script_dfhrtf}")
            return False

        # 1. Determine Output Mode (app-level setting from FilesTab)
        out_cfg = config.get('output', {})
        use_default = out_cfg.get('default', True)
        custom_dir = out_cfg.get('folder', '').strip()
        if not use_default and not custom_dir:
            error_callback("Output Folder: No custom folder selected. Choose a folder or switch to Default.")
            return False

        # Hoist delimiter and shared params so they're available for averaging even if no files ran
        delim_map = {"Tab": "\t", "Comma": ",", "Semicolon": ";", "Space": " "}
        delim = delim_map.get(df_cfg.get('text_delimiter', "Tab"), "\t")
        avg_params = config.get('averaging', {})
        # Inject the per-SOFA output resolution so the averaging pipeline uses the
        # same fractional-octave grid as generate_dfhrtf.py, keeping both file
        # types at identical point counts and frequency axis values.
        avg_params = {
            **avg_params,
            'output_octave_fraction': int(df_cfg.get('advanced', {}).get('output_octave_fraction', 48)),
        }

        # Format tilt as one decimal place for use in filename templates
        tilt_str = f"{float(df_cfg.get('tilt', 0.0)):.1f}"

        # 2. Determine Input Files
        resolved_inputs = []  # (input_path, base_name, sr_label, src_dir)
        sofa_cfg = config.get('sofa', {})

        if sofa_cfg.get('enabled'):
            log_callback("DFHRTF: Searching for generated mastered SOFA files...")
            for raw_path in files:
                if stop_checker and stop_checker(): return False
                base_name = os.path.splitext(os.path.basename(raw_path))[0]
                src_dir = os.path.dirname(raw_path)

                # Mastered SOFAs live where sofa_processor wrote them
                base_dir = src_dir if use_default else custom_dir
                mastered_dir = os.path.join(base_dir, SOFA_MASTERED_DIR)

                if not os.path.exists(mastered_dir):
                    log_callback(
                        f" [!] Skipping {base_name}: no mastered SOFA found "
                        f"(mastering likely failed)."
                    )
                    continue

                pattern = os.path.join(mastered_dir, f"*{base_name}*.sofa")
                candidates = glob.glob(pattern)
                if not candidates:
                    log_callback(
                        f" [!] Skipping {base_name}: no mastered SOFA found "
                        f"(mastering likely failed)."
                    )
                    continue

                uneq_candidates = [c for c in candidates if "uneq" in c.lower()]
                target_list = uneq_candidates if uneq_candidates else candidates

                def get_sr(path):
                    import re
                    match = re.search(r'(\d+)', os.path.basename(path))
                    return int(match.group(1)) if match else 0

                target_list.sort(key=get_sr, reverse=True)
                target_path = target_list[0]
                resolved_inputs.append((target_path, base_name, str(get_sr(target_path)), src_dir))
        else:
            for f in files:
                if stop_checker and stop_checker(): return False
                resolved_inputs.append((f, os.path.splitext(os.path.basename(f))[0], "raw", os.path.dirname(f)))

        # Track generated L/R files grouped by output directory for per-group averaging
        # groups[output_root] = {'l': [...], 'r': [...], 'folder': source_folder_name}
        groups = {}

        # Track which output dirs have already received a weighted-grid plot (Per Folder scope)
        seen_grid_dirs = set()

        total_inputs = len(resolved_inputs)
        log_root = custom_dir if not use_default else os.path.dirname(files[0])
        log_path = new_run_log("dfhrtf", log_root)
        log_callback(f"[Diagnostics log: {log_path}]")

        # Advanced dfhrtf params (with historically-correct defaults) — shared
        # across every job. Manifest mode has no argparse default fallback, so
        # this "common" dict must supply every value run_dfhrtf() reads off
        # `args` unconditionally, including avg_algo which
        # the CLI path only ever got via argparse defaults.
        dfhrtf_adv = df_cfg.get('advanced', {})
        smoothing_cfg = str(df_cfg.get('smoothing', 'Variable'))
        # 'Variable' -> 'variable'; '1/24' (or legacy 24) -> '24'
        smoothing_arg = 'variable' if smoothing_cfg.lower() == 'variable' else smoothing_cfg.split('/')[-1]

        common = {
            "avg_algo": df_cfg.get('per_sofa_avg_algo', 'Geo-Arc-Length'),
            "delimiter": delim,
            "smoothing": smoothing_arg,
            "tilt": float(df_cfg.get('tilt', 0.0)),
            "lr_avg_gamma": float(avg_params.get('softdtw_gamma_persofa', 0.01)),
            "lf_ext_dfhrtf": bool(df_cfg.get('extend_lf', False)),
            "lf_ext_dfhrtf_xover": float(df_cfg.get('xover', 1000.0)),
            "n_fft": int(dfhrtf_adv.get('n_fft', 16384)),
            "output_octave_fraction": int(dfhrtf_adv.get('output_octave_fraction', 48)),
            "norm_freq_hz": float(dfhrtf_adv.get('norm_freq_hz', 1000.0)),
            "lf_blend_low_oct": float(dfhrtf_adv.get('lf_blend_low_oct', 0.50)),
            "lf_blend_high_oct": float(dfhrtf_adv.get('lf_blend_high_oct', 1.0)),
            "stereo_angle": float(dfhrtf_adv.get('stereo_angle_deg', 30.0)),
            "weighting_mode": str(df_cfg.get('weighting_mode', 'Auto')).lower(),
            "grid_scale_mode": str(dfhrtf_adv.get('weighted_grid_scale_mode', 'manual')).lower(),
            "grid_scale_low": float(dfhrtf_adv.get('weighted_grid_rel_low', 0.0)),
            "grid_scale_high": float(dfhrtf_adv.get('weighted_grid_rel_high', 2.0)),
            "preserve_lr_balance": bool(dfhrtf_adv.get('preserve_lr_balance', True)),
            "var_smooth_freq_low": float(dfhrtf_adv.get('variable_smoothing_freq_low', 350.0)),
            "var_smooth_freq_hi": float(dfhrtf_adv.get('variable_smoothing_freq_hi', 6000.0)),
            "var_smooth_n_low": float(dfhrtf_adv.get('variable_smoothing_n_low', 2.0)),
            "var_smooth_n_hi": float(dfhrtf_adv.get('variable_smoothing_n_hi', 12.0)),
            "var_smooth_gamma": float(dfhrtf_adv.get('variable_smoothing_gamma', 0.88)),
            "var_smooth_mean_mode": str(dfhrtf_adv.get('variable_smoothing_mean_mode', 'db')).lower(),
        }

        # 3. Build Job List (sequential — no subprocesses, so groups/
        # seen_grid_dirs bookkeeping needs no locking)
        jobs = []  # each carries 'input' plus whichever *_out keys apply, and 'desc'
        for input_path, base_name, sr_label, src_dir in resolved_inputs:
            if stop_checker and stop_checker(): return False

            # Per-file DFHRTF output root
            base_dir = src_dir if use_default else custom_dir
            output_root = os.path.join(base_dir, DFHRTF_OUTPUT_DIR)
            os.makedirs(output_root, exist_ok=True)

            folder_name = os.path.basename(src_dir)

            job = {'input': input_path, 'desc': base_name}

            # Text Outputs
            ext = f".{df_cfg.get('text_extension', 'txt')}"
            template = df_cfg.get('text_filename', "{filename}_{samplerate}_{LR}")

            def get_text_path(lr_val, _base=base_name, _sr=sr_label, _ext=ext, _root=output_root, _tmpl=template, _folder=folder_name, _tilt=tilt_str):
                # {algo} is meaningless for per-SOFA outputs, so it is dropped
                # along with an adjacent separator rather than substituted.
                fname = render_filename(
                    _tmpl,
                    {
                        "filename": _base,
                        "folder": _folder,
                        "samplerate": _sr,
                        "tilt": _tilt,
                        "LR": lr_val,
                    },
                    ext=_ext,
                    drop={"algo"},
                )
                return os.path.join(_root, fname)

            if df_cfg.get('per_sofa_lr'):
                path_l = get_text_path("L")
                path_r = get_text_path("R")
                job['text_out_l'] = path_l
                job['text_out_r'] = path_r
                grp = groups.setdefault(output_root, {'l': [], 'r': [], 'folder': folder_name})
                grp['l'].append(path_l)
                grp['r'].append(path_r)

            if df_cfg.get('per_sofa_avg'):
                avg_tmpl = df_cfg.get('per_sofa_avg_filename', "{filename} avg LR")
                job['text_out_avg'] = get_text_path("Avg", _tmpl=avg_tmpl)

            # Graph Outputs
            if df_cfg.get('dfhrtf_graphs'):
                graph_template = df_cfg.get('graph_filename', "{filename}")
                graph_ext = f".{df_cfg.get('graph_extension', 'png')}"

                gname = render_filename(
                    graph_template,
                    {"filename": base_name, "folder": folder_name,
                     "tilt": tilt_str, "graphtype": "df"},
                    ext=graph_ext,
                )
                job['graph_out'] = os.path.join(output_root, gname)

            if df_cfg.get('weighted_grid'):
                graph_template = df_cfg.get('graph_filename', "{filename}")
                graph_ext = f".{df_cfg.get('graph_extension', 'png')}"
                scope = df_cfg.get('weighted_grid_scope', 'Per Folder')

                emit_weighted = (scope == 'All Files') or (output_root not in seen_grid_dirs)
                if emit_weighted:
                    seen_grid_dirs.add(output_root)
                    wname = render_filename(
                        graph_template,
                        {"filename": base_name, "folder": folder_name,
                         "tilt": tilt_str, "graphtype": "weighted"},
                    )

                    # Fallback: if {graphtype} wasn't used in the template, append '_weighted'
                    if "{graphtype}" not in graph_template:
                        if wname.lower().endswith(graph_ext.lower()):
                            wname = wname[:-len(graph_ext)] + "_weighted" + graph_ext
                        else:
                            wname += "_weighted" + graph_ext
                    else:
                        wname = render_filename(wname, {}, ext=graph_ext)

                    job['graph_weighted_out'] = os.path.join(output_root, wname)

            jobs.append(job)

        # Resolve worker count once, up front, so the post-step global averaging
        # phase can reuse it even when there are no chunk jobs to run (0 = auto =
        # half of CPU cores; mirrors run_chunks_parallel's sizing).
        proc_cfg = config.get('processing', {})
        raw_workers = int(proc_cfg.get('max_workers', 0) or 0)
        max_workers = raw_workers if raw_workers > 0 else max(1, (os.cpu_count() or 2) // 2)

        # Jobs that ended up with no output at all (e.g. weighted grid scope
        # "Per Folder" with no text/graph outputs: only the first file per
        # folder emits a grid) would just load a SOFA to produce nothing, so
        # they never reach a subprocess. They still count as done for progress.
        output_keys = ('text_out_l', 'text_out_r', 'text_out_avg',
                       'graph_out', 'graph_stereo_out', 'graph_weighted_out')
        # A job whose input is missing is kept so the run still reports it.
        jobs_to_run = [j for j in jobs
                       if any(j.get(k) for k in output_keys)
                       or not os.path.exists(j['input'])]
        n_skipped = len(jobs) - len(jobs_to_run)
        if n_skipped:
            log_callback(f"DFHRTF: {n_skipped} file(s) needed no output; skipped.")
            jobs = jobs_to_run
            if not jobs:
                progress_callback(100, "DFHRTF: no output needed")

        # 4. Execute Jobs (chunked + parallel)
        if jobs:
            chunk_size_cfg = int(proc_cfg.get('chunk_size', 16) or 16)
            chunk_size = effective_chunk_size(len(jobs), max_workers, chunk_size_cfg)

            log_callback(
                f"DFHRTF analysis: {len(jobs)} file(s), {max_workers} worker(s), "
                f"chunk size {chunk_size}."
            )

            tmp_dir = tempfile.mkdtemp(prefix="sofabuddy_manifest_")
            try:
                chunk_specs = []
                for i, job_chunk in enumerate(chunk_jobs(jobs, chunk_size)):
                    manifest_jobs = [
                        {k: v for k, v in j.items() if k != 'desc'} for j in job_chunk
                    ]
                    manifest_path = write_manifest(tmp_dir, common, manifest_jobs)
                    cmd = [sys.executable, "-u", script_dfhrtf, "--manifest", manifest_path]
                    chunk_specs.append({
                        'cmd': cmd,
                        'desc': f"DFHRTF chunk {i + 1} ({len(job_chunk)} files)",
                        'jobs': job_chunk,
                        'input_path': job_chunk[0]['input'],
                    })

                failed_files, failed_chunks, was_stopped = run_chunks_parallel(
                    chunk_specs, max_workers,
                    progress_callback, log_callback, stop_checker,
                    log_path, error_callback, env=blas_guarded_env(),
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if was_stopped:
                return False

            if failed_files or failed_chunks:
                chunk_job_total = sum(
                    len(c['jobs']) for c in chunk_specs if c['desc'] in failed_chunks
                )
                total_failed = len(failed_files) + chunk_job_total
                error_callback(
                    f"DFHRTF analysis: {total_failed}/{total_inputs} tasks failed "
                    f"— see inline blocks above and log: {log_path}"
                )
                for path, err in failed_files:
                    error_callback(f"  - {path}: {err}")
                for desc in failed_chunks:
                    error_callback(f"  - {desc}: chunk failed before reporting per-file results")

        # 5. Post-Processing: Averaging - Global averages (per output-folder group)
        # Default mode: one group per source folder -> per-folder average
        # Custom mode: all files share one output_root -> single global average
        if df_cfg.get('avg_all') and groups:
            algo = df_cfg.get('algo', 'Soft-DTW')
            exclude = df_cfg.get('exclude_outliers', True)
            ext = f".{df_cfg.get('text_extension', 'txt')}"
            short_algo = {"Arithmetic Mean": "arithmean", "Geo-Arc-Length": "geoarc", "Soft-DTW": "softdtw"}.get(algo, "softdtw")
            avg_template = df_cfg.get('avg_filename', "{folder} avg {LR}")
            avg_graph_template = df_cfg.get('avg_graph_filename', "{folder} avg")

            for out_dir, grp in groups.items():
                if not grp['l']:
                    continue
                if stop_checker and stop_checker(): return False

                label = "Per-Folder Average" if len(groups) > 1 else "Global Average"
                log_callback(f"\n--- Post-Processing: {label} ({os.path.basename(out_dir) or out_dir}) ---")
                os.makedirs(out_dir, exist_ok=True)

                def get_avg_path(lr_val, _dir=out_dir, _tmpl=avg_template, _ext=ext, _algo=short_algo, _tilt=tilt_str, _folder=grp['folder']):
                    fname = render_filename(
                        _tmpl,
                        {
                            "filename": "dfhrtfs_avg",
                            "folder": _folder,
                            "samplerate": "all",
                            "tilt": _tilt,
                            "LR": lr_val,
                            "algo": _algo,
                        },
                        ext=_ext,
                    )
                    return os.path.join(_dir, fname)

                res_l = run_averaging(grp['l'], algo, exclude, get_avg_path("L"), delim,
                                      log_callback, stop_checker, avg_params=avg_params,
                                      n_workers=max_workers)
                if stop_checker and stop_checker(): return False
                res_r = run_averaging(grp['r'], algo, exclude, get_avg_path("R"), delim,
                                      log_callback, stop_checker, avg_params=avg_params,
                                      n_workers=max_workers)

                if df_cfg.get('dfhrtf_graphs') and res_l and res_r:
                    try:
                        from core.plotting import plot_diffuse_field
                        gext = f".{df_cfg.get('graph_extension', 'png')}"
                        gname = render_filename(
                            avg_graph_template,
                            {"folder": grp['folder'], "tilt": tilt_str,
                             "graphtype": "df"},
                            ext=gext,
                        )
                        freqs_l, db_l = res_l
                        _, db_r = res_r
                        plot_diffuse_field(freqs_l, db_l, db_r,
                                           os.path.join(out_dir, gname), tilt_str)
                        log_callback(f" [+] Saved Average Graph: {gname}")
                    except Exception as e:
                        log_callback(f" [!] Average Graph Error: {str(e)}")

        return True

    except Exception as e:
        error_callback(f"DFHRTF Processor Error: {str(e)}")
        return False
