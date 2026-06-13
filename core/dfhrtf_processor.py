import os
import sys
import glob
import numpy as np

from core.subprocess_utils import new_run_log, run_task

# Shared folder-name constants (single definition; avoids silent drift between files)
SOFA_MASTERED_DIR = "sofa_mastered"
DFHRTF_OUTPUT_DIR = "DFHRTF"

# In-project imports (Deferred inside functions to speed up startup)

def run_averaging(file_list, algo_name, exclude_outliers, output_path, delimiter,
                  log_callback, stop_checker=None, avg_params=None):
    """
    Orchestrates the full ingestion -> interpolation -> outlier -> averaging chain.

    avg_params is an optional dict drawn from config['averaging'] and may contain:
      num_points, outlier_threshold_std, outlier_gamma,
      softdtw_gamma, geo_arc_y_scale
    All keys fall back to the values used historically if absent.
    """
    from core.averaging import DataIngestor, CommonAxisInterpolator, OutlierDetector, CurveAverager

    if not file_list:
        return

    # Unpack averaging params with the historically-correct runtime defaults
    p = avg_params or {}
    num_points       = int(p.get('num_points', 512))
    threshold_std    = float(p.get('outlier_threshold_std', 1.0))
    outlier_gamma    = float(p.get('outlier_gamma', 1.0))
    softdtw_gamma    = float(p.get('softdtw_gamma_global', 0.03))
    softdtw_max_iter = int(p.get('softdtw_max_iter', 25))
    softdtw_tol      = float(p.get('softdtw_tol', 0.001))
    geo_arc_y_scale  = float(p.get('geo_arc_y_scale', 0.01))

    try:
        log_callback(f"Averaging {len(file_list)} files using {algo_name}...")

        # 1. Ingest
        dfs = []
        for f in file_list:
            if stop_checker and stop_checker(): return
            dfs.append(DataIngestor.load_csv(f))

        # 2. Interpolate to common axis
        if stop_checker and stop_checker(): return
        interp_data = CommonAxisInterpolator.interpolate_to_common_axis(dfs, num_points=num_points)
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
                threshold_std=threshold_std,
                gamma=outlier_gamma
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
                tol=softdtw_tol
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

    except Exception as e:
        log_callback(f" [!] Averaging Error: {str(e)}")

def process_dfhrtf_exports(config, progress_callback, log_callback, error_callback, stop_checker=None):
    """
    Main processing logic for DFHRTF exports.
    Analyzes SOFA files and generates text/graph data.
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
                    log_callback(f" [!] Warning: Mastered dir not found for {base_name}. Using raw file.")
                    resolved_inputs.append((raw_path, base_name, "raw", src_dir))
                    continue

                pattern = os.path.join(mastered_dir, f"*{base_name}*.sofa")
                candidates = glob.glob(pattern)
                if not candidates:
                    resolved_inputs.append((raw_path, base_name, "raw", src_dir))
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
        # groups[output_root] = {'l': [...], 'r': [...]}
        groups = {}

        # Track which output dirs have already received a weighted-grid plot (Per Folder scope)
        seen_grid_dirs = set()

        # 3. Execution
        total_inputs = len(resolved_inputs)
        log_root = custom_dir if not use_default else os.path.dirname(files[0])
        log_path = new_run_log("dfhrtf", log_root)
        error_callback(f"[Diagnostics log: {log_path}]")

        failures = []  # list of (base_name, returncode)

        for i, (input_path, base_name, sr_label, src_dir) in enumerate(resolved_inputs):
            if stop_checker and stop_checker(): return False

            progress_val = int((i / total_inputs) * 100)
            progress_callback(progress_val, f"Analyzing: {base_name}")

            # Per-file DFHRTF output root
            base_dir = src_dir if use_default else custom_dir
            output_root = os.path.join(base_dir, DFHRTF_OUTPUT_DIR)
            os.makedirs(output_root, exist_ok=True)

            folder_name = os.path.basename(src_dir)

            # Pull advanced dfhrtf params (with historically-correct defaults)
            dfhrtf_adv = df_cfg.get('advanced', {})

            cmd = [sys.executable, "-u", script_dfhrtf, "--input", input_path]
            cmd.extend(["--smoothing", str(df_cfg.get('smoothing', 48))])
            cmd.extend(["--tilt", str(df_cfg.get('tilt', 0.0))])
            if df_cfg.get('extend_lf'):
                cmd.append("--lf_ext_dfhrtf")
                cmd.extend(["--lf_ext_dfhrtf_xover", str(df_cfg.get('xover', 375.0))])

            # Advanced params forwarded to the subprocess
            cmd.extend(["--n-fft",                str(int(dfhrtf_adv.get('n_fft', 16384)))])
            cmd.extend(["--output-octave-fraction", str(int(dfhrtf_adv.get('output_octave_fraction', 48)))])
            cmd.extend(["--norm-freq-hz",          str(float(dfhrtf_adv.get('norm_freq_hz', 1000.0)))])
            cmd.extend(["--lf-blend-low-oct",      str(float(dfhrtf_adv.get('lf_blend_low_oct', 0.50)))])
            cmd.extend(["--lf-blend-high-oct",     str(float(dfhrtf_adv.get('lf_blend_high_oct', 1.0)))])
            cmd.extend(["--stereo-angle",           str(float(dfhrtf_adv.get('stereo_angle_deg', 30.0)))])
            cmd.extend(["--lr-avg-gamma",           str(float(avg_params.get('softdtw_gamma_persofa', 0.01)))])

            cmd.extend(["--delimiter", delim])

            # Text Outputs
            ext = f".{df_cfg.get('text_extension', 'txt')}"
            template = df_cfg.get('text_filename', "{filename}_{samplerate}_{LR}")

            def get_text_path(lr_val, _base=base_name, _sr=sr_label, _ext=ext, _root=output_root, _tmpl=template, _folder=folder_name, _tilt=tilt_str):
                fname = _tmpl.replace("{filename}", _base)
                fname = fname.replace("{folder}", _folder)
                fname = fname.replace("{samplerate}", _sr)
                fname = fname.replace("{tilt}", _tilt)
                fname = fname.replace("{LR}", lr_val)
                # Remove {algo} and any preceding/succeeding underscores for per-sofa
                fname = fname.replace("_{algo}", "").replace("{algo}_", "").replace("{algo}", "")
                if not fname.endswith(_ext): fname += _ext
                return os.path.join(_root, fname)

            if df_cfg.get('per_sofa_lr'):
                path_l = get_text_path("L")
                path_r = get_text_path("R")
                cmd.extend(["--text-out-l", path_l, "--text-out-r", path_r])
                grp = groups.setdefault(output_root, {'l': [], 'r': []})
                grp['l'].append(path_l)
                grp['r'].append(path_r)

            if df_cfg.get('per_sofa_avg'):
                cmd.extend(["--text-out-avg", get_text_path("Avg")])
                cmd.extend(["--avg-algo", df_cfg.get('per_sofa_avg_algo', 'Arithmetic Mean')])

            # Graph Outputs
            if df_cfg.get('dfhrtf_graphs'):
                graph_template = df_cfg.get('graph_filename', "{filename}")
                graph_ext = f".{df_cfg.get('graph_extension', 'png')}"

                gname = graph_template.replace("{filename}", base_name).replace("{folder}", folder_name).replace("{tilt}", tilt_str).replace("{graphtype}", "df")
                if not gname.endswith(graph_ext): gname += graph_ext
                cmd.extend(["--graph-out", os.path.join(output_root, gname)])

            if df_cfg.get('stereo_graphs'):
                graph_template = df_cfg.get('graph_filename', "{filename}")
                graph_ext = f".{df_cfg.get('graph_extension', 'png')}"

                sname = graph_template.replace("{filename}", base_name).replace("{folder}", folder_name).replace("{tilt}", tilt_str).replace("{graphtype}", "stereo")

                # Fallback: If {graphtype} wasn't used in the template, append '_stereo' manually
                # so stereo graphs don't overwrite DF graphs.
                if "{graphtype}" not in graph_template:
                    if sname.endswith(graph_ext):
                        sname = sname[:-len(graph_ext)] + "_stereo" + graph_ext
                    else:
                        sname += "_stereo" + graph_ext
                else:
                    if not sname.endswith(graph_ext): sname += graph_ext

                cmd.extend(["--graph-stereo-out", os.path.join(output_root, sname)])

            if df_cfg.get('weighted_grid'):
                graph_template = df_cfg.get('graph_filename', "{filename}")
                graph_ext = f".{df_cfg.get('graph_extension', 'png')}"
                scope = df_cfg.get('weighted_grid_scope', 'Per Folder')

                emit_weighted = (scope == 'All Files') or (output_root not in seen_grid_dirs)
                if emit_weighted:
                    seen_grid_dirs.add(output_root)
                    wname = graph_template.replace("{filename}", base_name).replace("{folder}", folder_name).replace("{tilt}", tilt_str).replace("{graphtype}", "weighted")

                    # Fallback: if {graphtype} wasn't used in the template, append '_weighted'
                    if "{graphtype}" not in graph_template:
                        if wname.endswith(graph_ext):
                            wname = wname[:-len(graph_ext)] + "_weighted" + graph_ext
                        else:
                            wname += "_weighted" + graph_ext
                    else:
                        if not wname.endswith(graph_ext): wname += graph_ext

                    cmd.extend(["--graph-weighted-out", os.path.join(output_root, wname)])

            # Run Subprocess
            rc, _ = run_task(
                cmd,
                base_name,
                input_path,
                log_callback,
                stop_checker,
                log_path,
                error_callback=error_callback,
            )

            if rc is None:
                # stop_checker fired inside run_task
                return False

            if rc != 0:
                failures.append((base_name, rc))

        if failures:
            error_callback(
                f"DFHRTF analysis: {len(failures)}/{total_inputs} tasks failed "
                f"— see inline blocks above and log: {log_path}"
            )
            for name, rc in failures:
                error_callback(f"  - {name}  (exit {rc})")

        # 4. Post-Processing: Averaging (per output-folder group)
        # Default mode: one group per source folder -> per-folder average
        # Custom mode: all files share one output_root -> single global average
        if df_cfg.get('avg_all') and groups:
            algo = df_cfg.get('algo', 'Soft-DTW')
            exclude = df_cfg.get('exclude_outliers', False)
            ext = f".{df_cfg.get('text_extension', 'txt')}"
            short_algo = {"Arithmetic Mean": "arithmean", "Geo-Arc-Length": "geoarc", "Soft-DTW": "softdtw"}.get(algo, "softdtw")
            avg_template = df_cfg.get('text_filename', "{filename}_{samplerate}_{LR}")

            for out_dir, grp in groups.items():
                if not grp['l']:
                    continue
                if stop_checker and stop_checker(): return False

                label = "Per-Folder Average" if len(groups) > 1 else "Global Average"
                log_callback(f"\n--- Post-Processing: {label} ({os.path.basename(out_dir) or out_dir}) ---")
                os.makedirs(out_dir, exist_ok=True)

                def get_avg_path(lr_val, _dir=out_dir, _tmpl=avg_template, _ext=ext, _algo=short_algo, _tilt=tilt_str):
                    fname = _tmpl.replace("{filename}", "dfhrtfs_avg")
                    fname = fname.replace("{folder}", os.path.basename(_dir))
                    fname = fname.replace("{samplerate}", "all")
                    fname = fname.replace("{tilt}", _tilt)
                    fname = fname.replace("{LR}", lr_val)
                    fname = fname.replace("{algo}", _algo)
                    if not fname.endswith(_ext): fname += _ext
                    return os.path.join(_dir, fname)

                run_averaging(grp['l'], algo, exclude, get_avg_path("L"), delim,
                              log_callback, stop_checker, avg_params=avg_params)
                if stop_checker and stop_checker(): return False
                run_averaging(grp['r'], algo, exclude, get_avg_path("R"), delim,
                              log_callback, stop_checker, avg_params=avg_params)

        return True

    except Exception as e:
        error_callback(f"DFHRTF Processor Error: {str(e)}")
        return False
