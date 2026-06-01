import os
import sys
import subprocess
import glob
import numpy as np

# In-project imports (Deferred inside functions to speed up startup)

def run_averaging(file_list, algo_name, exclude_outliers, output_path, delimiter, log_callback, stop_checker=None):
    """Orchestrates the full ingestion -> interpolation -> outlier -> averaging chain."""
    from core.averaging import DataIngestor, CommonAxisInterpolator, OutlierDetector, CurveAverager

    if not file_list:
        return
    
    try:
        log_callback(f"Averaging {len(file_list)} files using {algo_name}...")
        
        # 1. Ingest
        dfs = []
        for f in file_list:
            if stop_checker and stop_checker(): return
            dfs.append(DataIngestor.load_csv(f))
            
        # 2. Interpolate to common axis
        if stop_checker and stop_checker(): return
        interp_data = CommonAxisInterpolator.interpolate_to_common_axis(dfs, num_points=512)
        common_freqs = interp_data['common_freqs']
        norm_freqs = interp_data['norm_freqs']
        norm_mags_2d = interp_data['norm_mags_2d']
        mag_min = interp_data['mag_min']
        mag_max = interp_data['mag_max']
        
        # 3. Outlier Detection
        final_mags_2d = norm_mags_2d
        if exclude_outliers:
            if stop_checker and stop_checker(): return
            valid_indices, excluded = OutlierDetector.filter_outliers(norm_mags_2d, algo_name, threshold_std=1.0)
            if excluded:
                log_callback(f" -> Excluded {len(excluded)} outliers.")
            final_mags_2d = norm_mags_2d[valid_indices]
            
        # 4. Average
        if stop_checker and stop_checker(): return
        if algo_name == "Arithmetic Mean":
            avg_norm = CurveAverager.arithmetic_mean(final_mags_2d)
        elif algo_name == "Geo-Arc-Length":
            # Pre-scale data to simulate a y-weight of 0.05 in the distance calculation
            scaled_mags = final_mags_2d * 0.05
            scaled_avg = CurveAverager.geometric_arc_length_blend(norm_freqs, scaled_mags)
            avg_norm = scaled_avg / 0.05
        elif algo_name == "Soft-DTW":
            log_callback(" -> Running Soft-DTW Barycenter (this may take a moment)...")
            # Note: tslearn SoftDTW doesn't natively support a simple stop callback mid-math easily here,
            # but we check between ears.
            avg_norm = CurveAverager.soft_dtw_barycenter(final_mags_2d, gamma=0.05)
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

        # 1. Determine Input Files
        resolved_inputs = []
        sofa_cfg = config.get('sofa', {})
        
        if sofa_cfg.get('enabled'):
            log_callback("DFHRTF: Searching for generated mastered SOFA files...")
            for raw_path in files:
                if stop_checker and stop_checker(): return False
                base_name = os.path.splitext(os.path.basename(raw_path))[0]
                input_dir = os.path.dirname(raw_path)
                mastered_dir = os.path.join(input_dir, "sofa_mastered")
                
                if not os.path.exists(mastered_dir):
                    log_callback(f" [!] Warning: Mastered dir not found for {base_name}. Using raw file.")
                    resolved_inputs.append((raw_path, base_name, "raw"))
                    continue

                pattern = os.path.join(mastered_dir, f"*{base_name}*.sofa")
                candidates = glob.glob(pattern)
                if not candidates:
                    resolved_inputs.append((raw_path, base_name, "raw"))
                    continue

                uneq_candidates = [c for c in candidates if "uneq" in c.lower()]
                target_list = uneq_candidates if uneq_candidates else candidates
                
                def get_sr(path):
                    import re
                    match = re.search(r'(\d+)', os.path.basename(path))
                    return int(match.group(1)) if match else 0
                
                target_list.sort(key=get_sr, reverse=True)
                target_path = target_list[0]
                resolved_inputs.append((target_path, base_name, str(get_sr(target_path))))
        else:
            for f in files:
                if stop_checker and stop_checker(): return False
                resolved_inputs.append((f, os.path.splitext(os.path.basename(f))[0], "raw"))

        # 2. Setup Output Dir
        input_dir = os.path.dirname(files[0])
        output_root = os.path.join(input_dir, "DFHRTF")
        os.makedirs(output_root, exist_ok=True)

        # Track generated files for post-averaging
        generated_l_files = []
        generated_r_files = []

        # 3. Execution
        total_inputs = len(resolved_inputs)
        for i, (input_path, base_name, sr_label) in enumerate(resolved_inputs):
            if stop_checker and stop_checker(): return False

            progress_val = int((i / total_inputs) * 100)
            progress_callback(progress_val, f"Analyzing: {base_name}")
            
            cmd = [sys.executable, "-u", script_dfhrtf, "--input", input_path]
            cmd.extend(["--smoothing", str(df_cfg.get('smoothing', 48))])
            cmd.extend(["--tilt", str(df_cfg.get('tilt', 0.0))])
            if df_cfg.get('extend_lf'):
                cmd.append("--lf_ext_dfhrtf")
                cmd.extend(["--lf_ext_dfhrtf_xover", str(df_cfg.get('xover', 375.0))])
            
            delim_map = {"Tab": "\t", "Comma": ",", "Semicolon": ";", "Space": " "}
            delim = delim_map.get(df_cfg.get('text_delimiter', "Tab"), "\t")
            cmd.extend(["--delimiter", delim])
            
            # Text Outputs
            ext = f".{df_cfg.get('text_extension', 'txt')}"
            template = df_cfg.get('text_filename', "{filename}_{samplerate}_{LR}")
            
            def get_text_path(lr_val):
                fname = template.replace("{filename}", base_name)
                fname = fname.replace("{samplerate}", sr_label)
                fname = fname.replace("{LR}", lr_val)
                if not fname.endswith(ext): fname += ext
                return os.path.join(output_root, fname)

            if df_cfg.get('per_sofa_lr'):
                path_l = get_text_path("L")
                path_r = get_text_path("R")
                cmd.extend(["--text-out-l", path_l, "--text-out-r", path_r])
                generated_l_files.append(path_l)
                generated_r_files.append(path_r)
            
            if df_cfg.get('per_sofa_avg'):
                cmd.extend(["--text-out-avg", get_text_path("Avg")])

            # Graph Outputs
            if df_cfg.get('dfhrtf_graphs'):
                graph_template = df_cfg.get('graph_filename', "{filename}")
                graph_ext = f".{df_cfg.get('graph_extension', 'png')}"
                
                # Replace both variables
                gname = graph_template.replace("{filename}", base_name).replace("{graphtype}", "df")
                
                if not gname.endswith(graph_ext): gname += graph_ext
                cmd.extend(["--graph-out", os.path.join(output_root, gname)])

            if df_cfg.get('stereo_graphs'):
                graph_template = df_cfg.get('graph_filename', "{filename}")
                graph_ext = f".{df_cfg.get('graph_extension', 'png')}"
                
                # Replace both variables
                sname = graph_template.replace("{filename}", base_name).replace("{graphtype}", "stereo")
                
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

            # Run Subprocess
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, creationflags=creation_flags
            )
            for line in iter(process.stdout.readline, ''):
                if stop_checker and stop_checker():
                    process.terminate()
                    process.wait()
                    return False
                if line: log_callback(line.rstrip('\n'))
            process.stdout.close()
            rc = process.wait()
            if rc != 0: error_callback(f"Analysis failed for {base_name}")

        # 4. Post-Processing: Global Averaging
        if df_cfg.get('avg_all') and generated_l_files:
            if stop_checker and stop_checker(): return False
            log_callback("\n--- Post-Processing: Global Averaging ---")
            avg_dir = os.path.join(input_dir, "averages")
            os.makedirs(avg_dir, exist_ok=True)
            
            algo = df_cfg.get('algo', 'Arithmetic Mean')
            exclude = df_cfg.get('exclude_outliers', False)
            ext = f".{df_cfg.get('text_extension', 'txt')}"
            
            run_averaging(generated_l_files, algo, exclude, os.path.join(avg_dir, f"Global_Avg_L{ext}"), delim, log_callback, stop_checker)
            if stop_checker and stop_checker(): return False
            run_averaging(generated_r_files, algo, exclude, os.path.join(avg_dir, f"Global_Avg_R{ext}"), delim, log_callback, stop_checker)

        return True

    except Exception as e:
        error_callback(f"DFHRTF Processor Error: {str(e)}")
        return False
