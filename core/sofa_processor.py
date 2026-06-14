import os
import sys

from core.dfhrtf_processor import SOFA_MASTERED_DIR  # shared constant
from core.subprocess_utils import new_run_log, run_task

def process_sofa_exports(config, progress_callback, log_callback, error_callback, stop_checker=None):
    """
    Main processing logic for SOFA exports.
    Breaks down the UI config into individual generate_sofa.py tasks.
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

        out_cfg = config.get('output', {})
        use_default = out_cfg.get('default', True)
        custom_dir = out_cfg.get('folder', '').strip()
        if not use_default and not custom_dir:
            error_callback("Output Folder: No custom folder selected. Choose a folder or switch to Default.")
            return False

        # 2. Build Task List
        tasks = []
        for file_path in files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Per-file output root: default -> each file's own folder; custom -> shared folder
            base_dir = os.path.dirname(file_path) if use_default else custom_dir
            out_root = os.path.join(base_dir, SOFA_MASTERED_DIR)
            os.makedirs(out_root, exist_ok=True)

            # Advanced SOFA params (with historically-correct defaults)
            sofa_adv = sofa_cfg.get('advanced', {})

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

                    cmd = [
                        sys.executable, "-u", script_master,
                        "--input", file_path,
                        "--output", out_path,
                        "--target-fs", str(sr),
                        "--hrir-length", str(int(sofa_cfg.get('hrir_length', 512)))
                    ]

                    if dfeq_on:
                        cmd.append("--dfeq")

                    if sofa_cfg.get('lf_extension'):
                        cmd.append("--lf-extend")
                        cmd.extend(["--lf-crossover", str(sofa_cfg.get('xover', 375.0))])

                    if sofa_cfg.get('normalize_distance'):
                        cmd.append("--harmonize-radius")
                        cmd.extend(["--target-radius", str(sofa_cfg.get('distance_m', 1.2))])

                    # Advanced params forwarded to the subprocess
                    cmd.extend(["--processing-length",
                                 str(int(sofa_adv.get('processing_length', 2048)))])
                    cmd.extend(["--norm-target-db",
                                 str(float(sofa_adv.get('norm_target_db', -1.0)))])
                    cmd.extend(["--onset-buffer",
                                 str(int(sofa_adv.get('onset_buffer', 32)))])
                    cmd.extend(["--onset-threshold",
                                 str(float(sofa_adv.get('onset_threshold_ratio', 0.05)))])
                    cmd.extend(["--blend-low-oct",
                                 str(float(sofa_adv.get('blend_low_oct', 0.50)))])
                    cmd.extend(["--blend-high-oct",
                                 str(float(sofa_adv.get('blend_high_oct', 0.75)))])

                    # Weighting: driven by dfhrtf config keys so one setting
                    # governs weighting in both the SOFA mastering and DFHRTF
                    # analysis stages.
                    df_cfg = config.get('dfhrtf', {})
                    dfhrtf_adv = df_cfg.get('advanced', {})
                    cmd.extend(["--weighting-mode",
                                 str(df_cfg.get('weighting_mode', 'Auto')).lower()])


                    tasks.append({
                        'cmd': cmd,
                        'desc': f"{base_name} ({sr}Hz, {eq_label})",
                        'input_path': file_path,
                    })

        # 3. Execute Tasks
        total_tasks = len(tasks)
        if total_tasks == 0:
            return True

        log_root = custom_dir if not use_default else os.path.dirname(files[0])
        log_path = new_run_log("sofa", log_root)
        log_callback(f"[Diagnostics log: {log_path}]")

        failures = []  # list of (desc, returncode)

        for i, task in enumerate(tasks):
            if stop_checker and stop_checker():
                return False

            progress_val = int((i / total_tasks) * 100)
            progress_callback(progress_val, f"Exporting: {task['desc']}")

            rc, _ = run_task(
                task['cmd'],
                task['desc'],
                task['input_path'],
                log_callback,
                stop_checker,
                log_path,
                error_callback=error_callback,
            )

            if rc is None:
                # stop_checker fired inside run_task
                return False

            if rc != 0:
                failures.append((task['desc'], rc))

        if failures:
            error_callback(
                f"SOFA export: {len(failures)}/{total_tasks} tasks failed "
                f"— see inline blocks above and log: {log_path}"
            )
            for desc, rc in failures:
                error_callback(f"  - {desc}  (exit {rc})")

        return True

    except Exception as e:
        error_callback(f"Processor Error: {str(e)}")
        return False
