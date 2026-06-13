import argparse
import os
import sys
import math
import numpy as np
import sofar as sf
import scipy.signal as signal
import scipy.fft as fft
from scipy.spatial import ConvexHull

# ================= R2 CONFIGURATION =================
PROCESSING_LENGTH = 2048
OUTPUT_LENGTH = 512
NORM_TARGET_DB = -1.0
SPEED_OF_SOUND = 343.0 # m/s
DEFAULT_ONSET_BUFFER = 32 # Samples of silence before the aligned impulse starts
# =================================================

def ensure_mesh2hrtf_import(m2h_path):
    try:
        import mesh2hrtf as m2h
        print("[+] Mesh2HRTF library imported successfully.")
        return m2h
    except ImportError:
        print(f"[i] Library not found. Adding path: {m2h_path}")
        sys.path.append(m2h_path)
        sys.path.append(os.path.dirname(m2h_path))
        
        try:
            import mesh2hrtf as m2h
            print("[+] Mesh2HRTF library imported successfully after path update.")
            return m2h
        except ImportError as e:
            print(f"[FATAL] Could not import mesh2hrtf. Check your path.\nError: {e}")
            sys.exit(1)

def run_project_export(m2h, project_path):
    print(f"   -> Processing: {os.path.basename(project_path)}...")
    try:
        m2h.output2hrtf(project_path)
    except Exception as e:
        print(f"[ERROR] Export failed for {project_path}: {e}")
        sys.exit(1)

def find_sofas_in_project(project_path):
    out_dir = os.path.join(project_path, "Output2HRTF")
    if not os.path.exists(out_dir):
        out_dir = project_path
        
    candidates = [f for f in os.listdir(out_dir) if f.endswith(".sofa")]
    if not candidates:
        print(f"[ERROR] No SOFA files found in {out_dir}")
        sys.exit(1)
    
    return [os.path.join(out_dir, c) for c in candidates if "HRIR" in c]

def get_dominant_radius(source_pos):
    r = source_pos[:, 2]
    radii_rounded = np.round(r, decimals=3)
    values, counts = np.unique(radii_rounded, return_counts=True)
    return values[np.argmax(counts)]

def spherical_to_cartesian(r, az, el):
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    x = r * np.cos(el_rad) * np.cos(az_rad)
    y = r * np.cos(el_rad) * np.sin(az_rad)
    z = r * np.sin(el_rad)
    return np.column_stack((x, y, z))

def calculate_geometric_weights(source_pos):
    print("      Calculating geometric weights...", end="", flush=True)
    az = source_pos[:, 0]
    el = source_pos[:, 1]
    
    dominant_r = get_dominant_radius(source_pos)
    valid_idx = np.abs(source_pos[:, 2] - dominant_r) < 0.05
    
    az_el_rounded = np.round(source_pos[valid_idx][:, 0:2], decimals=3)
    _, unique_idx = np.unique(az_el_rounded, axis=0, return_index=True)
    
    final_idx = np.where(valid_idx)[0][unique_idx]
    az_clean = az[final_idx]
    el_clean = el[final_idx]
    cart_coords = spherical_to_cartesian(1.0, az_clean, el_clean)
        
    try:
        hull = ConvexHull(cart_coords)
        weights = np.zeros(len(source_pos))
        temp_weights = np.zeros(len(cart_coords))
        for simplex in hull.simplices:
            A, B, C = cart_coords[simplex]
            cross_prod = np.cross(B - A, C - A)
            area = 0.5 * np.linalg.norm(cross_prod)
            temp_weights[simplex] += area / 3.0
        
        weights[final_idx] = temp_weights
        weights /= np.sum(weights)
        print(" Done.")
        return weights
    except:
        print(" Failed (Fallback to uniform).")
        return np.ones(len(source_pos)) / len(source_pos)

def find_global_onset(hrirs, threshold_ratio):
    """Finds the absolute first sample where energy crosses the threshold."""
    global_max = np.max(np.abs(hrirs))
    threshold = global_max * threshold_ratio
    max_over_space = np.max(np.abs(hrirs), axis=(0, 1))
    onset_idx = np.argmax(max_over_space > threshold)
    return onset_idx

def load_lf_target_base(filename, target_fs, n_fft):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(script_dir, "..", "..", "data", filename)
    
    if not os.path.exists(ref_path):
        # Fallback to local if not found in root data
        ref_path = os.path.join(script_dir, filename)
        if not os.path.exists(ref_path):
            return None

    try:
        data = np.loadtxt(ref_path)
        ref_freqs = data[:, 0]
        ref_db = data[:, 1]
        
        fft_freqs = fft.rfftfreq(n_fft, 1.0 / target_fs)
        target_db = np.interp(fft_freqs, ref_freqs, ref_db)
        return 10 ** (target_db / 20.0)
    except Exception as e:
        print(f"      [!] Error loading LF target: {e}")
        return None

def get_blend_weights(freqs, crossover_f, blend_low_oct=0.50, blend_high_oct=0.75):
    """Cosine crossfade: 0 below f_low, 1 above f_high, smooth blend in between."""
    f_low = crossover_f / (2**blend_low_oct)
    f_high = crossover_f * (2**blend_high_oct)
    weights = np.ones_like(freqs) 
    mask_target = freqs <= f_low
    mask_blend = (freqs > f_low) & (freqs < f_high)
    weights[mask_target] = 0.0
    
    if np.any(mask_blend):
        log_f = np.log2(freqs[mask_blend])
        log_low = np.log2(f_low)
        log_high = np.log2(f_high)
        # Cosine fade from 0 to 1
        phase = np.pi * (log_f - log_low) / (log_high - log_low)
        weights[mask_blend] = (1.0 - np.cos(phase)) / 2.0
    return weights

def master_sofa(raw_sofa, target_fs, apply_dfeq, output_path, args):
    print(f"\n--- Generating: {os.path.basename(output_path)} ---")

    # Resolve advanced params from args (with module-constant fallbacks for legacy calls)
    processing_length    = getattr(args, 'processing_length',    PROCESSING_LENGTH)
    norm_target_db       = getattr(args, 'norm_target_db',       NORM_TARGET_DB)
    onset_buffer         = getattr(args, 'onset_buffer',         DEFAULT_ONSET_BUFFER)
    onset_threshold      = getattr(args, 'onset_threshold',      0.05)
    blend_low_oct        = getattr(args, 'blend_low_oct',        0.50)
    blend_high_oct       = getattr(args, 'blend_high_oct',       0.75)

    out_sofa = raw_sofa.copy()
    hrirs = out_sofa.Data_IR.copy()
    src_fs = float(out_sofa.Data_SamplingRate)
    M, R, N = hrirs.shape

    # 1. Resample
    if abs(src_fs - target_fs) > 1.0:
        print(f"   1. Resampling: {src_fs:.0f} -> {target_fs} Hz")
        gcd = math.gcd(int(src_fs), int(target_fs))
        up = int(target_fs) // gcd
        down = int(src_fs) // gcd
        hrirs = signal.resample_poly(hrirs, up, down, axis=-1)
        N = hrirs.shape[-1]
    else:
        print(f"   1. Rate matches ({src_fs:.0f} Hz).")

    # 2. Pad to Processing Length
    print("   2. Preparing HRIRs...")
    if N < processing_length:
        pad_amt = processing_length - N
        hrirs = np.pad(hrirs, ((0,0), (0,0), (0, pad_amt)), mode='constant')
        N = processing_length
    elif N > processing_length:
        hrirs = hrirs[:, :, :processing_length]
        N = processing_length

    # 3. Global Impulse Alignment & Radius Harmonization
    global_onset = find_global_onset(hrirs, threshold_ratio=onset_threshold)
    base_shift = onset_buffer - global_onset
    
    if args.harmonize_radius:
        orig_r = get_dominant_radius(out_sofa.SourcePosition)
        target_r = args.target_radius
        physics_delay = int((target_r / SPEED_OF_SOUND) * target_fs)
        total_shift = base_shift + physics_delay
        gain_multiplier = orig_r / target_r
        
        print(f"   3. Harmonizing Radius: {orig_r:.2f}m -> {target_r:.2f}m (Delay: {physics_delay} samples, Gain: x{gain_multiplier:.3f})")
        hrirs *= gain_multiplier
    else:
        total_shift = base_shift
        print(f"   3. Global Impulse Alignment: Stripping {global_onset - onset_buffer} samples of padding.")

    # Execute the Time Shift (Preserving ITD)
    hrirs = np.roll(hrirs, total_shift, axis=-1)
    # Zero out the wrapped tail to prevent artifacts
    if total_shift < 0:
        hrirs[..., total_shift:] = 0.0
    elif total_shift > 0:
        hrirs[..., :total_shift] = 0.0

    # 4. Compute Raw Frequency Domain
    H_raw = fft.rfft(hrirs, n=N, axis=-1)
    freqs = fft.rfftfreq(N, 1.0 / target_fs)

    # 5. Calculate Global Diffuse Field Response (Raw)
    weights = calculate_geometric_weights(out_sofa.SourcePosition)
    weights_expanded = weights[:, np.newaxis, np.newaxis]
    
    P_raw = np.abs(H_raw)**2
    DF_power = np.sum(P_raw * weights_expanded, axis=0)
    DF_mag = np.sqrt(DF_power) # (R, N_bins)
    
    crossover_f = args.lf_crossover
    idx_xover = np.searchsorted(freqs, crossover_f)
    blend_weights = get_blend_weights(freqs, crossover_f,
                                      blend_low_oct=blend_low_oct,
                                      blend_high_oct=blend_high_oct)
    
    # Define Anchor Level (Level of Raw DF at crossover)
    anchor_level = np.mean(DF_mag[:, max(0, idx_xover-2):idx_xover+3], axis=-1, keepdims=True)
    
    # 6. Step A: Define the Unified Target Response
    # Default target: flat at anchor_level
    target_response = np.ones_like(DF_mag) * anchor_level
    
    if args.lf_extend:
        base_target = load_lf_target_base("lf_ref_mag.txt", target_fs, N)
        if base_target is not None:
            # Scale target to match anchor_level around crossover
            target_at_xover = np.mean(base_target[max(0, idx_xover-2):idx_xover+3])
            scaled_lf_target = base_target * (anchor_level / target_at_xover)
            # Unified target blends from the LF slope to the flat anchor level
            target_response = scaled_lf_target * (1.0 - blend_weights) + anchor_level * blend_weights
        else:
            print(f"   [!] Skipping LF Extension: Reference file 'lf_ref_mag.txt' not found.")

    # 7. Step B: Prepare Repaired/Spliced Signal
    H_rep = H_raw.copy()
    if args.lf_extend and base_target is not None:
        print(f"   4. Splicing Ideal LF Response (Crossover: {crossover_f} Hz)...")
        
        # Prepare Synthetic LF (Min-phase version of the scaled target)
        # Note: We use the scaled_lf_target which is already ear-specific
        H_synthetic = np.zeros_like(H_raw)
        for m in range(M):
            for r in range(R):
                log_base = np.log(scaled_lf_target[r] + 1e-12)
                full_log_base = np.concatenate([log_base, log_base[-2:0:-1]])
                min_phase_base = -np.imag(signal.hilbert(full_log_base))[:len(base_target)]
                
                # Align delay to raw phase at crossover
                phase_raw = np.unwrap(np.angle(H_raw[m, r, :]))
                phase_at_xover = np.mean(phase_raw[max(0, idx_xover-2):idx_xover+3])
                delay = (min_phase_base[idx_xover] - phase_at_xover) / (2 * np.pi * crossover_f)
                
                H_synthetic[m, r, :] = scaled_lf_target[r] * np.exp(1j * (min_phase_base - 2 * np.pi * freqs * delay))
        
        # Complex Splice
        H_rep = H_synthetic * (1.0 - blend_weights) + H_raw * blend_weights

    # 8. Step C: Apply Unified Corrective EQ (if DFEQ is ON)
    H_final = H_rep
    if apply_dfeq:
        print("   5. Applying Unified Diffuse Field EQ...")
        P_rep = np.abs(H_rep)**2
        DF_power_rep = np.sum(P_rep * weights_expanded, axis=0)
        DF_mag_rep = np.sqrt(DF_power_rep)
        
        # Corrective filter maps the repaired signal to the flat target
        flat_target = np.ones_like(DF_mag) * anchor_level
        eq_filter_mag = flat_target / (DF_mag_rep + 1e-12)
        
        log_eq = np.log(eq_filter_mag + 1e-12)
        full_log_eq = np.concatenate([log_eq, log_eq[..., -2:0:-1]], axis=-1)
        min_phase_eq = -np.imag(signal.hilbert(full_log_eq, axis=-1))
        min_phase_eq = min_phase_eq[..., :eq_filter_mag.shape[-1]]
        
        complex_eq = eq_filter_mag * np.exp(1j * min_phase_eq)
        H_final = H_rep * complex_eq[np.newaxis, :, :]
    else:
        print("   5. Skipping DFEQ.")

    # 9. Back to Time Domain
    hrirs = fft.irfft(H_final, n=N, axis=-1)

    # 9. Dynamic Crop & Fade
    if hasattr(args, 'hrir_length') and args.hrir_length is not None:
        final_output_length = args.hrir_length
    else:
        final_output_length = PROCESSING_LENGTH if (args.lf_extend and OUTPUT_LENGTH < 512) else OUTPUT_LENGTH
    
    if final_output_length < N:
        print(f"   7. Safely cropping causal tail to {final_output_length} samples and windowing...")
        hrirs = hrirs[:, :, :final_output_length]
        
        fade_in_len = 8
        if fade_in_len > 0:
            fade_in_curve = np.hanning(2 * fade_in_len)[:fade_in_len]
            hrirs[:, :, :fade_in_len] *= fade_in_curve
            
        fade_out_len = 32
        if fade_out_len > 0:
            fade_out_curve = np.hanning(2 * fade_out_len)[fade_out_len:] 
            hrirs[:, :, -fade_out_len:] *= fade_out_curve

    # 10. Normalize
    if not args.harmonize_radius:
        print(f"   8. Normalizing to {norm_target_db} dB...")
        peak = np.max(np.abs(hrirs))
        target_linear = 10 ** (norm_target_db / 20.0)
        if peak > 0:
            scale = target_linear / peak
            hrirs *= scale
            print(f"      Scaled by {scale:.4f}.")
    else:
        print("   8. Skipping Normalization (Preserving Harmonized Radius Volume).")

    # 11. Save
    out_sofa.Data_IR = hrirs.astype(np.float32)
    out_sofa.Data_SamplingRate = float(target_fs)
    # SOFA convention requires all *_Units values to be lowercase.
    # Some source files carry "Hertz" / "Metres" which sofar rejects on write.
    # Must iterate instance attributes (vars(obj)), not the class namespace —
    # sofar stores Data_SamplingRate_Units etc. on the instance, not the class.
    for _attr in list(vars(out_sofa)):
        if _attr.endswith("_Units"):
            try:
                _val = getattr(out_sofa, _attr)
                if isinstance(_val, str) and _val != _val.lower():
                    setattr(out_sofa, _attr, _val.lower())
            except Exception:
                pass
    sf.write_sofa(output_path, out_sofa)
    print(f"   [+] Saved {os.path.basename(output_path)}")

def merge_sofas(path_l, path_r):
    print("\n=== Merging Left and Right Projects ===")
    l_sofa = sf.read_sofa(path_l)
    r_sofa = sf.read_sofa(path_r)
    
    if l_sofa.Data_SamplingRate != r_sofa.Data_SamplingRate:
        print("[FATAL] Sampling rates do not match!")
        sys.exit(1)
        
    if l_sofa.Data_IR.shape[1] == 1 and r_sofa.Data_IR.shape[1] == 1:
        merged_ir = np.concatenate((l_sofa.Data_IR, r_sofa.Data_IR), axis=1)
        merged_recv_pos = np.vstack((l_sofa.ReceiverPosition, r_sofa.ReceiverPosition))
        
        d_l = np.atleast_2d(l_sofa.Data_Delay)
        d_r = np.atleast_2d(r_sofa.Data_Delay)
        if d_l.shape[0] != 1: d_l = d_l.T
        if d_r.shape[0] != 1: d_r = d_r.T
            
        merged_delay = np.concatenate((d_l, d_r), axis=1)
        
        merged = l_sofa.copy()
        merged.Data_IR = merged_ir
        merged.ReceiverPosition = merged_recv_pos
        merged.Data_Delay = merged_delay
        merged.GLOBAL_Title = "Merged HRTF (Mastered)"
        return merged
    else:
        print("[FATAL] Input SOFAs already have multiple receivers.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=False)
    parser.add_argument("--right", required=False)
    parser.add_argument("--m2h_path", required=False)
    parser.add_argument("--input", required=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only-48k", action="store_true")
    parser.add_argument("--dfeq", action="store_true", default=None)
    
    parser.add_argument("--target-fs", type=int, help="Explicit target samplerate (Hz) for standalone mode")
    parser.add_argument("--hrir-length", type=int, help="Override output HRIR length (Samples)")
    
    parser.add_argument("--lf-extend", action="store_true")
    parser.add_argument("--lf-crossover", type=float, default=375.0)

    parser.add_argument("--harmonize-radius", action="store_true")
    parser.add_argument("--target-radius", type=float, default=1.2)

    # Advanced tunable parameters (forwarded from sofa.advanced config section)
    parser.add_argument("--processing-length",  type=int,   default=PROCESSING_LENGTH,
                        help="FFT/processing buffer size in samples")
    parser.add_argument("--norm-target-db",      type=float, default=NORM_TARGET_DB,
                        help="Peak normalization target in dB")
    parser.add_argument("--onset-buffer",        type=int,   default=DEFAULT_ONSET_BUFFER,
                        help="Samples of silence before the aligned impulse")
    parser.add_argument("--onset-threshold",     type=float, default=0.05,
                        help="Fraction of global peak used for onset detection")
    parser.add_argument("--blend-low-oct",       type=float, default=0.50,
                        help="Octaves below crossover for crossfade start")
    parser.add_argument("--blend-high-oct",      type=float, default=0.75,
                        help="Octaves above crossover for crossfade end")

    args = parser.parse_args()

    targets = []
    if args.input:
        print("=== Standalone Mode: Mastering Single SOFA ===")
        if not os.path.exists(args.input):
            print(f"[ERROR] Input file missing: {args.input}")
            sys.exit(1)
            
        target_sofa = sf.read_sofa(args.input)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        targets.append((target_sofa, base_name))
        
    elif args.left and args.right and args.m2h_path:
        print("=== Orchestrator Mode: Generating and Merging ===")
        m2h = ensure_mesh2hrtf_import(args.m2h_path)
        run_project_export(m2h, args.left)
        run_project_export(m2h, args.right)

        sofas_l = find_sofas_in_project(args.left)
        sofas_r = find_sofas_in_project(args.right)
        
        pairs = []
        for sl in sofas_l:
            for sr in sofas_r:
                if os.path.basename(sl) == os.path.basename(sr):
                    pairs.append((sl, sr, os.path.basename(sl)))

        if not pairs:
            print("[ERROR] No matching left/right SOFA pairs found.")
            sys.exit(1)

        for sl, sr, name in pairs:
            print(f"   Merging Pair: {name}")
            target_sofa = merge_sofas(sl, sr)
            base_name = os.path.splitext(name)[0]
            targets.append((target_sofa, base_name))
        
    else:
        print("[ERROR] Invalid arguments. Provide either --input OR (--left, --right, --m2h_path).")
        sys.exit(1)

    print("\n=== Step 2: Mastering Outputs ===")
    
    # Define Job List based on arguments
    if args.target_fs:
        # Explicit/UI mode: Just run the one requested job
        apply_dfeq = args.dfeq if args.dfeq is not None else False
        jobs = [(args.target_fs, apply_dfeq, None)]
    elif args.only_48k:
        # Legacy/Shortcut mode: Only 48k, No DFEQ (unless explicitly requested)
        apply_dfeq = args.dfeq if args.dfeq is not None else False
        jobs = [(48000, apply_dfeq, None)]
    else:
        # Standard mode
        if args.dfeq is True:
            # Only DFEQ versions
            jobs = [(44100, True, "44100Hz_DFEQ.sofa"), (48000, True, "48000Hz_DFEQ.sofa")]
        elif args.dfeq is False:
            # Only Non-DFEQ versions
            jobs = [(44100, False, "44100Hz.sofa"), (48000, False, "48000Hz.sofa")]
        else:
            # Default: All 4
            jobs = [
                (44100, False, "44100Hz.sofa"),
                (44100, True,  "44100Hz_DFEQ.sofa"),
                (48000, False, "48000Hz.sofa"),
                (48000, True,  "48000Hz_DFEQ.sofa")
            ]

    for target_sofa, base_name in targets:
        for fs, dfeq, suffix in jobs:
            if suffix is None:
                if args.target_fs:
                    # In explicit UI mode, output path is already the final file path
                    out_path = args.output
                else:
                    out_path = os.path.join(args.output, f"{base_name}.sofa")
            else:
                out_path = os.path.join(args.output, f"{base_name}_{suffix}")
            master_sofa(target_sofa, fs, dfeq, out_path, args)

    print("\n[SUCCESS] All files generated.")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # Ensure a full traceback is always captured by the parent process
        input_hint = ""
        try:
            import sys as _sys
            for _i, _a in enumerate(_sys.argv):
                if _a == "--input" and _i + 1 < len(_sys.argv):
                    input_hint = f"\n  Input: {_sys.argv[_i + 1]}"
                    break
        except Exception:
            pass
        print(f"\n[UNCAUGHT EXCEPTION]{input_hint}", flush=True)
        traceback.print_exc()
        sys.exit(1)