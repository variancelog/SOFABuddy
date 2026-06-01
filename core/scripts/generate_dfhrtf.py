import argparse
import os
import sys
import numpy as np
import sofar as sf
import scipy.fft as fft
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import csv
import re

""" Note: To learn more about how this script computes Diffuse Field HRTF 
    (DFHRTF) from the SOFA files, please read `readme_dfhrtf_calculation.md 
    in the repo """


# ================= HELPER FUNCTIONS =================

def spherical_to_cartesian(r, az, el):
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    x = r * np.cos(el_rad) * np.cos(az_rad)
    y = r * np.cos(el_rad) * np.sin(az_rad)
    z = r * np.sin(el_rad)
    return np.column_stack((x, y, z))

def calculate_geometric_weights(source_pos, advanced_dfeq=True):
    """Calculates solid angle weights for spherical integration."""
    az = source_pos[:, 0]
    el = source_pos[:, 1]
    
    if advanced_dfeq:
        r = source_pos[:, 2]
        # Find dominant radius
        radii_rounded = np.round(r, decimals=3)
        values, counts = np.unique(radii_rounded, return_counts=True)
        dominant_r = values[np.argmax(counts)]
        
        # Filter coordinates deviating from dominant radius
        valid_idx = np.abs(r - dominant_r) < 0.05
        
        # Identify and remove duplicate angular coordinates
        az_el_rounded = np.round(source_pos[valid_idx][:, 0:2], decimals=3)
        _, unique_idx = np.unique(az_el_rounded, axis=0, return_index=True)
        
        final_idx = np.where(valid_idx)[0][unique_idx]
        
        az_clean = az[final_idx]
        el_clean = el[final_idx]
        cart_coords = spherical_to_cartesian(1.0, az_clean, el_clean)
    else:
        cart_coords = spherical_to_cartesian(1.0, az, el)
        final_idx = np.arange(len(source_pos))

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
    except:
        weights = np.ones(len(source_pos)) / len(source_pos)

    return weights

def find_closest_grid_point(source_pos, target_az, target_el):
    """Finds index of closest grid point using Cartesian Euclidean distance."""
    # target_cart will be shape (1, 3)
    target_cart = spherical_to_cartesian(1.0, np.array([target_az]), np.array([target_el]))
    
    az = source_pos[:, 0]
    el = source_pos[:, 1]
    source_cart = spherical_to_cartesian(1.0, az, el)
    
    # Calculate distances from each source to the single target
    distances = np.linalg.norm(source_cart - target_cart, axis=1)
    return np.argmin(distances)

def generate_fractional_octave_frequencies(start_freq, end_freq, fraction=6):
    """Generates log-spaced frequencies."""
    freqs = []
    f = start_freq
    while f <= end_freq:
        freqs.append(f)
        f *= 2**(1/fraction)
    return np.array(freqs)

def apply_spectral_tilt(freqs, magnitude_db, slope_per_octave, ref_freq=1000.0):
    """Applies a dB/octave tilt pivoting at ref_freq."""
    if slope_per_octave == 0:
        return magnitude_db
    num_octaves = np.log2(freqs / ref_freq)
    tilt_curve = num_octaves * slope_per_octave
    return magnitude_db + tilt_curve

def apply_smoothing(freqs, mags, fraction=24):
    """Applies 1/fraction octave moving-average smoothing."""
    smoothed = np.zeros_like(mags)
    for i, f in enumerate(freqs):
        f_lower = f / (2**(1/(2*fraction)))
        f_upper = f * (2**(1/(2*fraction)))
        mask = (freqs >= f_lower) & (freqs <= f_upper)
        if np.any(mask):
            smoothed[i] = np.mean(mags[mask])
        else:
            smoothed[i] = mags[i]
    return smoothed

def save_text_data(filename, freqs, mag, delimiter="\t"):
    """Saves magnitude data with custom delimiter. Includes header for CSVs."""
    try:
        is_csv = filename.lower().endswith(".csv")
        with open(filename, 'w', newline='') as ffile:
            if is_csv:
                ffile.write(f"Frequency (Hz){delimiter}Magnitude (dB)\n")
            
            for f, m in zip(freqs, mag):
                ffile.write(f"{f:.6f}{delimiter}{m:.3f}\n")
        print(f"   [+] Saved: {os.path.basename(filename)}")
    except IOError as e:
        print(f"[ERROR] Saving file {filename}: {e}")

def load_lf_reference(file_path):
    """Loads LF reference magnitude from a tab-delimited TXT file."""
    try:
        data = np.loadtxt(file_path)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"[ERROR] Loading LF reference '{file_path}': {e}")
        return None, None

def apply_lf_extension(target_freqs, val_db, ref_freqs, ref_db, xover_freq):
    """Applies LF extension with a Linkwitz-Riley type linear crossfade."""
    # Interpolate reference to match target frequencies
    interp_ref_db = np.interp(target_freqs, ref_freqs, ref_db)
    
    # Level match at crossover
    xover_idx = (np.abs(target_freqs - xover_freq)).argmin()
    offset = val_db[xover_idx] - interp_ref_db[xover_idx]
    matched_ref_db = interp_ref_db + offset
    
    # Convert to linear amplitude
    val_lin = 10**(val_db / 20.0)
    ref_lin = 10**(matched_ref_db / 20.0)
    
    # Crossfade weights (1.5 octave range)
    f_low = xover_freq / (2**0.5)
    f_high = xover_freq * (2**1.0)
    
    weights_ref = np.zeros_like(target_freqs)
    mask_low = target_freqs <= f_low
    mask_high = target_freqs >= f_high
    mask_mid = (target_freqs > f_low) & (target_freqs < f_high)
    
    weights_ref[mask_low] = 1.0
    weights_ref[mask_high] = 0.0
    
    # Logarithmic transition
    if np.any(mask_mid):
        log_f = np.log2(target_freqs[mask_mid])
        log_low = np.log2(f_low)
        log_high = np.log2(f_high)
        phase = np.pi * (log_f - log_low) / (log_high - log_low)
        weights_ref[mask_mid] = (1.0 + np.cos(phase)) / 2.0
        
    weights_val = 1.0 - weights_ref
    
    # Blend and convert back to dB
    combined_lin = (ref_lin * weights_ref) + (val_lin * weights_val)
    return 20 * np.log10(combined_lin + 1e-12)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input SOFA file")
    
    # Explicit Outputs (UI-driven)
    parser.add_argument("--text-out-l", help="Explicit path for Left ear text output")
    parser.add_argument("--text-out-r", help="Explicit path for Right ear text output")
    parser.add_argument("--text-out-avg", help="Explicit path for Average text output")
    parser.add_argument("--graph-out", help="Explicit path for Graph plot output")
    parser.add_argument("--graph-stereo-out", help="Explicit path for Stereo graph plot output")
    parser.add_argument("--delimiter", default="\t", help="Delimiter for text outputs")
    
    # Refinement Parameters
    parser.add_argument("--smoothing", type=int, default=48, help="Smoothing fraction (e.g. 24 for 1/24th octave)")
    parser.add_argument("--tilt", type=float, default=0.0, help="Spectral Tilt (dB/oct)")
    parser.add_argument("--lf_ext_dfhrtf", action="store_true", help="Enable Low Frequency extension using reference file")
    parser.add_argument("--lf_ext_dfhrtf_xover", type=float, default=375.0, help="Crossover frequency for LF extension")
    
    # Legacy/Misc
    parser.add_argument("--output_dir", help="Output folder (used if explicit paths not provided)")
    parser.add_argument("--prefix", type=str, default="", help="Optional file prefix")
    parser.add_argument("--squigify", action="store_true", help="Enable Squiglink optimized DFHRTF outputs")
    parser.add_argument("--sim_meas", action="store_true", help="Tag output as Simulated instead of Measured")
    parser.add_argument("--advanced-dfeq", action="store_true", default=True, help="Use advanced data sanitization")
    parser.add_argument("--only_plot", action="store_true", help="Generate only the plot, no text/CSV files")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file missing: {args.input}")
        sys.exit(1)

    # 1. Load Data
    sofa = sf.read_sofa(args.input)
    hrirs = sofa.Data_IR
    fs = float(sofa.Data_SamplingRate)
    
    # 2. Compute Diffuse Field Response
    weights = calculate_geometric_weights(sofa.SourcePosition, advanced_dfeq=args.advanced_dfeq)
    n_fft = 16384
    
    # FFT (Measurements, Receivers, Freqs)
    fft_data = np.fft.rfft(hrirs, n=n_fft, axis=2)
    power_spec = np.abs(fft_data)**2
    
    # Weighted Average across measurements (Axis 0)
    weights_expanded = weights[:, np.newaxis, np.newaxis]
    avg_power = np.sum(power_spec * weights_expanded, axis=0)
    
    avg_mag = np.sqrt(avg_power)
    avg_db = 20 * np.log10(avg_mag + 1e-12)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1/fs)

    # 3. Interpolate & Smooth
    target_freqs = generate_fractional_octave_frequencies(20, 20000, fraction=48)
    
    print(f"   -> Applying 1/{args.smoothing} Octave Smoothing...")
    smoothed_l = apply_smoothing(fft_freqs, avg_db[0], fraction=args.smoothing)
    smoothed_r = apply_smoothing(fft_freqs, avg_db[1], fraction=args.smoothing)
    
    val_l = np.interp(target_freqs, fft_freqs, smoothed_l)
    val_r = np.interp(target_freqs, fft_freqs, smoothed_r)

    # 4. Normalize (0 dB at 1 kHz)
    idx_1k = (np.abs(target_freqs - 1000.0)).argmin()
    val_l -= val_l[idx_1k]
    val_r -= val_r[idx_1k]

    # 4.5 LF Extension
    if args.lf_ext_dfhrtf:
        print(f"   -> Applying LF Extension (xover: {args.lf_ext_dfhrtf_xover}Hz)...")
        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "lf_ref_mag.txt")
        ref_freqs, ref_db = load_lf_reference(ref_path)
        if ref_freqs is not None:
            val_l = apply_lf_extension(target_freqs, val_l, ref_freqs, ref_db, args.lf_ext_dfhrtf_xover)
            val_r = apply_lf_extension(target_freqs, val_r, ref_freqs, ref_db, args.lf_ext_dfhrtf_xover)

    # 5. Apply Tilt
    if args.tilt != 0:
        print(f"   -> Applying {args.tilt} dB/oct tilt...")
        val_l = apply_spectral_tilt(target_freqs, val_l, args.tilt, 1000.0)
        val_r = apply_spectral_tilt(target_freqs, val_r, args.tilt, 1000.0)

    # 6. Calculate Average
    val_avg = (val_l + val_r) / 2.0

    # 7. Generate Plot
    if args.graph_out or args.only_plot or args.output_dir:
        plt.figure(figsize=(10, 6))
        plt.semilogx(target_freqs, val_l, label='Left Ear', linewidth=2, alpha=0.8)
        plt.semilogx(target_freqs, val_r, label='Right Ear', linewidth=2, alpha=0.8, linestyle='--')
        
        title_str = f"Diffuse Field HRTF (Normalized)\nSpectral Tilt: {args.tilt} dB/oct"
        plt.title(title_str)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.ylim(-25, 25)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.xlim(20, 20000)
        
        if args.graph_out:
            plt.savefig(args.graph_out, dpi=150)
            print(f"   [+] Saved Plot: {os.path.basename(args.graph_out)}")
        elif args.output_dir:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            plot_name = f"{args.prefix}{base_name} Tilt {args.tilt} LR.png"
            plt.savefig(os.path.join(args.output_dir, plot_name), dpi=150)
            print(f"   [+] Saved Plot: {plot_name}")
        
        plt.close()

    # 7.5 Stereo Graph (+/- 30 degrees)
    if args.graph_stereo_out:
        print("   -> Calculating +/-30 Stereo Response...")
        # Find closest grid points: Left (+30), Right (330 or -30)
        idx_30L = find_closest_grid_point(sofa.SourcePosition, 30.0, 0.0)
        idx_30R = find_closest_grid_point(sofa.SourcePosition, 330.0, 0.0)
        
        # Log what we found
        az_L, el_L = sofa.SourcePosition[idx_30L, 0], sofa.SourcePosition[idx_30L, 1]
        az_R, el_R = sofa.SourcePosition[idx_30R, 0], sofa.SourcePosition[idx_30R, 1]
        print(f"      Closest to L(+30,0): az={az_L:.1f}, el={el_L:.1f}")
        print(f"      Closest to R(-30,0): az={az_R:.1f}, el={el_R:.1f}")
        
        # Process L Ear
        mag_30L = 20 * np.log10(np.abs(np.fft.rfft(hrirs[idx_30L, 0, :], n=n_fft)) + 1e-12)
        smooth_30L = apply_smoothing(fft_freqs, mag_30L, fraction=args.smoothing)
        val_30L = np.interp(target_freqs, fft_freqs, smooth_30L)
        val_30L -= val_30L[idx_1k]
        
        # Process R Ear
        mag_30R = 20 * np.log10(np.abs(np.fft.rfft(hrirs[idx_30R, 1, :], n=n_fft)) + 1e-12)
        smooth_30R = apply_smoothing(fft_freqs, mag_30R, fraction=args.smoothing)
        val_30R = np.interp(target_freqs, fft_freqs, smooth_30R)
        val_30R -= val_30R[idx_1k]
        
        # Apply LF extension and tilt identically
        if args.lf_ext_dfhrtf:
            ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "lf_ref_mag.txt")
            ref_freqs, ref_db = load_lf_reference(ref_path)
            if ref_freqs is not None:
                val_30L = apply_lf_extension(target_freqs, val_30L, ref_freqs, ref_db, args.lf_ext_dfhrtf_xover)
                val_30R = apply_lf_extension(target_freqs, val_30R, ref_freqs, ref_db, args.lf_ext_dfhrtf_xover)
        
        if args.tilt != 0:
            val_30L = apply_spectral_tilt(target_freqs, val_30L, args.tilt, 1000.0)
            val_30R = apply_spectral_tilt(target_freqs, val_30R, args.tilt, 1000.0)
            
        # Plot
        plt.figure(figsize=(10, 6))
        plt.semilogx(target_freqs, val_30L, label='Left Ear (+30°)', linewidth=2, alpha=0.8)
        plt.semilogx(target_freqs, val_30R, label='Right Ear (-30°)', linewidth=2, alpha=0.8, linestyle='--')
        
        title_str = f"Stereo Response (+/-30°)\nSpectral Tilt: {args.tilt} dB/oct"
        plt.title(title_str)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.ylim(-25, 25)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.xlim(20, 20000)
        
        plt.savefig(args.graph_stereo_out, dpi=150)
        print(f"   [+] Saved Stereo Plot: {os.path.basename(args.graph_stereo_out)}")
        plt.close()

    if args.only_plot: return

    # 8. Save Text Files
    if args.text_out_l:
        save_text_data(args.text_out_l, target_freqs, val_l, args.delimiter)
    if args.text_out_r:
        save_text_data(args.text_out_r, target_freqs, val_r, args.delimiter)
    if args.text_out_avg:
        save_text_data(args.text_out_avg, target_freqs, val_avg, args.delimiter)
    
    # Handle legacy saving if requested via output_dir
    if not any([args.text_out_l, args.text_out_r, args.text_out_avg]) and args.output_dir:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        sim_meas_str = "Simulated" if args.sim_meas else "Measured"
        ext = ".txt" if args.squigify else ".csv"
        
        l_path = os.path.join(args.output_dir, f"{args.prefix}{args.tilt:g} {base_name} {sim_meas_str} L{ext}")
        r_path = os.path.join(args.output_dir, f"{args.prefix}{args.tilt:g} {base_name} {sim_meas_str} R{ext}")
        save_text_data(l_path, target_freqs, val_l, args.delimiter)
        save_text_data(r_path, target_freqs, val_r, args.delimiter)
        
        if not args.squigify:
            avg_path = os.path.join(args.output_dir, f"{args.prefix}{args.tilt:g} {base_name} {sim_meas_str} Avg{ext}")
            save_text_data(avg_path, target_freqs, val_avg, args.delimiter)

if __name__ == "__main__":
    main()
