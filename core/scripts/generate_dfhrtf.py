import argparse
import os
import sys
import warnings
# Silence requests' RequestsDependencyWarning (chardet 6.x > its supported ceiling).
# Must precede `import sofar`, which eagerly imports requests and would otherwise
# emit the warning before any later filter is registered.
warnings.filterwarnings("ignore", message=r".*doesn't match a supported version.*")
import numpy as np  # noqa: E402
import sofar as sf  # noqa: E402
import scipy.fft as fft  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import csv  # noqa: E402
import re  # noqa: E402

# Bootstrap project root onto sys.path so core.* imports work when this script
# is launched as a subprocess (no cwd / PYTHONPATH guarantees from the caller).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.geometry_weights import (  # noqa: E402
    spherical_to_cartesian,
    get_dominant_radius,
    calculate_geometric_weights,
)

# Marker area (pts²) for the weighted-grid plot.  Size is constant so that
# comparisons across files differ only in color, not marker footprint.
WEIGHTED_GRID_NODE_SIZE = 20


# ================= HELPER FUNCTIONS =================


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

def apply_lf_extension(target_freqs, val_db, ref_freqs, ref_db, xover_freq,
                       blend_low_oct=0.50, blend_high_oct=1.0):
    """Applies LF extension with a cosine crossfade."""
    # Interpolate reference to match target frequencies
    interp_ref_db = np.interp(target_freqs, ref_freqs, ref_db)

    # Level match at crossover
    xover_idx = (np.abs(target_freqs - xover_freq)).argmin()
    offset = val_db[xover_idx] - interp_ref_db[xover_idx]
    matched_ref_db = interp_ref_db + offset

    # Convert to linear amplitude
    val_lin = 10**(val_db / 20.0)
    ref_lin = 10**(matched_ref_db / 20.0)

    # Crossfade weights (configurable octave widths)
    f_low = xover_freq / (2**blend_low_oct)
    f_high = xover_freq * (2**blend_high_oct)
    
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
    parser.add_argument("--avg-algo", default="Arithmetic Mean", help="L/R averaging algorithm: 'Arithmetic Mean', 'Geo-Arc-Length', or 'Soft-DTW'")
    parser.add_argument("--graph-out", help="Explicit path for Graph plot output")
    parser.add_argument("--graph-stereo-out", help="Explicit path for Stereo graph plot output")
    parser.add_argument("--graph-weighted-out", help="Explicit path for Weighted Grid plot output")
    parser.add_argument("--delimiter", default="\t", help="Delimiter for text outputs")
    
    # Refinement Parameters
    parser.add_argument("--smoothing", type=int, default=48, help="Smoothing fraction (e.g. 24 for 1/24th octave)")
    parser.add_argument("--tilt", type=float, default=0.0, help="Spectral Tilt (dB/oct)")
    parser.add_argument("--lr-avg-gamma", type=float, default=0.01, help="Soft-DTW gamma for per-SOFA L/R average")
    parser.add_argument("--lf_ext_dfhrtf", action="store_true", help="Enable Low Frequency extension using reference file")
    parser.add_argument("--lf_ext_dfhrtf_xover", type=float, default=375.0, help="Crossover frequency for LF extension")

    # Advanced tunable parameters (forwarded from dfhrtf.advanced config section)
    parser.add_argument("--n-fft",                  type=int,   default=16384,
                        help="FFT size for diffuse-field computation")
    parser.add_argument("--output-octave-fraction",  type=int,   default=48,
                        help="Resolution of output frequency grid (1/N octave)")
    parser.add_argument("--norm-freq-hz",            type=float, default=1000.0,
                        help="0 dB reference frequency for normalization (Hz)")
    parser.add_argument("--lf-blend-low-oct",        type=float, default=0.50,
                        help="Octaves below crossover for LF extension crossfade start")
    parser.add_argument("--lf-blend-high-oct",       type=float, default=1.0,
                        help="Octaves above crossover for LF extension crossfade end")
    parser.add_argument("--stereo-angle",            type=float, default=30.0,
                        help="Azimuth angle (degrees) for stereo graph L/R measurement")
    
    parser.add_argument("--advanced-dfeq", action="store_true", default=True, help="Use advanced data sanitization")
    parser.add_argument("--weighting-mode", default="auto", choices=["auto", "geometric", "uniform"],
                        help="Grid weighting mode: auto (detect balanced grids), geometric, or uniform 1/N")

    parser.add_argument("--grid-scale-mode", default="auto", choices=["auto", "manual"],
                        help="Weighted-grid color range: auto (1st-99th pct) or manual fixed low/high")
    parser.add_argument("--grid-scale-low",  type=float, default=0.001,
                        help="Manual color-range minimum (default 0.001)")
    parser.add_argument("--grid-scale-high", type=float, default=0.003,
                        help="Manual color-range maximum (default 0.003)")
    parser.add_argument("--preserve-lr-balance", action="store_true",
                        help="Normalize L and R with a single shared 1 kHz reference "
                             "(mean of L/R) to preserve the real inter-aural level offset")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file missing: {args.input}")
        sys.exit(1)

    # 1. Load Data
    sofa = sf.read_sofa(args.input)
    hrirs = sofa.Data_IR
    fs = float(sofa.Data_SamplingRate)

    # Resolve advanced params from args
    n_fft            = args.n_fft
    norm_freq_hz     = args.norm_freq_hz
    lf_blend_low_oct = args.lf_blend_low_oct
    lf_blend_high_oct= args.lf_blend_high_oct
    stereo_angle     = args.stereo_angle

    # 2. Compute Diffuse Field Response
    weights = calculate_geometric_weights(
        sofa.SourcePosition,
        advanced_dfeq=args.advanced_dfeq,
        weighting_mode=args.weighting_mode,
    )

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
    target_freqs = generate_fractional_octave_frequencies(20, 20000,
                                                          fraction=args.output_octave_fraction)

    print(f"   -> Applying 1/{args.smoothing} Octave Smoothing...")
    smoothed_l = apply_smoothing(fft_freqs, avg_db[0], fraction=args.smoothing)
    smoothed_r = apply_smoothing(fft_freqs, avg_db[1], fraction=args.smoothing)

    val_l = np.interp(target_freqs, fft_freqs, smoothed_l)
    val_r = np.interp(target_freqs, fft_freqs, smoothed_r)

    # 4. Normalize (0 dB at norm_freq_hz)
    # preserve_lr_balance: use a single shared mean reference so the real
    # inter-aural level difference (ILD) is preserved across L and R.
    idx_norm = (np.abs(target_freqs - norm_freq_hz)).argmin()
    if args.preserve_lr_balance:
        ref = (val_l[idx_norm] + val_r[idx_norm]) / 2.0   # shared → preserves ILD
        val_l -= ref
        val_r -= ref
    else:
        val_l -= val_l[idx_norm]
        val_r -= val_r[idx_norm]

    # 4.5 LF Extension
    if args.lf_ext_dfhrtf:
        print(f"   -> Applying LF Extension (xover: {args.lf_ext_dfhrtf_xover}Hz)...")
        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "lf_ref_mag.txt")
        ref_freqs, ref_db = load_lf_reference(ref_path)
        if ref_freqs is not None:
            val_l = apply_lf_extension(target_freqs, val_l, ref_freqs, ref_db,
                                       args.lf_ext_dfhrtf_xover,
                                       blend_low_oct=lf_blend_low_oct,
                                       blend_high_oct=lf_blend_high_oct)
            val_r = apply_lf_extension(target_freqs, val_r, ref_freqs, ref_db,
                                       args.lf_ext_dfhrtf_xover,
                                       blend_low_oct=lf_blend_low_oct,
                                       blend_high_oct=lf_blend_high_oct)

    # 5. Apply Tilt
    if args.tilt != 0:
        print(f"   -> Applying {args.tilt} dB/oct tilt...")
        val_l = apply_spectral_tilt(target_freqs, val_l, args.tilt, norm_freq_hz)
        val_r = apply_spectral_tilt(target_freqs, val_r, args.tilt, norm_freq_hz)

    # 6. Calculate Average
    if args.avg_algo == "Arithmetic Mean":
        val_avg = (val_l + val_r) / 2.0
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        from core.averaging import CurveAverager, CommonAxisInterpolator
        mags_2d = np.vstack([val_l, val_r])
        mag_min, mag_max = float(mags_2d.min()), float(mags_2d.max())
        if mag_max > mag_min:
            norm = (mags_2d - mag_min) / (mag_max - mag_min)
        else:
            norm = np.zeros_like(mags_2d)
        if args.avg_algo == "Geo-Arc-Length":
            log_f = np.log10(target_freqs)
            norm_freqs = (log_f - log_f.min()) / (log_f.max() - log_f.min())
            avg_norm = CurveAverager.geometric_arc_length_blend(norm_freqs, norm * 0.01) / 0.01
        elif args.avg_algo == "Soft-DTW":
            print("   -> Running Soft-DTW L/R average...")
            avg_norm = CurveAverager.soft_dtw_barycenter(norm, gamma=args.lr_avg_gamma)
        else:
            avg_norm = CurveAverager.arithmetic_mean(norm)
        val_avg = CommonAxisInterpolator.unnormalize_magnitude(avg_norm, mag_min, mag_max)

    # 7. Generate Plot
    if args.graph_out:
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

        plt.savefig(args.graph_out, dpi=150)
        print(f"   [+] Saved Plot: {os.path.basename(args.graph_out)}")
        plt.close()

    # 7.5 Stereo Graph (+/- stereo_angle degrees)
    # To be added to an HRTF or Graph-specific Tab later
    if args.graph_stereo_out:
        az_left  = stereo_angle
        az_right = 360.0 - stereo_angle  # e.g. 330° for 30°
        print(f"   -> Calculating +/-{stereo_angle:.0f}° Stereo Response...")
        # Find closest grid points: Left (+az), Right (-az / mirror)
        idx_30L = find_closest_grid_point(sofa.SourcePosition, az_left, 0.0)
        idx_30R = find_closest_grid_point(sofa.SourcePosition, az_right, 0.0)

        # Log what we found
        az_L, el_L = sofa.SourcePosition[idx_30L, 0], sofa.SourcePosition[idx_30L, 1]
        az_R, el_R = sofa.SourcePosition[idx_30R, 0], sofa.SourcePosition[idx_30R, 1]
        print(f"      Closest to L(+{az_left:.0f},0): az={az_L:.1f}, el={el_L:.1f}")
        print(f"      Closest to R(-{stereo_angle:.0f},0): az={az_R:.1f}, el={el_R:.1f}")

        # Process L Ear
        mag_30L = 20 * np.log10(np.abs(np.fft.rfft(hrirs[idx_30L, 0, :], n=n_fft)) + 1e-12)
        smooth_30L = apply_smoothing(fft_freqs, mag_30L, fraction=args.smoothing)
        val_30L = np.interp(target_freqs, fft_freqs, smooth_30L)

        # Process R Ear
        mag_30R = 20 * np.log10(np.abs(np.fft.rfft(hrirs[idx_30R, 1, :], n=n_fft)) + 1e-12)
        smooth_30R = apply_smoothing(fft_freqs, mag_30R, fraction=args.smoothing)
        val_30R = np.interp(target_freqs, fft_freqs, smooth_30R)

        # Normalize — shared reference preserves the stereo pair's ILD
        if args.preserve_lr_balance:
            ref_s = (val_30L[idx_norm] + val_30R[idx_norm]) / 2.0
            val_30L -= ref_s
            val_30R -= ref_s
        else:
            val_30L -= val_30L[idx_norm]
            val_30R -= val_30R[idx_norm]

        # Apply LF extension and tilt identically
        if args.lf_ext_dfhrtf:
            ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "lf_ref_mag.txt")
            ref_freqs, ref_db = load_lf_reference(ref_path)
            if ref_freqs is not None:
                val_30L = apply_lf_extension(target_freqs, val_30L, ref_freqs, ref_db,
                                             args.lf_ext_dfhrtf_xover,
                                             blend_low_oct=lf_blend_low_oct,
                                             blend_high_oct=lf_blend_high_oct)
                val_30R = apply_lf_extension(target_freqs, val_30R, ref_freqs, ref_db,
                                             args.lf_ext_dfhrtf_xover,
                                             blend_low_oct=lf_blend_low_oct,
                                             blend_high_oct=lf_blend_high_oct)

        if args.tilt != 0:
            val_30L = apply_spectral_tilt(target_freqs, val_30L, args.tilt, norm_freq_hz)
            val_30R = apply_spectral_tilt(target_freqs, val_30R, args.tilt, norm_freq_hz)
            
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

    # 7.6 Weighted Grid Plot (cleaned point set, color encodes solid-angle weight)
    if args.graph_weighted_out:
        print("   -> Plotting Weighted Grid...")
        src = sofa.SourcePosition
        cart = spherical_to_cartesian(1.0, src[:, 0], src[:, 1])

        mask = weights > 0                      # surviving (cleaned) points only
        w = weights[mask]
        cart_clean = cart[mask]

        # Color range: manual fixed bounds or auto 1st–99th percentile
        if args.grid_scale_mode == "manual":
            vmin, vmax = args.grid_scale_low, args.grid_scale_high
        else:
            vmin = np.percentile(w, 1)
            vmax = np.percentile(w, 99)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            cart_clean[:, 0], cart_clean[:, 1], cart_clean[:, 2],
            c=w, s=WEIGHTED_GRID_NODE_SIZE, cmap='viridis', alpha=0.85, vmin=vmin, vmax=vmax
        )
        plt.colorbar(sc, ax=ax, label='Solid-angle weight', shrink=0.6, pad=0.1)
        ax.set_title(f"Weighted grid — {int(mask.sum())} nodes")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect([1, 1, 1])
        plt.savefig(args.graph_weighted_out, dpi=150, bbox_inches='tight')
        print(f"   [+] Saved Weighted Grid: {os.path.basename(args.graph_weighted_out)}")
        plt.close()

    # 8. Save Text Files
    if args.text_out_l:
        save_text_data(args.text_out_l, target_freqs, val_l, args.delimiter)
    if args.text_out_r:
        save_text_data(args.text_out_r, target_freqs, val_r, args.delimiter)
    if args.text_out_avg:
        save_text_data(args.text_out_avg, target_freqs, val_avg, args.delimiter)

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
