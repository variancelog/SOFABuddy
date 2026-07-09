import numpy as np
import sofar as sf
import scipy.fft as fft
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull

def spherical_to_cartesian(r, az, el):
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    x = r * np.cos(el_rad) * np.cos(az_rad)
    y = r * np.cos(el_rad) * np.sin(az_rad)
    z = r * np.sin(el_rad)
    return np.column_stack((x, y, z))

def get_dominant_radius(source_pos):
    r = source_pos[:, 2]
    radii_rounded = np.round(r, decimals=3)
    values, counts = np.unique(radii_rounded, return_counts=True)
    return values[np.argmax(counts)]

def calculate_geometric_weights(source_pos):
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
        return weights
    except:
        return np.ones(len(source_pos)) / len(source_pos)

def analyze_and_plot_df(file_path, crossover_f=450.0):
    print(f"\n--- Analyzing: {os.path.basename(file_path)} ---")
    sofa = sf.read_sofa(file_path)
    fs = float(sofa.Data_SamplingRate)
    hrirs = sofa.Data_IR
    M, R, N = hrirs.shape
    
    H = fft.rfft(hrirs, axis=-1)
    freqs = fft.rfftfreq(N, 1/fs)
    
    # Calculate DF using GEOMETRIC weights
    weights = calculate_geometric_weights(sofa.SourcePosition)
    weights_expanded = weights[:, np.newaxis, np.newaxis]
    
    P = np.abs(H)**2
    DF_power = np.sum(P * weights_expanded, axis=0)
    DF_mag = np.sqrt(DF_power)
    DF_db = 20 * np.log10(DF_mag + 1e-12)
    
    plt.figure(figsize=(10, 6))
    for r in range(R):
        plt.semilogx(freqs, DF_db[r], label=f'Ear {r}')
    plt.axvline(x=crossover_f, color='r', linestyle='--', label='Crossover')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.title(f"Diffuse Field Magnitude (Geometric): {os.path.basename(file_path)}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(20, 20000)
    
    out_dir = "zzsofa_raw/comparisons"
    out_path = os.path.join(out_dir, f"df_mag_check_GEO_{os.path.basename(file_path)}.png")
    plt.savefig(out_path)
    
    # Print some stats (Average across ears)
    idx_xover = np.searchsorted(freqs, crossover_f)
    DF_db_avg = np.mean(DF_db, axis=0)
    level_at_xover = DF_db_avg[idx_xover]
    avg_above = np.mean(DF_db_avg[idx_xover:])
    avg_below = np.mean(DF_db_avg[max(0, idx_xover-50):idx_xover])
    
    std_above = np.std(DF_db_avg[idx_xover:])
    
    print(f"Avg DF Std Dev above {crossover_f}Hz: {std_above:.4f} dB")
    print(f"Level at xover: {level_at_xover:.2f} dB")
    print(f"Avg level above: {avg_above:.2f} dB")
    print(f"Step size (Above - BelowWindow): {avg_above - avg_below:.2f} dB")

if __name__ == "__main__":
    analyze_and_plot_df('zzsofa_raw/sofa_mastered/ARI_NH0012.sofa')
    analyze_and_plot_df('zzsofa_raw/sofa_mastered/VL01b_HRIR_48000Hz.sofa')
