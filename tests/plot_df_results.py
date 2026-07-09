import numpy as np
import sofar as sf
import scipy.fft as fft
import matplotlib.pyplot as plt
import os

def analyze_and_plot_df(file_path, crossover_f=450.0):
    print(f"\n--- Analyzing: {os.path.basename(file_path)} ---")
    sofa = sf.read_sofa(file_path)
    fs = float(sofa.Data_SamplingRate)
    hrirs = sofa.Data_IR
    M, R, N = hrirs.shape
    
    H = fft.rfft(hrirs, axis=-1)
    freqs = fft.rfftfreq(N, 1/fs)
    
    # Calculate DF
    P = np.abs(H)**2
    DF_power = np.mean(P, axis=(0, 1))
    DF_mag = np.sqrt(DF_power)
    DF_db = 20 * np.log10(DF_mag + 1e-12)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs, DF_db)
    plt.axvline(x=crossover_f, color='r', linestyle='--', label='Crossover')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.title(f"Diffuse Field Magnitude: {os.path.basename(file_path)}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(20, 20000)
    
    out_dir = "zzsofa_raw/comparisons"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_path = os.path.join(out_dir, f"df_mag_check_{os.path.basename(file_path)}.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    
    # Print some stats
    idx_xover = np.searchsorted(freqs, crossover_f)
    level_at_xover = DF_db[idx_xover]
    avg_above = np.mean(DF_db[idx_xover:idx_xover+50]) # nearby above
    avg_below = np.mean(DF_db[max(0, idx_xover-50):idx_xover]) # nearby below
    
    print(f"Level at xover: {level_at_xover:.2f} dB")
    print(f"Avg level nearby above: {avg_above:.2f} dB")
    print(f"Avg level nearby below: {avg_below:.2f} dB")
    print(f"Step size: {avg_above - avg_below:.2f} dB")

if __name__ == "__main__":
    # Ensure comparisons directory exists
    if not os.path.exists("zzsofa_raw/comparisons"):
        os.makedirs("zzsofa_raw/comparisons")
        
    analyze_and_plot_df('zzsofa_raw/sofa_mastered/ARI_NH0012_48000Hz_DFEQ.sofa')
    analyze_and_plot_df('zzsofa_raw/sofa_mastered/VL01b_HRIR_48000Hz_48000Hz_DFEQ.sofa')
