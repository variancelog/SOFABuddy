import numpy as np
import sofar as sf
import scipy.fft as fft
import os

def calculate_geometric_weights(source_pos):
    # Simplified version for verification
    # Just to get a rough DF estimate
    return np.ones(len(source_pos)) / len(source_pos)

def analyze_df(file_path, crossover_f=450.0):
    print(f"\n--- Analyzing: {os.path.basename(file_path)} ---")
    sofa = sf.read_sofa(file_path)
    fs = float(sofa.Data_SamplingRate)
    hrirs = sofa.Data_IR
    M, R, N = hrirs.shape
    
    H = fft.rfft(hrirs, axis=-1)
    freqs = fft.rfftfreq(N, 1/fs)
    
    # Calculate DF
    P = np.abs(H)**2
    # Use uniform weights for simple verification
    DF_power = np.mean(P, axis=(0, 1))
    DF_mag = np.sqrt(DF_power)
    DF_db = 20 * np.log10(DF_mag + 1e-12)
    
    idx_xover = np.searchsorted(freqs, crossover_f)
    
    df_above = DF_db[idx_xover:]
    df_below = DF_db[:idx_xover]
    
    std_above = np.std(df_above)
    range_above = np.max(df_above) - np.min(df_above)
    
    print(f"DF Std Dev above {crossover_f}Hz: {std_above:.2f} dB")
    print(f"DF Range above {crossover_f}Hz: {range_above:.2f} dB")
    
    # Check phase continuity at crossover
    # We'll check a few directions
    print("\nPhase Continuity at crossover (rad dev):")
    for idx in [0, M//4, M//2]:
        for r in range(R):
            phase = np.unwrap(np.angle(H[idx, r, :]))
            # Local slope
            diff = np.diff(phase)
            # Check jump at idx_xover
            jump = np.abs(phase[idx_xover] - phase[idx_xover-1])
            # Average jump nearby
            avg_jump = np.mean(np.abs(diff[max(0, idx_xover-10):idx_xover+10]))
            print(f"  Pos {idx}, Ear {r}: Jump={jump:.4f}, AvgNearby={avg_jump:.4f}")

if __name__ == "__main__":
    analyze_df('zzsofa_raw/sofa_mastered/ARI_NH0012_48000Hz_DFEQ.sofa')
    analyze_df('zzsofa_raw/sofa_mastered/VL01b_HRIR_48000Hz_48000Hz_DFEQ.sofa')
