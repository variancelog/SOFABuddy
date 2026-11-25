import argparse
import contextlib
import io
import json
import os
import sys
import math
import warnings
# Silence requests' RequestsDependencyWarning (chardet 6.x > its supported ceiling).
# Must precede `import sofar`, which eagerly imports requests and would otherwise
# emit the warning before any later filter is registered.
warnings.filterwarnings("ignore", message=r".*doesn't match a supported version.*")
import numpy as np  # noqa: E402
import sofar as sf  # noqa: E402
import scipy.fft as fft  # noqa: E402

# Bootstrap project root onto sys.path so core.* imports work when this script
# is launched as a subprocess (no cwd / PYTHONPATH guarantees from the caller).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.geometry_weights import (  # noqa: E402
    spherical_to_cartesian,
    get_dominant_radius,
    calculate_geometric_weights,
)
from core.sofa_io import read_sofa_robust  # noqa: E402

# ================= R2 CONFIGURATION =================
PROCESSING_LENGTH = 2048
OUTPUT_LENGTH = 512
NORM_TARGET_DB = -1.0
DEFAULT_ONSET_BUFFER = 32 # Samples of silence before the aligned impulse starts
# =================================================


def _hilbert(x):
    """Analytic signal along the last axis (same algorithm as scipy.signal.hilbert).

    Reimplemented on scipy.fft so that importing scipy.signal (~1.5 s of
    subprocess start-up, mostly scipy.stats/interpolate) is only paid when
    resampling is actually required.
    """
    N = x.shape[-1]
    Xf = fft.fft(x, N, axis=-1)
    if N % 2 == 0:
        Xf[..., 1:N // 2] *= 2.0
        Xf[..., N // 2 + 1:N] = 0.0
    else:
        Xf[..., 1:(N + 1) // 2] *= 2.0
        Xf[..., (N + 1) // 2:N] = 0.0
    return fft.ifft(Xf, axis=-1)


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

def _prepare_master(raw_sofa, target_fs, args):
    """Steps 1-6: everything up to and including the LF splice.

    Depends only on (raw_sofa, target_fs, args) -- not on whether DFEQ is applied --
    so the uneq and dfeq variants of one input/samplerate share it.  Returns a
    bundle dict that _finish_master() treats as read-only.
    """
    # Resolve advanced params from args (with module-constant fallbacks for legacy calls)
    processing_length    = getattr(args, 'processing_length',    PROCESSING_LENGTH)
    onset_buffer         = getattr(args, 'onset_buffer',         DEFAULT_ONSET_BUFFER)
    onset_threshold      = getattr(args, 'onset_threshold',      0.05)
    blend_low_oct        = getattr(args, 'blend_low_oct',        0.50)
    blend_high_oct       = getattr(args, 'blend_high_oct',       0.75)
    debug_dump           = getattr(args, 'debug_dump',           None)  # diag-only; see --debug-dump

    # Diagnostic capture (only populated when debug_dump is set; zero effect otherwise)
    dbg = {} if debug_dump else None

    hrirs = raw_sofa.Data_IR.copy()
    src_fs = float(raw_sofa.Data_SamplingRate)
    M, R, N = hrirs.shape

    # 1. Resample
    if abs(src_fs - target_fs) > 1.0:
        print(f"   1. Resampling: {src_fs:.0f} -> {target_fs} Hz")
        gcd = math.gcd(int(src_fs), int(target_fs))
        up = int(target_fs) // gcd
        down = int(src_fs) // gcd
        import scipy.signal as signal  # deferred: slow import, only needed here
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
        orig_r = get_dominant_radius(raw_sofa.SourcePosition)
        target_r = args.target_radius
        gain_multiplier = orig_r / target_r

        # Distance normalization changes only the level (inverse-distance gain)
        # and the declared radius metadata (rewritten before save). No absolute
        # time-of-arrival delay is encoded: real SOFA files strip propagation
        # delay and declare distance via SourcePosition alone, and encoding an
        # absolute TOA here (~209 samples for 1.5 m @ 48 kHz) pushes the impulse
        # onset past short output windows, destroying the frequency response.
        print(f"   3. Harmonizing Radius: {orig_r:.2f}m -> {target_r:.2f}m (Gain: x{gain_multiplier:.3f})")
        hrirs *= gain_multiplier
    else:
        print(f"   3. Global Impulse Alignment: Stripping {global_onset - onset_buffer} samples of padding.")

    total_shift = base_shift

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

    # 5. Calculate Weights for Diffuse Field Calculation (Raw)
    weights = calculate_geometric_weights(
        raw_sofa.SourcePosition,
        advanced_dfeq=True,
        weighting_mode=args.weighting_mode,
    )
    weights_expanded = weights[:, np.newaxis, np.newaxis]
    
    mag_raw = np.abs(H_raw)
    P_raw = mag_raw**2
    DF_power = np.sum(P_raw * weights_expanded, axis=0)
    DF_mag = np.sqrt(DF_power) # (R, N_bins)
    
    # 6. Low Frequency Extension
    # 6.a. LF Extension Parameters
    crossover_f = args.lf_crossover
    idx_xover = np.searchsorted(freqs, crossover_f)
    blend_weights = get_blend_weights(freqs, crossover_f,
                                      blend_low_oct=blend_low_oct,
                                      blend_high_oct=blend_high_oct)
     
    # 6.b. Define Anchor Level (Level of Raw DF at LF extension crossover)
    anchor_level = np.mean(DF_mag[:, max(0, idx_xover-2):idx_xover+3], axis=-1, keepdims=True)
    
       
    # 6.c. Define the Unified Target Response
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

    ##
    # 6.d.2 Apply Minimum Phase LF Splicing
    H_rep = H_raw  # never mutated in place below; no copy needed
    if args.lf_extend and base_target is not None and scaled_lf_target is not None:
        print(f"   4. Splicing Ideal LF Response (Crossover: {crossover_f} Hz using Phase Blending)...")
        
        # --- A. Extract Minimum Phase and Excess Phase of Raw HRIR ---
        log_mag_raw = np.log(mag_raw + 1e-12)
        full_log_mag_raw = np.concatenate([log_mag_raw, log_mag_raw[..., -2:0:-1]], axis=-1)
        min_phase_raw = -np.imag(_hilbert(full_log_mag_raw))[..., :H_raw.shape[-1]]
        
        phase_raw = np.unwrap(np.angle(H_raw), axis=-1)
        excess_phase = phase_raw - min_phase_raw
        
        H_raw_min = mag_raw * np.exp(1j * min_phase_raw)

        # --- B. Create Pure Minimum-Phase Synthetic LF ---
        # Depends only on the receiver, so compute once per r and broadcast over M.
        log_base = np.log(scaled_lf_target + 1e-12)
        full_log_base = np.concatenate([log_base, log_base[..., -2:0:-1]], axis=-1)
        min_phase_base = -np.imag(_hilbert(full_log_base))[..., :len(base_target)]
        H_synthetic_min = scaled_lf_target * np.exp(1j * min_phase_base)   # (R, bins)
        
        # --- C. Blend in the Minimum Phase Domain (Solves Comb Filtering) ---
        H_rep_min = H_synthetic_min * (1.0 - blend_weights) + H_raw_min * blend_weights
        
        # --- D. Fix Time Dispersion (Blend Excess Phase) ---
        # Replace the chaotic sub-crossover excess phase with a pure linear delay
        # matching the time-of-arrival. The delay is estimated from the UNWRAP-
        # IMMUNE group-delay slope, not a single-point read of the DC-anchored
        # np.unwrap above: on MEASURED files the lowest bins are near-zero-
        # magnitude noise, so np.unwrap can commit a spurious 2*pi slip that
        # offsets the excess phase at the crossover and inflates the extrapolated
        # delay by k*(fs/crossover) samples (~128 @ 48k/1000). That mispositions
        # the spliced LF — chaotic sub-1kHz phase plus extra crop truncation.
        # Adjacent-bin excess-phase increments wrap to (-pi, pi], so any global
        # slip cancels; magnitude weighting ignores noisy near-null bins.
        # SIMULATED files (clean phase) are unaffected — the estimate matches the
        # old single-point value to a fraction of a sample.
        f_lo_fit = crossover_f / (2 ** blend_low_oct)   # exact-match edge (f_low)
        f_hi_fit = crossover_f * 2.0
        fit_idx = np.where((freqs >= f_lo_fit) & (freqs <= f_hi_fit))[0]
        if fit_idx.size >= 2:
            # Unit-modulus excess phasor, evaluated only on the fit bins (elementwise,
            # so identical to computing it over every bin and indexing).
            excess_c = (H_raw[..., fit_idx] / (mag_raw[..., fit_idx] + 1e-12)) \
                * np.exp(-1j * min_phase_raw[..., fit_idx])
            dphi = np.angle(excess_c[..., 1:] * np.conj(excess_c[..., :-1]))
            d_f = np.diff(freqs[fit_idx])
            w_fit = np.minimum(mag_raw[..., fit_idx[1:]], mag_raw[..., fit_idx[:-1]])
            num = np.sum(w_fit * (-dphi) * (2 * np.pi * d_f), axis=-1)
            den = np.sum(w_fit * (2 * np.pi * d_f) ** 2, axis=-1)
            delay_arr = num / (den + 1e-30)             # (M, R) seconds
        else:  # degenerate grid — fall back to the legacy single-point estimate
            ex_xover = np.mean(excess_phase[..., max(0, idx_xover-2):idx_xover+3], axis=-1)
            delay_arr = -ex_xover / (2 * np.pi * crossover_f)
        # Safety net: reject non-causal or ill-conditioned (near-null) estimates.
        # Generous ceiling — the group-delay fit is the real defense; this only
        # catches pathological positions, not a genuine large ITD.
        max_delay_sec = (onset_buffer + 128) / target_fs
        delay_arr = np.clip(delay_arr, 0.0, max_delay_sec)

        # Extrapolate each position's delay as a pure linear phase slope for the LF
        linear_excess_phase = -2 * np.pi * freqs[None, None, :] * delay_arr[..., None]
        if debug_dump:
            delay_sec_arr = delay_arr

        # Crossfade the phase: Linear delay (LF) -> True spatial phase (HF)
        blended_excess_phase = linear_excess_phase * (1.0 - blend_weights) + excess_phase * blend_weights
        
        # --- E. Final Reassembly ---
        H_rep = H_rep_min * np.exp(1j * blended_excess_phase)
    ## End: Test pure min phase splicing

    if debug_dump:
        # Frequency-domain snapshot (post-splice, pre-DFEQ). Splice-only arrays
        # are captured only when the LF path actually ran.
        dbg.update(dict(
            freqs=freqs, H_raw=H_raw, H_rep=H_rep,
            anchor_level=anchor_level, blend_weights=blend_weights,
            idx_xover=idx_xover, crossover_f=crossover_f,
            global_onset=global_onset, total_shift=total_shift,
        ))
        if args.lf_extend and 'scaled_lf_target' in dir() and scaled_lf_target is not None:
            dbg.update(dict(
                scaled_lf_target=scaled_lf_target,
                excess_phase=excess_phase,
                delay_sec_arr=delay_sec_arr,
            ))

    return dict(
        N=N, H_rep=H_rep, weights_expanded=weights_expanded, DF_mag=DF_mag,
        anchor_level=anchor_level, dbg=dbg,
        orig_r=orig_r if args.harmonize_radius else None,
    )


def _prepare_master_logged(raw_sofa, target_fs, args):
    """_prepare_master plus a record of its printed lines (for replay on reuse)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            prep = _prepare_master(raw_sofa, target_fs, args)
    finally:
        print(buf.getvalue(), end='')
    prep['log'] = buf.getvalue()
    return prep


def master_sofa(raw_sofa, target_fs, apply_dfeq, output_path, args, _prep_cache=None):
    """Master one output.  `_prep_cache` (a dict, used by run_manifest) keeps the
    most recent steps-1-6 bundle so consecutive jobs for the same input object,
    target_fs and args (e.g. the uneq and dfeq variants) skip recomputing it."""
    print(f"\n--- Generating: {os.path.basename(output_path)} ---")
    prep = None
    if _prep_cache is not None and not getattr(args, 'debug_dump', None):
        if (_prep_cache.get('raw') is raw_sofa and _prep_cache.get('args') is args
                and _prep_cache.get('fs') == target_fs):
            prep = _prep_cache['prep']
            print(prep['log'], end='')
        else:
            _prep_cache.clear()  # drop the old bundle before building the new one
            prep = _prepare_master_logged(raw_sofa, target_fs, args)
            _prep_cache.update(raw=raw_sofa, args=args, fs=target_fs, prep=prep)
    else:
        prep = _prepare_master_logged(raw_sofa, target_fs, args)
    _finish_master(raw_sofa, prep, target_fs, apply_dfeq, output_path, args)


def _finish_master(raw_sofa, prep, target_fs, apply_dfeq, output_path, args):
    """Steps 7-13 (DFEQ onward).  Never mutates the arrays in `prep`."""
    norm_target_db       = getattr(args, 'norm_target_db',       NORM_TARGET_DB)
    onset_buffer         = getattr(args, 'onset_buffer',         DEFAULT_ONSET_BUFFER)
    ild_fix              = getattr(args, 'ild_fix',              False)
    ild_threshold        = getattr(args, 'ild_threshold',        1.5)
    ild_low_freq         = getattr(args, 'ild_low_freq',         300.0)
    ild_high_freq        = getattr(args, 'ild_high_freq',        6000.0)
    debug_dump           = getattr(args, 'debug_dump',           None)
    N, H_rep, weights_expanded = prep['N'], prep['H_rep'], prep['weights_expanded']
    DF_mag, anchor_level, orig_r = prep['DF_mag'], prep['anchor_level'], prep['orig_r']
    dbg = dict(prep['dbg']) if debug_dump else None
    out_sofa = raw_sofa.copy()
    if args.harmonize_radius:
        target_r = args.target_radius

  # 7. Step C: Apply Unified Diffuse Field EQ (if DFEQ is ON)
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
        min_phase_eq = -np.imag(_hilbert(full_log_eq))
        min_phase_eq = min_phase_eq[..., :eq_filter_mag.shape[-1]]
        
        complex_eq = eq_filter_mag * np.exp(1j * min_phase_eq)
        H_final = H_rep * complex_eq[np.newaxis, :, :]
    else:
        print("   5. Skipping DFEQ.")

    # 8. Back to Time Domain
    hrirs = fft.irfft(H_final, n=N, axis=-1)
    if debug_dump:
        dbg['hrirs_precrop'] = hrirs.copy()

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
        # Preserve the physical 1/r level relationship across the batch, but
        # guard against clipping: the inverse-distance gain (orig_r/target_r)
        # can push peaks past full scale. Only scale down when that happens.
        peak = np.max(np.abs(hrirs))
        if peak > 1.0:
            target_linear = 10 ** (norm_target_db / 20.0)
            scale = target_linear / peak
            hrirs *= scale
            print(f"   8. [!] Harmonized gain overshoots full scale by {20*np.log10(peak):+.2f} dB; "
                  f"scaling down {20*np.log10(scale):.2f} dB to a {norm_target_db} dB peak. "
                  f"This file's absolute level no longer matches the 1/r batch relationship.")
        else:
            print("   8. Skipping Normalization (Preserving Harmonized Radius Volume).")

    # 11. Save
    out_sofa.Data_IR = hrirs.astype(np.float32)

    # 12. ILD Balance — runs on the fully-assembled mastered sofar object so it
    # applies to the final combined signal (post-EQ, post-normalize).  Guard on
    # R == 2: apply_ild_fix indexes channels 0 and 1 directly.
    if ild_fix and out_sofa.Data_IR.shape[1] == 2:
        from core.ild_audit import diffuse_field_ild, apply_ild_fix
        ild_report = diffuse_field_ild(
            out_sofa,
            weighting_mode=args.weighting_mode,
            estimate_band=(ild_low_freq, ild_high_freq),
        )
        ild_est = ild_report["ild_estimate_db"]
        if abs(ild_est) > ild_threshold:
            print(f"   9. Balancing ILD: {ild_est:+.2f} dB offset corrected.")
            apply_ild_fix(out_sofa, ild_est)
            out_sofa.Data_IR = out_sofa.Data_IR.astype(np.float32)
        else:
            print(f"   9. ILD within threshold ({ild_est:+.2f} dB <= {ild_threshold:.1f} dB); no correction.")

    # 13. Rewrite SourcePosition radii so the file's declared distance matches
    # its processing. Scale the whole radius column so the dominant radius maps
    # exactly to target_r; off-radius outliers keep their proportional deviation
    # (and remain excluded by the ±0.05 m dominant-radius filter downstream).
    # Runs AFTER ILD balancing, which recomputes weights from SourcePosition:
    # rewriting first could let previously-excluded points pass the dominant-
    # radius filter and shift the ILD estimate.
    if args.harmonize_radius and orig_r != 0:
        sp = np.asarray(out_sofa.SourcePosition, dtype=float).copy()
        sp[:, 2] *= (target_r / orig_r)
        out_sofa.SourcePosition = sp

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
    if debug_dump:
        dbg['hrirs_final'] = np.asarray(out_sofa.Data_IR, dtype=float)
        dbg['target_fs'] = float(target_fs)
        dbg['onset_buffer'] = int(onset_buffer)
        os.makedirs(os.path.dirname(os.path.abspath(debug_dump)), exist_ok=True)
        np.savez_compressed(debug_dump, **dbg)
        print(f"   [debug] Dumped diagnostics to {debug_dump}")

    sf.write_sofa(output_path, out_sofa)
    print(f"   [+] Saved {os.path.basename(output_path)}")

def run_manifest(manifest_path):
    """
    Batch-process multiple SOFA mastering jobs from one JSON manifest:
      {"common": {...shared flags, same dest names as the CLI...},
       "jobs": [{"input": ..., "output": ..., "target_fs": ..., "dfeq": ...}, ...]}

    `common` supplies every attribute master_sofa() reads off `args` (both the
    getattr(args, ..., default)-based advanced params and the directly-accessed
    ones: harmonize_radius, target_radius, weighting_mode, lf_crossover,
    lf_extend, hrir_length) — it is built once as an argparse.Namespace and
    reused across jobs, exactly like the single-file CLI path's parsed args.

    Consecutive jobs sharing the same input SOFA reuse the already-loaded
    object (read-each-input-once optimization for chunks built from one
    source file's SR/EQ variants). Each job is wrapped in try/except so one
    failure doesn't abort its siblings; prints a `[JOB OK]`/`[JOB FAIL]`
    sentinel line per job for the caller to parse. Returns a process exit
    code (nonzero if any job failed).
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    common_ns = argparse.Namespace(**manifest.get("common", {}))
    exit_code = 0
    cached_input = None
    cached_sofa = None
    prep_cache = {}  # one-entry steps-1-6 cache; see master_sofa()

    for job in manifest.get("jobs", []):
        input_path = job.get("input")
        try:
            if input_path != cached_input:
                if not os.path.exists(input_path):
                    raise FileNotFoundError(f"Input file missing: {input_path}")
                cached_sofa = read_sofa_robust(input_path)
                cached_input = input_path

            master_sofa(cached_sofa, job["target_fs"], bool(job.get("dfeq", False)),
                        job["output"], common_ns, _prep_cache=prep_cache)
            print(f"[JOB OK] {input_path}")
        except Exception as e:
            exit_code = 1
            print(f"[JOB FAIL] {input_path}: {e}")

    return exit_code


def build_parser():
    """Construct the CLI argument parser.

    Module-level (rather than inline in main()) so tests can derive a complete
    default Namespace via `vars(build_parser().parse_args([]))` instead of
    hand-copying defaults. `--input`/`--output` are deliberately not
    argparse-`required`; main() enforces them with parser.error() after
    parsing, so parse_args([]) yields a full default namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to a JSON manifest for chunked/parallel batch processing "
                             "(runs multiple jobs in this one process; skips --input/--output)")
    parser.add_argument("--input", required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--dfeq", action="store_true", default=None)
    
    parser.add_argument("--target-fs", type=int, help="Explicit target samplerate (Hz) for standalone mode")
    parser.add_argument("--hrir-length", type=int, help="Override output HRIR length (Samples)")
    
    parser.add_argument("--lf-extend", action="store_true")
    parser.add_argument("--lf-crossover", type=float, default=1000.0)

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

    # Weighting (shared with generate_dfhrtf.py via dfhrtf config keys)
    parser.add_argument("--weighting-mode",      default="auto",
                        choices=["auto", "geometric", "uniform"],
                        help="Grid weighting: auto (design-test), geometric, or uniform 1/N")

    # ILD balancing (forwarded from sofa config keys)
    parser.add_argument("--ild-fix",       action="store_true",
                        help="Balance inter-aural level differences after mastering.")
    parser.add_argument("--ild-threshold", type=float, default=1.5,
                        help="ILD correction threshold in dB (default: 1.5).")
    parser.add_argument("--ild-low-freq",  type=float, default=300.0,
                        help="ILD estimation band low edge in Hz (default: 300.0).")
    parser.add_argument("--ild-high-freq", type=float, default=6000.0,
                        help="ILD estimation band high edge in Hz (default: 6000.0).")

    # Diagnostics: dump intermediate splice/crop arrays to an .npz. Off by
    # default; no effect on normal or GUI runs.
    parser.add_argument("--debug-dump", type=str, default=None,
                        help="Path to write an .npz of internal splice/crop arrays for diagnostics.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.manifest:
        sys.exit(run_manifest(args.manifest))

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --manifest is given")

    targets = []
    print("=== Standalone Mode: Mastering Single SOFA ===")
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file missing: {args.input}")
        sys.exit(1)

    target_sofa = read_sofa_robust(args.input)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    targets.append((target_sofa, base_name))

    print("\n=== Step 2: Mastering Outputs ===")
    
    # Define Job List based on arguments
    if args.target_fs:
        # Explicit/UI mode: Just run the one requested job
        apply_dfeq = args.dfeq if args.dfeq is not None else False
        jobs = [(args.target_fs, apply_dfeq, None)]
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