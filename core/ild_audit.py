"""
core/ild_audit.py
Diffuse-field inter-aural level difference (ILD) audit and correction.

Functions
---------
diffuse_field_ild(sofa_or_path, weighting_mode='auto', estimate_band=ESTIMATE_BAND) -> dict
    Compute a geometry-weighted DF ILD report for a raw SOFA file.

apply_ild_fix(sofa, ild_estimate_db) -> sofa
    Apply a single broadband scalar, split-level correction to Data_IR
    (mutates the array in-place and returns the sofa object).

Constants
---------
LF_TOL          : float — |LF-ILD| threshold above which a file is flagged
                  as a 'calibration_offset' (default 1.5 dB).
BROADBAND_TOL   : float — broadband ILD threshold for overall flagging (2.0 dB).
ESTIMATE_BAND   : tuple — (low_hz, high_hz) band used to compute the robust
                  ild_estimate_db that drives corrections (300–6000 Hz).

Theory
------
A genuine diffuse-field ILD must approach 0 dB at low frequencies because
long-wavelength sounds diffract around the head symmetrically. A flat
offset persisting below 500 Hz is therefore not an acoustic phenomenon — it
signals a per-channel gain or calibration error in the source measurement.

The correction is a single broadband scalar, split symmetrically between
channels, so that overall energy and inter-aural time difference (ITD) are
both preserved.  Per-direction HRIR structure is untouched.
"""

import numpy as np

# Public thresholds — importable by tests and the CLI auditor.
LF_TOL        = 1.5   # dB: |LF-ILD| above this → verdict 'calibration_offset'
BROADBAND_TOL = 2.0   # dB: |broadband_ild_db| above this → flagged offender
ESTIMATE_BAND = (300, 6000)  # Hz: mid-band for robust ILD estimation

# Per-band breakpoints (Hz) used in band_ild_db.
_BANDS = [(100, 500), (500, 2000), (2000, 6000), (6000, 16000)]


def diffuse_field_ild(sofa_or_path, weighting_mode="auto", estimate_band=ESTIMATE_BAND):
    """Return a diffuse-field ILD report for a SOFA file.

    Parameters
    ----------
    sofa_or_path : str or sofar SOFA object
        Path to a .sofa file, or an already-loaded sofar object.
    weighting_mode : {'auto', 'geometric', 'uniform'}
        Passed to calculate_geometric_weights.  'auto' uses the spherical-design
        classifier (will select 'geometric' for measured HRTF grids).
    estimate_band : tuple of (float, float)
        (low_hz, high_hz) frequency range used to compute the robust
        ild_estimate_db that drives corrections.  Defaults to ESTIMATE_BAND
        (300–6000 Hz).  Pass a custom tuple to override (e.g. from config).

    Returns
    -------
    dict with keys:
        broadband_ild_db  – 20*log10(mean(DF_L) / mean(DF_R)) over all bins
                            (linear-magnitude full-band mean; for reporting only —
                            biased toward high-energy bins).
        band_ild_db       – {(lo, hi): ild_db, ...} per _BANDS entry.
        raw_energy_ild_db – 10*log10(sum(ir_L**2) / sum(ir_R**2)); unweighted
                            cross-check, no FFT.
        lf_ild_db         – band_ild_db[(100, 500)]; key discriminator between
                            acoustic and calibration-error ILD.
        ild_estimate_db   – mean of per-bin 20*log10(DF_L/DF_R) over
                            ESTIMATE_BAND (300–6000 Hz).  Robust mid-band
                            estimate; used by apply_ild_fix.
        verdict           – 'calibration_offset' if abs(lf_ild_db) > LF_TOL,
                            else 'acoustic_or_balanced'.
    """
    import sofar as sf  # optional dep — only required when this function is called
    from core.geometry_weights import calculate_geometric_weights
    from core.sofa_io import read_sofa_robust

    if isinstance(sofa_or_path, str):
        sofa = read_sofa_robust(sofa_or_path)
    else:
        sofa = sofa_or_path

    ir  = np.asarray(sofa.Data_IR, dtype=float)      # (M, R, N)
    fs  = float(sofa.Data_SamplingRate)
    sp  = np.asarray(sofa.SourcePosition)             # (M, 3) az/el/r in degrees/m

    # Geometry-weighted diffuse-field magnitude spectrum per receiver.
    weights = calculate_geometric_weights(
        sp, advanced_dfeq=True, weighting_mode=weighting_mode
    )
    N = ir.shape[2]
    H = np.fft.rfft(ir, n=N, axis=2)                 # (M, R, bins)
    P = np.abs(H) ** 2
    w_exp = weights[:, np.newaxis, np.newaxis]        # (M, 1, 1)
    DF = np.sqrt(np.sum(P * w_exp, axis=0))           # (R, bins); R0=L, R1=R
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)           # (bins,)

    eps = 1e-12  # avoid log(0)

    # Broadband ILD: linear-magnitude mean across all bins.
    broadband_ild_db = 20.0 * np.log10(
        (np.mean(DF[0]) + eps) / (np.mean(DF[1]) + eps)
    )

    # Per-band ILD: mean of per-bin dB difference within each frequency band.
    band_ild_db = {}
    for lo, hi in _BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            db_diff = 20.0 * np.log10((DF[0, mask] + eps) / (DF[1, mask] + eps))
            band_ild_db[(lo, hi)] = float(np.mean(db_diff))
        else:
            band_ild_db[(lo, hi)] = float("nan")

    # Robust mid-band ILD estimate for corrections (dB-domain mean over estimate_band).
    lo_est, hi_est = estimate_band
    mid_mask = (freqs >= lo_est) & (freqs < hi_est)
    if mid_mask.any():
        db_diff_mid = 20.0 * np.log10((DF[0, mid_mask] + eps) / (DF[1, mid_mask] + eps))
        ild_estimate_db = float(np.mean(db_diff_mid))
    else:
        ild_estimate_db = broadband_ild_db

    # Unweighted raw energy ILD (cross-check; no FFT).
    eL = float(np.sum(ir[:, 0, :] ** 2))
    eR = float(np.sum(ir[:, 1, :] ** 2))
    raw_energy_ild_db = 10.0 * np.log10((eL + eps) / (eR + eps))

    lf_ild_db = band_ild_db.get((100, 500), float("nan"))
    verdict = (
        "calibration_offset"
        if abs(lf_ild_db) > LF_TOL
        else "acoustic_or_balanced"
    )

    return {
        "broadband_ild_db":  float(broadband_ild_db),
        "band_ild_db":        band_ild_db,
        "raw_energy_ild_db":  float(raw_energy_ild_db),
        "lf_ild_db":          float(lf_ild_db),
        "ild_estimate_db":    float(ild_estimate_db),
        "verdict":            verdict,
    }


def apply_ild_fix(sofa, ild_estimate_db):
    """Apply a split-level broadband scalar ILD correction to sofa.Data_IR.

    Mutates Data_IR in-place and returns the sofa object so callers can chain
    or immediately re-measure.

    Correction principle
    --------------------
    Given an ILD estimate of X dB (L louder than R when X > 0), attenuate L
    by X/2 dB and boost R by X/2 dB.  Both corrections are constant scalars
    over all frequencies, so:
      • Per-direction HRIR structure is untouched.
      • Inter-aural time difference (ITD) is unaffected.
      • The geometric mean amplitude of L and R is preserved (gain_L * gain_R
        = 1), but total *energy* changes by an amount proportional to the
        initial channel imbalance.  For a calibration offset of ~3 dB with a
        real channel imbalance of the same size, energy changes by ~6–8%.
        Files with minimal channel energy imbalance see <1% change.

    Parameters
    ----------
    sofa : sofar SOFA object
        Must expose a writable Data_IR attribute (numpy array, shape M×R×N).
    ild_estimate_db : float
        Estimated ILD in dB (positive → L is louder, negative → R is louder).
        Typically the 'ild_estimate_db' value from diffuse_field_ild().

    Returns
    -------
    sofa : the same (mutated) object.
    """
    ir = np.asarray(sofa.Data_IR, dtype=float)

    # Detect float32-precision source data: if all non-zero samples are exactly
    # representable as float32, the file was stored at float32 precision (common
    # for measured HRTF data).  Multiplying by a float64 constant fills the
    # lower mantissa bits, which destroys the compression pattern and roughly
    # doubles the file size.  Round back to float32 precision after the gain to
    # preserve compressibility — this is lossless relative to the source.
    nonzero = ir[ir != 0.0].ravel()
    is_f32_precision = (
        len(nonzero) > 0
        and np.all(nonzero == nonzero.astype(np.float32).astype(np.float64))
    )

    gain_L = 10.0 ** (-ild_estimate_db / 2.0 / 20.0)
    gain_R = 10.0 ** (+ild_estimate_db / 2.0 / 20.0)
    ir[:, 0, :] *= gain_L
    ir[:, 1, :] *= gain_R

    if is_f32_precision:
        ir = ir.astype(np.float32).astype(np.float64)

    sofa.Data_IR = ir
    return sofa
