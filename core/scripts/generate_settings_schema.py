"""
core/scripts/generate_settings_schema.py

Generates  core/default_settings.schema.json  from the SCHEMA validation dict
in core/config.py.  Run this after adding or changing any SCHEMA entry:

    python -m core.scripts.generate_settings_schema

build_schema() is also importable by the pytest drift-guard test so CI can
detect any desync between SCHEMA and the committed schema file without writing
to disk.
"""

import json
import sys
from pathlib import Path

# Make `core` importable whether we're run as a module (-m) or a plain script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import SCHEMA

_SCHEMA_OUT = Path(__file__).parent.parent / "default_settings.schema.json"

# ── Descriptions ──────────────────────────────────────────────────────────────
# Human-readable descriptions for every SCHEMA key.  These are the only part
# of the schema that can't be derived mechanically; everything else (types,
# bounds, enums) is generated from config.py SCHEMA.

DESCRIPTIONS: dict[str, str] = {
    # output
    "output.default": (
        "Use per-source-folder output locations. When true, each SOFA file is "
        "written alongside its source folder; when false the single custom "
        "'folder' path is used as the shared root."
    ),
    "output.folder": (
        "Custom output root folder path.  Only used when 'default' is false."
    ),

    # sofa — surfaced settings
    "sofa.enabled": (
        "Enable the SOFA mastering stage. Input SOFA files are resampled, "
        "globally aligned, and optionally radius-normalised and DF-equalised "
        "before DFHRTF analysis."
    ),
    "sofa.samplerates": (
        "Target samplerates (Hz) to export. Each selected samplerate produces "
        "a separate mastered SOFA file."
    ),
    "sofa.phase_mode": (
        "Phase treatment for the HRIR output. 'Original Phase' preserves the "
        "measured phase response unmodified."
    ),
    "sofa.lf_extension": (
        "Extend the HRIR below the measurement's reliable low-frequency limit "
        "by blending in a synthetic min-phase signal derived from the reference "
        "LF curve (data/lf_ref_mag.txt)."
    ),
    "sofa.xover": (
        "Crossover frequency (Hz) for LF extension blending. The synthetic LF "
        "signal is spliced in below this frequency; blend width is controlled "
        "by advanced.blend_low_oct / blend_high_oct."
    ),
    "sofa.uneq": (
        "Output an un-equalized (raw diffuse-field) SOFA file in addition to "
        "the DF-equalized variant."
    ),
    "sofa.dfeq": (
        "Apply diffuse-field equalization — each HRIR is divided by the "
        "geometry-weighted DF average response so the net speaker-to-ear "
        "transfer is flat on average."
    ),
    "sofa.normalize_distance": (
        "Harmonize all HRIRs to a common listening radius using physics-based "
        "delay correction (sample-accurate fractional delay) and inverse-square "
        "gain scaling."
    ),
    "sofa.distance_m": (
        "Target listening radius in metres for distance normalization. "
        "Typical values are 1.0 – 2.0 m."
    ),
    "sofa.hrir_length": (
        "Output HRIR length in samples after processing and windowing. "
        "Shorter lengths reduce CPU/memory cost at the expense of LF resolution."
    ),
    "sofa.ild_fix": (
        "Balance inter-aural level differences (ILD) in mastered SOFA files. "
        "Applies a split-level broadband scalar correction when the estimated "
        "diffuse-field ILD exceeds ild_threshold."
    ),
    "sofa.ild_threshold": (
        "ILD correction threshold in dB. Files whose estimated diffuse-field "
        "ILD (300–6000 Hz mean) exceeds this value are corrected. Calibration "
        "offsets are typically ≥ 1.5 dB; acoustic ILD is usually < 0.5 dB."
    ),
    "sofa.naming": (
        "Output filename template for mastered SOFA files. "
        "Supported variables: {folder}, {filename}, {samplerate}, {eq}."
    ),

    # sofa — advanced tunables
    "sofa.advanced.processing_length": (
        "Internal FFT processing length in samples. Must be ≥ hrir_length; "
        "larger values reduce time-domain aliasing during convolution steps."
    ),
    "sofa.advanced.norm_target_db": (
        "Peak normalization target level in dBFS applied to each output SOFA "
        "file after all processing."
    ),
    "sofa.advanced.onset_buffer": (
        "Extra samples prepended before the detected HRIR onset when aligning "
        "impulse responses. Provides headroom for pre-ringing."
    ),
    "sofa.advanced.onset_threshold_ratio": (
        "Fraction of the HRIR's peak amplitude used to detect the onset sample "
        "(first sample that exceeds peak × threshold)."
    ),
    "sofa.advanced.blend_low_oct": (
        "Lower fade width in octaves below sofa.xover. The synthetic LF signal "
        "starts blending in at xover / 2^blend_low_oct."
    ),
    "sofa.advanced.blend_high_oct": (
        "Upper fade width in octaves above sofa.xover. The original signal is "
        "fully restored at xover × 2^blend_high_oct."
    ),
    "sofa.advanced.ild_low_freq": (
        "Lower edge (Hz) of the frequency band used to estimate the "
        "diffuse-field ILD for correction. Default 300 Hz."
    ),
    "sofa.advanced.ild_high_freq": (
        "Upper edge (Hz) of the frequency band used to estimate the "
        "diffuse-field ILD for correction. Default 6000 Hz."
    ),

    # dfhrtf — surfaced settings
    "dfhrtf.enabled": (
        "Enable the DFHRTF extraction stage. Computes geometry-weighted "
        "spectral averages from SOFA files, applies smoothing and tilt, and "
        "writes text and/or graph outputs."
    ),
    "dfhrtf.per_sofa_lr": (
        "Write separate per-channel (L and R) DFHRTF text files for each "
        "SOFA file."
    ),
    "dfhrtf.per_sofa_avg": (
        "Write a single averaged DFHRTF text file per SOFA file, combining L "
        "and R via the algorithm selected in per_sofa_avg_algo."
    ),
    "dfhrtf.per_sofa_avg_algo": (
        "Algorithm used to average the L and R DFHRTF curves when writing "
        "per-SOFA average outputs."
    ),
    "dfhrtf.text_filename": (
        "Text output filename template. "
        "Supported variables: {folder}, {filename}, {samplerate}, {tilt}, {LR}."
    ),
    "dfhrtf.text_delimiter": (
        "Column delimiter written in DFHRTF text output files."
    ),
    "dfhrtf.text_extension": (
        "File extension for DFHRTF text output files."
    ),
    "dfhrtf.dfhrtf_graphs": (
        "Generate a DFHRTF magnitude plot for each processed SOFA file."
    ),
    "dfhrtf.weighted_grid": (
        "Generate a geometry-weighted measurement grid visualization plot."
    ),
    "dfhrtf.weighted_grid_scope": (
        "Whether the weighted grid plot is computed per source folder or once "
        "across all input files."
    ),
    "dfhrtf.graph_filename": (
        "Graph output filename template. "
        "Supported variables: {folder}, {filename}, {samplerate}, {tilt}, "
        "{graphtype}."
    ),
    "dfhrtf.graph_extension": (
        "Image format for graph output files."
    ),
    "dfhrtf.smoothing": (
        "Fractional-octave smoothing bandwidth applied to the DFHRTF magnitude "
        "curve (e.g. 24 = 1/24-octave smoothing)."
    ),
    "dfhrtf.tilt": (
        "Spectral tilt applied to the DFHRTF output in dB/octave. Negative "
        "values roll off the high frequencies; 0.0 means no tilt."
    ),
    "dfhrtf.extend_lf": (
        "Extend the DFHRTF below its reliable LF limit by blending in the "
        "reference LF curve (data/lf_ref_mag.txt)."
    ),
    "dfhrtf.xover": (
        "Crossover frequency (Hz) for DFHRTF LF extension blending."
    ),
    "dfhrtf.avg_all": (
        "Compute a global average DFHRTF across all processed SOFA files after "
        "per-file extraction completes."
    ),
    "dfhrtf.algo": (
        "Algorithm used for global multi-file averaging."
    ),
    "dfhrtf.exclude_outliers": (
        "Remove curves whose pairwise Soft-DTW distance from the median exceeds "
        "averaging.outlier_threshold_std standard deviations before global "
        "averaging."
    ),

    # dfhrtf — advanced tunables
    "dfhrtf.advanced.n_fft": (
        "FFT size for DFHRTF spectral computation. Larger values give finer "
        "frequency resolution but increase processing time."
    ),
    "dfhrtf.advanced.output_octave_fraction": (
        "Output frequency grid resolution in fractional octaves for DFHRTF text "
        "files (e.g. 48 = 1/48-octave spacing)."
    ),
    "dfhrtf.advanced.norm_freq_hz": (
        "Reference frequency (Hz) at which the DFHRTF magnitude is normalized "
        "to 0 dB before writing outputs."
    ),
    "dfhrtf.advanced.lf_blend_low_oct": (
        "Lower fade width in octaves below dfhrtf.xover for DFHRTF LF "
        "extension blending."
    ),
    "dfhrtf.advanced.lf_blend_high_oct": (
        "Upper fade width in octaves above dfhrtf.xover for DFHRTF LF "
        "extension blending."
    ),
    "dfhrtf.advanced.stereo_angle_deg": (
        "Azimuth angle (degrees) used when extracting L/R off-axis curves for "
        "stereo graph outputs."
    ),

    # averaging
    "averaging.num_points": (
        "Number of log-spaced frequency points onto which all DFHRTF curves "
        "are resampled before averaging."
    ),
    "averaging.outlier_threshold_std": (
        "Outlier rejection threshold in standard deviations above the median "
        "pairwise distance. Curves exceeding median + N×std are removed."
    ),
    "averaging.softdtw_gamma_persofa": (
        "Soft-DTW gamma parameter for per-SOFA L+R barycenter averaging. "
        "Smaller values approach hard-DTW; larger values approach Euclidean "
        "mean."
    ),
    "averaging.softdtw_gamma_global": (
        "Soft-DTW gamma parameter for global multi-file barycenter averaging."
    ),
    "averaging.softdtw_max_iter": (
        "Maximum L-BFGS-B iterations for the Soft-DTW barycenter optimizer."
    ),
    "averaging.softdtw_tol": (
        "Convergence tolerance for the Soft-DTW barycenter optimizer. "
        "Iteration stops when the gradient norm falls below this value."
    ),
    "averaging.geo_arc_y_scale": (
        "Y-axis scale factor applied before arc-length parameterization in the "
        "geometric arc-length blend algorithm. Balances frequency vs. magnitude "
        "contribution to curve distance."
    ),
}

# Descriptions for intermediate object nodes (groups of settings)
_OBJECT_DESCRIPTIONS: dict[str, str] = {
    "output": "Output path configuration.",
    "sofa": "SOFA mastering stage settings.",
    "sofa.advanced": (
        "Advanced SOFA mastering tunables. Not surfaced in the UI — "
        "edit this file directly."
    ),
    "dfhrtf": "DFHRTF extraction stage settings.",
    "dfhrtf.advanced": (
        "Advanced DFHRTF extraction tunables. Not surfaced in the UI — "
        "edit this file directly."
    ),
    "averaging": (
        "Global averaging pipeline settings. Not surfaced in the UI — "
        "edit this file directly."
    ),
}


# ── Schema builder ────────────────────────────────────────────────────────────

def _spec_to_prop(spec: dict) -> dict:
    """Convert a single SCHEMA spec dict to a JSON Schema property object."""
    t = spec.get("type")
    choices = spec.get("choices")
    item_type = spec.get("item_type")
    min_val = spec.get("min")
    max_val = spec.get("max")

    prop: dict = {}

    if t is list:
        # Array field (e.g. samplerates)
        prop["type"] = "array"
        if choices is not None:
            prop["items"] = {"enum": choices}
        elif item_type is not None:
            _item_type_map = {int: "integer", float: "number", str: "string", bool: "boolean"}
            prop["items"] = {"type": _item_type_map.get(item_type, "string")}
        prop["uniqueItems"] = True
        prop["minItems"] = 1
    elif t is bool:
        prop["type"] = "boolean"
    elif t is int:
        if choices is not None:
            prop["enum"] = choices
        else:
            prop["type"] = "integer"
            if min_val is not None:
                prop["minimum"] = min_val
            if max_val is not None:
                prop["maximum"] = max_val
    elif t is float:
        if choices is not None:
            prop["enum"] = choices
        else:
            prop["type"] = "number"
            if min_val is not None:
                prop["minimum"] = min_val
            if max_val is not None:
                prop["maximum"] = max_val
    elif t is str:
        if choices is not None:
            prop["enum"] = choices
        else:
            prop["type"] = "string"
    elif isinstance(t, tuple):
        # e.g. (int, float) for hrir_length — use "number" to avoid false
        # positives on values like 512.0 that Python will happily coerce
        prop["type"] = "number"
        if min_val is not None:
            prop["minimum"] = min_val
        if max_val is not None:
            prop["maximum"] = max_val

    return prop


def _ensure_object(props: dict, key: str, obj_path: str) -> dict:
    """Return props[key], creating it as an object node if absent."""
    if key not in props:
        node: dict = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        }
        desc = _OBJECT_DESCRIPTIONS.get(obj_path)
        if desc:
            node["description"] = desc
        props[key] = node
    return props[key]


def build_schema() -> dict:
    """Build and return the full JSON Schema dict from config.py SCHEMA."""
    root_props: dict = {
        "$schema": {"type": "string"},
        "schema_version": {
            "type": "integer",
            "const": 1,
            "description": (
                "Internal version identifier used by the migration path in "
                "config.py. Do not change manually."
            ),
        },
    }

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SOFA Buddy Settings",
        "description": (
            "Factory defaults for SOFA Buddy.  Surfaced settings are managed "
            "by the UI; advanced.* and averaging.* keys must be edited here."
        ),
        "type": "object",
        "additionalProperties": False,
        "properties": root_props,
    }

    for dotted_path, spec in SCHEMA.items():
        segments = dotted_path.split(".")

        # Walk / create intermediate object nodes
        current_props = root_props
        for i, seg in enumerate(segments[:-1]):
            parent_path = ".".join(segments[: i + 1])
            obj = _ensure_object(current_props, seg, parent_path)
            current_props = obj["properties"]

        # Build the leaf property
        leaf_key = segments[-1]
        prop = _spec_to_prop(spec)
        desc = DESCRIPTIONS.get(dotted_path)
        if desc:
            prop["description"] = desc
        current_props[leaf_key] = prop

    return schema


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    schema = build_schema()
    with open(_SCHEMA_OUT, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)
        fh.write("\n")  # trailing newline
    print(f"Written: {_SCHEMA_OUT}")


if __name__ == "__main__":
    main()
