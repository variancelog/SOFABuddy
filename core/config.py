"""
core/config.py

Cascaded JSON configuration management for SOFA Buddy.

  default_settings.json  (bundled alongside this file, read-only)
      Full schema with factory defaults — used as fallback and "Reset" target.

  user_settings.json  (per-user OS app-config dir, written by the UI)
      Live session state.  Written on app close and on Start Processing.
      The imported-files list is NEVER written here.

Qt-free API  (safe to import in subprocess scripts without a QApplication):
  load_defaults() -> dict
  schema_default(section, key) -> value
  load_merged(user_path=None) -> (cfg, warnings)
  validate(cfg) -> (cfg, warnings)

Qt-dependent API  (requires a running QApplication):
  manager() -> ConfigManager   — process-wide singleton
"""

import copy
import json
import os
import tempfile
from pathlib import Path

_DEFAULTS_PATH = Path(__file__).parent / "default_settings.json"
_SCHEMA_PATH   = Path(__file__).parent / "default_settings.schema.json"

# Keys removed from the settings; stale copies in a user's saved file are
# dropped on load. NOT migrated: the replacement key keeps its factory default.
_RETIRED_KEYS = (
    "dfhrtf.advanced.variable_smoothing_mean_mode",  # now dfhrtf.advanced.smoothing_mean_mode
    # Replaced by dfhrtf.advanced.variable_smoothing_profile. These DO hold tuned
    # user values, so _migrate() converts them (schema v1 -> v2) before this drop.
    "dfhrtf.advanced.variable_smoothing_freq_low",
    "dfhrtf.advanced.variable_smoothing_freq_hi",
    "dfhrtf.advanced.variable_smoothing_n_low",
    "dfhrtf.advanced.variable_smoothing_n_hi",
    "dfhrtf.advanced.variable_smoothing_gamma",
    # Replaced by the adaptive hybrid_s_factor / hybrid_s_min / hybrid_s_max.
    # Never shipped in a release, so no migration or schema bump.
    "dfhrtf.advanced.hybrid_spline_smoothness",
)

SCHEMA_VERSION = 2

# ── JSON-schema bounds reader ─────────────────────────────────────────────────

_schema_cache: dict | None = None


def _load_json_schema() -> dict:
    """Load + cache default_settings.schema.json.  Returns {} if unreadable."""
    global _schema_cache
    if _schema_cache is None:
        try:
            with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
                _schema_cache = json.load(fh)
        except (OSError, json.JSONDecodeError):
            _schema_cache = {}
    return _schema_cache


def schema_bounds(dotted_path: str, fallback: tuple = (None, None)) -> tuple:
    """Return (minimum, maximum) for a dotted config path from the JSON schema.

    Walks the nested ``properties`` tree; returns *fallback* if the path is
    absent or the schema file cannot be read.  Safe to call at module import
    time — the schema file is never a hard runtime dependency.
    """
    node = _load_json_schema()
    for key in dotted_path.split("."):
        props = node.get("properties") if isinstance(node, dict) else None
        if not isinstance(props, dict) or key not in props:
            return fallback
        node = props[key]
    if not isinstance(node, dict):
        return fallback
    return (node.get("minimum", fallback[0]), node.get("maximum", fallback[1]))


# ── Factory-default reader ────────────────────────────────────────────────────

_defaults_cache: dict | None = None


def _load_defaults_cached() -> dict:
    """Load + cache default_settings.json.  Returns {} if unreadable."""
    global _defaults_cache
    if _defaults_cache is None:
        try:
            _defaults_cache = load_defaults()
        except (OSError, json.JSONDecodeError):
            _defaults_cache = {}
    return _defaults_cache


def schema_default(section: str, key: str, fallback=None):
    """Return the factory default for ``<section>.<key>`` from default_settings.json.

    *section* and *key* may each be dotted (e.g. ``schema_default("dfhrtf",
    "advanced.n_fft")``).  Returns *fallback* if the path is absent or the
    defaults file cannot be read.

    The returned value is a ``deepcopy`` — mutable defaults (e.g.
    ``sofa.samplerates``) can be handed to callers without risk of the cached
    factory copy being mutated.
    """
    value, found = _get_nested(_load_defaults_cached(), f"{section}.{key}")
    if not found:
        return fallback
    return copy.deepcopy(value)


# ── Validation schema ─────────────────────────────────────────────────────────
# Each dotted-path key maps to a spec dict with zero or more of:
#   type        : expected Python type (or tuple of types)
#   item_type   : for list values, the expected element type
#   choices     : valid values (or, for list, valid element values)
#   min / max   : numeric range (inclusive); value is clamped, not reverted
# A missing key in the config is silently ignored during validation (the merged
# defaults already supplied it).

# Resolve xover bounds from the schema once at import time.
# Literals here are the fallback used only if the schema file is unreadable.
_SOFA_XOVER   = schema_bounds("sofa.xover",   (200.0, 500.0))
_DFHRTF_XOVER = schema_bounds("dfhrtf.xover", (200.0, 500.0))

SCHEMA: dict[str, dict] = {
    # -- output --
    "output.default":  {"type": bool},
    "output.folder":   {"type": str},

    # -- sofa (surfaced settings) --
    "sofa.enabled":             {"type": bool},
    "sofa.samplerates":         {"type": list, "item_type": int,
                                  "choices": [44100, 48000, 96000]},
    # Inert placeholder (only one choice exists today) — consider surfacing
    # additional phase modes as real options in settings.json when added.
    "sofa.phase_mode":          {"type": str,
                                  "choices": ["Original Phase"]},
    "sofa.lf_extension":        {"type": bool},
    "sofa.xover":               {"type": float, "min": _SOFA_XOVER[0],   "max": _SOFA_XOVER[1]},
    "sofa.uneq":                {"type": bool},
    "sofa.dfeq":                {"type": bool},
    "sofa.normalize_distance":  {"type": bool},
    "sofa.distance_m":          {"type": float, "min": 1.0, "max": 2.0},
    "sofa.hrir_length":         {"type": (int, float), "min": 128, "max": 1024},
    "sofa.ild_fix":             {"type": bool},
    "sofa.ild_threshold":       {"type": float, "min": 0.1, "max": 3.0},
    "sofa.naming":              {"type": str},

    # -- sofa.advanced (hidden tunables) --
    "sofa.advanced.processing_length":      {"type": int, "min": 512, "max": 8192},
    "sofa.advanced.norm_target_db":         {"type": float, "min": -12.0, "max": 0.0},
    "sofa.advanced.onset_buffer":           {"type": int, "min": 0, "max": 256},
    "sofa.advanced.onset_threshold_ratio":  {"type": float, "min": 0.001, "max": 0.5},
    "sofa.advanced.blend_low_oct":          {"type": float, "min": 0.1, "max": 2.0},
    "sofa.advanced.blend_high_oct":         {"type": float, "min": 0.1, "max": 2.0},
    "sofa.advanced.ild_low_freq":           {"type": float, "min": 20.0,   "max": 1000.0},
    "sofa.advanced.ild_high_freq":          {"type": float, "min": 1000.0, "max": 20000.0},

    # -- dfhrtf (surfaced settings) --
    "dfhrtf.enabled":             {"type": bool},
    "dfhrtf.per_sofa_lr":         {"type": bool},
    "dfhrtf.per_sofa_avg":        {"type": bool},
    "dfhrtf.per_sofa_avg_algo":   {"type": str,
                                    "choices": ["Arithmetic Mean", "Geo-Arc-Length", "Soft-DTW"]},
    "dfhrtf.per_sofa_avg_filename": {"type": str},
    "dfhrtf.text_filename":         {"type": str},
    "dfhrtf.avg_filename":          {"type": str},
    "dfhrtf.avg_graph_filename":    {"type": str},
    "dfhrtf.text_delimiter":      {"type": str,
                                    "choices": ["Tab", "Comma", "Semicolon", "Space"]},
    "dfhrtf.text_extension":      {"type": str, "choices": ["txt", "csv"]},
    "dfhrtf.dfhrtf_graphs":       {"type": bool},
    "dfhrtf.weighting_mode":      {"type": str,
                                    "choices": ["Auto", "Geometric", "Uniform"]},
    "dfhrtf.weighted_grid":       {"type": bool},
    "dfhrtf.weighted_grid_scope": {"type": str,
                                    "choices": ["Per Folder", "All Files"]},
    "dfhrtf.graph_filename":      {"type": str},
    "dfhrtf.graph_extension":     {"type": str, "choices": ["png", "jpg"]},
    "dfhrtf.smoothing":           {"type": str,
                                    "choices": ["Variable", "Hybrid", "1/3", "1/6", "1/12", "1/24", "1/48"]},
    "dfhrtf.tilt":                {"type": float, "min": -2.0, "max": 1.0},
    "dfhrtf.extend_lf":           {"type": bool},
    "dfhrtf.xover":               {"type": float, "min": _DFHRTF_XOVER[0], "max": _DFHRTF_XOVER[1]},
    "dfhrtf.avg_all":             {"type": bool},
    "dfhrtf.algo":                {"type": str,
                                    "choices": ["Arithmetic Mean", "Geo-Arc-Length", "Soft-DTW"]},
    "dfhrtf.exclude_outliers":    {"type": bool},

    # -- dfhrtf.advanced (hidden tunables) --
    "dfhrtf.advanced.n_fft":                {"type": int, "min": 1024, "max": 65536},
    "dfhrtf.advanced.output_octave_fraction": {"type": int, "min": 1, "max": 96},
    "dfhrtf.advanced.norm_freq_hz":          {"type": float, "min": 100.0, "max": 20000.0},
    "dfhrtf.advanced.lf_blend_low_oct":      {"type": float, "min": 0.1, "max": 2.0},
    "dfhrtf.advanced.lf_blend_high_oct":     {"type": float, "min": 0.1, "max": 2.0},
    "dfhrtf.advanced.stereo_angle_deg":      {"type": float, "min": 0.0, "max": 90.0},

    "dfhrtf.advanced.weighted_grid_scale_mode":    {"type": str, "choices": ["auto", "manual"]},
    "dfhrtf.advanced.weighted_grid_rel_low":       {"type": float, "min": 0.0, "max": 10.0},
    "dfhrtf.advanced.weighted_grid_rel_high":      {"type": float, "min": 0.0, "max": 10.0},
    "dfhrtf.advanced.preserve_lr_balance":         {"type": bool},

    # A list of {freq, n, gamma?} objects: validated by _validate_smoothing_profile()
    # (the generic machinery only handles lists of scalars), and described in the
    # generated JSON schema by a hand-written fragment (generate_settings_schema.py).
    "dfhrtf.advanced.variable_smoothing_profile":   {"type": list, "custom": "smoothing_profile"},
    "dfhrtf.advanced.smoothing_window":             {"type": str,   "choices": ["gaussian", "flat"]},
    "dfhrtf.advanced.smoothing_mean_mode":          {"type": str,   "choices": ["power", "db", "cubic"]},

    # Hybrid smoothing (core/hybrid_smoothing.py); ranges mirror HybridParams.
    "dfhrtf.advanced.hybrid_xover_hz":              {"type": float, "min": 5500.0, "max": 9000.0},
    "dfhrtf.advanced.hybrid_xover_width_oct":       {"type": float, "min": 0.01, "max": 0.5},
    "dfhrtf.advanced.hybrid_tv_weight":             {"type": float, "min": 0.05, "max": 5.0},
    "dfhrtf.advanced.hybrid_s_factor":              {"type": float, "min": 0.5, "max": 10.0},
    "dfhrtf.advanced.hybrid_s_min":                 {"type": float, "min": 0.5, "max": 30.0},
    "dfhrtf.advanced.hybrid_s_max":                 {"type": float, "min": 0.5, "max": 30.0},
    "dfhrtf.advanced.hybrid_huber_window":          {"type": int,   "choices": [5, 7, 9, 11, 13, 15]},
    "dfhrtf.advanced.hybrid_huber_delta":           {"type": float, "min": 0.05, "max": 0.30},
    "dfhrtf.advanced.hybrid_huber_iter":            {"type": int,   "min": 1, "max": 20},

    # -- averaging --
    "averaging.num_points":            {"type": int, "min": 64, "max": 4096},
    "averaging.outlier_threshold_std": {"type": float, "min": 0.1, "max": 10.0},
    "averaging.softdtw_gamma_persofa": {"type": float, "min": 0.0001, "max": 10.0},
    "averaging.softdtw_gamma_global":  {"type": float, "min": 0.0001, "max": 10.0},
    "averaging.softdtw_max_iter":      {"type": int, "min": 5, "max": 200},
    "averaging.softdtw_tol":           {"type": float, "min": 1e-8, "max": 1.0},
    "averaging.geo_arc_y_scale":       {"type": float, "min": 0.0001, "max": 10.0},

    # -- processing (parallel/chunked batch execution) --
    "processing.max_workers": {"type": int, "min": 0, "max": 64},
    "processing.chunk_size":  {"type": int, "min": 1, "max": 256},
}


# ── Qt-free helpers ───────────────────────────────────────────────────────────

def load_defaults() -> dict:
    """Load and return the bundled default_settings.json (read-only source of truth)."""
    with open(_DEFAULTS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* on top of *base*.  Returns a new deep copy."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        elif key in result and isinstance(result[key], dict):
            continue    # a non-object (null, number, ...) where a section belongs: keep the defaults
        else:
            result[key] = copy.deepcopy(val)
    return result


def _get_nested(cfg: dict, dotted_path: str):
    """Return (value, found) for a dotted-path key in a nested dict."""
    node = cfg
    for k in dotted_path.split("."):
        if not isinstance(node, dict) or k not in node:
            return None, False
        node = node[k]
    return node, True


def _set_nested(cfg: dict, dotted_path: str, value) -> None:
    """Write *value* into a nested dict at *dotted_path*, creating sub-dicts as needed."""
    keys = dotted_path.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def validate(cfg: dict, notices: list | None = None) -> tuple:
    """
    Walk the SCHEMA against *cfg* (a deep copy is made internally) and:
      - coerce borderline numeric types where possible
      - clamp out-of-range numerics to [min, max]
      - revert invalid choices / un-coercible types to the default value
      - filter invalid items from list-with-choices fields
      - drop the 'files' key if present (never persisted)
      - drop retired keys (see _RETIRED_KEYS)

    Returns (sanitized_cfg, warnings) where warnings is a list[str].  If a
    *notices* list is given, schema migrations append their user-facing messages
    to it (see _migrate()).
    """
    defaults = load_defaults()
    warnings: list = []
    if not isinstance(cfg, dict):
        warnings.append("Config: settings were not a JSON object, using defaults.")
        cfg = {}
    cfg = copy.deepcopy(cfg)

    # Schema-version migration (v1 -> v2: variable-smoothing profile). Never
    # raises: a file v0.7.0 tolerated must still start the app.
    from_version = _coerce_version(cfg.get("schema_version"))
    if from_version < SCHEMA_VERSION:
        try:
            cfg = _migrate(cfg, from_version, notices)
        except Exception as exc:    # defensive: odd hand-edited values
            warnings.append(f"Config: could not migrate old settings ({exc}).")
            cfg["schema_version"] = SCHEMA_VERSION
    else:
        cfg["schema_version"] = from_version

    for path, spec in SCHEMA.items():
        value, found = _get_nested(cfg, path)
        if not found:
            continue  # Absent keys are fine — defaults already merged in

        default_val, _ = _get_nested(defaults, path)

        if spec.get("custom") == "smoothing_profile":
            from core.smoothing import profile_from_dicts, profile_to_dicts
            try:
                value = profile_to_dicts(profile_from_dicts(value))
            except ValueError as exc:
                warnings.append(
                    f"Config: '{path}' is invalid ({exc}), reverted to default.")
                value = copy.deepcopy(default_val)
            _set_nested(cfg, path, value)
            continue

        expected_type = spec.get("type")
        choices      = spec.get("choices")
        item_type    = spec.get("item_type")
        min_val      = spec.get("min")
        max_val      = spec.get("max")
        fixed        = False

        # 1. Type check / coercion
        if expected_type is not None and not fixed:
            if isinstance(expected_type, tuple):
                type_ok = isinstance(value, expected_type)
            elif expected_type == list:
                type_ok = isinstance(value, list)
            else:
                type_ok = isinstance(value, expected_type)

            if not type_ok:
                try:
                    if expected_type == float:
                        value = float(value)
                    elif expected_type == int:
                        value = int(float(value))  # handle "512.0" -> 512
                    elif expected_type == bool:
                        # Don't silently accept 0/1 as bool — must already be bool
                        if not isinstance(value, bool):
                            raise TypeError
                    else:
                        raise TypeError
                except (TypeError, ValueError):
                    warnings.append(
                        f"Config: '{path}' has wrong type "
                        f"(got {type(value).__name__}), reverted to default."
                    )
                    value = default_val
                    fixed = True

        # 2. List item type coercion / filtering (e.g. samplerates)
        if not fixed and isinstance(value, list) and item_type is not None:
            coerced = []
            for item in value:
                try:
                    coerced.append(item_type(item))
                except (TypeError, ValueError):
                    pass
            value = coerced

        # 3. Choices validation
        if not fixed and choices is not None:
            if isinstance(value, list):
                filtered = [v for v in value if v in choices]
                if len(filtered) != len(value):
                    removed = [v for v in value if v not in choices]
                    warnings.append(
                        f"Config: '{path}' had invalid items {removed!r}, filtered out."
                    )
                value = filtered
            else:
                if value not in choices:
                    warnings.append(
                        f"Config: '{path}' = {value!r} is not a valid choice "
                        f"(expected one of {choices}), reverted to default."
                    )
                    value = default_val
                    fixed = True

        # 4. Numeric range clamping
        if not fixed and isinstance(value, (int, float)):
            if min_val is not None and value < min_val:
                warnings.append(
                    f"Config: '{path}' = {value} is below minimum {min_val}, clamped."
                )
                value = type(value)(min_val)
            if max_val is not None and value > max_val:
                warnings.append(
                    f"Config: '{path}' = {value} is above maximum {max_val}, clamped."
                )
                value = type(value)(max_val)

        _set_nested(cfg, path, value)

    # hybrid_s_min must not exceed hybrid_s_max; clamping can't repair a swapped
    # pair, so put both back to the defaults.
    s_lo, lo_found = _get_nested(cfg, "dfhrtf.advanced.hybrid_s_min")
    s_hi, hi_found = _get_nested(cfg, "dfhrtf.advanced.hybrid_s_max")
    if lo_found and hi_found and s_lo > s_hi:
        warnings.append(
            f"Config: 'dfhrtf.advanced.hybrid_s_min' = {s_lo} exceeds "
            f"'dfhrtf.advanced.hybrid_s_max' = {s_hi}, both reverted to defaults.")
        for key in ("hybrid_s_min", "hybrid_s_max"):
            _set_nested(cfg, f"dfhrtf.advanced.{key}", _get_nested(defaults, f"dfhrtf.advanced.{key}")[0])

    # Never persist the files list
    cfg.pop("files", None)

    # Keys that no longer exist are dropped silently, not migrated.
    for old_path in _RETIRED_KEYS:
        *parents, leaf = old_path.split(".")
        node, found = _get_nested(cfg, ".".join(parents))
        if found and isinstance(node, dict):
            node.pop(leaf, None)

    return cfg, warnings


# v1 (v0.7.0) variable-smoothing keys and their factory values, for migration.
_V1_VARIABLE_DEFAULTS = {
    "freq_low": 350.0, "freq_hi": 6000.0, "n_low": 2.0, "n_hi": 12.0, "gamma": 0.88,
}
# v1 factory filename templates, replaced by ones containing {smoothing}.
_V1_TEMPLATE_DEFAULTS = {
    "text_filename": "{filename} {LR}",
    "graph_filename": "{graphtype} {filename}",
}
# The new default profile's 14 kHz -> 17.5 kHz tail (gamma on the arrival point).
_TAIL_POINTS = ((14000.0, 1.0), (17500.0, 1.6))


def _coerce_version(value) -> int:
    """schema_version as an int. A missing key or anything unusable (null, bool,
    junk text, a list) means the old format, version 1."""
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 1
    return 1


def _migrate(cfg: dict, from_version: int, notices: list | None = None) -> dict:
    """Bring a loaded config up to SCHEMA_VERSION.  Mutates and returns *cfg*.

    v1 -> v2: the five variable_smoothing_{freq_low,freq_hi,n_low,n_hi,gamma}
    keys become the variable_smoothing_profile point list, and the two old
    default filename templates gain {smoothing}.  User-facing messages are
    appended to *notices*. A file without a schema_version is the old format and
    migrates like v1.
    """
    if from_version < 2:
        _migrate_v1_to_v2(cfg, notices)
    cfg["schema_version"] = SCHEMA_VERSION
    return cfg


def _migrate_v1_to_v2(cfg: dict, notices: list | None) -> None:
    from core.smoothing import (
        format_profile, profile_from_dicts, profile_to_dicts,
    )

    defaults = load_defaults()
    # Sections that aren't objects (null, a number, ...) count as absent.
    if not isinstance(cfg.get("dfhrtf"), dict):
        cfg["dfhrtf"] = copy.deepcopy(defaults["dfhrtf"])
    dfhrtf = cfg["dfhrtf"]
    if not isinstance(dfhrtf.get("advanced"), dict):
        dfhrtf["advanced"] = copy.deepcopy(defaults["dfhrtf"]["advanced"])
    adv = dfhrtf["advanced"]

    # -- Variable-smoothing keys -> profile --
    old_present = any(f"variable_smoothing_{k}" in adv for k in _V1_VARIABLE_DEFAULTS)
    old = {}
    for k, factory in _V1_VARIABLE_DEFAULTS.items():
        try:
            old[k] = float(adv.get(f"variable_smoothing_{k}", factory))
        except (TypeError, ValueError):
            old[k] = factory

    default_profile = copy.deepcopy(defaults["dfhrtf"]["advanced"]["variable_smoothing_profile"])
    detail = ""
    if not old_present or old == _V1_VARIABLE_DEFAULTS:
        adv["variable_smoothing_profile"] = default_profile
        detail = ("Your Variable smoothing now uses the new default profile, which "
                  "adds extra smoothing from 14 to 17.5 kHz.")
    else:
        pts = [{"freq": old["freq_low"], "n": old["n_low"]},
               {"freq": old["freq_hi"], "n": old["n_hi"], "gamma": old["gamma"]}]
        # Do-nothing points that repeat n_hi: same smoothing as before, but the
        # list already has the new structure for the user to edit. Only points
        # above freq_hi can be added (frequencies must increase).
        for freq, gamma in _TAIL_POINTS:
            if freq > old["freq_hi"]:
                p = {"freq": freq, "n": old["n_hi"]}
                if freq == _TAIL_POINTS[-1][0] and old["freq_hi"] < _TAIL_POINTS[0][0]:
                    p["gamma"] = gamma
                pts.append(p)
        try:
            adv["variable_smoothing_profile"] = profile_to_dicts(profile_from_dicts(pts))
        except ValueError:
            adv["variable_smoothing_profile"] = default_profile
            detail = ("Your previous Variable settings could not be converted, so the "
                      "new default profile is used (it adds extra smoothing from 14 to "
                      "17.5 kHz).")
        else:
            kept = format_profile(profile_from_dicts(pts[:2]))
            extra = len(pts) - 2
            added = {0: "No extra points were needed.",
                     1: "One extra point was added that holds the same smoothing "
                        "width, so it doesn't change.",
                     2: "Two extra points were added that hold the same smoothing "
                        "width, so it doesn't change."}[extra]
            detail = (f"Your previous Variable settings were kept ({kept}). {added} "
                      "Reset settings to use the new default, which adds extra "
                      "smoothing from 14 to 17.5 kHz.")
    for k in _V1_VARIABLE_DEFAULTS:
        adv.pop(f"variable_smoothing_{k}", None)

    # -- Old default filename templates -> ones with {smoothing} --
    updated = []
    for key, old_tpl in _V1_TEMPLATE_DEFAULTS.items():
        if dfhrtf.get(key) == old_tpl:
            dfhrtf[key] = defaults["dfhrtf"][key]
            updated.append(f"{key} -> \"{dfhrtf[key]}\"")

    if notices is not None:
        text = ("Smoothing settings updated.\n\n"
                "SOFA Buddy now smooths with a Gaussian window, and Variable smoothing "
                "is now a list of points (settings key "
                "dfhrtf.advanced.variable_smoothing_profile).\n\n" + detail)
        if updated:
            text += ("\n\nYour default output filename templates now include "
                     "{smoothing}: " + "; ".join(updated) + ".")
        text += ("\n\nTo reproduce older output exactly, set "
                 "dfhrtf.advanced.smoothing_window to \"flat\" in the settings file.")
        notices.append(text)


def load_merged(user_path=None, notices: list | None = None) -> tuple:
    """
    Load defaults and deep-merge the user file on top (if it exists). Qt-free.
    Returns (merged_cfg, warnings).  Schema-migration messages for a user file
    from an older version are appended to *notices* when a list is given.
    """
    cfg = load_defaults()
    if user_path and os.path.isfile(user_path):
        try:
            with open(user_path, "r", encoding="utf-8") as fh:
                user_cfg = json.load(fh)
            if not isinstance(user_cfg, dict):
                raise ValueError("top level is not a JSON object")
            cfg = _deep_merge(cfg, user_cfg)
            # The version comes from the USER file (not the merge, which would
            # inherit the defaults' current one): a missing key is the old format.
            cfg["schema_version"] = _coerce_version(user_cfg.get("schema_version"))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            return validate(cfg)[0], [
                f"Config: Could not read user settings ({exc}), using defaults."
            ]
    return validate(cfg, notices)


# ── ConfigManager (Qt-dependent) ─────────────────────────────────────────────

class ConfigManager:
    """
    Process-wide config manager for the UI.  Requires a running QApplication
    (uses QStandardPaths to resolve the per-user config dir).

    Obtain the singleton via  `manager()` at the bottom of this module.
    """

    _FILE_NAME = "user_settings.json"

    def __init__(self):
        self._cached: dict | None = None
        self._warnings: list = []
        self._notices: list = []

    # -- Path resolution --

    def user_path(self) -> str:
        """
        Platform-appropriate path for user_settings.json.
          Windows : %LOCALAPPDATA%\\<AppName>\\user_settings.json
          Linux   : ~/.config/<AppName>/user_settings.json
          macOS   : ~/Library/Preferences/<AppName>/user_settings.json

        Requires app.setApplicationName() / app.setOrganizationName() to have
        been called on the QApplication before this is invoked.
        """
        from PySide6.QtCore import QStandardPaths
        config_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation
        )
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, self._FILE_NAME)

    # -- Load / cache --

    def load(self) -> dict:
        """
        Load and cache the merged config on first call; return cached copy thereafter.
        Call reset() to force a reload.
        """
        if self._cached is None:
            self._notices = []
            try:
                self._cached, self._warnings = load_merged(self.user_path(), self._notices)
            except Exception as exc:    # never let settings stop the app starting
                self._notices = []
                self._cached, _ = validate(load_defaults())
                self._warnings = [f"Config: could not load user settings ({exc}), using defaults."]
        return copy.deepcopy(self._cached)

    def get(self) -> dict:
        """Convenience alias for load()."""
        return self.load()

    @property
    def warnings(self) -> list:
        """Validation warnings from the most recent load (empty list if none)."""
        self.load()  # ensure loaded
        return list(self._warnings)

    @property
    def notices(self) -> list:
        """One-time user-facing messages from migrating an older settings file
        (empty on a fresh install or an up-to-date file). Show them, then
        save() so the file is current and they never repeat."""
        self.load()
        return list(self._notices)

    # -- Save --

    def save(self, cfg: dict) -> None:
        """
        Validate *cfg* and atomically write user_settings.json.
        'files' is always stripped.  The cache is updated on success.
        """
        to_save = copy.deepcopy(cfg)
        to_save.pop("files", None)

        sanitized, _ = validate(to_save)
        sanitized["schema_version"] = SCHEMA_VERSION

        path = self.user_path()
        dir_ = os.path.dirname(path)

        # Atomic write: temp file in the same dir + os.replace
        fd, tmp_path = tempfile.mkstemp(
            dir=dir_, suffix=".tmp", prefix="user_settings_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(sanitized, fh, indent=2)
            os.replace(tmp_path, path)
        except OSError:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._cached = copy.deepcopy(sanitized)

    # -- Reset --

    def reset(self) -> dict:
        """Delete user_settings.json and reload from bundled defaults."""
        try:
            os.unlink(self.user_path())
        except FileNotFoundError:
            pass
        self._cached = None
        self._warnings = []
        self._notices = []
        return self.load()

    # -- Snapshot helper --

    def snapshot(self, output_cfg, tab_sofa, tab_dfhrtf) -> dict:
        """
        Build a full persistable config dict from the current tab state.
        Live surfaced values are merged over the cached advanced/averaging
        sections so that JSON-only tunables are preserved across saves.
        `output_cfg` is the output-folder dict from the main window.
        """
        cached = self.load()

        sofa_live   = tab_sofa.get_config()    # no 'advanced' key
        dfhrtf_live = tab_dfhrtf.get_config()  # no 'advanced' key

        # Start from cache (which has advanced), then overlay live surfaced values
        result_sofa   = copy.deepcopy(cached.get("sofa",   {}))
        result_dfhrtf = copy.deepcopy(cached.get("dfhrtf", {}))
        result_sofa.update(sofa_live)
        result_dfhrtf.update(dfhrtf_live)

        return {
            "schema_version": SCHEMA_VERSION,
            "output":   output_cfg,
            "sofa":     result_sofa,
            "dfhrtf":   result_dfhrtf,
            "averaging": copy.deepcopy(cached.get("averaging", {})),
            "processing": copy.deepcopy(cached.get("processing", {})),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: ConfigManager | None = None


def manager() -> ConfigManager:
    """Return (creating if necessary) the process-wide ConfigManager singleton."""
    global _instance
    if _instance is None:
        _instance = ConfigManager()
    return _instance
