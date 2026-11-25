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
                                    "choices": ["Variable", "1/3", "1/6", "1/12", "1/24", "1/48"]},
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

    "dfhrtf.advanced.variable_smoothing_freq_low":  {"type": float, "min": 20.0,  "max": 2000.0},
    "dfhrtf.advanced.variable_smoothing_freq_hi":   {"type": float, "min": 500.0, "max": 20000.0},
    "dfhrtf.advanced.variable_smoothing_n_low":     {"type": float, "min": 1.0,   "max": 48.0},
    "dfhrtf.advanced.variable_smoothing_n_hi":      {"type": float, "min": 1.0,   "max": 48.0},
    "dfhrtf.advanced.variable_smoothing_gamma":     {"type": float, "min": 0.1,   "max": 10.0},
    "dfhrtf.advanced.variable_smoothing_mean_mode": {"type": str,   "choices": ["db", "cubic"]},

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


def validate(cfg: dict) -> tuple:
    """
    Walk the SCHEMA against *cfg* (a deep copy is made internally) and:
      - coerce borderline numeric types where possible
      - clamp out-of-range numerics to [min, max]
      - revert invalid choices / un-coercible types to the default value
      - filter invalid items from list-with-choices fields
      - drop the 'files' key if present (never persisted)

    Returns (sanitized_cfg, warnings) where warnings is a list[str].
    """
    defaults = load_defaults()
    cfg = copy.deepcopy(cfg)
    warnings: list = []

    # Schema-version migration (no-op at v1)
    from_version = cfg.get("schema_version", 0)
    if from_version != 1:
        cfg = _migrate(cfg, from_version)

    for path, spec in SCHEMA.items():
        value, found = _get_nested(cfg, path)
        if not found:
            continue  # Absent keys are fine — defaults already merged in

        default_val, _ = _get_nested(defaults, path)
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

    # Never persist the files list
    cfg.pop("files", None)

    return cfg, warnings


def _migrate(cfg: dict, from_version: int) -> dict:
    """Stub for future schema migrations.  Currently a no-op for v0 → v1."""
    # Example pattern for a future rename:
    #   if from_version == 0:
    #       old_val = cfg.get("dfhrtf", {}).pop("old_key", None)
    #       if old_val is not None:
    #           cfg.setdefault("dfhrtf", {})["new_key"] = old_val
    cfg["schema_version"] = 1
    return cfg


def load_merged(user_path=None) -> tuple:
    """
    Load defaults and deep-merge the user file on top (if it exists). Qt-free.
    Returns (merged_cfg, warnings).
    """
    cfg = load_defaults()
    if user_path and os.path.isfile(user_path):
        try:
            with open(user_path, "r", encoding="utf-8") as fh:
                user_cfg = json.load(fh)
            cfg = _deep_merge(cfg, user_cfg)
        except (json.JSONDecodeError, OSError) as exc:
            return validate(cfg)[0], [
                f"Config: Could not read user settings ({exc}), using defaults."
            ]
    return validate(cfg)


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
            self._cached, self._warnings = load_merged(self.user_path())
        return copy.deepcopy(self._cached)

    def get(self) -> dict:
        """Convenience alias for load()."""
        return self.load()

    @property
    def warnings(self) -> list:
        """Validation warnings from the most recent load (empty list if none)."""
        self.load()  # ensure loaded
        return list(self._warnings)

    # -- Save --

    def save(self, cfg: dict) -> None:
        """
        Validate *cfg* and atomically write user_settings.json.
        'files' is always stripped.  The cache is updated on success.
        """
        to_save = copy.deepcopy(cfg)
        to_save.pop("files", None)

        sanitized, _ = validate(to_save)
        sanitized["schema_version"] = 1

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
            "schema_version": 1,
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
