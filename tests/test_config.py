"""
tests/test_config.py

Unit tests for core/config.py — the cascaded JSON configuration system.
Covers: load_defaults, deep merge, validation (clamp / revert / filter),
atomic round-trip, files-key exclusion, reset, and tab set_config identity.

Run with:
    pytest tests/test_config.py
"""

import copy
import json
import os
import tempfile
import sys
from pathlib import Path

import pytest

from core import config as C


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level ConfigManager singleton before each test."""
    C._instance = None
    yield
    C._instance = None


@pytest.fixture()
def tmp_user_dir(tmp_path):
    """Return a temp directory for user_settings.json."""
    return tmp_path


def write_user(tmp_user_dir, data: dict) -> str:
    path = str(tmp_user_dir / "user_settings.json")
    with open(path, "w") as fh:
        json.dump(data, fh)
    return path


# ── 1. load_defaults ─────────────────────────────────────────────────────────

class TestLoadDefaults:
    def test_returns_dict_with_top_level_keys(self):
        d = C.load_defaults()
        for key in ("schema_version", "output", "sofa", "dfhrtf", "averaging"):
            assert key in d, f"Missing top-level key: {key}"

    def test_sofa_has_advanced(self):
        d = C.load_defaults()
        assert "advanced" in d["sofa"]
        assert "processing_length" in d["sofa"]["advanced"]

    def test_dfhrtf_has_advanced(self):
        d = C.load_defaults()
        assert "advanced" in d["dfhrtf"]
        assert "n_fft" in d["dfhrtf"]["advanced"]

    def test_averaging_has_all_keys(self):
        d = C.load_defaults()
        avg = d["averaging"]
        for key in ("num_points", "outlier_threshold_std",
                    "softdtw_gamma_persofa", "softdtw_gamma_global",
                    "softdtw_max_iter", "softdtw_tol", "geo_arc_y_scale"):
            assert key in avg

    def test_default_values_match_runtime(self):
        """Ensure the defaults reflect the values actually used at runtime, not misleading placeholders."""
        d = C.load_defaults()
        assert d["averaging"]["outlier_threshold_std"] == pytest.approx(0.85)
        assert d["averaging"]["softdtw_gamma_persofa"] == pytest.approx(0.06)
        assert d["averaging"]["softdtw_gamma_global"] == pytest.approx(0.06)


# ── 2. Deep merge ─────────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_partial_user_override_preserves_defaults(self, tmp_user_dir):
        xover = C.SCHEMA["sofa.xover"]["min"] + 100.0  # in-band override value
        user = {"sofa": {"xover": xover}}
        path = write_user(tmp_user_dir, user)
        cfg, _ = C.load_merged(path)
        # User override applied
        assert cfg["sofa"]["xover"] == pytest.approx(xover)
        # Default values preserved
        assert cfg["sofa"]["dfeq"] is True
        assert cfg["averaging"]["num_points"] == 1024

    def test_nested_advanced_partial_override(self, tmp_user_dir):
        user = {"sofa": {"advanced": {"norm_target_db": -3.0}}}
        path = write_user(tmp_user_dir, user)
        cfg, _ = C.load_merged(path)
        # Overridden
        assert cfg["sofa"]["advanced"]["norm_target_db"] == pytest.approx(-3.0)
        # Other advanced keys preserved from defaults
        assert cfg["sofa"]["advanced"]["processing_length"] == 2048

    def test_averaging_full_override(self, tmp_user_dir):
        user = {"averaging": {"num_points": 256, "softdtw_gamma_global": 0.1}}
        path = write_user(tmp_user_dir, user)
        cfg, _ = C.load_merged(path)
        assert cfg["averaging"]["num_points"] == 256
        assert cfg["averaging"]["softdtw_gamma_global"] == pytest.approx(0.1)
        # Keys not overridden stay at defaults
        assert cfg["averaging"]["softdtw_max_iter"] == 15

    def test_missing_user_file_returns_defaults(self):
        cfg, warnings = C.load_merged("/nonexistent/path/user_settings.json")
        defaults = C.load_defaults()
        assert cfg["sofa"]["xover"] == pytest.approx(defaults["sofa"]["xover"])
        assert warnings == []


# ── 3. Validation ─────────────────────────────────────────────────────────────

class TestValidation:
    def _merged_with_patch(self, patch: dict) -> tuple:
        """Apply a patch dict over defaults and validate."""
        base = C.load_defaults()
        patched = C._deep_merge(base, patch)
        return C.validate(patched)

    # Numeric clamping
    def test_xover_above_max_is_clamped(self):
        cfg, warnings = self._merged_with_patch({"sofa": {"xover": 9999.0}})
        assert cfg["sofa"]["xover"] == pytest.approx(C.SCHEMA["sofa.xover"]["max"])
        assert any("xover" in w for w in warnings)

    def test_xover_below_min_is_clamped(self):
        cfg, warnings = self._merged_with_patch({"sofa": {"xover": 10.0}})
        assert cfg["sofa"]["xover"] == pytest.approx(C.SCHEMA["sofa.xover"]["min"])
        assert any("xover" in w for w in warnings)

    def test_norm_target_db_clamped(self):
        cfg, warnings = self._merged_with_patch(
            {"sofa": {"advanced": {"norm_target_db": 5.0}}}
        )
        assert cfg["sofa"]["advanced"]["norm_target_db"] == pytest.approx(0.0)
        assert any("norm_target_db" in w for w in warnings)

    # Choices revert
    def test_bad_delimiter_reverts_to_default(self):
        cfg, warnings = self._merged_with_patch({"dfhrtf": {"text_delimiter": "Pipe"}})
        assert cfg["dfhrtf"]["text_delimiter"] == "Tab"
        assert any("text_delimiter" in w for w in warnings)

    def test_bad_algo_reverts_to_default(self):
        cfg, warnings = self._merged_with_patch({"dfhrtf": {"algo": "KMeans"}})
        assert cfg["dfhrtf"]["algo"] == C.load_defaults()["dfhrtf"]["algo"]
        assert any("algo" in w for w in warnings)

    def test_smoothing_bad_value_reverts(self):
        cfg, warnings = self._merged_with_patch({"dfhrtf": {"smoothing": 7}})
        assert cfg["dfhrtf"]["smoothing"] == "Variable"  # default
        assert any("smoothing" in w for w in warnings)

    def test_smoothing_legacy_int_reverts(self):
        # Pre-dropdown user files stored an int (e.g. 24); fails the str type
        # check and reverts to the new default.
        cfg, warnings = self._merged_with_patch({"dfhrtf": {"smoothing": 24}})
        assert cfg["dfhrtf"]["smoothing"] == "Variable"
        assert any("smoothing" in w for w in warnings)

    def test_smoothing_valid_fraction_passes(self):
        cfg, warnings = self._merged_with_patch({"dfhrtf": {"smoothing": "1/12"}})
        assert cfg["dfhrtf"]["smoothing"] == "1/12"
        assert not any("smoothing" in w for w in warnings)

    # Samplerate list filtering
    def test_samplerates_invalid_items_filtered(self):
        cfg, warnings = self._merged_with_patch({"sofa": {"samplerates": [48000, 99999]}})
        assert 99999 not in cfg["sofa"]["samplerates"]
        assert 48000 in cfg["sofa"]["samplerates"]
        assert any("samplerates" in w for w in warnings)

    # Type coercion
    def test_float_as_string_coerced(self):
        xover = C.SCHEMA["sofa.xover"]["min"] + 100.0  # in-band override value
        cfg, warnings = self._merged_with_patch({"sofa": {"xover": str(xover)}})
        assert cfg["sofa"]["xover"] == pytest.approx(xover)
        # No warning if coercible
        assert not any("xover" in w for w in warnings)

    def test_bool_wrong_type_reverts(self):
        cfg, warnings = self._merged_with_patch({"sofa": {"dfeq": "yes"}})
        defaults = C.load_defaults()
        assert cfg["sofa"]["dfeq"] == defaults["sofa"]["dfeq"]
        assert any("dfeq" in w for w in warnings)

    # Files key stripped
    def test_files_key_stripped_from_output(self):
        base = C.load_defaults()
        base["files"] = ["/some/file.sofa"]
        sanitized, _ = C.validate(base)
        assert "files" not in sanitized


# ── 4. Atomic round-trip ──────────────────────────────────────────────────────

class TestAtomicRoundTrip:
    def test_save_then_load_is_identity(self, tmp_user_dir, monkeypatch):
        """save() followed by load_merged() should produce the same config."""
        mgr = C.ConfigManager()
        user_path = str(tmp_user_dir / "user_settings.json")
        monkeypatch.setattr(mgr, "user_path", lambda: user_path)

        # Load defaults, tweak a value, save
        xover = C.SCHEMA["sofa.xover"]["min"] + 120.0  # in-band override value
        cfg = mgr.load()
        cfg["sofa"]["xover"] = xover
        cfg["averaging"]["num_points"] = 256
        mgr.save(cfg)

        # Reload and check
        mgr2 = C.ConfigManager()
        monkeypatch.setattr(mgr2, "user_path", lambda: user_path)
        reloaded = mgr2.load()
        assert reloaded["sofa"]["xover"] == pytest.approx(xover)
        assert reloaded["averaging"]["num_points"] == 256
        # schema_version must be written
        assert reloaded["schema_version"] == 1

    def test_files_never_written_to_disk(self, tmp_user_dir, monkeypatch):
        mgr = C.ConfigManager()
        user_path = str(tmp_user_dir / "user_settings.json")
        monkeypatch.setattr(mgr, "user_path", lambda: user_path)

        cfg = mgr.load()
        cfg["files"] = ["/path/to/file.sofa"]
        mgr.save(cfg)

        with open(user_path) as fh:
            written = json.load(fh)
        assert "files" not in written

    def test_corrupt_user_file_falls_back_to_defaults(self, tmp_user_dir, monkeypatch):
        user_path = str(tmp_user_dir / "user_settings.json")
        with open(user_path, "w") as fh:
            fh.write("{{{ not valid json }}")

        cfg, warnings = C.load_merged(user_path)
        assert any("Could not read" in w for w in warnings)
        defaults = C.load_defaults()
        assert cfg["sofa"]["xover"] == pytest.approx(defaults["sofa"]["xover"])


# ── 5. Reset ──────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_restores_defaults(self, tmp_user_dir, monkeypatch):
        mgr = C.ConfigManager()
        user_path = str(tmp_user_dir / "user_settings.json")
        monkeypatch.setattr(mgr, "user_path", lambda: user_path)

        # Change and save
        cfg = mgr.load()
        cfg["sofa"]["xover"] = 450.0
        mgr.save(cfg)

        # Reset
        reset_cfg = mgr.reset()
        defaults = C.load_defaults()
        assert reset_cfg["sofa"]["xover"] == pytest.approx(defaults["sofa"]["xover"])
        assert not os.path.exists(user_path)

    def test_reset_when_no_file_doesnt_crash(self, tmp_user_dir, monkeypatch):
        mgr = C.ConfigManager()
        user_path = str(tmp_user_dir / "nonexistent_user_settings.json")
        monkeypatch.setattr(mgr, "user_path", lambda: user_path)
        # Should not raise
        cfg = mgr.reset()
        assert "sofa" in cfg


# ── 6. Tab set_config / get_config identity ───────────────────────────────────

@pytest.mark.usefixtures("qapp")
class TestTabRoundTrip:
    """pytest-qt round-trip: tab.set_config(tab.get_config()) is identity."""

    def test_sofa_tab_identity(self):
        from ui.tabs import SofaTab
        tab = SofaTab()
        original = tab.get_config()
        tab.set_config(original)
        restored = tab.get_config()
        assert restored == original

    def test_dfhrtf_tab_identity(self):
        from ui.tabs import DfhrtfTab
        tab = DfhrtfTab()
        original = tab.get_config()
        tab.set_config(original)
        restored = tab.get_config()
        assert restored == original

    def test_main_window_output_identity(self):
        from ui.main_window import SofaBuddyMainWindow
        window = SofaBuddyMainWindow()
        original = window.get_output_config()
        window.set_output_config(original)
        restored = window.get_output_config()
        assert restored == original

    def test_sofa_tab_restores_non_default_values(self):
        from ui.tabs import SofaTab
        xover = C.SCHEMA["sofa.xover"]["min"] + 100.0  # in-band override value
        tab = SofaTab()
        cfg = tab.get_config()
        cfg["xover"] = xover
        cfg["samplerates"] = [44100, 96000]
        cfg["dfeq"] = False
        cfg["uneq"] = True
        cfg["enabled"] = True
        tab.set_config(cfg)
        restored = tab.get_config()
        assert restored["xover"] == pytest.approx(xover)
        assert 44100 in restored["samplerates"]
        assert 96000 in restored["samplerates"]
        assert restored["dfeq"] is False
        assert restored["uneq"] is True
        assert restored["enabled"] is True

    def test_dfhrtf_tab_restores_smoothing(self):
        from ui.tabs import DfhrtfTab
        tab = DfhrtfTab()
        cfg = tab.get_config()
        cfg["smoothing"] = "1/6"
        tab.set_config(cfg)
        assert tab.get_config()["smoothing"] == "1/6"


# ── 7. JSON Schema drift guard ────────────────────────────────────────────────

def _all_leaf_paths(d: dict, prefix: str = ""):
    """Yield every dotted-path key for leaf (non-dict) values in a nested dict."""
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _all_leaf_paths(v, path)
        else:
            yield path


def _schema_prop_at(schema: dict, segments: list) -> dict | None:
    """Follow *segments* through the schema's nested 'properties' tree.
    Returns the property dict if found, else None."""
    node = schema
    for seg in segments:
        props = node.get("properties", {})
        if seg not in props:
            return None
        node = props[seg]
    return node


class TestSchemaDriftGuard:
    """Ensure default_settings.schema.json stays in sync with config.py SCHEMA."""

    def test_committed_schema_matches_generated(self):
        """The committed schema file must equal what build_schema() produces.

        Fails if SCHEMA in config.py was changed without re-running the generator.
        Fix: python -m core.scripts.generate_settings_schema
        """
        from core.scripts.generate_settings_schema import build_schema
        built = build_schema()
        schema_path = (
            Path(__file__).parent.parent / "core" / "default_settings.schema.json"
        )
        with open(schema_path, "r", encoding="utf-8") as fh:
            committed = json.load(fh)
        assert built == committed, (
            "default_settings.schema.json is out of sync with config.py SCHEMA. "
            "Run:  python -m core.scripts.generate_settings_schema"
        )

    def test_all_default_setting_keys_covered_by_schema(self):
        """Every leaf key in default_settings.json (except '$schema') must be
        reachable in the generated schema's property tree.

        Fails if a key is added to default_settings.json / SCHEMA but the
        generator was not re-run.
        """
        from core.scripts.generate_settings_schema import build_schema
        schema = build_schema()
        defaults = C.load_defaults()

        missing = []
        for path in _all_leaf_paths(defaults):
            if path == "$schema":
                continue  # meta-key, not a setting
            segments = path.split(".")
            if _schema_prop_at(schema, segments) is None:
                missing.append(path)

        assert not missing, (
            f"Keys in default_settings.json not covered by schema: {missing}\n"
            "Run:  python -m core.scripts.generate_settings_schema"
        )
