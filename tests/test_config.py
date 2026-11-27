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

    def test_retired_mean_mode_key_is_dropped_not_migrated(self):
        cfg, warnings = self._merged_with_patch(
            {"dfhrtf": {"advanced": {"variable_smoothing_mean_mode": "db"}}})
        adv = cfg["dfhrtf"]["advanced"]
        assert "variable_smoothing_mean_mode" not in adv
        assert adv["smoothing_mean_mode"] == "power"    # factory default, not the old "db"
        assert warnings == []

    # variable_smoothing_profile (list of objects: custom validator)
    def test_default_profile_is_valid_and_untouched(self):
        cfg, warnings = self._merged_with_patch({})
        assert cfg["dfhrtf"]["advanced"]["variable_smoothing_profile"] == \
            C.load_defaults()["dfhrtf"]["advanced"]["variable_smoothing_profile"]
        assert warnings == []

    @pytest.mark.parametrize("bad", [
        [],
        "350:2",
        [{"freq": 500.0, "n": 2.0}, {"freq": 500.0, "n": 4.0}],       # not increasing
        [{"freq": 900.0, "n": 2.0}, {"freq": 500.0, "n": 4.0}],
        [{"freq": 500.0, "n": 0.5}],                                   # n out of range
        [{"freq": 500.0, "n": 60.0}],
        [{"freq": 500.0, "n": 2.0}, {"freq": 900.0, "n": 4.0, "gamma": 20.0}],
        [{"freq": 500.0, "n": 2.0}, {"freq": 900.0, "n": 4.0, "gamma": 0.01}],
        [{"freq": 500.0, "n": 2.0, "gamma": 0.9}],                     # gamma on first point
        [{"freq": 500.0, "n": 2.0, "colour": "red"}],                  # unknown key
        [{"freq": 5.0, "n": 2.0}],                                     # freq out of range
        [{"freq": 25000.0, "n": 2.0}],
    ])
    def test_invalid_profile_reverts_to_default_with_warning(self, bad):
        cfg, warnings = self._merged_with_patch(
            {"dfhrtf": {"advanced": {"variable_smoothing_profile": bad}}})
        assert cfg["dfhrtf"]["advanced"]["variable_smoothing_profile"] == \
            C.load_defaults()["dfhrtf"]["advanced"]["variable_smoothing_profile"]
        assert any("variable_smoothing_profile" in w for w in warnings)

    def test_valid_custom_profile_is_kept(self):
        prof = [{"freq": 200, "n": 3}, {"freq": 4000.0, "n": 24.0, "gamma": 2.0}]
        cfg, warnings = self._merged_with_patch(
            {"dfhrtf": {"advanced": {"variable_smoothing_profile": prof}}})
        assert cfg["dfhrtf"]["advanced"]["variable_smoothing_profile"] == [
            {"freq": 200.0, "n": 3.0}, {"freq": 4000.0, "n": 24.0, "gamma": 2.0}]
        assert warnings == []

    def test_bad_smoothing_window_reverts(self):
        cfg, warnings = self._merged_with_patch(
            {"dfhrtf": {"advanced": {"smoothing_window": "triangular"}}})
        assert cfg["dfhrtf"]["advanced"]["smoothing_window"] == "gaussian"
        assert any("smoothing_window" in w for w in warnings)

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
        assert reloaded["schema_version"] == C.SCHEMA_VERSION == 2

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


# ── 4b. Migration v1 -> v2 (variable-smoothing profile) ──────────────────────

V1_DEFAULT_KEYS = {
    "variable_smoothing_freq_low": 350.0, "variable_smoothing_freq_hi": 6000.0,
    "variable_smoothing_n_low": 2.0, "variable_smoothing_n_hi": 12.0,
    "variable_smoothing_gamma": 0.88,
}
OLD_KEYS = tuple(V1_DEFAULT_KEYS)


def v1_user_file(tmp_user_dir, adv_overrides=None, dfhrtf_overrides=None, drop_old=False):
    """Write a user_settings.json as v0.7.0 saved it (schema_version 1, five old keys,
    old default templates)."""
    adv = {} if drop_old else dict(V1_DEFAULT_KEYS)
    adv["variable_smoothing_mean_mode"] = "db"          # also retired in v0.7.0 files
    adv.update(adv_overrides or {})
    dfhrtf = {"text_filename": "{filename} {LR}", "graph_filename": "{graphtype} {filename}",
              "advanced": adv}
    dfhrtf.update(dfhrtf_overrides or {})
    return write_user(tmp_user_dir, {"schema_version": 1, "dfhrtf": dfhrtf})


def pts(cfg):
    return cfg["dfhrtf"]["advanced"]["variable_smoothing_profile"]


class TestSmoothingMigration:
    DEFAULT_PROFILE = C.load_defaults()["dfhrtf"]["advanced"]["variable_smoothing_profile"]

    def test_untouched_user_gets_the_new_default_profile(self, tmp_user_dir):
        notices = []
        cfg, warnings = C.load_merged(v1_user_file(tmp_user_dir), notices)
        assert pts(cfg) == self.DEFAULT_PROFILE
        assert warnings == []
        assert len(notices) == 1
        assert "new default profile" in notices[0]
        assert "Gaussian" in notices[0]
        assert 'smoothing_window' in notices[0] and '"flat"' in notices[0]

    def test_old_keys_retired_and_version_bumped(self, tmp_user_dir):
        for overrides in ({}, {"variable_smoothing_n_hi": 24.0}):
            cfg, _ = C.load_merged(v1_user_file(tmp_user_dir, overrides), [])
            adv = cfg["dfhrtf"]["advanced"]
            for k in OLD_KEYS + ("variable_smoothing_mean_mode",):
                assert k not in adv
            assert cfg["schema_version"] == 2
            assert adv["smoothing_window"] == "gaussian"

    def test_customised_user_keeps_their_curve_plus_do_nothing_points(self, tmp_user_dir):
        path = v1_user_file(tmp_user_dir, {
            "variable_smoothing_freq_low": 200.0, "variable_smoothing_freq_hi": 4000.0,
            "variable_smoothing_n_low": 3.0, "variable_smoothing_n_hi": 20.0,
            "variable_smoothing_gamma": 1.3})
        notices = []
        cfg, warnings = C.load_merged(path, notices)
        assert pts(cfg) == [
            {"freq": 200.0, "n": 3.0},
            {"freq": 4000.0, "n": 20.0, "gamma": 1.3},
            {"freq": 14000.0, "n": 20.0},
            {"freq": 17500.0, "n": 20.0, "gamma": 1.6},
        ]
        assert warnings == []
        assert len(notices) == 1
        assert "previous Variable settings were kept" in notices[0]
        assert "200:3,4000:20:1.3" in notices[0]
        assert "Reset" in notices[0]

    def test_customised_curve_is_numerically_identical_to_the_old_one(self, tmp_user_dir):
        import numpy as np
        from core.smoothing import SmoothingParams, profile_from_dicts, smoothing_profile
        path = v1_user_file(tmp_user_dir, {
            "variable_smoothing_freq_low": 200.0, "variable_smoothing_freq_hi": 4000.0,
            "variable_smoothing_n_low": 3.0, "variable_smoothing_n_hi": 20.0,
            "variable_smoothing_gamma": 1.3})
        cfg, _ = C.load_merged(path, [])
        new = SmoothingParams(points=profile_from_dicts(pts(cfg)))
        old = SmoothingParams(points=((200.0, 3.0, 1.0), (4000.0, 20.0, 1.3)))
        f = np.geomspace(20.0, 20000.0, 500)
        # equal to rounding (above freq_hi the old form is w_lo + (w_hi - w_lo), 1 ulp off w_hi)
        np.testing.assert_allclose(smoothing_profile(f, new), smoothing_profile(f, old),
                                   rtol=1e-14, atol=0)

    @pytest.mark.parametrize("freq_hi, expected_extra", [
        (10000.0, [{"freq": 14000.0, "n": 20.0}, {"freq": 17500.0, "n": 20.0, "gamma": 1.6}]),
        (14000.0, [{"freq": 17500.0, "n": 20.0}]),     # edge: freq_hi already >= 14000
        (16000.0, [{"freq": 17500.0, "n": 20.0}]),
        (17500.0, []),
        (20000.0, []),
    ])
    def test_customised_freq_hi_edge_cases_only_add_points_above_it(
            self, tmp_user_dir, freq_hi, expected_extra):
        path = v1_user_file(tmp_user_dir, {"variable_smoothing_freq_hi": freq_hi,
                                           "variable_smoothing_n_hi": 20.0})
        notices = []
        cfg, warnings = C.load_merged(path, notices)
        got = pts(cfg)
        assert got[:2] == [{"freq": 350.0, "n": 2.0},
                           {"freq": freq_hi, "n": 20.0, "gamma": 0.88}]
        assert got[2:] == expected_extra
        assert warnings == []
        assert len(notices) == 1

    def test_unconvertible_old_values_fall_back_to_default_with_a_notice(self, tmp_user_dir):
        # freq_low above freq_hi was accepted by the old degenerate-band code
        path = v1_user_file(tmp_user_dir, {"variable_smoothing_freq_low": 1500.0,
                                           "variable_smoothing_freq_hi": 800.0})
        notices = []
        cfg, _ = C.load_merged(path, notices)
        assert pts(cfg) == self.DEFAULT_PROFILE
        assert "could not be converted" in notices[0]

    def test_v1_file_without_the_old_keys_is_treated_as_untouched(self, tmp_user_dir):
        notices = []
        cfg, _ = C.load_merged(v1_user_file(tmp_user_dir, drop_old=True), notices)
        assert pts(cfg) == self.DEFAULT_PROFILE
        assert len(notices) == 1

    def test_old_default_templates_are_updated_and_mentioned(self, tmp_user_dir):
        notices = []
        cfg, _ = C.load_merged(v1_user_file(tmp_user_dir), notices)
        d = C.load_defaults()["dfhrtf"]
        assert cfg["dfhrtf"]["text_filename"] == d["text_filename"] == "{filename} {LR} {smoothing}"
        assert cfg["dfhrtf"]["graph_filename"] == d["graph_filename"]
        assert "{smoothing}" in notices[0]

    def test_customised_templates_are_left_alone(self, tmp_user_dir):
        path = v1_user_file(tmp_user_dir, dfhrtf_overrides={
            "text_filename": "{filename} mine", "graph_filename": "{graphtype} {filename}"})
        notices = []
        cfg, _ = C.load_merged(path, notices)
        assert cfg["dfhrtf"]["text_filename"] == "{filename} mine"
        assert cfg["dfhrtf"]["graph_filename"] == C.load_defaults()["dfhrtf"]["graph_filename"]
        assert "text_filename" not in notices[0]
        assert "graph_filename" in notices[0]

    def test_fresh_install_and_current_file_produce_no_notice(self, tmp_user_dir):
        notices = []
        C.load_merged(str(tmp_user_dir / "nope.json"), notices)
        assert notices == []
        notices = []
        C.load_merged(write_user(tmp_user_dir, {"schema_version": 2}), notices)
        assert notices == []

    def test_manager_notices_then_save_then_second_load_is_silent(self, tmp_user_dir, monkeypatch):
        path = v1_user_file(tmp_user_dir)
        mgr = C.ConfigManager()
        monkeypatch.setattr(mgr, "user_path", lambda: path)
        cfg = mgr.load()
        assert len(mgr.notices) == 1
        mgr.save(cfg)
        with open(path) as fh:
            on_disk = json.load(fh)
        assert on_disk["schema_version"] == 2
        assert not any(k in on_disk["dfhrtf"]["advanced"] for k in OLD_KEYS)
        assert on_disk["dfhrtf"]["advanced"]["variable_smoothing_profile"] == self.DEFAULT_PROFILE

        mgr2 = C.ConfigManager()
        monkeypatch.setattr(mgr2, "user_path", lambda: path)
        mgr2.load()
        assert mgr2.notices == []
        assert mgr2.warnings == []

    def test_reset_gives_the_real_default_profile_and_no_notice(self, tmp_user_dir, monkeypatch):
        path = v1_user_file(tmp_user_dir, {"variable_smoothing_n_hi": 20.0})
        mgr = C.ConfigManager()
        monkeypatch.setattr(mgr, "user_path", lambda: path)
        assert len(mgr.notices) == 1
        cfg = mgr.reset()
        assert pts(cfg) == self.DEFAULT_PROFILE
        assert mgr.notices == []

    def test_saved_snapshot_is_written_at_the_current_version(self, tmp_user_dir, monkeypatch):
        mgr = C.ConfigManager()
        path = str(tmp_user_dir / "user_settings.json")
        monkeypatch.setattr(mgr, "user_path", lambda: path)
        mgr.save(mgr.load())
        with open(path) as fh:
            assert json.load(fh)["schema_version"] == C.SCHEMA_VERSION


class TestMigrationRobustness:
    """Files v0.7.0 tolerated must never stop the app starting."""
    DEFAULT_PROFILE = C.load_defaults()["dfhrtf"]["advanced"]["variable_smoothing_profile"]

    @pytest.mark.parametrize("data", [
        {"schema_version": "1", "dfhrtf": {"advanced": {"variable_smoothing_n_hi": 20.0}}},
        {"schema_version": None},
        {"schema_version": [1]},
        {"schema_version": True},
        {"schema_version": "banana"},
        {"schema_version": 1, "dfhrtf": {"advanced": None}},
        {"schema_version": 1, "dfhrtf": None},
        {"schema_version": 1, "dfhrtf": "oops"},
        {"schema_version": 1, "dfhrtf": {"advanced": {"variable_smoothing_n_hi": None,
                                                       "variable_smoothing_gamma": "x"}}},
    ])
    def test_odd_files_load_and_end_up_at_the_current_version(self, tmp_user_dir, data):
        notices = []
        cfg, _ = C.load_merged(write_user(tmp_user_dir, data), notices)
        assert cfg["schema_version"] == C.SCHEMA_VERSION
        assert isinstance(cfg["dfhrtf"]["advanced"]["variable_smoothing_profile"], list)
        assert cfg["dfhrtf"]["smoothing"]     # section intact
        assert len(notices) == 1

    def test_string_version_still_migrates_customised_keys(self, tmp_user_dir):
        path = write_user(tmp_user_dir, {"schema_version": "1", "dfhrtf": {"advanced": {
            "variable_smoothing_n_hi": 20.0}}})
        cfg, _ = C.load_merged(path, [])
        assert pts(cfg)[1] == {"freq": 6000.0, "n": 20.0, "gamma": 0.88}

    def test_top_level_list_falls_back_to_defaults(self, tmp_user_dir):
        path = str(tmp_user_dir / "user_settings.json")
        with open(path, "w") as fh:
            json.dump([1, 2, 3], fh)
        cfg, warnings = C.load_merged(path, [])
        assert any("Could not read" in w for w in warnings)
        assert cfg["schema_version"] == C.SCHEMA_VERSION

    def test_validate_never_raises_on_garbage(self):
        for junk in ([], "x", None, 5):
            cfg, warnings = C.validate(junk)
            assert cfg["schema_version"] == C.SCHEMA_VERSION and warnings
        cfg, _ = C.validate({"schema_version": 1, "dfhrtf": {"advanced": None}})
        assert pts(cfg) == self.DEFAULT_PROFILE

    def test_manager_load_falls_back_when_loading_explodes(self, tmp_user_dir, monkeypatch):
        mgr = C.ConfigManager()
        monkeypatch.setattr(mgr, "user_path", lambda: str(tmp_user_dir / "x.json"))
        def boom(*a, **k):
            raise RuntimeError("kaboom")
        monkeypatch.setattr(C, "load_merged", boom)
        cfg = mgr.load()
        assert cfg["schema_version"] == C.SCHEMA_VERSION
        assert any("kaboom" in w for w in mgr.warnings)
        assert mgr.notices == []

    # A user file without schema_version is the OLD format (the merge must not
    # inherit the defaults' current version and skip the migration).
    def test_missing_schema_version_migrates_with_notice(self, tmp_user_dir):
        path = write_user(tmp_user_dir, {"dfhrtf": {"advanced": {
            "variable_smoothing_freq_hi": 4000.0, "variable_smoothing_n_hi": 20.0}}})
        notices = []
        cfg, _ = C.load_merged(path, notices)
        assert pts(cfg)[1] == {"freq": 4000.0, "n": 20.0, "gamma": 0.88}
        assert len(notices) == 1 and "previous Variable settings were kept" in notices[0]
        assert "variable_smoothing_n_hi" not in cfg["dfhrtf"]["advanced"]

    @pytest.mark.parametrize("freq_hi, phrase", [
        (10000.0, "Two extra points were added that hold the same"),
        (16000.0, "One extra point was added that holds the same"),
        (17500.0, "No extra points were needed."),
    ])
    def test_notice_grammar(self, tmp_user_dir, freq_hi, phrase):
        path = v1_user_file(tmp_user_dir, {"variable_smoothing_freq_hi": freq_hi,
                                           "variable_smoothing_n_hi": 20.0})
        notices = []
        C.load_merged(path, notices)
        assert phrase in notices[0]
        assert "needed that" not in notices[0]

    def test_profile_rejects_strings_and_bools(self):
        for bad in ([{"freq": "350", "n": 2.0}], [{"freq": 350.0, "n": True}],
                    [{"freq": 350.0, "n": 2.0}, {"freq": 900.0, "n": 3.0, "gamma": "1"}]):
            cfg, warnings = C.validate(C._deep_merge(
                C.load_defaults(), {"dfhrtf": {"advanced": {"variable_smoothing_profile": bad}}}))
            assert pts(cfg) == self.DEFAULT_PROFILE
            assert any("variable_smoothing_profile" in w for w in warnings)


@pytest.mark.usefixtures("qapp")
class TestMainWindowSurvivesBadSettings:
    @pytest.mark.parametrize("content", ['[1, 2]', '{"schema_version": 1, "dfhrtf": null}',
                                         '{"schema_version": 1, "dfhrtf": {"advanced": null}}',
                                         '{"schema_version": "1"}'])
    def test_window_opens(self, tmp_user_dir, monkeypatch, content):
        from ui.main_window import SofaBuddyMainWindow
        path = str(tmp_user_dir / "user_settings.json")
        with open(path, "w") as fh:
            fh.write(content)
        mgr = C.manager()
        monkeypatch.setattr(mgr, "user_path", lambda: path)
        window = SofaBuddyMainWindow()
        assert window.tab_dfhrtf.get_config()["smoothing"] == "Variable"


@pytest.mark.usefixtures("qapp")
class TestMigrationPopup:
    def test_popup_shows_once_then_saves(self, tmp_user_dir, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        from ui.main_window import SofaBuddyMainWindow

        path = v1_user_file(tmp_user_dir)
        mgr = C.manager()
        monkeypatch.setattr(mgr, "user_path", lambda: path)
        monkeypatch.delenv("SOFA_BUDDY_NO_POPUPS")
        shown = []
        monkeypatch.setattr(QMessageBox, "information",
                            lambda parent, title, text: shown.append(text))
        win = SofaBuddyMainWindow()
        assert shown == []                        # nothing modal during construction (splash is up)
        win.show_startup_notices()
        assert len(shown) == 1 and "Smoothing settings updated" in shown[0]
        win.show_startup_notices()                # runs once
        assert len(shown) == 1
        with open(path) as fh:
            assert json.load(fh)["schema_version"] == 2

        C._instance = None
        mgr2 = C.manager()
        monkeypatch.setattr(mgr2, "user_path", lambda: path)
        SofaBuddyMainWindow().show_startup_notices()
        assert len(shown) == 1                    # second launch: silent

    def test_no_popup_on_a_fresh_install(self, tmp_user_dir, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        from ui.main_window import SofaBuddyMainWindow

        mgr = C.manager()
        monkeypatch.setattr(mgr, "user_path", lambda: str(tmp_user_dir / "fresh.json"))
        monkeypatch.delenv("SOFA_BUDDY_NO_POPUPS")
        monkeypatch.setattr(QMessageBox, "information",
                            lambda *a: pytest.fail("popup on a fresh install"))
        SofaBuddyMainWindow().show_startup_notices()
        assert not (tmp_user_dir / "fresh.json").exists()


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
