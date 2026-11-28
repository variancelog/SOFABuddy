"""Defaults-parity guards between the backend CLIs and default_settings.json.

Two distinct contracts live here:

1. **The variable-smoothing profile, the smoothing window and the averaging
   domain are contractually synced four ways** — `SmoothingParams` dataclass
   defaults, `default_settings.json`, the `--var-smooth-profile` /
   `--smoothing-window` / `--smoothing-mean-mode` argparse defaults, and
   `core/dfhrtf_processor.py`'s fallbacks. A GUI run and a bare CLI run must
   smooth identically. The nine Hybrid-smoothing parameters (`hybrid_*`) are
   synced the same way (test_hybrid_smoothing_four_way_sync). (The profile is a list in the JSON and a compact string
   on the CLI, so it is compared as the canonical string.)

2. **Everything else may legitimately diverge.** The CLIs default independently
   of the GUI's factory settings (lowercase enum spellings, literal delimiters,
   store_true flags that are off unless passed, historical tuning). Those are
   enumerated in `_KNOWN_INTENTIONAL_DIVERGENCES` with a rationale each, and the
   test asserts the *actual* divergence set equals the documented one — so a new,
   undocumented divergence fails loudly.
"""

import re
from pathlib import Path

import pytest

import core.config as C
from core.hybrid_smoothing import HybridParams
from core.smoothing import SmoothingParams


_REPO_ROOT = Path(__file__).parent.parent


# --------------------------------------------------------------------------- #
# 1. Variable smoothing: four-way sync
# --------------------------------------------------------------------------- #
# What is synced: the width profile, the window and the averaging domain. Each
# is compared across SmoothingParams, default_settings.json, the argparse
# default and dfhrtf_processor's fallback, as a canonical profile string /
# plain value.
_PROCESSOR_PATH = _REPO_ROOT / "core" / "dfhrtf_processor.py"


_HYBRID_KEYS = ("xover_hz", "xover_width_oct", "tv_weight", "s_factor", "s_min", "s_max",
                "huber_window", "huber_delta", "huber_iter")


def _processor_fallbacks():
    """Read the `dfhrtf_adv.get(..., <literal>)` fallbacks (and the profile
    fallback literal) out of core/dfhrtf_processor.py.

    A source scan rather than a call: the fallbacks live inside
    `process_dfhrtf_exports()`'s `common` dict, which cannot be built without a
    real file list, output tree, and run log. Scanning the literals is both more
    robust and less brittle than faking all of that.
    """
    src = _PROCESSOR_PATH.read_text(encoding="utf-8")
    found = {}
    for key in ("smoothing_window", "smoothing_mean_mode"):
        m = re.search(
            r"dfhrtf_adv\.get\(\s*['\"]" + key + r"['\"]\s*,\s*([^)]+?)\s*\)", src)
        assert m, f"no fallback found for {key} in {_PROCESSOR_PATH.name}"
        found[key] = eval(m.group(1), {"__builtins__": {}})  # a plain literal
    # The profile fallback is the string literal used when the settings have no
    # profile: `"350:2,..." if profile_cfg is None else ...`.
    m = re.search(r"['\"]([0-9.:,]+)['\"]\s+if\s+profile_cfg\s+is\s+None", src)
    assert m, f"no profile fallback found in {_PROCESSOR_PATH.name}"
    found["variable_smoothing_profile"] = m.group(1)
    for key in _HYBRID_KEYS:
        name = "hybrid_" + key
        m = re.search(
            r"dfhrtf_adv\.get\(\s*['\"]" + name + r"['\"]\s*,\s*([^)]+?)\s*\)", src)
        assert m, f"no fallback found for {name} in {_PROCESSOR_PATH.name}"
        found[name] = eval(m.group(1), {"__builtins__": {}})
    return found


def test_variable_smoothing_four_way_sync():
    from core.smoothing import format_profile, parse_profile, profile_from_dicts

    params = SmoothingParams()
    json_adv = C.load_defaults()["dfhrtf"]["advanced"]
    from core.scripts.generate_dfhrtf import build_parser
    cli = vars(build_parser().parse_args([]))
    fallbacks = _processor_fallbacks()

    def canon(profile_str):
        return format_profile(parse_profile(profile_str))

    profile = {
        "SmoothingParams": format_profile(params.points),
        "default_settings.json": format_profile(
            profile_from_dicts(json_adv["variable_smoothing_profile"])),
        "argparse": canon(cli["var_smooth_profile"]),
        "dfhrtf_processor fallback": canon(fallbacks["variable_smoothing_profile"]),
    }
    window = {
        "SmoothingParams": params.window,
        "default_settings.json": json_adv["smoothing_window"],
        "argparse": cli["smoothing_window"],
        "dfhrtf_processor fallback": fallbacks["smoothing_window"],
    }
    mean_mode = {
        "SmoothingParams": params.mean_mode,
        "default_settings.json": json_adv["smoothing_mean_mode"],
        "argparse": cli["smoothing_mean_mode"],
        "dfhrtf_processor fallback": fallbacks["smoothing_mean_mode"],
    }
    mismatches = {name: sources
                  for name, sources in (("profile", profile), ("window", window),
                                        ("mean_mode", mean_mode))
                  if len(set(sources.values())) != 1}

    assert not mismatches, (
        "Variable-smoothing defaults are out of sync. All four must agree "
        "(core/smoothing.py SmoothingParams, core/default_settings.json, "
        "generate_dfhrtf.py --var-smooth-profile / --smoothing-window / "
        "--smoothing-mean-mode argparse defaults, "
        f"core/dfhrtf_processor.py fallbacks):\n{mismatches}"
    )


def test_hybrid_smoothing_four_way_sync():
    """Hybrid parameters: HybridParams, default_settings.json, the --hybrid-*
    argparse defaults and dfhrtf_processor's fallbacks must all agree."""
    params = HybridParams()
    json_adv = C.load_defaults()["dfhrtf"]["advanced"]
    from core.scripts.generate_dfhrtf import build_parser
    cli = vars(build_parser().parse_args([]))
    fallbacks = _processor_fallbacks()

    mismatches = {}
    for key in _HYBRID_KEYS:
        name = "hybrid_" + key
        sources = {
            "HybridParams": getattr(params, key),
            "default_settings.json": json_adv[name],
            "argparse": cli[name],
            "dfhrtf_processor fallback": fallbacks[name],
        }
        if len(set(sources.values())) != 1:
            mismatches[name] = sources
    assert not mismatches, (
        "Hybrid-smoothing defaults are out of sync. All four must agree "
        "(core/hybrid_smoothing.py HybridParams, core/default_settings.json, "
        "generate_dfhrtf.py --hybrid-* argparse defaults, "
        f"core/dfhrtf_processor.py fallbacks):\n{mismatches}"
    )


def test_smoothing_keys_are_not_listed_as_intentional_divergences():
    """The synced smoothing dests must never be parked in the
    intentional-divergence table."""
    listed = _SYNCED_SMOOTHING_DESTS & set(_KNOWN_INTENTIONAL_DIVERGENCES["dfhrtf"])
    assert not listed, (
        f"{sorted(listed)} are contractually synced (see "
        "test_variable_smoothing_four_way_sync) and must not be documented as "
        "intentional divergences."
    )


_SYNCED_SMOOTHING_DESTS = {"var_smooth_profile", "smoothing_window", "smoothing_mean_mode",
                           *("hybrid_" + k for k in _HYBRID_KEYS)}


# --------------------------------------------------------------------------- #
# 2. Everything else: documented CLI-vs-GUI divergences
# --------------------------------------------------------------------------- #
# argparse dest -> dotted path in default_settings.json. Only mapped dests are
# compared; unmapped dests (--input, --output, per-job paths, store_true
# switches with no JSON twin, ...) have no factory-settings counterpart.
_SOFA_ARG_TO_JSON = {
    "lf_crossover":      "sofa.xover",
    "target_radius":     "sofa.distance_m",
    "hrir_length":       "sofa.hrir_length",
    "ild_threshold":     "sofa.ild_threshold",
    "processing_length": "sofa.advanced.processing_length",
    "norm_target_db":    "sofa.advanced.norm_target_db",
    "onset_buffer":      "sofa.advanced.onset_buffer",
    "onset_threshold":   "sofa.advanced.onset_threshold_ratio",
    "blend_low_oct":     "sofa.advanced.blend_low_oct",
    "blend_high_oct":    "sofa.advanced.blend_high_oct",
    "ild_low_freq":      "sofa.advanced.ild_low_freq",
    "ild_high_freq":     "sofa.advanced.ild_high_freq",
    # generate_sofa shares the grid-weighting flag with the DFHRTF config section.
    "weighting_mode":    "dfhrtf.weighting_mode",
}

_DFHRTF_ARG_TO_JSON = {
    "avg_algo":               "dfhrtf.per_sofa_avg_algo",
    "delimiter":              "dfhrtf.text_delimiter",
    "smoothing":              "dfhrtf.smoothing",
    "tilt":                   "dfhrtf.tilt",
    "lf_ext_dfhrtf_xover":    "dfhrtf.xover",
    "weighting_mode":         "dfhrtf.weighting_mode",
    "n_fft":                  "dfhrtf.advanced.n_fft",
    "output_octave_fraction": "dfhrtf.advanced.output_octave_fraction",
    "norm_freq_hz":           "dfhrtf.advanced.norm_freq_hz",
    "lf_blend_low_oct":       "dfhrtf.advanced.lf_blend_low_oct",
    "lf_blend_high_oct":      "dfhrtf.advanced.lf_blend_high_oct",
    "stereo_angle":           "dfhrtf.advanced.stereo_angle_deg",
    "grid_scale_mode":        "dfhrtf.advanced.weighted_grid_scale_mode",
    "grid_scale_low":         "dfhrtf.advanced.weighted_grid_rel_low",
    "grid_scale_high":        "dfhrtf.advanced.weighted_grid_rel_high",
    "preserve_lr_balance":    "dfhrtf.advanced.preserve_lr_balance",
    "smoothing_window":       "dfhrtf.advanced.smoothing_window",
    "smoothing_mean_mode":    "dfhrtf.advanced.smoothing_mean_mode",
    **{"hybrid_" + k: "dfhrtf.advanced.hybrid_" + k for k in _HYBRID_KEYS},
}

# dest -> (cli_default, json_default, rationale)
# The CLI defaults independently of the GUI's factory settings; only the
# documented variable-smoothing keys are contractually synced.
_KNOWN_INTENTIONAL_DIVERGENCES = {
    "sofa": {
        "target_radius":  (1.2, 1.5,
                           "CLI keeps its historical 1.2 m radius; the GUI factory "
                           "default moved to 1.5 m."),
        "hrir_length":    (None, 256.0,
                           "CLI None means 'no override — keep the input length'; the "
                           "GUI always sends an explicit length."),
        "ild_threshold":  (1.5, 1.2,
                           "CLI keeps the original 1.5 dB threshold; the GUI factory "
                           "default was lowered to 1.2 dB."),
        "blend_low_oct":  (0.50, 0.33,
                           "CLI keeps the original wider LF crossfade; the GUI factory "
                           "default was narrowed with the 1 kHz crossover change."),
        "blend_high_oct": (0.75, 1.0,
                           "Same crossfade retune as blend_low_oct."),
        "ild_low_freq":   (300.0, 350.0,
                           "CLI keeps the original 300-6000 Hz ILD estimation band; the "
                           "GUI factory band is 350-3500 Hz."),
        "ild_high_freq":  (6000.0, 3500.0,
                           "Same ILD band retune as ild_low_freq."),
        "weighting_mode": ("auto", "Auto",
                           "CLI enums are lowercase; the GUI stores the title-case label "
                           "it shows in the dropdown."),
    },
    "dfhrtf": {
        "avg_algo":          ("Arithmetic Mean", "Geo-Arc-Length",
                              "CLI keeps the cheap arithmetic mean as its default; the "
                              "GUI factory default is Geo-Arc-Length."),
        "delimiter":         ("\t", "Tab",
                              "CLI takes the literal delimiter character; the GUI stores "
                              "the human-readable name and translates it."),
        "smoothing":         ("variable", "Variable",
                              "CLI enums are lowercase; the GUI stores the title-case "
                              "dropdown label."),
        "weighting_mode":    ("auto", "Auto",
                              "CLI enums are lowercase; the GUI stores the title-case "
                              "dropdown label."),
        "lf_blend_low_oct":  (0.50, 0.33,
                              "CLI keeps the original wider LF crossfade; the GUI factory "
                              "default was narrowed with the 1 kHz crossover change."),
        "preserve_lr_balance": (False, True,
                              "store_true flag: off unless passed on the CLI; the GUI "
                              "factory default turns ILD preservation on."),
    },
}


def _actual_divergences(cli_defaults, mapping):
    defaults = C.load_defaults()
    out = {}
    for dest, path in mapping.items():
        json_val, found = C._get_nested(defaults, path)
        assert found, f"{path} missing from default_settings.json"
        if cli_defaults[dest] != json_val:
            out[dest] = (cli_defaults[dest], json_val)
    return out


@pytest.mark.parametrize("section", ["sofa", "dfhrtf"])
def test_cli_vs_factory_divergences_are_exactly_the_documented_set(section):
    mapping = _SOFA_ARG_TO_JSON if section == "sofa" else _DFHRTF_ARG_TO_JSON
    if section == "sofa":
        from core.scripts.generate_sofa import build_parser
    else:
        from core.scripts.generate_dfhrtf import build_parser
    cli = vars(build_parser().parse_args([]))

    actual = _actual_divergences(cli, mapping)
    documented = {k: (v[0], v[1]) for k, v in _KNOWN_INTENTIONAL_DIVERGENCES[section].items()}

    new = {k: v for k, v in actual.items() if k not in documented}
    gone = {k: v for k, v in documented.items() if k not in actual}
    changed = {
        k: {"documented": documented[k], "actual": actual[k]}
        for k in set(actual) & set(documented)
        if actual[k] != documented[k]
    }

    assert not (new or gone or changed), (
        f"{section}: CLI-vs-default_settings.json divergences no longer match the "
        f"documented set.\n"
        f"  NEW (undocumented) : {new}\n"
        f"  RESOLVED (drop from _KNOWN_INTENTIONAL_DIVERGENCES): {gone}\n"
        f"  VALUE CHANGED      : {changed}"
    )


def test_divergence_mappings_reference_real_argparse_dests():
    """Guards the mapping tables themselves against a renamed CLI flag."""
    from core.scripts.generate_dfhrtf import build_parser as dfhrtf_parser
    from core.scripts.generate_sofa import build_parser as sofa_parser

    sofa_dests = set(vars(sofa_parser().parse_args([])))
    dfhrtf_dests = set(vars(dfhrtf_parser().parse_args([])))

    assert not set(_SOFA_ARG_TO_JSON) - sofa_dests
    assert not set(_DFHRTF_ARG_TO_JSON) - dfhrtf_dests
    assert not set(_KNOWN_INTENTIONAL_DIVERGENCES["sofa"]) - set(_SOFA_ARG_TO_JSON)
    assert not set(_KNOWN_INTENTIONAL_DIVERGENCES["dfhrtf"]) - set(_DFHRTF_ARG_TO_JSON)
