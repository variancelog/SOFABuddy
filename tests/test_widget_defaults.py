"""Widget-default drift guards.

`core/default_settings.json` is the single source of truth for what a freshly
constructed tab shows.  These tests pin that contract from both ends:

* a newly built tab's `get_config()` must equal the factory JSON, so no
  construction literal in `ui/tabs.py` can quietly disagree with it;
* `set_config({})` — the "every key missing" path — must land on the same
  factory values, so no `cfg.get(key, <literal>)` fallback can drift either;
* every surfaced-in-JSON key must actually be surfaced by `get_config()`,
  except an explicitly documented JSON-only set, so a newly added JSON key
  cannot silently go unwired in the UI.

Modelled on `tests/test_config.py::TestSchemaDriftGuard`.
"""

import pytest

import core.config as C
from ui.tabs import DfhrtfTab, SofaTab


pytestmark = pytest.mark.qt


# Top-level keys under `sofa` / `dfhrtf` in default_settings.json that are
# deliberately NOT surfaced by the corresponding tab's get_config().
_JSON_ONLY = {
    # The whole advanced block is JSON-only tunables by design.
    "sofa": {"advanced"},
    "dfhrtf": {
        "advanced",
        # Average-output filename templates: JSON-only, no UI control yet.
        # See core/naming.py and README.md "Filename Template Variables".
        "avg_filename",
        "avg_graph_filename",
        "per_sofa_avg_filename",
    },
}


def _factory(section):
    return C.load_defaults()[section]


def _expected_for(got, section):
    """The factory values restricted to the keys the tab actually surfaces."""
    factory = _factory(section)
    missing = sorted(set(got) - set(factory))
    assert not missing, (
        f"{section} tab surfaces key(s) with no entry in default_settings.json: "
        f"{missing}"
    )
    return {k: factory[k] for k in got}


# --------------------------------------------------------------------------- #
# Construction defaults
# --------------------------------------------------------------------------- #
def test_sofa_tab_construction_defaults_match_factory_json(qtbot):
    tab = SofaTab()
    qtbot.addWidget(tab)
    got = tab.get_config()
    assert got == _expected_for(got, "sofa")


def test_dfhrtf_tab_construction_defaults_match_factory_json(qtbot):
    tab = DfhrtfTab()
    qtbot.addWidget(tab)
    got = tab.get_config()
    assert got == _expected_for(got, "dfhrtf")


# --------------------------------------------------------------------------- #
# set_config() fallbacks
# --------------------------------------------------------------------------- #
def test_dfhrtf_smoothing_dropdown_offers_and_round_trips_hybrid(qtbot):
    tab = DfhrtfTab()
    qtbot.addWidget(tab)
    tab.set_config({"smoothing": "Hybrid"})
    assert tab.get_config()["smoothing"] == "Hybrid"
    tab.set_config({})
    assert tab.get_config()["smoothing"] == _factory("dfhrtf")["smoothing"]


def test_sofa_set_config_fallbacks_are_factory_values(qtbot):
    tab = SofaTab()
    qtbot.addWidget(tab)
    tab.set_config({})
    got = tab.get_config()
    assert got == _expected_for(got, "sofa")


def test_dfhrtf_set_config_fallbacks_are_factory_values(qtbot):
    tab = DfhrtfTab()
    qtbot.addWidget(tab)
    tab.set_config({})
    got = tab.get_config()
    assert got == _expected_for(got, "dfhrtf")


# --------------------------------------------------------------------------- #
# Surface coverage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section, cls", [("sofa", SofaTab), ("dfhrtf", DfhrtfTab)])
def test_every_factory_key_is_surfaced_or_documented_json_only(section, cls, qtbot):
    """A new key in default_settings.json must either get a UI control or be
    added to `_JSON_ONLY` with a reason — never silently go unsurfaced."""
    tab = cls()
    qtbot.addWidget(tab)
    unsurfaced = set(_factory(section)) - set(tab.get_config())
    assert unsurfaced == _JSON_ONLY[section], (
        f"{section}: unsurfaced keys {sorted(unsurfaced)} != documented JSON-only "
        f"set {sorted(_JSON_ONLY[section])}. Either wire the key into "
        f"ui/tabs.py's {cls.__name__}.get_config()/set_config(), or add it to "
        f"_JSON_ONLY here with a one-line rationale."
    )


# --------------------------------------------------------------------------- #
# Dependent enable-state derived from the (now correct) defaults
# --------------------------------------------------------------------------- #
def test_sofa_xover_slider_enable_state_tracks_lf_extension(qtbot):
    """`lf_extension` defaults to True, so the crossover slider must be live at
    construction. Regression guard for the hand-set `setEnabled(False)` that
    used to desync from the factory default."""
    tab = SofaTab()
    qtbot.addWidget(tab)
    lf_default = C.load_defaults()["sofa"]["lf_extension"]
    assert tab.btn_lf.get_value() is lf_default
    # isEnabledTo() ignores the (disabled-by-default) parent container.
    assert tab.slider_xover.isEnabledTo(tab.container) is lf_default
