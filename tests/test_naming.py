"""Tests for core.naming.render_filename — the shared filename-template renderer."""

import os

import pytest

from core.naming import render_filename


# --- Token substitution -----------------------------------------------------

@pytest.mark.parametrize("token,value,expected", [
    ("filename", "subject01", "subject01"),
    ("folder", "HUTUBS", "HUTUBS"),
    ("samplerate", 48000, "48000"),
    ("eq", "dfeq", "dfeq"),
    ("tilt", "-0.5", "-0.5"),
    ("LR", "L", "L"),
    ("graphtype", "weighted", "weighted"),
    ("algo", "softdtw", "softdtw"),
])
def test_each_token_substitutes(token, value, expected):
    assert render_filename("{%s}" % token, {token: value}) == expected


def test_multiple_tokens_in_one_template():
    out = render_filename(
        "{folder}_{filename}_{samplerate}_{eq}",
        {"folder": "HUTUBS", "filename": "pp1", "samplerate": 44100, "eq": "uneq"},
    )
    assert out == "HUTUBS_pp1_44100_uneq"


def test_repeated_token_substitutes_every_occurrence():
    assert render_filename("{LR}-{LR}", {"LR": "R"}) == "R-R"


def test_non_string_values_are_coerced():
    assert render_filename("{samplerate}", {"samplerate": 48000}) == "48000"


# --- Unknown tokens ---------------------------------------------------------

def test_unknown_token_passes_through_unchanged():
    # Matches the historical .replace()-chain behavior: a token the caller does
    # not supply is left verbatim rather than erroring or silently vanishing.
    assert render_filename("{filename}_{foo}", {"filename": "pp1"}) == "pp1_{foo}"


def test_extra_tokens_not_in_template_are_ignored():
    assert render_filename("{filename}", {"filename": "pp1", "eq": "dfeq"}) == "pp1"


# --- drop -------------------------------------------------------------------

def test_drop_removes_token_and_underscore_separator():
    assert render_filename("{filename}_{algo}", {"filename": "avg"}, drop={"algo"}) == "avg"


def test_drop_removes_token_and_hyphen_separator():
    # This is the case the old triple-replace hack got wrong: it only stripped
    # "_{algo}"/"{algo}_" and would have left a dangling "avg-".
    assert render_filename("{filename}-{algo}", {"filename": "avg"}, drop={"algo"}) == "avg"


def test_drop_removes_token_and_space_separator():
    assert render_filename("{filename} {algo}", {"filename": "avg"}, drop={"algo"}) == "avg"


def test_drop_with_leading_token_removes_trailing_separator():
    assert render_filename("{algo}_{filename}", {"filename": "avg"}, drop={"algo"}) == "avg"


def test_drop_bare_token_with_no_separator():
    assert render_filename("{algo}", {"filename": "avg"}, drop={"algo"}) == ""


def test_drop_in_the_middle_leaves_one_separator():
    out = render_filename("{filename}_{algo}_{LR}",
                          {"filename": "avg", "LR": "L"}, drop={"algo"})
    assert out == "avg_L"


def test_drop_wins_over_tokens():
    out = render_filename("{filename}_{algo}",
                          {"filename": "avg", "algo": "softdtw"}, drop={"algo"})
    assert out == "avg"


def test_drop_absent_token_is_a_noop():
    assert render_filename("{filename}", {"filename": "avg"}, drop={"algo"}) == "avg"


# --- ext --------------------------------------------------------------------

def test_ext_appended_when_missing():
    assert render_filename("{filename}", {"filename": "pp1"}, ext=".txt") == "pp1.txt"


def test_ext_not_doubled_when_already_present():
    assert render_filename("{filename}.txt", {"filename": "pp1"}, ext=".txt") == "pp1.txt"


def test_ext_check_is_case_insensitive():
    assert render_filename("{filename}.TXT", {"filename": "pp1"}, ext=".txt") == "pp1.TXT"
    assert render_filename("{filename}.txt", {"filename": "pp1"}, ext=".TXT") == "pp1.txt"


def test_ext_appended_after_drop_cleanup():
    out = render_filename("{filename}_{algo}", {"filename": "avg"},
                          ext=".txt", drop={"algo"})
    assert out == "avg.txt"


def test_no_ext_leaves_name_bare():
    assert render_filename("{filename}", {"filename": "pp1"}) == "pp1"


def test_ext_matched_from_substituted_value():
    # The extension check runs on the rendered name, not the raw template.
    assert render_filename("{filename}", {"filename": "pp1.sofa"}, ext=".sofa") == "pp1.sofa"


# --- No double substitution -------------------------------------------------

def test_literal_token_in_value_is_not_resubstituted():
    # A source filename that literally contains "{eq}" must survive verbatim:
    # substitution is a single pass per token with no re-scanning.
    out = render_filename("{filename}_{eq}",
                          {"filename": "weird{eq}name", "eq": "dfeq"})
    assert out == "weird{eq}name_dfeq"


def test_token_value_containing_another_token_is_not_expanded():
    out = render_filename("{filename}", {"filename": "{samplerate}", "samplerate": 48000})
    assert out == "{samplerate}"


def test_dropped_token_appearing_in_a_value_survives():
    out = render_filename("{filename}_{algo}",
                          {"filename": "a{algo}b"}, drop={"algo"})
    assert out == "a{algo}b"


# --- Known gap: no path sanitization ---------------------------------------

@pytest.mark.xfail(
    reason="KNOWN GAP: render_filename does not sanitize path traversal. Token "
           "values containing '..' or path separators flow straight into the "
           "rendered filename, so a crafted {filename}/{folder} can escape the "
           "intended output directory. This test asserts the desired behavior "
           "and will turn green automatically once sanitization is added.",
    strict=True,
)
def test_path_traversal_in_tokens_is_sanitized():
    out = render_filename("{folder}_{filename}",
                          {"folder": "..", "filename": "../../etc/passwd"},
                          ext=".txt")
    # Desired: the rendered name is a single path component that cannot escape
    # the directory it is joined into.
    assert os.sep not in out
    assert "/" not in out
    assert ".." not in out


@pytest.mark.parametrize("dropdown", ["Variable", "Hybrid", "1/3", "1/6", "1/12", "1/24", "1/48"])
def test_smoothing_token_replaces_slash(dropdown):
    # dfhrtf_processor builds the value as dropdown.replace("/", "-")
    out = render_filename("{filename} {smoothing}",
                          {"filename": "pp1", "smoothing": dropdown.replace("/", "-")},
                          ext=".txt")
    assert out == "pp1 %s.txt" % dropdown.replace("/", "-")
    assert "/" not in out
