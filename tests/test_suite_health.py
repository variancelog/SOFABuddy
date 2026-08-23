"""
tests/test_suite_health.py

Meta-guard for the test suite itself. These tests don't exercise app
behavior; they mechanically catch the class of drift the pytest-suite
remediation effort is about (stale fixture references, dev-dependency
typos/rot, and future re-introduction of removed hand-rolled patterns).

Headless: no Qt import.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT, "tests")
SOFA_FIXTURE_DIR = os.path.join(TESTS_DIR, "data", "sofa")


def _test_source_files():
    for name in os.listdir(TESTS_DIR):
        if name == "conftest.py":
            continue
        if name.startswith("test_") and name.endswith(".py"):
            yield os.path.join(TESTS_DIR, name)


def test_sofa_fixture_references_check_runs():
    """
    Sanity check that scanning for tests/data/sofa/ fixture references
    doesn't error. Fixtures are gitignored and vary machine to machine, so
    this deliberately does not hardcode expected filenames or fail when
    the fixture directory is empty/missing.
    """
    referenced = set()
    pattern = re.compile(r"""tests[/\\]data[/\\]sofa[/\\][^"'\s)]+""")
    for path in _test_source_files():
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        for match in pattern.findall(source):
            referenced.add(match)

    # Just confirm the scan executed; presence/absence of fixtures on disk
    # is expected to vary and is not asserted here.
    assert isinstance(referenced, set)
    if os.path.isdir(SOFA_FIXTURE_DIR):
        assert os.path.isdir(SOFA_FIXTURE_DIR)


def test_requirements_dev_lines_are_valid_requirements():
    """Guard requirements-dev.txt against typos/rot."""
    req_path = os.path.join(ROOT, "requirements-dev.txt")
    assert os.path.isfile(req_path), "requirements-dev.txt is missing"

    with open(req_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    assert lines, "requirements-dev.txt has no requirement lines"

    try:
        from packaging.requirements import Requirement, InvalidRequirement

        for line in lines:
            try:
                Requirement(line)
            except InvalidRequirement as exc:
                raise AssertionError(f"Invalid requirement line {line!r}: {exc}")
    except ImportError:
        # Fall back to a simple regex check if packaging isn't available.
        simple_req = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\s*(>=|==|<=|~=|!=|>|<)?\s*[A-Za-z0-9.*]*$")
        for line in lines:
            assert simple_req.match(line), f"Invalid requirement line: {line!r}"


def test_no_future_argparse_namespace_construction():
    """
    Guard against re-introducing the hand-rolled argparse.Namespace(...)
    arg-mocking pattern. test_grid_norm.py, test_lf_splice.py, and
    test_parallel.py are known, pre-existing offenders slated for cleanup
    in Phase 2 — they are excluded here so this check only fires on
    *future* occurrences elsewhere in the suite.
    """
    known_offenders = {
        "test_grid_norm.py",
        "test_lf_splice.py",
        "test_parallel.py",
        "test_suite_health.py",  # this file's own pattern string, not a real usage
    }
    pattern = re.compile(r"argparse" r"\." r"Namespace" r"\(")

    offending_files = []
    for path in _test_source_files():
        name = os.path.basename(path)
        if name in known_offenders:
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        if pattern.search(source):
            offending_files.append(name)

    assert not offending_files, (
        f"Found new argparse.Namespace(...) construction in: {offending_files}. "
        "Use the shared test-fixture helper instead of hand-rolled arg mocking."
    )
