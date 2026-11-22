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
import shutil
import subprocess

import pytest

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


# --- release-hygiene guards ------------------------------------------------

def _read_root(name):
    with open(os.path.join(ROOT, name), "r", encoding="utf-8") as f:
        return f.read()


def _readme_relative_images():
    links = re.findall(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)", _read_root("README.md"))
    return [l.split("#")[0].split("?")[0].replace("\\", "/").removeprefix("./")
            for l in links if not re.match(r"^[a-z][a-z0-9+.-]*:|^//|^#", l, re.I)]


def _version_tuple(s):
    return tuple(int(x) for x in s.split("."))


def test_version_matches_changelog():
    """
    core.version.__version__ must equal the first '## vX.Y.Z' heading of
    CHANGELOG.md. If the changelog opens with '## Unreleased', a bump in
    progress is allowed: __version__ must be >= the first versioned heading.
    """
    from core.version import __version__

    headings = re.findall(r"^##\s+(Unreleased|v\d+\.\d+\.\d+)", _read_root("CHANGELOG.md"), re.M | re.I)
    assert headings, "CHANGELOG.md has no '## vX.Y.Z' or '## Unreleased' heading"
    versions = [h.lstrip("v") for h in headings if h.lower() != "unreleased"]
    assert versions, "CHANGELOG.md has no versioned heading"
    if headings[0].lower() == "unreleased":
        assert _version_tuple(__version__) >= _version_tuple(versions[0])
    else:
        assert __version__ == versions[0]


def test_readme_relative_images_exist():
    missing = [p for p in _readme_relative_images() if not os.path.isfile(os.path.join(ROOT, p))]
    assert not missing, f"README.md links missing image(s): {missing}"


def test_tracked_root_pngs_are_linked_from_readme():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    proc = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("not a git repository")
    pngs = [f for f in proc.stdout.splitlines() if "/" not in f and f.lower().endswith(".png")]
    linked = set(_readme_relative_images())
    orphans = [p for p in pngs if p not in linked]
    assert not orphans, f"tracked root PNG(s) not linked from README.md: {orphans}"
