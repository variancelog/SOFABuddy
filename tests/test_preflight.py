"""
tests/test_preflight.py

Headless tests for tools/preflight.py using throwaway git repos in tmp_path.
No network, and the pytest check is never run (check functions are called
directly, or main() is used with --fast).
"""
import subprocess

import pytest

from tools import preflight as pf


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _write(repo, rel, text="x\n"):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _commit(repo, msg="c"):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", msg)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "commit.gpgsign", "false")
    _git(r, "config", "tag.gpgsign", "false")
    _write(r, "core/version.py", '__version__ = "0.1.0"\n')
    _write(r, "CHANGELOG.md", "# Changelog\n\n## v0.1.0 (2026-01-01)\n- first\n")
    _write(r, "README.md", "# App\n\n![shot](shot.png)\n")
    _write(r, "shot.png", "png")
    _write(r, "ui/w.py", "a = 1\n")
    _commit(r, "init")
    _git(r, "tag", "v0.1.0")
    return r


def _status(result):
    return result[0]


def test_clean_vs_dirty_tree(repo):
    assert _status(pf.check_clean_tree(str(repo))) == "PASS"
    _write(repo, "new.txt")
    assert _status(pf.check_clean_tree(str(repo))) == "WARN"


def test_last_tag_picks_highest_semver_and_excludes(repo):
    _git(repo, "tag", "v0.10.0")
    _git(repo, "tag", "v0.9.0")
    _git(repo, "tag", "not-a-version")
    assert pf.last_tag(str(repo)) == "v0.10.0"
    assert pf.last_tag(str(repo), exclude="v0.10.0") == "v0.9.0"


def test_changelog_fails_when_code_changed_without_it(repo):
    _write(repo, "core/x.py", "y = 2\n")
    _commit(repo)
    assert _status(pf.check_changelog(str(repo), "v0.1.0")) == "FAIL"


def test_changelog_passes_when_both_changed(repo):
    _write(repo, "core/x.py", "y = 2\n")
    _write(repo, "CHANGELOG.md", "## v0.1.0 (2026-01-01)\n- more\n")
    _commit(repo)
    assert _status(pf.check_changelog(str(repo), "v0.1.0")) == "PASS"


def test_changelog_passes_when_only_docs_changed(repo):
    _write(repo, "notes.txt")
    _commit(repo)
    assert _status(pf.check_changelog(str(repo), "v0.1.0")) == "PASS"


def test_changelog_warns_without_tag(repo):
    assert _status(pf.check_changelog(str(repo), None)) == "WARN"


def test_version_ok_mismatch_and_unreleased(repo):
    assert _status(pf.check_version(str(repo))) == "PASS"
    _write(repo, "core/version.py", '__version__ = "0.2.0"\n')
    assert _status(pf.check_version(str(repo))) == "FAIL"
    _write(repo, "CHANGELOG.md", "## Unreleased\n- wip\n\n## v0.1.0 (2026-01-01)\n")
    assert _status(pf.check_version(str(repo))) == "WARN"


def test_readme_images_ok(repo):
    assert _status(pf.check_readme_images(str(repo))) == "PASS"


def test_readme_broken_image_fails(repo):
    _write(repo, "README.md", "![a](shot.png)\n![b](missing.png)\n![c](https://example.com/x.png)\n")
    status, _, detail = pf.check_readme_images(str(repo))
    assert status == "FAIL" and "missing.png" in detail and "example.com" not in detail


def test_orphan_root_png_fails(repo):
    _write(repo, "orphan.png", "png")
    _commit(repo)
    status, _, detail = pf.check_readme_images(str(repo))
    assert status == "FAIL" and "orphan.png" in detail


def test_untracked_or_nested_png_is_not_orphan(repo):
    _write(repo, "docs/other.png", "png")
    _commit(repo)
    _write(repo, "untracked.png", "png")
    assert _status(pf.check_readme_images(str(repo))) == "PASS"


@pytest.fixture
def with_origin(repo, tmp_path):
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True, capture_output=True)
    _git(repo, "remote", "add", "origin", str(bare))
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    _git(repo, "push", "-q", "-u", "origin", branch)
    return repo


def test_outgoing_no_upstream_warns(repo):
    assert _status(pf.check_outgoing_files(str(repo))) == "WARN"


def test_outgoing_clean_passes(with_origin):
    _write(with_origin, "ok.txt")
    _commit(with_origin)
    assert _status(pf.check_outgoing_files(str(with_origin))) == "PASS"


def test_outgoing_sofa_fails(with_origin):
    _write(with_origin, "data/head.sofa", "bin")
    _commit(with_origin)
    status, _, detail = pf.check_outgoing_files(str(with_origin))
    assert status == "FAIL" and "head.sofa" in detail


def test_outgoing_secret_names_fail(with_origin):
    _write(with_origin, ".env", "K=1")
    _write(with_origin, "cfg/user_settings.json", "{}")
    _commit(with_origin)
    status, _, detail = pf.check_outgoing_files(str(with_origin))
    assert status == "FAIL" and ".env" in detail and "user_settings.json" in detail


def test_outgoing_large_file_fails(with_origin):
    big = with_origin / "big.bin"
    big.write_bytes(b"0" * (5 * 1024 * 1024 + 1))
    _commit(with_origin)
    assert _status(pf.check_outgoing_files(str(with_origin))) == "FAIL"


def test_release_version_check(repo):
    assert _status(pf.check_release_version(str(repo), "v0.1.0")) == "PASS"
    assert _status(pf.check_release_version(str(repo), "v0.2.0")) == "FAIL"


def test_release_tag_absent_passes(with_origin):
    status, _, detail = pf.check_release_tag(str(with_origin), "v0.2.0")
    assert status == "PASS" and "not tagged yet" in detail


def test_release_tag_at_head_passes(with_origin):
    _git(with_origin, "tag", "v0.2.0")
    _git(with_origin, "push", "-q", "origin", "v0.2.0")
    status, _, detail = pf.check_release_tag(str(with_origin), "v0.2.0")
    assert status == "PASS" and "already tagged" in detail


def test_release_tag_elsewhere_fails(with_origin):
    _git(with_origin, "tag", "v0.2.0")
    _git(with_origin, "push", "-q", "origin", "v0.2.0")
    _write(with_origin, "more.txt")
    _commit(with_origin)
    assert _status(pf.check_release_tag(str(with_origin), "v0.2.0")) == "FAIL"


def test_release_tag_unreachable_origin_warns(repo):
    assert _status(pf.check_release_tag(str(repo), "v0.2.0")) == "WARN"


def test_screenshots_warn_when_ui_changed_without_png(repo):
    _write(repo, "ui/w.py", "a = 2\n")
    _commit(repo)
    assert _status(pf.check_screenshots(str(repo), "v0.1.0")) == "WARN"


def test_screenshots_pass_when_png_also_changed(repo):
    _write(repo, "ui/w.py", "a = 2\n")
    _write(repo, "shot.png", "new png")
    _commit(repo)
    assert _status(pf.check_screenshots(str(repo), "v0.1.0")) == "PASS"


def test_screenshots_pass_when_ui_unchanged(repo):
    _write(repo, "core/x.py")
    _commit(repo)
    assert _status(pf.check_screenshots(str(repo), "v0.1.0")) == "PASS"


def test_main_exit_codes(repo, capsys):
    assert pf.main(["--fast"], repo=str(repo)) == 0
    _write(repo, "core/x.py")
    _commit(repo)
    assert pf.main(["--fast"], repo=str(repo)) == 1
    out = capsys.readouterr().out
    assert "FAIL changelog:" in out and "preflight:" in out
