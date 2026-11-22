"""Pre-push / pre-release preflight checks (stdlib only).

Usage: python tools/preflight.py [--fast] [--release vX.Y.Z]

Prints one line per check ("PASS|WARN|FAIL <name>: <detail>") and a summary.
Exit code 1 if any check FAILs, else 0. Each check returns (status, name, detail).
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
HEADING_RE = re.compile(r"^##\s+(v?\d+\.\d+\.\d+|Unreleased)\b", re.M | re.I)
IMG_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)")
FORBIDDEN_GLOBS = ["*.sofa", "*user_settings.json", "settings.local.json", ".env", "*credentials*"]
MAX_BYTES = 5 * 1024 * 1024
CODE_PATHS = ("core/", "ui/", "main.pyw")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


# --- helpers ---------------------------------------------------------------

def git(repo, *args, timeout=30):
    """Return (returncode, stdout). Missing git / timeout -> (127, '')."""
    try:
        p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""
    return p.returncode, p.stdout


def _read(repo, name):
    try:
        with open(os.path.join(repo, name), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def semver(s):
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)$", s)
    return tuple(int(x) for x in m.groups()) if m else None


def last_tag(repo, exclude=None):
    """Highest local vX.Y.Z tag, optionally excluding one."""
    _, out = git(repo, "tag", "--list")
    tags = [t.strip() for t in out.split() if TAG_RE.match(t.strip()) and t.strip() != exclude]
    return max(tags, key=semver) if tags else None


def changelog_top(repo):
    """First '## ' heading of CHANGELOG.md: 'Unreleased', 'X.Y.Z', or None."""
    text = _read(repo, "CHANGELOG.md")
    m = HEADING_RE.search(text or "")
    if not m:
        return None
    return "Unreleased" if m.group(1).lower() == "unreleased" else m.group(1).lstrip("vV")


def changelog_first_version(repo):
    """First versioned heading (skipping Unreleased), as 'X.Y.Z', or None."""
    for m in HEADING_RE.finditer(_read(repo, "CHANGELOG.md") or ""):
        if m.group(1).lower() != "unreleased":
            return m.group(1).lstrip("vV")
    return None


def code_version(repo):
    m = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", _read(repo, "core/version.py") or "", re.M)
    return m.group(1) if m else None


def readme_images(repo):
    """Relative image paths linked from README.md (forward slashes, no anchors)."""
    out = []
    for target in IMG_RE.findall(_read(repo, "README.md") or ""):
        if re.match(r"^[a-z][a-z0-9+.-]*:|^//|^#", target, re.I):
            continue
        out.append(target.split("#")[0].split("?")[0].replace("\\", "/").removeprefix("./"))
    return out


def tracked_root_pngs(repo):
    rc, out = git(repo, "ls-files")
    if rc != 0:
        return None
    return [f for f in out.splitlines() if "/" not in f and f.lower().endswith(".png")]


# --- push-mode checks ------------------------------------------------------

def check_clean_tree(repo):
    rc, out = git(repo, "status", "--porcelain")
    if rc != 0:
        return WARN, "clean-tree", "git unavailable"
    if out.strip():
        return WARN, "clean-tree", f"{len(out.splitlines())} uncommitted change(s) won't be pushed"
    return PASS, "clean-tree", "working tree clean"


def check_changelog(repo, tag):
    name = "changelog"
    if not tag:
        return WARN, name, "no release tag found; skipped"
    rc, out = git(repo, "diff", "--name-only", f"{tag}..HEAD")
    if rc != 0:
        return WARN, name, f"could not diff against {tag}"
    files = out.splitlines()
    touched = [f for f in files if f.startswith(CODE_PATHS)]
    if touched and "CHANGELOG.md" not in files:
        return FAIL, name, f"app code changed since {tag} but CHANGELOG.md was not updated"
    return PASS, name, f"since {tag}: " + ("CHANGELOG.md updated" if touched else "no app code changes")


def check_version(repo):
    name = "version"
    ver, top = code_version(repo), changelog_top(repo)
    if ver is None:
        return FAIL, name, "could not parse __version__ from core/version.py"
    if top is None:
        return FAIL, name, "no version heading found in CHANGELOG.md"
    if top == "Unreleased":
        return WARN, name, f"CHANGELOG top is Unreleased (code is {ver}); release not cut yet"
    if ver != top:
        return FAIL, name, f"core/version.py is {ver} but CHANGELOG top heading is {top}"
    return PASS, name, f"{ver} matches CHANGELOG"


def check_readme_images(repo):
    name = "readme-images"
    if _read(repo, "README.md") is None:
        return WARN, name, "no README.md"
    links = readme_images(repo)
    missing = [p for p in links if not os.path.isfile(os.path.join(repo, p))]
    pngs = tracked_root_pngs(repo)
    orphans = [p for p in (pngs or []) if p not in links]
    problems = []
    if missing:
        problems.append("broken links: " + ", ".join(missing))
    if orphans:
        problems.append("not linked from README: " + ", ".join(orphans))
    if problems:
        return FAIL, name, "; ".join(problems)
    return PASS, name, f"{len(links)} image link(s) resolve; no orphan root PNGs"


def check_outgoing_files(repo):
    name = "outgoing-files"
    rc, _ = git(repo, "rev-parse", "--abbrev-ref", "@{u}")
    if rc != 0:
        return WARN, name, "no upstream configured; skipped"
    rc, out = git(repo, "diff", "--name-only", "--diff-filter=AM", "@{u}..HEAD")
    if rc != 0:
        return WARN, name, "could not diff against upstream"
    bad = []
    for path in out.splitlines():
        base = path.rsplit("/", 1)[-1]
        if any(fnmatch.fnmatch(base.lower(), g) for g in FORBIDDEN_GLOBS):
            bad.append(path)
            continue
        rc, size = git(repo, "cat-file", "-s", f"HEAD:{path}")
        if rc == 0 and size.strip().isdigit() and int(size) > MAX_BYTES:
            bad.append(f"{path} (>{MAX_BYTES // 2**20} MB)")
    if bad:
        return FAIL, name, "unwanted outgoing file(s): " + ", ".join(bad)
    return PASS, name, "no forbidden or oversized files in outgoing commits"


def check_tests(repo):
    name = "tests"
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=repo,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as e:
        return FAIL, name, f"could not run pytest: {e}"
    lines = [l for l in p.stdout.strip().splitlines() if l.strip()]
    last = lines[-1] if lines else "no output"
    return (PASS if p.returncode == 0 else FAIL), name, last


# --- release-mode checks ---------------------------------------------------

def check_release_version(repo, release):
    name = "release-version"
    arg, ver, top = release.lstrip("vV"), code_version(repo), changelog_first_version(repo)
    if arg == ver == top:
        return PASS, name, f"{release} matches core/version.py and CHANGELOG"
    return FAIL, name, f"arg={arg} core/version.py={ver} CHANGELOG={top}"


def check_release_tag(repo, release):
    name = "release-tag"
    rc, out = git(repo, "ls-remote", "--tags", "origin", f"refs/tags/{release}", f"refs/tags/{release}^{{}}", timeout=15)
    if rc != 0:
        return WARN, name, "could not reach origin; skipped"
    refs = {}
    for line in out.splitlines():
        sha, _, ref = line.partition("\t")
        refs[ref.strip()] = sha
    remote = refs.get(f"refs/tags/{release}^{{}}") or refs.get(f"refs/tags/{release}")
    if not remote:
        return PASS, name, "not tagged yet - tag after pushing"
    _, head = git(repo, "rev-parse", "HEAD")
    if remote == head.strip():
        return PASS, name, "already tagged at HEAD"
    return FAIL, name, f"{release} exists on origin at {remote[:8]}, not HEAD ({head.strip()[:8]})"


def check_screenshots(repo, tag):
    name = "screenshots"
    if not tag:
        return WARN, name, "no previous tag; skipped"
    _, ui = git(repo, "diff", "--name-only", f"{tag}..HEAD", "--", "ui/")
    if not ui.strip():
        return PASS, name, f"no UI changes since {tag}"
    _, changed = git(repo, "diff", "--name-only", f"{tag}..HEAD")
    shots = set(readme_images(repo))
    if shots & set(changed.splitlines()):
        return PASS, name, f"UI changed since {tag}; README screenshots also updated"
    return WARN, name, f"UI changed since {tag}; screenshots may be stale"


# --- driver ----------------------------------------------------------------

def run_checks(repo=REPO_ROOT, fast=False, release=None):
    tag = last_tag(repo, exclude=release)
    results = [
        check_clean_tree(repo),
        check_changelog(repo, tag),
        check_version(repo),
        check_readme_images(repo),
        check_outgoing_files(repo),
    ]
    if not fast:
        results.append(check_tests(repo))
    if release:
        results += [check_release_version(repo, release),
                    check_release_tag(repo, release),
                    check_screenshots(repo, tag)]
    return results


def main(argv=None, repo=REPO_ROOT):
    ap = argparse.ArgumentParser(description="SOFA Buddy preflight checks")
    ap.add_argument("--fast", action="store_true", help="skip the pytest run")
    ap.add_argument("--release", metavar="vX.Y.Z", help="also run release checks")
    args = ap.parse_args(argv)
    if args.release and not TAG_RE.match(args.release):
        ap.error("--release must look like vX.Y.Z")
    results = run_checks(repo, fast=args.fast, release=args.release)
    for status, name, detail in results:
        print(f"{status} {name}: {detail}")
    counts = {s: sum(1 for r in results if r[0] == s) for s in (PASS, WARN, FAIL)}
    print(f"preflight: {counts[PASS]} passed, {counts[WARN]} warning(s), {counts[FAIL]} failure(s)")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
