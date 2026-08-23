"""
tools/extract_grid_fixtures.py

Extract the measurement-grid geometry (the SOFA ``SourcePosition`` array) from
every real ``.sofa`` file in ``tests/data/sofa/`` into a small committed ``.npy``
fixture under ``tests/data/grids/``.

Why
---
The real ``.sofa`` corpus is gitignored (``*.sofa`` in ``.gitignore``) because the
files are 1.6-11 MB each. Tests that only need the *grid* — e.g. the weighting
classifier's "real measured grids must classify as geometric" guard — therefore
skipped on every clean clone. A ``SourcePosition`` array is (M, 3) float64, i.e.
a few KB to ~40 KB, and is not matched by the ``*.sofa`` ignore rule, so it can
be committed and the test can always run.

Usage
-----
    python tools/extract_grid_fixtures.py            # write tests/data/grids/*.npy
    python tools/extract_grid_fixtures.py --check    # verify only, write nothing

Files are read with ``core.sofa_io.read_sofa_robust()`` — the same loader the
production code uses — so legacy/non-conformant files are handled identically.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SOFA_DIR = os.path.join(ROOT, "tests", "data", "sofa")
GRID_DIR = os.path.join(ROOT, "tests", "data", "grids")


def extract(sofa_path):
    """Return the (M, 3) SourcePosition array from *sofa_path* as float64."""
    from core.sofa_io import read_sofa_robust

    sofa = read_sofa_robust(sofa_path)
    sp = np.asarray(sofa.SourcePosition, dtype=np.float64)
    if sp.ndim != 2 or sp.shape[1] != 3:
        raise ValueError(f"{sofa_path}: unexpected SourcePosition shape {sp.shape}")
    return sp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sofa-dir", default=SOFA_DIR)
    ap.add_argument("--out-dir", default=GRID_DIR)
    ap.add_argument("--check", action="store_true",
                    help="Read and report only; do not write .npy files.")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.sofa_dir):
        print(f"[!] No SOFA source directory: {args.sofa_dir}")
        return 1

    names = sorted(f for f in os.listdir(args.sofa_dir) if f.endswith(".sofa"))
    if not names:
        print(f"[!] No .sofa files in {args.sofa_dir}")
        return 1

    if not args.check:
        os.makedirs(args.out_dir, exist_ok=True)

    failures = []
    for name in names:
        src = os.path.join(args.sofa_dir, name)
        stem = os.path.splitext(name)[0]
        dst = os.path.join(args.out_dir, stem + ".npy")
        try:
            sp = extract(src)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((name, exc))
            print(f"[FAIL] {name}: {exc}")
            continue
        if args.check:
            print(f"[OK]   {name}: SourcePosition {sp.shape} (would write {dst})")
        else:
            np.save(dst, sp)
            print(f"[OK]   {name}: SourcePosition {sp.shape} -> "
                  f"{os.path.relpath(dst, ROOT)} ({os.path.getsize(dst)} bytes)")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
