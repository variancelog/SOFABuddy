"""
tests/audit_diffuse_ild.py
Standalone CLI auditor for diffuse-field inter-aural level difference (ILD)
in raw SOFA files.

Usage
-----
    # Audit all .sofa files in a folder (defaults to tests/data/sofa/)
    python tests/audit_diffuse_ild.py

    # Audit a specific folder with a tighter threshold
    python tests/audit_diffuse_ild.py path/to/sofa/files --diff 1.5

    # Flag AND correct offenders (writes <name>_ildfix.sofa alongside sources)
    python tests/audit_diffuse_ild.py --fix

    # Geometric weighting mode, no corrections
    python tests/audit_diffuse_ild.py --weighting-mode geometric

Exit code
---------
    0  — all files within threshold (or all offenders successfully corrected
          when --fix is set).
    1  — one or more unfixed offenders remain.
"""

import argparse
import os
import sys

# Bootstrap import path so this script can run from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _parse_args():
    p = argparse.ArgumentParser(
        description="Audit SOFA files for diffuse-field ILD calibration offsets."
    )
    p.add_argument(
        "folder",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__)),
        help="Folder containing .sofa files (default: file dir)",
    )
    p.add_argument(
        "--diff",
        type=float,
        default=1.8,
        metavar="dB",
        help="ILD threshold in dB: files with |est-ILD| above this are flagged "
             "as offenders (default: 1.8).",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Write a corrected copy (<name>-n.sofa) for each offender. "
             "Source files are never overwritten.",
    )
    p.add_argument(
        "--weighting-mode",
        choices=["auto", "geometric", "uniform"],
        default="auto",
        help="Geometry weighting mode passed to the DF calculator (default: auto).",
    )
    return p.parse_args()


def _short(name, width=34):
    """Truncate/pad a filename stem to a fixed display width."""
    if len(name) > width:
        name = name[: width - 1] + "…"
    return f"{name:{width}s}"


def main():
    args = _parse_args()

    try:
        import sofar as sf
    except ImportError:
        print("ERROR: 'sofar' is required for this script.  pip install sofar")
        sys.exit(1)

    from core.ild_audit import (
        diffuse_field_ild,
        apply_ild_fix,
        BROADBAND_TOL,
    )

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"ERROR: folder not found: {folder}")
        sys.exit(1)

    sofa_files = sorted(f for f in os.listdir(folder) if f.endswith(".sofa")
                        and not f.endswith("-n.sofa"))
    if not sofa_files:
        print(f"No .sofa files found in {folder}")
        sys.exit(0)

    # Header
    col_w = 34
    print()
    print(
        f"{'file':{col_w}s}  {'rawILD':>7s}  {'wILD':>7s}  "
        f"{'LF-ILD':>7s}  {'est-ILD':>8s}  {'verdict':<22s}"
        + ("  fix->residual" if args.fix else "")
    )
    print("-" * (col_w + 2 + 7 + 2 + 7 + 2 + 7 + 2 + 8 + 2 + 22
                 + (16 if args.fix else 0)))

    offenders_remaining = 0

    for fname in sofa_files:
        path = os.path.join(folder, fname)
        stem = fname[:-5]  # strip .sofa
        try:
            sofa = sf.read_sofa(path)
        except Exception as exc:
            print(f"{_short(stem)}  ERROR loading: {exc}")
            continue

        try:
            r = diffuse_field_ild(sofa, weighting_mode=args.weighting_mode)
        except Exception as exc:
            print(f"{_short(stem)}  ERROR in audit: {exc}")
            continue

        is_offender = abs(r["ild_estimate_db"]) > args.diff
        flag = " <==" if is_offender else ""

        fix_col = ""
        if args.fix and is_offender:
            out_path = os.path.join(folder, stem + "-n.sofa")
            try:
                apply_ild_fix(sofa, r["ild_estimate_db"])
                sf.write_sofa(out_path, sofa)
                # Re-measure residual from the written file.
                sofa2 = sf.read_sofa(out_path)
                r2 = diffuse_field_ild(sofa2, weighting_mode=args.weighting_mode)
                fix_col = f"  {r2['ild_estimate_db']:+.2f} dB"
            except Exception as exc:
                fix_col = f"  FIX-ERROR: {exc}"
                offenders_remaining += 1
        elif is_offender:
            offenders_remaining += 1

        print(
            f"{_short(stem)}  "
            f"{r['raw_energy_ild_db']:+7.2f}  "
            f"{r['broadband_ild_db']:+7.2f}  "
            f"{r['lf_ild_db']:+7.2f}  "
            f"{r['ild_estimate_db']:+8.2f}  "
            f"{r['verdict']:<22s}"
            f"{flag}{fix_col}"
        )

    print()
    if offenders_remaining:
        print(
            f"  {offenders_remaining} file(s) exceed the {args.diff} dB threshold"
            + (" (unfixed — re-run with --fix to correct)" if not args.fix else "")
            + "."
        )
        sys.exit(1)
    else:
        msg = "All files within threshold."
        if args.fix:
            msg += "  Corrected files written as *-n.sofa."
        print(" ", msg)
        sys.exit(0)


if __name__ == "__main__":
    main()
