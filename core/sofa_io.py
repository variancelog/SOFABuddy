"""
Shared SOFA file loading utilities.

Provides read_sofa_robust(), which:
  1. Strips legacy NC_GLOBAL. attribute prefixes (old SOFA C++ API 1.x files).
  2. Attempts a verified sofar read; on ValueError (convention check failed due to
     non-standard custom entries) falls back to verify=False with a clear warning.
  3. Passes the loaded object through sanitize_sofa() to repair known
     convention violations in-memory before returning — original files are
     never modified.

Both backend scripts import from here so the logic lives in one place.
"""

import os
import tempfile
import warnings

warnings.filterwarnings("ignore", message=r".*doesn't match a supported version.*")
import sofar as sf  # noqa: E402


def _normalize_legacy_sofa(input_path):
    """
    Old SOFA files written by the SOFA C++ API 1.x (e.g. IRCAM BILI) store global
    attributes with an 'NC_GLOBAL.' prefix in the attribute name instead of bare names.
    sofar expects bare names like 'SOFAConventions' and fails with AttributeError.
    This creates a temp copy with the prefix stripped so sofar can read the file.
    Returns (path_to_read, temp_path_or_None); caller must delete temp_path when done.
    """
    import netCDF4 as nc4
    with nc4.Dataset(input_path, 'r') as src:
        attrs = src.ncattrs()
        is_legacy = any(a.startswith('NC_GLOBAL.') for a in attrs) and 'SOFAConventions' not in attrs
        if not is_legacy:
            return input_path, None

    fd, tmp_path = tempfile.mkstemp(suffix='.sofa')
    os.close(fd)
    try:
        with nc4.Dataset(input_path, 'r') as src, nc4.Dataset(tmp_path, 'w', format='NETCDF4') as dst:
            for name, dim in src.dimensions.items():
                dst.createDimension(name, None if dim.isunlimited() else len(dim))

            # Build a lookup of variable-scoped attrs stored as globals (e.g. "Data.SamplingRate.Units")
            var_attr_globals = {}
            for a in attrs:
                parts = a.rsplit('.', 1)
                if len(parts) == 2 and parts[0] in src.variables:
                    var_attr_globals.setdefault(parts[0], {})[parts[1]] = getattr(src, a)

            for name, var in src.variables.items():
                nv = dst.createVariable(name, var.datatype, var.dimensions)
                nv[:] = var[:]
                for a in var.ncattrs():
                    setattr(nv, a, getattr(var, a))
                for extra_attr, val in var_attr_globals.get(name, {}).items():
                    if extra_attr not in var.ncattrs():
                        setattr(nv, extra_attr, val)

            for a in attrs:
                if a in var_attr_globals or (a.rsplit('.', 1)[0] if '.' in a else None) in var_attr_globals:
                    continue  # already handled as variable attr
                val = getattr(src, a)
                bare = a[len('NC_GLOBAL.'):] if a.startswith('NC_GLOBAL.') else a
                if bare and bare != '_NCProperties':
                    setattr(dst, bare, val)
    except Exception:
        os.unlink(tmp_path)
        raise
    return tmp_path, tmp_path


def sanitize_sofa(sofa):
    """
    Repair known SOFA convention violations in-memory.
    Non-destructive: the source file is never modified. Extend with new rules
    as non-conformant databases are encountered.
    """
    import numpy as np

    # Rule: Emitter dimension must be 1 for SimpleFreeFieldHRIR.
    # Some databases (e.g. Princeton 3D3A) declare SimpleFreeFieldHRIR but store
    # multiple emitter positions (E > 1), which sofar rejects on write:
    #   "Dimension E is of size N but must be 1 if GLOBAL_SOFAConventions is SimpleFreeFieldHRIR"
    # Data.IR is (M, R, N) with no emitter dependence, so collapsing E to 1
    # via the convention default (origin, cartesian) is lossless.
    try:
        ep = np.asarray(sofa.EmitterPosition)
        if ep.ndim >= 1 and ep.shape[0] != 1:
            print(
                f"[!] Sanitizing: collapsing EmitterPosition from E={ep.shape[0]} "
                "to E=1 (convention requires E=1 for SimpleFreeFieldHRIR; "
                "emitter data is not used in HRIR processing)"
            )
            sofa.EmitterPosition = np.zeros((1, 3, 1))
    except AttributeError:
        pass

    return sofa


def read_sofa_robust(path):
    """
    Load a SOFA file tolerantly:
      - Normalises legacy NC_GLOBAL. attribute prefixes.
      - Tries a verified read first (sofar default).
      - If sofar raises ValueError because the file contains non-standard custom
        entries (e.g. Princeton 3D3A room metadata), falls back to verify=False
        and prints a warning so the issue is visible in the log.
      - Passes the result through sanitize_sofa() to repair known convention
        violations in-memory before returning.
      - Re-raises any other exception unchanged.
    """
    read_path, tmp_path = _normalize_legacy_sofa(path)
    try:
        try:
            sofa = sf.read_sofa(read_path)
        except ValueError:
            print(
                "[!] SOFA verification failed; re-reading with verify=False "
                "(file contains custom/non-standard entries that sofar cannot verify)"
            )
            sofa = sf.read_sofa(read_path, verify=False)
    finally:
        if tmp_path:
            os.unlink(tmp_path)
    return sanitize_sofa(sofa)
