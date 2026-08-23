"""Tests for `core/sofa_io.py` — the tolerant SOFA loader.

`read_sofa_robust()` runs first on every file a user drops into the app, so a
bug here surfaces as "your file won't open". These tests walk its fallback
ladder rung by rung with real on-disk netCDF files (no mocking of sofar's
reader except where a rung can only be reached synthetically):

1. **Normal read** — a conformant file is read strictly, with no fallback and
   no temp file.
2. **Legacy `NC_GLOBAL.` prefixes** — a file written the way the SOFA C++ API
   1.x wrote them (globals prefixed, variable attrs stored as dotted globals)
   makes a plain `sofar.read_sofa()` raise `AttributeError`; `read_sofa_robust`
   normalizes it into a temp copy and reads it.
3. **`verify=False` fallback** — a file whose `RoomType` is `"free-field"`
   makes a strict read raise `ValueError`; the loader retries unverified.

Plus `sanitize_sofa()`'s in-memory repairs, the guarantee that the *source
file on disk is never modified*, and that the legacy temp copy is unlinked
(a Windows file-lock regression there would leak one temp file per input file
per run).

The netCDF rebuild helpers below deliberately construct the malformed files by
hand rather than through sofar — sofar refuses to *write* exactly the files
this module exists to *read*.
"""

import glob
import hashlib
import os
import tempfile
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", message=r".*doesn't match a supported version.*")

import sofar as sf  # noqa: E402

from core import sofa_io  # noqa: E402
from core.sofa_io import read_sofa_robust, sanitize_sofa  # noqa: E402


# --------------------------------------------------------------------------- #
# netCDF rebuild helpers
# --------------------------------------------------------------------------- #
def _rebuild(src_path, dst_path, *, prefix_globals=False,
             var_attrs_as_globals=(), dim_overrides=None, var_data=None,
             global_overrides=None):
    """Copy a netCDF SOFA file, optionally deforming it.

    prefix_globals
        Rename every global attribute `X` -> `NC_GLOBAL.X` (legacy layout).
    var_attrs_as_globals
        Sequence of `"Variable.Attr"` names to *move* off the variable and
        store as a bare dotted global instead (also legacy layout).
    dim_overrides / var_data / global_overrides
        Dimension sizes, variable payloads and global attribute values to
        replace while copying.
    """
    import netCDF4 as nc4

    dim_overrides = dim_overrides or {}
    var_data = var_data or {}
    global_overrides = global_overrides or {}
    moved = {tuple(name.rsplit(".", 1)) for name in var_attrs_as_globals}

    with nc4.Dataset(src_path) as src, nc4.Dataset(dst_path, "w", format="NETCDF4") as dst:
        for name, dim in src.dimensions.items():
            size = dim_overrides.get(
                name, None if dim.isunlimited() else len(dim))
            dst.createDimension(name, size)

        for name, var in src.variables.items():
            nv = dst.createVariable(name, var.datatype, var.dimensions)
            nv[:] = var_data[name] if name in var_data else var[:]
            for attr in var.ncattrs():
                if (name, attr) in moved:
                    dst.setncattr(f"{name}.{attr}", getattr(var, attr))
                else:
                    setattr(nv, attr, getattr(var, attr))

        for attr in src.ncattrs():
            if attr == "_NCProperties":
                continue
            value = global_overrides.get(attr, getattr(src, attr))
            dst.setncattr(("NC_GLOBAL." + attr) if prefix_globals else attr, value)


@pytest.fixture
def base_sofa(tmp_path, synthetic_sofa_file):
    """A small, fully conformant SimpleFreeFieldHRIR file on disk."""
    return synthetic_sofa_file(tmp_path / "base.sofa", n_points=12,
                               n_taps=16, n_samples=64)


@pytest.fixture
def legacy_sofa(tmp_path, base_sofa):
    """`base_sofa` rewritten in the SOFA C++ API 1.x layout."""
    dst = str(tmp_path / "legacy.sofa")
    _rebuild(base_sofa, dst, prefix_globals=True,
             var_attrs_as_globals=("Data.SamplingRate.Units",))
    return dst


@pytest.fixture
def roomtype_sofa(tmp_path, base_sofa):
    """`base_sofa` with the RWTH-Aachen-style `RoomType = "free-field"`."""
    dst = str(tmp_path / "roomtype.sofa")
    _rebuild(base_sofa, dst, global_overrides={"RoomType": "free-field"})
    return dst


@pytest.fixture
def multi_emitter_sofa(tmp_path, base_sofa):
    """`base_sofa` declaring SimpleFreeFieldHRIR but storing E=2 emitters."""
    dst = str(tmp_path / "emitters.sofa")
    _rebuild(base_sofa, dst, dim_overrides={"E": 2},
             var_data={"EmitterPosition": np.zeros((2, 3, 1))})
    return dst


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class _Bare:
    """Minimal attribute bag: missing attributes raise AttributeError, exactly
    like a sofar Sofa object that lacks the entry."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# --------------------------------------------------------------------------- #
# 1. Fallback ladder — rung by rung
# --------------------------------------------------------------------------- #
class TestFallbackLadder:
    def test_conformant_file_reads_normally(self, base_sofa):
        """A clean file loads with its payload intact."""
        sofa = read_sofa_robust(base_sofa)
        assert sofa.GLOBAL_SOFAConventions == "SimpleFreeFieldHRIR"
        assert np.asarray(sofa.Data_IR).shape == (12, 2, 64)
        assert float(sofa.Data_SamplingRate) == 48000.0

    def test_conformant_file_uses_no_fallback(self, base_sofa, monkeypatch):
        """The verified read is the *only* read for a conformant file."""
        calls = []
        real = sf.read_sofa
        monkeypatch.setattr(
            sofa_io.sf, "read_sofa",
            lambda p, **kw: (calls.append(kw), real(p, **kw))[1])

        read_sofa_robust(base_sofa)

        assert len(calls) == 1
        assert calls[0].get("verify", True) is True

    def test_legacy_file_is_unreadable_without_normalization(self, legacy_sofa):
        """Guard: the legacy fixture really does defeat a plain sofar read."""
        with pytest.raises(AttributeError):
            sf.read_sofa(legacy_sofa)

    def test_legacy_nc_global_prefixes_are_normalized(self, legacy_sofa):
        """`NC_GLOBAL.`-prefixed globals become bare names and the file loads."""
        sofa = read_sofa_robust(legacy_sofa)
        assert sofa.GLOBAL_SOFAConventions == "SimpleFreeFieldHRIR"
        assert np.asarray(sofa.Data_IR).shape == (12, 2, 64)

    def test_legacy_dotted_globals_are_reattached_to_variables(self, legacy_sofa):
        """`Data.SamplingRate.Units` stored as a global is restored as a
        variable attribute, not dropped or left as a stray global."""
        sofa = read_sofa_robust(legacy_sofa)
        assert sofa.Data_SamplingRate_Units == "hertz"

    def test_strict_read_fails_on_bad_room_type(self, roomtype_sofa):
        """Guard: the fixture really does trip sofar's verification."""
        with pytest.raises(ValueError):
            sf.read_sofa(roomtype_sofa)

    def test_falls_back_to_verify_false(self, roomtype_sofa, monkeypatch, capsys):
        """On ValueError the loader retries with verify=False and says so."""
        calls = []
        real = sf.read_sofa
        monkeypatch.setattr(
            sofa_io.sf, "read_sofa",
            lambda p, **kw: (calls.append(kw), real(p, **kw))[1])

        sofa = read_sofa_robust(roomtype_sofa)

        assert [kw.get("verify", True) for kw in calls] == [True, False]
        assert "verify=False" in capsys.readouterr().out
        assert np.asarray(sofa.Data_IR).shape == (12, 2, 64)

    def test_non_value_error_is_re_raised_unchanged(self, base_sofa, monkeypatch):
        """Only ValueError triggers the fallback; anything else propagates."""
        def boom(path, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(sofa_io.sf, "read_sofa", boom)
        with pytest.raises(RuntimeError, match="disk on fire"):
            read_sofa_robust(base_sofa)


# --------------------------------------------------------------------------- #
# 2. sanitize_sofa() repairs
# --------------------------------------------------------------------------- #
class TestSanitizeEmitterPosition:
    def test_collapses_multi_emitter_to_e1(self, multi_emitter_sofa, capsys):
        """End-to-end: an E=2 SimpleFreeFieldHRIR file is collapsed to E=1."""
        sofa = read_sofa_robust(multi_emitter_sofa)
        assert np.asarray(sofa.EmitterPosition).shape == (1, 3, 1)
        assert "collapsing EmitterPosition" in capsys.readouterr().out

    def test_single_emitter_is_left_alone(self, base_sofa, capsys):
        """A conformant E=1 file is not touched or warned about."""
        sofa = read_sofa_robust(base_sofa)
        assert np.asarray(sofa.EmitterPosition).shape == (1, 3, 1)
        assert "collapsing EmitterPosition" not in capsys.readouterr().out

    def test_collapsed_position_is_the_convention_default(self, multi_emitter_sofa):
        """The replacement is the convention default (origin, cartesian)."""
        sofa = read_sofa_robust(multi_emitter_sofa)
        assert np.allclose(np.asarray(sofa.EmitterPosition), 0.0)

    def test_missing_emitter_position_is_tolerated(self):
        """A Sofa object with no EmitterPosition passes through, not crashes."""
        obj = _Bare(GLOBAL_RoomType="free field")
        assert sanitize_sofa(obj) is obj


class TestSanitizeRoomType:
    @pytest.mark.parametrize("raw, expected", [
        ("free-field", "free field"),
        ("free_field", "free field"),
        ("Free-Field", "free field"),
        ("FREE FIELD", "free field"),
        ("  free   field  ", "free field"),
        ("Reverberant", "reverberant"),
        ("SHOEBOX", "shoebox"),
        ("DAE", "dae"),
    ])
    def test_normalizes_known_variants(self, raw, expected):
        """Hyphens/underscores/case/whitespace collapse to the four legal values."""
        obj = _Bare(GLOBAL_RoomType=raw)
        assert sanitize_sofa(obj).GLOBAL_RoomType == expected

    @pytest.mark.parametrize("value", ["free field", "reverberant", "shoebox", "dae"])
    def test_already_valid_values_are_untouched(self, value, capsys):
        obj = _Bare(GLOBAL_RoomType=value)
        assert sanitize_sofa(obj).GLOBAL_RoomType == value
        assert "Sanitizing" not in capsys.readouterr().out

    @pytest.mark.parametrize("value", ["living room", "anechoic", "free-fieldish"])
    def test_unknown_values_are_left_for_sofar_to_report(self, value):
        """Normalization must not invent a legal value out of an illegal one."""
        obj = _Bare(GLOBAL_RoomType=value)
        assert sanitize_sofa(obj).GLOBAL_RoomType == value

    def test_non_string_room_type_is_tolerated(self):
        obj = _Bare(GLOBAL_RoomType=3)
        assert sanitize_sofa(obj).GLOBAL_RoomType == 3

    def test_missing_room_type_is_tolerated(self):
        obj = _Bare(EmitterPosition=np.zeros((1, 3, 1)))
        assert sanitize_sofa(obj) is obj

    def test_end_to_end_room_type_repair(self, roomtype_sofa, capsys):
        """The RWTH-Aachen case, all the way through the loader: the file
        cannot be read strictly, and the value that made sofar reject it comes
        back repaired so a later write_sofa() would succeed."""
        sofa = read_sofa_robust(roomtype_sofa)
        assert sofa.GLOBAL_RoomType == "free field"
        assert "GLOBAL_RoomType 'free-field' -> 'free field'" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 3. The source file is never modified
# --------------------------------------------------------------------------- #
class TestSourceFileIsNeverModified:
    @pytest.mark.parametrize("fixture_name", [
        "base_sofa", "legacy_sofa", "roomtype_sofa", "multi_emitter_sofa"])
    def test_bytes_on_disk_are_unchanged(self, fixture_name, request):
        """Every rung of the ladder — including the two that trigger in-memory
        repairs — must leave the user's file byte-identical."""
        path = request.getfixturevalue(fixture_name)
        before = _sha256(path)
        before_mtime = os.stat(path).st_mtime_ns

        read_sofa_robust(path)

        assert _sha256(path) == before
        assert os.stat(path).st_mtime_ns == before_mtime

    def test_repairs_really_happened_in_memory(self, multi_emitter_sofa):
        """Guard for the test above: the file whose bytes are unchanged is one
        the loader genuinely rewrote in memory, so 'unchanged' is not vacuous."""
        import netCDF4 as nc4

        sofa = read_sofa_robust(multi_emitter_sofa)
        assert np.asarray(sofa.EmitterPosition).shape == (1, 3, 1)
        with nc4.Dataset(multi_emitter_sofa) as ds:
            assert len(ds.dimensions["E"]) == 2  # still E=2 on disk

    def test_source_file_still_exists_and_reloads(self, legacy_sofa):
        """The legacy path unlinks a temp copy — never the input itself."""
        read_sofa_robust(legacy_sofa)
        assert os.path.exists(legacy_sofa)
        read_sofa_robust(legacy_sofa)  # still readable a second time


# --------------------------------------------------------------------------- #
# 4. The temp copy is unlinked
# --------------------------------------------------------------------------- #
class TestTempFileCleanup:
    def test_legacy_read_leaves_no_temp_file(self, legacy_sofa, tmp_path, monkeypatch):
        """A Windows file-lock regression here would leak one temp file per
        input file per run. Redirect the temp dir so only this call's temp
        files can appear in it, and assert it ends empty."""
        temp_dir = tmp_path / "tmpdir"
        temp_dir.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))

        made = []
        real_mkstemp = tempfile.mkstemp
        monkeypatch.setattr(
            tempfile, "mkstemp",
            lambda *a, **kw: (lambda r: (made.append(r[1]), r)[1])(real_mkstemp(*a, **kw)))

        read_sofa_robust(legacy_sofa)

        # Non-vacuity: the legacy path really did create a temp copy...
        assert len(made) == 1, "legacy normalization did not create a temp file"
        # ...and it is gone.
        assert not os.path.exists(made[0])
        assert os.listdir(str(temp_dir)) == []

    def test_temp_file_removed_even_when_read_fails(self, legacy_sofa, tmp_path,
                                                    monkeypatch):
        """The unlink lives in a `finally`; a failing read must not leak."""
        temp_dir = tmp_path / "tmpdir"
        temp_dir.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
        monkeypatch.setattr(
            sofa_io.sf, "read_sofa",
            lambda p, **kw: (_ for _ in ()).throw(RuntimeError("nope")))

        with pytest.raises(RuntimeError):
            read_sofa_robust(legacy_sofa)

        assert os.listdir(str(temp_dir)) == []

    def test_non_legacy_read_creates_no_temp_file(self, base_sofa, tmp_path,
                                                  monkeypatch):
        """A conformant file is read in place — no copy, nothing to clean up."""
        temp_dir = tmp_path / "tmpdir"
        temp_dir.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))

        read_sofa_robust(base_sofa)

        assert os.listdir(str(temp_dir)) == []

    def test_global_temp_dir_is_not_polluted(self, legacy_sofa):
        """Belt-and-braces against the real system temp dir, using the actual
        `mkstemp(suffix='.sofa')` naming the module uses."""
        pattern = os.path.join(tempfile.gettempdir(), "*.sofa")
        before = set(glob.glob(pattern))

        read_sofa_robust(legacy_sofa)

        assert set(glob.glob(pattern)) - before == set()
