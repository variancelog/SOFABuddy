"""
tests/test_sofa_utils.py
Tests for core.sofa_utils — the four lightweight netCDF4 readers that
`ui/tabs.py::FilesTab` calls **on the main thread** when a user selects or
drags in a file.

Two concerns:

1. Happy path — each reader returns correct values for a known synthetic SOFA.
2. Graceful degradation — none of the four may ever raise, whatever garbage the
   user drops on the Files tab. Each has its own "safe failure" sentinel:

   | function              | sentinel on failure |
   |-----------------------|---------------------|
   | `get_sofa_metadata`   | `None`              |
   | `calculate_max_itd`   | `0.0`               |
   | `calculate_grid_radius` | `0.0`             |
   | `get_grid_cartesian`  | `None`              |

   Note the third failure mode below (a *valid* netCDF file missing the
   expected variables) is not an exception path at all: `nc.Dataset` opens it
   fine and each function takes its own missing-variable branch. Those branches
   return the *same* sentinels except `get_sofa_metadata`, which returns a
   populated dict of zeros rather than `None` — asserted explicitly.
"""
import netCDF4 as nc
import numpy as np
import pytest
import sofar as sf

from core.sofa_utils import (
    boundary_ring_elevations,
    draw_elevation_rings,
    apply_unit_sphere_axes,
    grid_coverage,
    calculate_grid_radius,
    calculate_max_itd,
    get_grid_cartesian,
    elevation_range,
    elevation_ring_xyz,
    format_elevation_range,
    format_elevation_span,
    get_sofa_metadata,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
FS = 44100
N_POINTS = 96
N_SAMPLES = 256
RADIUS_M = 1.25


@pytest.fixture
def good_sofa(tmp_path, synthetic_sofa_file):
    """A valid synthetic SOFA on disk with known fs / M / N / radius."""
    return synthetic_sofa_file(
        tmp_path / "good.sofa",
        fs=FS,
        n_points=N_POINTS,
        n_samples=N_SAMPLES,
        radius_m=RADIUS_M,
    )


def test_get_sofa_metadata_happy(good_sofa):
    """Metadata matches exactly what the fixture was built with."""
    meta = get_sofa_metadata(good_sofa)
    assert {k: meta[k] for k in ("samplerate", "length", "points")} == {
        "samplerate": float(FS),
        "length": N_SAMPLES,
        "points": N_POINTS,
    }
    # Fibonacci sphere: elevation extent is just inside the poles.
    assert -90.0 < meta["el_min"] < -80.0
    assert 80.0 < meta["el_max"] < 90.0


def test_calculate_grid_radius_happy(good_sofa):
    """Dominant SourcePosition radius is the fixture's radius."""
    assert calculate_grid_radius(good_sofa) == pytest.approx(RADIUS_M)


def test_calculate_grid_radius_picks_dominant_not_mean(tmp_path, synthetic_sofa):
    """With a handful of off-sphere outliers the *modal* radius still wins."""
    s = synthetic_sofa(radius_m=RADIUS_M, n_points=N_POINTS)
    src = np.array(s.SourcePosition, dtype=float)
    src[:5, 2] = 10.0  # outliers far from the measurement sphere
    s.SourcePosition = src
    path = str(tmp_path / "outliers.sofa")
    sf.write_sofa(path, s)

    assert calculate_grid_radius(path) == pytest.approx(RADIUS_M)


def test_get_grid_cartesian_happy(good_sofa):
    """Grid is projected onto the *unit* sphere, with radius/count reported."""
    grid = get_grid_cartesian(good_sofa)
    assert set(grid) == {"cart", "radius", "n_nodes", "el_min", "el_max", "coverage"}
    assert grid["n_nodes"] == N_POINTS
    assert grid["radius"] == pytest.approx(RADIUS_M)
    assert grid["cart"].shape == (N_POINTS, 3)
    # Every point sits on the unit sphere regardless of measurement radius.
    assert np.allclose(np.linalg.norm(grid["cart"], axis=1), 1.0)


def test_get_grid_cartesian_matches_az_el_convention(tmp_path, synthetic_sofa):
    """az=0/el=0 -> +x (front); az=90 -> +y (left); el=90 -> +z (up)."""
    s = synthetic_sofa(n_points=3)
    s.SourcePosition = np.array(
        [[0.0, 0.0, 1.0], [90.0, 0.0, 1.0], [0.0, 90.0, 1.0]], dtype=float
    )
    path = str(tmp_path / "axes.sofa")
    sf.write_sofa(path, s)

    cart = get_grid_cartesian(path)["cart"]
    assert np.allclose(cart[0], [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(cart[1], [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(cart[2], [0.0, 0.0, 1.0], atol=1e-9)


def test_calculate_max_itd_zero_when_ears_aligned(good_sofa):
    """The synthetic fixture puts both ears at the same onset -> zero ITD."""
    assert calculate_max_itd(good_sofa) == pytest.approx(0.0)


def test_calculate_max_itd_known_offset(tmp_path, synthetic_sofa):
    """A hand-injected 10-sample L/R peak offset reads back as 10/fs ms."""
    fs = 48000
    offset = 10
    s = synthetic_sofa(fs=fs, n_points=8, n_samples=N_SAMPLES)
    ir = np.zeros_like(np.array(s.Data_IR, dtype=float))
    ir[:, 0, 40] = 1.0
    ir[:, 1, 40 + offset] = 1.0
    s.Data_IR = ir
    path = str(tmp_path / "itd.sofa")
    sf.write_sofa(path, s)

    assert calculate_max_itd(path) == pytest.approx(offset / fs * 1000.0)


def test_calculate_max_itd_reports_the_maximum(tmp_path, synthetic_sofa):
    """With mixed per-position offsets the largest one is reported."""
    fs = 48000
    s = synthetic_sofa(fs=fs, n_points=8, n_samples=N_SAMPLES)
    ir = np.zeros_like(np.array(s.Data_IR, dtype=float))
    offsets = [0, 3, -7, 15, 2, -1, 4, 0]
    for m, off in enumerate(offsets):
        ir[m, 0, 60] = 1.0
        ir[m, 1, 60 + off] = 1.0
    s.Data_IR = ir
    path = str(tmp_path / "itd_mixed.sofa")
    sf.write_sofa(path, s)

    assert calculate_max_itd(path) == pytest.approx(max(map(abs, offsets)) / fs * 1000.0)


# ---------------------------------------------------------------------------
# Graceful degradation — the Files tab must never crash on user input
# ---------------------------------------------------------------------------

@pytest.fixture
def missing_file(tmp_path):
    """A path that does not exist."""
    return str(tmp_path / "does_not_exist.sofa")


@pytest.fixture
def text_file(tmp_path):
    """A plain-text file wearing a .sofa extension (a user drag-drop mistake)."""
    p = tmp_path / "not_really.sofa"
    p.write_text("hello world\nthis is definitely not netCDF\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def barren_netcdf(tmp_path):
    """A structurally valid netCDF4 file with none of the SOFA variables.

    Opens cleanly under `nc.Dataset`, so this exercises each function's
    missing-variable branch rather than its except-clause.
    """
    p = str(tmp_path / "barren.sofa")
    with nc.Dataset(p, "w", format="NETCDF4") as ds:
        ds.createDimension("I", 1)
        v = ds.createVariable("Unrelated", "f8", ("I",))
        v[:] = [1.0]
    return p


BAD_INPUTS = ["missing_file", "text_file", "barren_netcdf"]


@pytest.mark.parametrize("bad_input", BAD_INPUTS)
@pytest.mark.parametrize(
    "func",
    [get_sofa_metadata, calculate_max_itd, calculate_grid_radius, get_grid_cartesian],
    ids=lambda f: f.__name__,
)
def test_readers_never_raise_on_bad_input(func, bad_input, request):
    """All 4 readers x 3 bad inputs: no exception ever escapes to the caller."""
    path = request.getfixturevalue(bad_input)
    func(path)  # must not raise


@pytest.mark.parametrize("bad_input", BAD_INPUTS)
def test_get_sofa_metadata_sentinel(bad_input, request):
    """`None` for unreadable files; a zeroed dict for a readable-but-barren one."""
    path = request.getfixturevalue(bad_input)
    result = get_sofa_metadata(path)
    if bad_input == "barren_netcdf":
        assert result == {"samplerate": 0.0, "length": 0, "points": 0,
                          "el_min": None, "el_max": None, "coverage": None}
    else:
        assert result is None


@pytest.mark.parametrize("bad_input", BAD_INPUTS)
@pytest.mark.parametrize(
    "func", [calculate_max_itd, calculate_grid_radius], ids=lambda f: f.__name__
)
def test_float_readers_sentinel(func, bad_input, request):
    """`calculate_max_itd` / `calculate_grid_radius` degrade to exactly 0.0."""
    path = request.getfixturevalue(bad_input)
    result = func(path)
    assert isinstance(result, float)
    assert result == 0.0


@pytest.mark.parametrize("bad_input", BAD_INPUTS)
def test_get_grid_cartesian_sentinel(bad_input, request):
    """`get_grid_cartesian` degrades to None in all three cases."""
    assert get_grid_cartesian(request.getfixturevalue(bad_input)) is None


@pytest.mark.parametrize(
    "func",
    [get_sofa_metadata, calculate_max_itd, calculate_grid_radius, get_grid_cartesian],
    ids=lambda f: f.__name__,
)
def test_readers_never_raise_on_directory_path(tmp_path, func):
    """A directory dragged onto the Files tab is also survivable."""
    func(str(tmp_path))


# ---------------------------------------------------------------------------
# Partial-corruption: a real SOFA missing just one expected variable
# ---------------------------------------------------------------------------

def _sofa_without(tmp_path, good_sofa, drop, name):
    """Copy `good_sofa` into a new netCDF4 file omitting variable `drop`."""
    out = str(tmp_path / name)
    with nc.Dataset(good_sofa, "r") as src, nc.Dataset(out, "w", format="NETCDF4") as dst:
        for dname, dim in src.dimensions.items():
            dst.createDimension(dname, len(dim))
        for vname, var in src.variables.items():
            if vname == drop:
                continue
            new = dst.createVariable(vname, var.datatype, var.dimensions)
            new[:] = var[:]
    return out


def test_missing_data_ir_degrades(tmp_path, good_sofa):
    """No Data.IR: metadata zeroes M/N but keeps fs; ITD -> 0.0; grid unaffected."""
    path = _sofa_without(tmp_path, good_sofa, "Data.IR", "no_ir.sofa")

    meta = get_sofa_metadata(path)
    assert meta["samplerate"] == pytest.approx(float(FS))
    assert meta["length"] == 0 and meta["points"] == 0
    assert calculate_max_itd(path) == 0.0
    # SourcePosition survives, so the grid readers still work.
    assert calculate_grid_radius(path) == pytest.approx(RADIUS_M)
    assert get_grid_cartesian(path)["n_nodes"] == N_POINTS


def test_missing_source_position_degrades(tmp_path, good_sofa):
    """No SourcePosition: grid readers hit their sentinels; IR readers unaffected."""
    path = _sofa_without(tmp_path, good_sofa, "SourcePosition", "no_src.sofa")

    assert calculate_grid_radius(path) == 0.0
    assert get_grid_cartesian(path) is None
    assert get_sofa_metadata(path)["points"] == N_POINTS
    assert calculate_max_itd(path) == pytest.approx(0.0)


def test_missing_samplerate_degrades(tmp_path, good_sofa):
    """No Data.SamplingRate: metadata reports 0.0 fs; ITD bails out at 0.0."""
    path = _sofa_without(tmp_path, good_sofa, "Data.SamplingRate", "no_fs.sofa")

    meta = get_sofa_metadata(path)
    assert meta["samplerate"] == 0.0
    assert meta["points"] == N_POINTS and meta["length"] == N_SAMPLES
    assert calculate_max_itd(path) == 0.0


# ---------------------------------------------------------------------------
# Elevation coverage helpers
# ---------------------------------------------------------------------------
M_ = chr(0x2212)   # proper minus sign
D_ = chr(0xB0)     # degree sign


@pytest.mark.parametrize(
    "el, expected",
    [
        ([-90.0, 0.0, 90.0], (-90.0, 90.0)),          # full sphere
        ([0.0, 45.0, 90.0], (0.0, 90.0)),             # upper hemisphere
        ([10.0, 10.0, 10.0], (10.0, 10.0)),           # single ring
        ([-0.0, 30.0], (0.0, 30.0)),                  # -0.0 normalised
        ([-0.04, 30.04], (0.0, 30.0)),                # rounds to 0.1, -0.0 -> 0.0
        ([-22.46, 5.0], (-22.5, 5.0)),
        ([np.nan, -40.0, 90.0, np.nan], (-40.0, 90.0)),
    ],
)
def test_elevation_range_values(el, expected):
    lo, hi = elevation_range(np.array(el))
    assert (lo, hi) == expected
    assert str(lo) != "-0.0" and str(hi) != "-0.0"


def test_elevation_range_masked_array_drops_masked():
    arr = np.ma.masked_array([-90.0, 0.0, 45.0, 90.0], mask=[True, False, False, True])
    assert elevation_range(arr) == (0.0, 45.0)


@pytest.mark.parametrize("empty", [np.array([]), np.array([np.nan, np.nan]),
                                   np.ma.masked_all(3)])
def test_elevation_range_empty_is_none(empty):
    assert elevation_range(empty) is None


@pytest.mark.parametrize(
    "lo, hi, expected",
    [
        (-90.0, 90.0, f"Elevation: {M_}90{D_} to +90{D_}"),
        (-40.0, 90.0, f"Elevation: {M_}40{D_} to +90{D_}"),
        (0.0, 90.0, f"Elevation: 0{D_} to +90{D_}"),
        (-22.5, 90.0, f"Elevation: {M_}22.5{D_} to +90{D_}"),
        (12.5, 12.5, f"Elevation: +12.5{D_} to +12.5{D_}"),
    ],
)
def test_format_elevation_range(lo, hi, expected):
    assert format_elevation_range(lo, hi) == expected


def test_format_elevation_span_has_no_label():
    assert format_elevation_span(-40.0, 90.0) == f"{M_}40{D_} to +90{D_}"


@pytest.mark.parametrize("el", [-40.0, 0.0, 33.3])
def test_elevation_ring_xyz_on_unit_sphere_at_elevation(el):
    x, y, z = elevation_ring_xyz(el, n=90)
    assert x.shape == (90,)
    assert np.allclose(x**2 + y**2 + z**2, 1.0)
    assert np.allclose(np.degrees(np.arcsin(z)), el)


def _fib(n, el_lo=-90.0, el_hi=90.0):
    """Near-uniform Fibonacci grid restricted to an elevation band (az, el deg)."""
    k = np.arange(n) + 0.5
    el = np.degrees(np.arcsin(1 - 2 * k / n))
    az = np.degrees(k * np.pi * (3 - np.sqrt(5))) % 360
    keep = (el >= el_lo) & (el <= el_hi)
    return az[keep], el[keep]


def _rows(el_values, n_az=72):
    az = np.arange(n_az) * 360.0 / n_az
    return (np.tile(az, len(el_values)), np.repeat(el_values, n_az))


def test_coverage_exact_pole_rows_full_sphere():
    az, el = _rows(np.arange(-90, 91, 10.0))
    cov = grid_coverage(az, el)
    assert cov.full_sphere and (cov.el_min, cov.el_max) == (-90.0, 90.0)


def test_coverage_fibonacci_nodes_short_of_poles_is_full_sphere():
    az, el = _fib(480)
    assert el.max() < 89.0 and el.min() > -89.0       # no node on a pole
    cov = grid_coverage(az, el)
    assert cov.full_sphere and not cov.top_open and not cov.bottom_open


def test_coverage_upper_hemisphere_bottom_open():
    az, el = _fib(600, el_lo=0.0)
    cov = grid_coverage(az, el)
    assert cov.bottom_open and not cov.top_open and not cov.full_sphere


def test_coverage_ari_like_both_open():
    az, el = _rows(np.arange(-30, 81, 5.0))
    cov = grid_coverage(az, el)
    assert cov.top_open and cov.bottom_open
    assert (cov.el_min, cov.el_max) == (-30.0, 80.0)


def test_coverage_masked_and_nan_input():
    az, el = _rows(np.arange(-90, 91, 10.0))
    az_m = np.ma.masked_array(az, mask=(el > 60))
    el_m = np.ma.masked_array(el, mask=(el > 60))
    cov = grid_coverage(az_m, el_m)
    assert cov.el_max == 60.0 and cov.top_open
    el_n = el.copy()
    el_n[:5] = np.nan
    assert grid_coverage(az, el_n) is not None   # NaN rows dropped, no raise


def test_coverage_single_point():
    cov = grid_coverage([0.0], [90.0])
    assert cov.spacing_deg is None and not cov.top_open and cov.bottom_open
    cov = grid_coverage([0.0], [10.0])
    assert cov.top_open and cov.bottom_open


@pytest.mark.parametrize("empty", [[], [np.nan]])
def test_coverage_empty_is_none(empty):
    assert grid_coverage(empty, empty) is None


def test_coverage_duplicates_do_not_zero_spacing():
    az, el = _rows(np.arange(-30, 81, 5.0))
    cov = grid_coverage(np.tile(az, 2), np.tile(el, 2))
    assert cov.spacing_deg > 1.0


def test_boundary_ring_elevations_follow_open_flags():
    from core.grid_info import GridCoverage
    assert boundary_ring_elevations(GridCoverage(-90, 90, 5, False, False)) == []
    assert boundary_ring_elevations(GridCoverage(-83.4, 83.4, 8.7, False, False)) == []
    assert boundary_ring_elevations(GridCoverage(0, 90, 5, False, True)) == [0]
    assert boundary_ring_elevations(GridCoverage(-40, 80, 5, True, True)) == [-40, 80]
    assert boundary_ring_elevations(GridCoverage(10, 10, None, True, True)) == [10]


def test_format_full_sphere_suffix_only_when_full():
    assert format_elevation_range(-83.4, 83.4, True) == (
        f"Elevation: {M_}83.4{D_} to +83.4{D_} (full sphere)")
    assert format_elevation_span(-83.4, 83.4, True).endswith("(full sphere)")
    assert "full sphere" not in format_elevation_range(-30.0, 80.0)


def test_get_grid_cartesian_reports_elevation_extent(tmp_path, synthetic_sofa):
    s = synthetic_sofa(n_points=3)
    s.SourcePosition = np.array(
        [[0.0, -40.0, 1.0], [90.0, 10.0, 1.0], [0.0, 90.0, 1.0]], dtype=float
    )
    path = str(tmp_path / "partial.sofa")
    sf.write_sofa(path, s)

    grid = get_grid_cartesian(path)
    assert (grid["el_min"], grid["el_max"]) == (-40.0, 90.0)
    meta = get_sofa_metadata(path)
    assert (meta["el_min"], meta["el_max"]) == (-40.0, 90.0)
    assert grid["coverage"] == meta["coverage"]


def test_weighted_grid_png_written(tmp_path, synthetic_sofa):
    """Smoke test: backend script writes the weighted-grid PNG for a partial grid."""
    import subprocess
    import sys
    from pathlib import Path

    s = synthetic_sofa(n_points=300)
    src = np.array(s.SourcePosition, dtype=float)
    keep = src[:, 1] >= -40
    s.Data_IR = np.array(s.Data_IR)[keep]
    s.SourcePosition = src[keep]
    sofa_path = str(tmp_path / "partial.sofa")
    sf.write_sofa(sofa_path, s)
    png = tmp_path / "weighted.png"

    script = Path(__file__).resolve().parents[1] / "core" / "scripts" / "generate_dfhrtf.py"
    r = subprocess.run(
        [sys.executable, str(script), "--input", sofa_path,
         "--graph-weighted-out", str(png)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert png.exists() and png.stat().st_size > 0


def test_rings_drawn_for_open_ends_and_axes_fixed():
    """Rings follow the open flags, and apply_unit_sphere_axes fixes the limits
    to exactly +-1.05 regardless of the scatter's extent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.grid_info import GridCoverage

    rng = np.random.default_rng(0)
    az = np.radians(rng.uniform(-60, 60, 200))
    el = np.radians(rng.uniform(-40, 40, 200))
    x, y, z = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)

    fig = plt.figure()
    try:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z)
        draw_elevation_rings(ax, GridCoverage(-40, 40, 5, True, True))
        assert len(ax.lines) == 2
        draw_elevation_rings(ax, GridCoverage(-40, 40, 5, False, False))
        assert len(ax.lines) == 2
        apply_unit_sphere_axes(ax)
        for lim in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()):
            assert lim == (-1.05, 1.05)
    finally:
        plt.close(fig)
