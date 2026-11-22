"""Dependency-free (numpy-only) helpers for describing a measurement grid's
elevation coverage. Re-exported from core.sofa_utils; the backend scripts import
from here directly so they don't pull in netCDF4.
"""
from dataclasses import dataclass

import numpy as np

MINUS = chr(0x2212)   # proper minus sign
DEG = chr(0xB0)


def elevation_range(el_deg):
    """Return (el_min, el_max) in degrees, rounded to 0.1, or None if no valid values.

    Accepts ndarray / masked array / sequence; masked and NaN/inf values are dropped.
    """
    if np.ma.isMaskedArray(el_deg):
        el_deg = el_deg.compressed()
    arr = np.asarray(el_deg, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo = round(float(arr.min()), 1) + 0.0   # "+ 0.0" turns -0.0 into 0.0
    hi = round(float(arr.max()), 1) + 0.0
    return lo, hi


def _fmt_deg(v):
    text = f"{abs(v):.1f}".removesuffix(".0")
    if v > 0:
        return f"+{text}{DEG}"
    if v < 0:
        return f"{MINUS}{text}{DEG}"
    return f"{text}{DEG}"


@dataclass(frozen=True)
class GridCoverage:
    """Elevation extent of a grid plus which poles are left open.

    A pole is "open" when its empty polar cap (90 - el_max, or 90 + el_min)
    exceeds the grid's median nearest-neighbour angular spacing. Quadrature
    grids (e.g. t-designs) have no node on a pole but are still full-sphere.
    """
    el_min: float
    el_max: float
    spacing_deg: float | None
    top_open: bool
    bottom_open: bool

    @property
    def full_sphere(self):
        return not (self.top_open or self.bottom_open)


# With fewer than 2 unique points there is no spacing; a pole counts as
# covered only if a node sits (almost) on it.
_SINGLE_POINT_POLE_TOL_DEG = 0.5


def grid_coverage(az_deg, el_deg):
    """Analyse a grid's elevation coverage; None if there are no valid points."""
    az = np.ma.filled(np.ma.asarray(az_deg, dtype=float), np.nan).ravel()
    el = np.ma.filled(np.ma.asarray(el_deg, dtype=float), np.nan).ravel()
    ok = np.isfinite(az) & np.isfinite(el)
    az, el = az[ok], el[ok]
    if el.size == 0:
        return None
    rng = elevation_range(el)

    a, e = np.radians(az), np.radians(el)
    pts = np.column_stack((np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)))
    pts = np.unique(np.round(pts, 6), axis=0)

    spacing = None
    if len(pts) >= 2:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pts).query(pts, k=2)
        chord = float(np.median(d[:, 1]))
        spacing = float(np.degrees(2 * np.arcsin(min(chord / 2, 1.0))))
    tol = spacing if spacing is not None else _SINGLE_POINT_POLE_TOL_DEG
    top_gap = 90.0 - float(el.max())
    bottom_gap = 90.0 + float(el.min())
    return GridCoverage(rng[0], rng[1], spacing, top_gap > tol, bottom_gap > tol)


def format_elevation_span(el_min, el_max, full_sphere=False):
    """Label-less form, e.g. '-40 deg to +90 deg' (for the Files tab row)."""
    text = f"{_fmt_deg(el_min)} to {_fmt_deg(el_max)}"
    return text + " (full sphere)" if full_sphere else text


def format_elevation_range(el_min, el_max, full_sphere=False):
    """Plot-title form: 'Elevation: <span>'."""
    return f"Elevation: {format_elevation_span(el_min, el_max, full_sphere)}"


def elevation_ring_xyz(el_deg, n=180):
    """Unit-sphere circle of constant elevation as a (3, n) array [x; y; z]."""
    az = np.linspace(0.0, 2 * np.pi, n)
    el = np.radians(el_deg)
    return np.vstack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                      np.full(n, np.sin(el))))


def boundary_ring_elevations(coverage):
    """Elevations (deg) at which to draw boundary rings: el_min if the bottom
    is open, el_max if the top is open (de-duplicated when equal)."""
    out = []
    if coverage.bottom_open:
        out.append(coverage.el_min)
    if coverage.top_open and coverage.el_max not in out:
        out.append(coverage.el_max)
    return out


RING_STYLE = dict(color="#d62728", linestyle="--", linewidth=2.0, alpha=0.95)
# Drawn above the scatter: mplot3d otherwise depth-sorts the line beneath
# dense/large markers and the ring all but disappears.
RING_ZORDER = 10
AXIS_LIM = 1.05


def apply_unit_sphere_axes(ax):
    """Fixed +-1.05 limits and a cubic box so grids are visually comparable."""
    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    ax.set_zlim(-AXIS_LIM, AXIS_LIM)
    ax.set_box_aspect([1, 1, 1])


def draw_elevation_rings(ax, coverage):
    """Draw dashed boundary rings (open ends only) on a 3D matplotlib axis.
    Callers apply apply_unit_sphere_axes() afterwards, so ring extent never
    affects the framing."""
    rings = boundary_ring_elevations(coverage)
    if rings:
        ax.computed_zorder = False
    for e in rings:
        x, y, z = elevation_ring_xyz(e)
        ax.plot(x, y, z, zorder=RING_ZORDER, **RING_STYLE)
