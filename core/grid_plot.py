"""Weighted-grid plot (DFHRTF "Weighted Grid" PNG).

Visual parity with the maintainer's plot_grid.py (--color relative): nodes are
coloured by relative solid-angle weight (weight * N_kept, 1 = mean) on a fixed
range, zero-weight nodes are grey, and the colorbar marks the grid's min/max
(or "all = 1" when uniform).

matplotlib is not imported here at all: the caller creates the figure (via the
backend script's Agg-forcing _get_pyplot()) and passes it in.
"""
import numpy as np

from core.grid_info import (
    RING_STYLE, RING_ZORDER, boundary_ring_elevations, elevation_ring_xyz,
    format_elevation_span,
)

FIGSIZE = (10.5, 8)
DEFAULT_REL_RANGE = (0.0, 2.0)
_MARKER_STYLE = dict(marker="o", s=18, depthshade=True, edgecolors="k",
                     linewidths=0.3)
_TIMES = chr(0xD7)


def sphere_tag(coverage):
    """'Full Sphere' / 'Partial Sphere' from the grid coverage (None -> '')."""
    if coverage is None:
        return ""
    return "Full Sphere" if coverage.full_sphere else "Partial Sphere"


def weights_summary(rel_lo, rel_hi, weighting_mode="auto"):
    """Second title line. rel_lo/rel_hi are relative weights (1 = mean) or None."""
    if rel_lo is None or rel_hi is None:
        return "Weights: none"
    if is_uniform(rel_lo, rel_hi):
        if str(weighting_mode).lower() == "auto":
            return "Weights: Uniform (1/N, spherical design)"
        return "Weights: Uniform (1/N)"
    return f"Weights: Non-uniform, {rel_lo:.2f} to {rel_hi:.2f} {_TIMES} mean"


def is_uniform(rel_lo, rel_hi):
    return (rel_hi - rel_lo) < 1e-6


def grid_title(name, radius, n_kept, n_excluded, coverage):
    """First title line, e.g. 'X Grid (Radius: 1.2m; Nodes: 440; Elevation: ... [Full Sphere])'."""
    nodes = f"{n_kept}" + (f" (+{n_excluded} excluded)" if n_excluded > 0 else "")
    text = f"{name} Grid (Radius: {radius:g}m; Nodes: {nodes}"
    if coverage is not None:
        span = format_elevation_span(coverage.el_min, coverage.el_max)
        text += f"; Elevation: {span} [{sphere_tag(coverage)}]"
    return text + ")"


def _resolve_range(rel_kept, vmin, vmax, auto_scale):
    """Colour range (vmin, vmax) in relative units."""
    if auto_scale and rel_kept.size:
        lo, hi = (float(v) for v in np.percentile(rel_kept, [1, 99]))
        if hi - lo < 1e-9:
            lo, hi = lo - 1e-3, hi + 1e-3
        return lo, hi
    try:
        lo, hi = float(vmin), float(vmax)
    except (TypeError, ValueError):
        return DEFAULT_REL_RANGE
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        return DEFAULT_REL_RANGE
    return lo, hi


def plot_weighted_grid(fig, xyz, weights, *, name, radius, coverage,
                       vmin=0.0, vmax=2.0, auto_scale=False,
                       weighting_mode="auto"):
    """Draw the weighted grid into 'fig' (cleared by the caller) and return it.

    xyz       (N, 3) node positions at 'radius', all nodes (zero-weight included)
    weights   (N,) DFHRTF weights; nodes with weight <= 0 are drawn grey
    coverage  GridCoverage of the kept nodes, or None
    vmin/vmax fixed colour range in relative units (1 = mean weight)
    auto_scale  use the 1st-99th percentile of the relative weights instead
    """
    xyz = np.asarray(xyz, dtype=float)
    w = np.asarray(weights, dtype=float)
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0

    kept = np.isfinite(w) & (w > 0)
    n_kept = int(kept.sum())
    n_excluded = int(len(w) - n_kept)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    ax = fig.add_subplot(111, projection="3d")
    if n_excluded:
        ax.scatter(x[~kept], y[~kept], z[~kept], c="lightgrey", **_MARKER_STYLE)

    rel_lo = rel_hi = None
    if n_kept:
        rel = w[kept] * n_kept
        rel_lo, rel_hi = float(rel.min()), float(rel.max())
        lo, hi = _resolve_range(rel, vmin, vmax, auto_scale)
        sc = ax.scatter(x[kept], y[kept], z[kept], c=rel, cmap="viridis",
                        vmin=lo, vmax=hi, **_MARKER_STYLE)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1,
                          label="Relative solid-angle weight (1 = mean)")
        # Mark the grid's actual span on the colorbar (clipped to the bar).
        marks = ([(1.0, "all = 1")] if is_uniform(rel_lo, rel_hi) else
                 [(rel_lo, f"min {rel_lo:.2f}"), (rel_hi, f"max {rel_hi:.2f}")])
        for v, text in marks:
            yv = min(max(v, lo), hi)
            cb.ax.axhline(yv, color="k", linewidth=2.0)
            cb.ax.text(-0.25, yv, text, transform=cb.ax.get_yaxis_transform(),
                       ha="right", va="center", fontsize=9)

    if coverage is not None:
        rings = boundary_ring_elevations(coverage)
        if rings:
            ax.computed_zorder = False
        for e in rings:
            rx, ry, rz = np.asarray(elevation_ring_xyz(e)) * radius
            ax.plot(rx, ry, rz, zorder=RING_ZORDER, **RING_STYLE)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect([1, 1, 1])

    fig.suptitle(grid_title(name, radius, n_kept, n_excluded, coverage)
                 + "\n" + weights_summary(rel_lo, rel_hi, weighting_mode))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.tight_layout()
    return fig
