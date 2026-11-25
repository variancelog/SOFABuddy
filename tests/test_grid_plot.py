"""Unit tests for core.grid_plot.plot_weighted_grid (Agg backend, no subprocess)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from core.geometry_weights import spherical_to_cartesian
from core.grid_info import GridCoverage, grid_coverage
from core.grid_plot import plot_weighted_grid

FULL = GridCoverage(-80.0, 80.0, 10.0, False, False)
PARTIAL = GridCoverage(-40.0, 80.0, 10.0, True, True)


def _grid(n=60, radius=1.2):
    rng = np.random.default_rng(1)
    az = rng.uniform(0, 360, n)
    el = rng.uniform(-80, 80, n)
    return spherical_to_cartesian(radius, az, el)


def _plot(weights, xyz=None, **kw):
    weights = np.asarray(weights, dtype=float)
    xyz = _grid(len(weights)) if xyz is None else xyz
    kw.setdefault("name", "G")
    kw.setdefault("radius", 1.2)
    kw.setdefault("coverage", FULL)
    fig = plt.figure(figsize=(10.5, 8))
    plot_weighted_grid(fig, xyz, weights, **kw)
    return fig


def _ax(fig):
    return fig.axes[0]


def _title(fig):
    return fig._suptitle.get_text()


def _mark_texts(fig):
    return [t.get_text() for t in fig.axes[1].texts]


def _scatters(fig):
    return _ax(fig).collections


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _nonuniform(n=60):
    w = np.linspace(0.5, 1.5, n)
    return w / w.sum()


def test_relative_values_and_clipping():
    n = 50
    w = np.ones(n)
    w[0] = 5.0           # relative far above 2
    w /= w.sum()
    fig = _plot(w, vmin=0.0, vmax=2.0)
    sc = _scatters(fig)[0]
    rel = np.asarray(sc.get_array())
    assert np.allclose(rel, w * n)
    assert sc.get_clim() == (0.0, 2.0)
    assert rel.max() > 2.0
    assert "max" in " ".join(_mark_texts(fig))


def test_uniform_mark_and_auto_wording():
    n = 40
    w = np.full(n, 1.0 / n)
    fig = _plot(w, weighting_mode="auto")
    assert _mark_texts(fig) == ["all = 1"]
    assert "Weights: Uniform (1/N, spherical design)" in _title(fig)
    fig = _plot(w, weighting_mode="uniform")
    assert _mark_texts(fig) == ["all = 1"]
    assert _title(fig).endswith("Weights: Uniform (1/N)")
    assert "spherical" not in _title(fig)


def test_min_max_marks_nonuniform():
    w = _nonuniform()
    fig = _plot(w)
    rel = w * len(w)
    assert _mark_texts(fig) == [f"min {rel.min():.2f}", f"max {rel.max():.2f}"]
    assert f"Non-uniform, {rel.min():.2f} to {rel.max():.2f}" in _title(fig)
    assert fig.axes[1].get_ylabel() == "Relative solid-angle weight (1 = mean)"


def test_zero_weight_nodes_grey_and_excluded_title():
    w = _nonuniform(60)
    w[:7] = 0.0
    w /= w.sum()
    fig = _plot(w)
    cols = _scatters(fig)
    assert len(cols) == 2
    grey = cols[0]
    assert len(grey._offsets3d[0]) == 7
    assert np.allclose(grey.get_facecolor()[0][:3], matplotlib.colors.to_rgb("lightgrey"))
    assert len(cols[1]._offsets3d[0]) == 53
    assert "Nodes: 53 (+7 excluded)" in _title(fig)
    # relative weights use the kept count
    assert np.allclose(np.asarray(cols[1].get_array()), w[7:] * 53)


def test_title_without_exclusions_and_sphere_tags():
    fig = _plot(_nonuniform(60))
    t = _title(fig).splitlines()[0]
    assert "Nodes: 60;" in t and "excluded" not in t
    assert t.startswith("G Grid (Radius: 1.2m;") and t.endswith("[Full Sphere])")
    fig = _plot(_nonuniform(60), coverage=PARTIAL)
    assert _title(fig).splitlines()[0].endswith("[Partial Sphere])")


def test_axes_exactly_radius_and_labels():
    for r in (1.2, 0.35):
        fig = _plot(_nonuniform(), radius=r)
        ax = _ax(fig)
        for lim in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()):
            assert lim == (-r, r)
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("X (m)", "Y (m)", "Z (m)")
        plt.close(fig)


def test_rings_only_for_open_ends_and_scaled():
    fig = _plot(_nonuniform(), coverage=PARTIAL, radius=1.5)
    lines = _ax(fig).lines
    assert len(lines) == 2
    xs, ys, _ = lines[0]._verts3d
    assert np.allclose(np.hypot(xs, ys), 1.5 * np.cos(np.radians(-40.0)))
    assert len(_ax(_plot(_nonuniform(), coverage=FULL)).lines) == 0


def test_all_zero_weights_do_not_crash():
    fig = _plot(np.zeros(20), coverage=None)
    assert len(fig.axes) == 1                       # no colorbar
    assert len(_scatters(fig)) == 1
    assert _title(fig).endswith("Weights: none")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_radius_falls_back_to_one(bad):
    fig = _plot(_nonuniform(), radius=bad)
    assert _ax(fig).get_xlim() == (-1.0, 1.0)


def test_inverted_rel_range_falls_back():
    fig = _plot(_nonuniform(), vmin=3.0, vmax=1.0)
    assert _scatters(fig)[0].get_clim() == (0.0, 2.0)
    fig = _plot(_nonuniform(), vmin=1.0, vmax=1.0)
    assert _scatters(fig)[0].get_clim() == (0.0, 2.0)


def test_auto_scale_percentiles_and_uniform_widening():
    w = _nonuniform(200)
    rel = w * 200
    fig = _plot(w, auto_scale=True)
    lo, hi = np.percentile(rel, [1, 99])
    assert np.allclose(_scatters(fig)[0].get_clim(), (lo, hi))
    n = 30
    fig = _plot(np.full(n, 1.0 / n), auto_scale=True)
    lo, hi = _scatters(fig)[0].get_clim()
    assert np.isclose(lo, 1.0 - 1e-3) and np.isclose(hi, 1.0 + 1e-3)
    assert _mark_texts(fig) == ["all = 1"]


def test_figure_size_and_real_coverage():
    az = np.linspace(0, 350, 36)
    az = np.tile(az, 5)
    el = np.repeat([-30.0, -10.0, 10.0, 30.0, 50.0], 36)
    xyz = spherical_to_cartesian(1.0, az, el)
    w = np.full(len(az), 1.0 / len(az))
    fig = _plot(w, xyz=xyz, coverage=grid_coverage(az, el), radius=1.0)
    assert tuple(fig.get_size_inches()) == (10.5, 8)
    assert "[Partial Sphere]" in _title(fig)
