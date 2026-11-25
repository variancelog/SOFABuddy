"""Guards the vectorised ConvexHull weights and the weight cache in
core/geometry_weights.py against a reference copy of the original per-triangle
loop, on the committed real-grid fixtures (numpy/scipy only, no Qt)."""
import glob
import os

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from core import geometry_weights as gw

GRID_DIR = os.path.join(os.path.dirname(__file__), "data", "grids")
GRIDS = sorted(glob.glob(os.path.join(GRID_DIR, "*.npy")))


def _reference_geometric(source_pos):
    """The original loop implementation (advanced_dfeq=True, geometric)."""
    az, el = source_pos[:, 0], source_pos[:, 1]
    dom = gw.get_dominant_radius(source_pos)
    valid = np.abs(source_pos[:, 2] - dom) < 0.05
    rounded = np.round(source_pos[valid][:, 0:2], decimals=3)
    _, uidx = np.unique(rounded, axis=0, return_index=True)
    fidx = np.where(valid)[0][uidx]
    cart = gw.spherical_to_cartesian(1.0, az[fidx], el[fidx])
    k = 1000
    i = np.arange(0, k, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    th = np.pi * (1 + 5 ** 0.5) * i
    ghost = np.column_stack((np.cos(th) * np.sin(phi),
                             np.sin(th) * np.sin(phi), np.cos(phi)))
    ghost = ghost[ghost[:, 2] < (np.min(cart[:, 2]) - 0.01)]
    closed = np.vstack((cart, ghost))
    hull = ConvexHull(closed)
    tw = np.zeros(len(closed))
    for s in hull.simplices:
        A, B, C = closed[s]
        tw[s] += 0.5 * np.linalg.norm(np.cross(B - A, C - A)) / 3.0
    w = np.zeros(len(source_pos))
    w[fidx] = tw[:len(cart)]
    return w / np.sum(w)


@pytest.mark.parametrize("path", GRIDS, ids=[os.path.basename(p) for p in GRIDS])
def test_vectorised_weights_match_loop(path):
    sp = np.load(path)
    gw._WEIGHT_CACHE.clear()
    got = gw.calculate_geometric_weights(sp, True, "geometric")
    np.testing.assert_allclose(got, _reference_geometric(sp), rtol=0, atol=1e-15)


def test_cache_returns_independent_copies():
    sp = np.load(GRIDS[0])
    gw._WEIGHT_CACHE.clear()
    w1 = gw.calculate_geometric_weights(sp, True, "geometric")
    w1[:] = 0.0
    w2 = gw.calculate_geometric_weights(sp, True, "geometric")
    assert np.isclose(w2.sum(), 1.0)


def test_cache_keyed_on_positions():
    sp = np.load(GRIDS[0])
    gw._WEIGHT_CACHE.clear()
    a = gw.calculate_geometric_weights(sp, True, "geometric")
    # Same shape, one node moved: must not be served the cached weights of sp.
    sp2 = sp.copy()
    sp2[0, 1] = sp2[0, 1] - 7.0 if sp2[0, 1] > 0 else sp2[0, 1] + 7.0
    b = gw.calculate_geometric_weights(sp2, True, "geometric")
    assert not np.array_equal(a, b)
    np.testing.assert_allclose(b, _reference_geometric(sp2), rtol=0, atol=1e-15)
    # A different mode on cached positions still honours the mode.
    u = gw.calculate_geometric_weights(sp, True, "uniform")
    assert np.ptp(u[u > 0]) == 0.0 and not np.array_equal(u, a)
