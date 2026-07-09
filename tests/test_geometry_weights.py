"""
tests/test_geometry_weights.py
Regression tests for core.geometry_weights — the shared diffuse-field
weighting module used by both generate_sofa.py and generate_dfhrtf.py.
"""
import sys
import os
import numpy as np
import pytest

# Ensure project root is on sys.path so core.* resolves without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.geometry_weights import (
    spherical_to_cartesian,
    get_dominant_radius,
    calculate_geometric_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _full_sphere(n_az=12, n_el=7, radius=1.0):
    """Return a SourcePosition array for a roughly-uniform full sphere."""
    rows = []
    for el in np.linspace(-60, 90, n_el):
        for az in np.linspace(0, 330, n_az):
            rows.append([az, el, radius])
    return np.array(rows)


def _hemi_sphere(n_az=12, n_el=4, radius=1.0):
    """Return a SourcePosition array for an upper hemisphere (el >= 0)."""
    rows = []
    for el in np.linspace(0, 90, n_el):
        for az in np.linspace(0, 330, n_az):
            rows.append([az, el, radius])
    return np.array(rows)


def _mixed_radii_sphere(radius=1.0, n_outlier=5):
    """Full sphere with a handful of points at a different radius."""
    sp = _full_sphere()
    outliers = sp[:n_outlier].copy()
    outliers[:, 2] = radius + 0.5   # far off the dominant radius
    return np.vstack([sp, outliers])


# ---------------------------------------------------------------------------
# spherical_to_cartesian
# ---------------------------------------------------------------------------

def test_spherical_to_cartesian_front():
    """az=0, el=0 should map to (1, 0, 0)."""
    xyz = spherical_to_cartesian(1.0, np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(xyz, [[1.0, 0.0, 0.0]], atol=1e-12)


def test_spherical_to_cartesian_top():
    """el=90 should map to (0, 0, 1) regardless of azimuth."""
    xyz = spherical_to_cartesian(1.0, np.array([0.0, 90.0]), np.array([90.0, 90.0]))
    np.testing.assert_allclose(xyz[:, 2], [1.0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# get_dominant_radius
# ---------------------------------------------------------------------------

def test_dominant_radius_pure():
    sp = _full_sphere(radius=1.5)
    assert get_dominant_radius(sp) == pytest.approx(1.5, abs=1e-3)


def test_dominant_radius_with_outliers():
    sp = _mixed_radii_sphere(radius=1.0, n_outlier=5)
    # Majority are at radius 1.0; outliers at 1.5 should not win.
    assert get_dominant_radius(sp) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# calculate_geometric_weights — basic contract
# ---------------------------------------------------------------------------

def test_weights_sum_to_one_full_sphere():
    sp = _full_sphere()
    w = calculate_geometric_weights(sp)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_weights_nonnegative():
    sp = _full_sphere()
    w = calculate_geometric_weights(sp)
    assert np.all(w >= 0)


def test_weights_length_matches_input():
    sp = _full_sphere()
    w = calculate_geometric_weights(sp)
    assert len(w) == len(sp)


# ---------------------------------------------------------------------------
# uniform mode
# ---------------------------------------------------------------------------

def test_uniform_mode_equals_one_over_n():
    sp = _full_sphere()
    w = calculate_geometric_weights(sp, weighting_mode='uniform')
    kept = w[w > 0]
    expected = 1.0 / len(kept)
    np.testing.assert_allclose(kept, expected, rtol=1e-9)


def test_uniform_mode_sum_to_one():
    sp = _full_sphere()
    w = calculate_geometric_weights(sp, weighting_mode='uniform')
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# geometric mode — should differ from uniform on an open/hemispherical grid
# ---------------------------------------------------------------------------

def test_geometric_differs_from_uniform_on_hemisphere():
    sp = _hemi_sphere()
    w_geo = calculate_geometric_weights(sp, weighting_mode='geometric')
    w_uni = calculate_geometric_weights(sp, weighting_mode='uniform')
    # On a hemispherical grid the ghost-node hull gives unequal weights;
    # they must not be elementwise equal to 1/N.
    assert not np.allclose(w_geo, w_uni, rtol=1e-3), (
        "Geometric weights on a hemisphere should differ from uniform 1/N"
    )


def test_geometric_mode_sum_to_one():
    sp = _hemi_sphere()
    w = calculate_geometric_weights(sp, weighting_mode='geometric')
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# auto mode — CV-based snap
# ---------------------------------------------------------------------------

def test_auto_returns_valid_weights_for_regular_grid(capsys):
    """A full, regular sphere gets valid weights regardless of the auto decision."""
    sp = _full_sphere(n_az=24, n_el=13)
    w_auto = calculate_geometric_weights(sp, weighting_mode='auto')
    assert w_auto.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(w_auto >= 0)


# ---------------------------------------------------------------------------
# outlier radius cleaning
# ---------------------------------------------------------------------------

def test_outlier_radii_get_zero_weight():
    """Points far from the dominant radius should receive weight 0."""
    sp = _mixed_radii_sphere(radius=1.0, n_outlier=5)
    n_main = len(_full_sphere())
    w = calculate_geometric_weights(sp, advanced_dfeq=True)
    # The 5 outlier points (appended at the end) must have weight 0
    assert np.all(w[n_main:] == 0.0)


# ---------------------------------------------------------------------------
# hull-failure fallback — degenerate input (all points identical)
# ---------------------------------------------------------------------------

def test_fallback_on_degenerate_input():
    """When ConvexHull fails, weights should still sum to 1."""
    # Three collinear points — hull will fail
    sp = np.array([[0.0, 0.0, 1.0]] * 10)
    w = calculate_geometric_weights(sp, advanced_dfeq=False)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(w >= 0)
