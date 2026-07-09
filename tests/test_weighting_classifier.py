"""
tests/test_weighting_classifier.py
Objective comparison of the new spherical-design quadrature classifier against
the legacy geo-weight-CV heuristic for the auto weighting mode.

Requires only numpy and scipy (no Qt, no SOFA files for the core battery).
SOFA-file tests are skipped automatically if netCDF4/sofar is unavailable.

Grid battery
------------
  Equal-weight designs (expect → 'uniform'):
    N-Design_29/30/50/55, T-Design_21           (Mesh2HRTF Nodes.txt)

  Non-equal-weight grids (expect → 'geometric'):
    Lebedev_N10, Sphere_ICO, Sphere_UV, Default (Mesh2HRTF Nodes.txt)
    Equiangular full sphere                     (synthetic)
    Near-uniform Fibonacci sphere               (synthetic)
    Real measured SOFA grids                    (tests/data/sofa/, optional)
"""
import os
import sys
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.geometry_weights import (
    calculate_geometric_weights,
    spherical_to_cartesian,
    _is_equal_weight_design,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SOFA_DIR = os.path.join(DATA_DIR, 'sofa')


# ---------------------------------------------------------------------------
# Loaders / converters
# ---------------------------------------------------------------------------

def _load_nodes(subdir):
    """Load a Mesh2HRTF Nodes.txt from tests/data/<subdir>/Nodes.txt.

    Format: first line is node count; remaining lines are  id x y z.
    Returns unit-normalised (N, 3) Cartesian coords.
    """
    path = os.path.join(DATA_DIR, subdir, 'Nodes.txt')
    data = np.loadtxt(path, skiprows=1)   # skip the count header
    xyz = data[:, 1:4]                    # drop id column
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / norms                    # project onto unit sphere


def _nodes_as_sourcepos(xyz):
    """Convert unit-sphere Cartesian (N,3) → SOFA-style (az°, el°, r=1)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    az_deg = np.degrees(np.arctan2(y, x))
    el_deg = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    r = np.ones(len(xyz))
    return np.column_stack((az_deg, el_deg, r))


# ---------------------------------------------------------------------------
# Synthetic grid helpers
# ---------------------------------------------------------------------------

def _equiangular_sphere(n_az=24, n_el=13):
    """Regular equiangular grid — NOT a spherical design."""
    rows = []
    for el in np.linspace(-90, 90, n_el):
        for az in np.linspace(0, 360 - 360 / n_az, n_az):
            rows.append([az, el, 1.0])
    return np.array(rows)


def _fibonacci_sphere(n=500):
    """Near-uniform Fibonacci/golden-spiral sphere — NOT an exact design."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)          # colatitude
    theta = np.pi * (1 + 5 ** 0.5) * indices      # azimuth
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    xyz = np.column_stack((x, y, z))
    return _nodes_as_sourcepos(xyz)


# ---------------------------------------------------------------------------
# Reproduce the OLD CV heuristic (self-contained so it survives the removal)
# ---------------------------------------------------------------------------

def _old_cv_decision(source_pos, threshold=0.12):
    """Return 'uniform' or 'geometric' using the legacy geo-weight-CV rule."""
    w = calculate_geometric_weights(source_pos, weighting_mode='geometric')
    wa = w[w > 0]
    if len(wa) == 0 or wa.mean() == 0:
        return 'geometric'
    cv = float(wa.std() / wa.mean())
    return 'uniform' if cv < threshold else 'geometric'


def _new_decision(source_pos):
    """Return 'uniform' or 'geometric' using the new spherical-design test."""
    az, el = source_pos[:, 0], source_pos[:, 1]
    cart = spherical_to_cartesian(1.0, az, el)
    return 'uniform' if _is_equal_weight_design(cart) else 'geometric'


# ---------------------------------------------------------------------------
# Labeled battery
# ---------------------------------------------------------------------------
# Each entry: (name, source_pos_array, expected_label)
# Built lazily so individual tests can use it without loading everything.

_DESIGN_DIRS = [
    ('N-Design_29', 'uniform'),
    ('N-Design_30', 'uniform'),
    ('N-Design_50', 'uniform'),
    ('N-Design_55', 'uniform'),
    ('T-Design_21', 'uniform'),
]

_NON_DESIGN_DIRS = [
    ('Lebedev_N10', 'geometric'),
    ('Sphere_ICO',  'geometric'),
    ('Sphere_UV',   'geometric'),
    ('Default',     'geometric'),
]


def _build_file_battery():
    """Return [(name, sp, expected)] for all Mesh2HRTF Nodes.txt grids."""
    items = []
    for (subdir, label) in _DESIGN_DIRS + _NON_DESIGN_DIRS:
        xyz = _load_nodes(subdir)
        sp = _nodes_as_sourcepos(xyz)
        items.append((subdir, sp, label))
    return items


def _build_synthetic_battery():
    """Return [(name, sp, expected)] for programmatically generated grids."""
    return [
        ('Equiangular_24x13',  _equiangular_sphere(24, 13), 'geometric'),
        ('Fibonacci_500',      _fibonacci_sphere(500),       'geometric'),
    ]


def _build_full_battery():
    return _build_file_battery() + _build_synthetic_battery()


# ---------------------------------------------------------------------------
# Test 1 — new classifier is 100% accurate on the whole battery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,sp,expected", _build_full_battery(),
                         ids=[x[0] for x in _build_full_battery()])
def test_new_classifier_correct(name, sp, expected):
    """New spherical-design test must give the right label for every grid."""
    got = _new_decision(sp)
    assert got == expected, (
        f"{name}: expected '{expected}', new classifier returned '{got}'"
    )


# ---------------------------------------------------------------------------
# Test 2 — the old CV heuristic misclassifies at least one real design
# ---------------------------------------------------------------------------

def test_old_cv_heuristic_misclassifies_a_real_design():
    """N-Design_29 has geo-weight CV > 12 % — the old heuristic returns
    'geometric' for a grid where 1/N is provably exact.  This is the
    headline failure the new method eliminates."""
    xyz = _load_nodes('N-Design_29')
    sp = _nodes_as_sourcepos(xyz)

    old_label = _old_cv_decision(sp, threshold=0.12)
    new_label = _new_decision(sp)

    assert old_label == 'geometric', (
        f"N-Design_29: expected old heuristic to misclassify as 'geometric', "
        f"got '{old_label}' — the test premise may no longer hold"
    )
    assert new_label == 'uniform', (
        f"N-Design_29: new classifier should return 'uniform', got '{new_label}'"
    )


# ---------------------------------------------------------------------------
# Test 3 — SH quadrature residuals have a clean 9-order gap
# ---------------------------------------------------------------------------

def _worst_sh_residual(cart_coords, test_order=8):
    """Return the largest |4π/N Σ Yℓm| over ℓ=1..test_order, m=-ℓ..ℓ."""
    # Reimplement inline so the test is self-contained w.r.t. the compat shim.
    try:
        from scipy.special import sph_harm_y as _shy
        def _Y(l, m, col, az): return _shy(l, abs(m), col, az)
    except ImportError:
        from scipy.special import sph_harm as _shy
        def _Y(l, m, col, az): return _shy(abs(m), l, az, col)

    N = len(cart_coords)
    x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
    r = np.linalg.norm(cart_coords, axis=1)
    az = np.arctan2(y, x)
    col = np.arccos(np.clip(z / r, -1.0, 1.0))
    worst = 0.0
    for l in range(1, test_order + 1):
        for m in range(-l, l + 1):
            Yc = _Y(l, m, col, az)
            Yr = (np.sqrt(2) * (-1) ** m * Yc.imag if m < 0
                  else np.sqrt(2) * (-1) ** m * Yc.real if m > 0
                  else Yc.real)
            res = abs(4 * math.pi / N * np.sum(Yr))
            if res > worst:
                worst = res
    return worst


def test_residual_separation_has_clean_margin():
    """Design residuals ~1e-11; non-design residuals ~1e0.  Clean gap at 1e-6."""
    design_residuals = {}
    non_design_residuals = {}

    for (subdir, label) in _DESIGN_DIRS:
        xyz = _load_nodes(subdir)
        design_residuals[subdir] = _worst_sh_residual(xyz)

    for (subdir, label) in _NON_DESIGN_DIRS:
        xyz = _load_nodes(subdir)
        non_design_residuals[subdir] = _worst_sh_residual(xyz)

    max_design = max(design_residuals.values())
    min_non_design = min(non_design_residuals.values())

    assert max_design < 1e-6, (
        f"Some design grid has residual above 1e-6: "
        f"{max(design_residuals, key=design_residuals.get)} → {max_design:.2e}"
    )
    assert min_non_design > 1e-6, (
        f"Some non-design grid slipped below the 1e-6 threshold: "
        f"{min(non_design_residuals, key=non_design_residuals.get)} → {min_non_design:.2e}"
    )
    # Verify the gap is genuinely large, not marginal
    assert min_non_design / max_design > 1e4, (
        f"Gap between designs ({max_design:.2e}) and non-designs "
        f"({min_non_design:.2e}) is narrower than expected 4 orders of magnitude"
    )


# ---------------------------------------------------------------------------
# Test 4 — new method's chosen weights minimize quadrature error
# ---------------------------------------------------------------------------

def _sh_quadrature_error(weights, cart_coords, test_order=8):
    """RMS quadrature error |4π Σ wᵢ Yℓm(i)| over ℓ=1..test_order."""
    try:
        from scipy.special import sph_harm_y as _shy
        def _Y(l, m, col, az): return _shy(l, abs(m), col, az)
    except ImportError:
        from scipy.special import sph_harm as _shy
        def _Y(l, m, col, az): return _shy(abs(m), l, az, col)

    N_kept = len(weights[weights > 0])
    x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
    r = np.linalg.norm(cart_coords, axis=1)
    az = np.arctan2(y, x)
    col = np.arccos(np.clip(z / r, -1.0, 1.0))
    errors = []
    for l in range(1, test_order + 1):
        for m in range(-l, l + 1):
            Yc = _Y(l, m, col, az)
            Yr = (np.sqrt(2) * (-1) ** m * Yc.imag if m < 0
                  else np.sqrt(2) * (-1) ** m * Yc.real if m > 0
                  else Yc.real)
            # Weighted integral: 4π Σ wᵢ Yℓm(i) — exact designs → 0
            errors.append(abs(4 * math.pi * np.dot(weights, Yr)))
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def test_chosen_weights_minimize_quadrature_error():
    """New method's auto-chosen weights reduce quadrature error on the grids
    where the two classifiers disagree.

    When old→geometric but new→uniform (genuine designs): new must be strictly
    better — uniform weights integrate SHs to near-machine-zero on a design,
    while geometric weights add hull-estimation error.

    When old→uniform but new→geometric (near-uniform non-designs that the CV
    heuristic accidentally matches): the old heuristic lucked into the lower-
    error choice, but the new classifier's answer is still *correct*
    (Fibonacci/equiangular are not designs).  We do not assert that geometric
    is better than uniform in this direction — only that we fix the
    design→geometric misclassification, which is the critical failure.
    """
    # --- Design grids: new→uniform must beat old→geometric ---------------
    for (subdir, _label) in _DESIGN_DIRS:
        xyz = _load_nodes(subdir)
        cart = xyz  # already unit-normalised
        sp = _nodes_as_sourcepos(xyz)

        old_label = _old_cv_decision(sp)
        new_label = _new_decision(sp)
        assert new_label == 'uniform', f"{subdir}: new classifier should give 'uniform'"

        if old_label == 'geometric':
            # This is the meaningful comparison: old picked geometric (wrong),
            # new picks uniform (right) — new must have lower error.
            w_old = calculate_geometric_weights(sp, weighting_mode='geometric')
            w_new = calculate_geometric_weights(sp, weighting_mode='auto')
            err_old = _sh_quadrature_error(w_old, cart)
            err_new = _sh_quadrature_error(w_new, cart)
            assert err_new < err_old, (
                f"{subdir}: uniform (new) error {err_new:.4e} should be < "
                f"geometric (old) error {err_old:.4e}"
            )

    # --- Canonical case: N-Design_29 uniform → near machine-zero ----------
    xyz = _load_nodes('N-Design_29')
    cart_29 = xyz
    sp_29 = _nodes_as_sourcepos(xyz)

    w_old_geo = calculate_geometric_weights(sp_29, weighting_mode='geometric')
    w_new_auto = calculate_geometric_weights(sp_29, weighting_mode='auto')

    err_geo = _sh_quadrature_error(w_old_geo, cart_29)
    err_uni = _sh_quadrature_error(w_new_auto, cart_29)

    assert err_uni < err_geo, (
        f"N-Design_29: uniform weights (new) error {err_uni:.4e} should be "
        f"less than geometric weights (old) error {err_geo:.4e}"
    )
    # Uniform on a design should integrate SHs to near machine-zero
    assert err_uni < 1e-9, (
        f"N-Design_29 with uniform weights: expected near-zero error, got {err_uni:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 5 — end-to-end auto mode matches the classifier's decision
# ---------------------------------------------------------------------------

def test_auto_end_to_end_design_gets_uniform():
    """auto mode on a genuine design must return exact 1/N weights."""
    xyz = _load_nodes('T-Design_21')
    sp = _nodes_as_sourcepos(xyz)
    N = len(sp)

    w = calculate_geometric_weights(sp, weighting_mode='auto')
    kept = w[w > 0]
    expected = 1.0 / len(kept)
    np.testing.assert_allclose(kept, expected, rtol=1e-9,
                               err_msg="T-Design_21 in auto mode should give exact 1/N weights")


def test_auto_end_to_end_non_design_gets_geometric():
    """auto mode on a non-design must NOT return exact uniform weights."""
    sp = _equiangular_sphere(24, 13)
    N = len(sp)
    w = calculate_geometric_weights(sp, weighting_mode='auto')
    kept = w[w > 0]
    uniform_val = 1.0 / len(kept)
    assert not np.allclose(kept, uniform_val, rtol=1e-6), (
        "Equiangular sphere in auto mode should receive geometric (non-uniform) weights"
    )


# ---------------------------------------------------------------------------
# Test 6 — known limitation: low-degree polyhedra below the test order
# ---------------------------------------------------------------------------

def _octahedron_sourcepos():
    """6-point octahedron — a degree-3 spherical design, below test order 6."""
    pts = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
                   dtype=float)
    return _nodes_as_sourcepos(pts)


def _icosahedron_sourcepos():
    """12-point icosahedron — a degree-5 spherical design, below test order 6."""
    phi = (1 + math.sqrt(5)) / 2
    verts = []
    for a in [-1, 1]:
        for b in [-1, 1]:
            verts += [[0, a, b * phi], [a, b * phi, 0], [b * phi, 0, a]]
    xyz = np.array(verts, dtype=float)
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz /= norms
    return _nodes_as_sourcepos(xyz)


def test_design_order_limitation_documented():
    """Octahedron (deg-3) and icosahedron (deg-5) are genuine designs but fall
    below the fixed test order (6/8 requiring ≥49/81 points).
    _is_equal_weight_design returns False for them — documented known limitation.

    In practice these regular polyhedra have far too few points (6 and 12) to be
    useful HRTF measurement grids; the ghost-node ConvexHull method is also
    unreliable at such small counts.  This test only asserts that the classifier
    returns False and does NOT make quality assertions about the weights — the
    ghost-node hull is not designed for point sets this small."""
    for name, sp in [('octahedron', _octahedron_sourcepos()),
                     ('icosahedron', _icosahedron_sourcepos())]:
        az, el = sp[:, 0], sp[:, 1]
        cart = spherical_to_cartesian(1.0, az, el)
        result = _is_equal_weight_design(cart)
        assert result is False, (
            f"{name}: expected False (known limitation — fewer than 49 points "
            f"for test order 6), but got True"
        )


# ---------------------------------------------------------------------------
# Test 7 (optional) — real SOFA files are classified as non-designs
# ---------------------------------------------------------------------------

def _load_sofa_sourcepos(sofa_path):
    """Return SourcePosition array from a SOFA file (requires sofar/netCDF4)."""
    import sofar
    sofa = sofar.read_sofa(sofa_path)
    return sofa.SourcePosition


@pytest.mark.skipif(
    not os.path.isdir(SOFA_DIR) or not os.listdir(SOFA_DIR),
    reason="tests/data/sofa/ is empty or missing"
)
def test_real_sofa_grids_classified_as_geometric():
    """All real measured SOFA grids should be classified 'geometric' (not designs)."""
    sofar = pytest.importorskip('sofar', reason="sofar not installed")

    sofa_files = [f for f in os.listdir(SOFA_DIR) if f.endswith('.sofa')]
    assert sofa_files, "No .sofa files found in tests/data/sofa/"

    misclassified = []
    for fname in sofa_files:
        path = os.path.join(SOFA_DIR, fname)
        try:
            sp = _load_sofa_sourcepos(path)
        except Exception as e:
            pytest.skip(f"Could not load {fname}: {e}")
            continue
        label = _new_decision(sp)
        if label != 'geometric':
            misclassified.append(fname)

    assert not misclassified, (
        f"These real SOFA grids were incorrectly classified as 'uniform' designs: "
        f"{misclassified}"
    )
