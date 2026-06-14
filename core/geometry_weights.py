"""
core/geometry_weights.py
Shared geometric weighting helpers for diffuse-field computation.

Used by both backend subprocess scripts (generate_sofa.py and generate_dfhrtf.py).
Depends only on numpy and scipy so subprocess import cost is zero beyond what
those scripts already import.
"""
import numpy as np
from scipy.spatial import ConvexHull


def spherical_to_cartesian(r, az, el):
    """Convert spherical (r, az, el) in degrees to Cartesian (x, y, z)."""
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    x = r * np.cos(el_rad) * np.cos(az_rad)
    y = r * np.cos(el_rad) * np.sin(az_rad)
    z = r * np.sin(el_rad)
    return np.column_stack((x, y, z))


def get_dominant_radius(source_pos):
    """Return the most-common measurement radius from a SourcePosition array."""
    r = source_pos[:, 2]
    radii_rounded = np.round(r, decimals=3)
    values, counts = np.unique(radii_rounded, return_counts=True)
    return values[np.argmax(counts)]


def calculate_geometric_weights(source_pos, advanced_dfeq=True,
                                weighting_mode='auto', balance_weight_cv=0.12):
    """Calculate solid-angle weights for spherical diffuse-field integration.

    Parameters
    ----------
    source_pos : ndarray, shape (M, 3)
        SOFA SourcePosition array — columns are (azimuth °, elevation °, radius m).
    advanced_dfeq : bool
        When True (default), apply data sanitisation before computing weights:
        filter to the dominant measurement radius (±0.05 m) and deduplicate
        angular positions.  Set False only when the source positions are already
        known to be clean.
    weighting_mode : {'auto', 'geometric', 'uniform'}
        'auto'      — compute ghost-node ConvexHull weights, then snap to
                      uniform 1/N if their CV is below *balance_weight_cv*
                      (correct behaviour for balanced grids such as spherical
                      t-designs).
        'geometric' — always use ghost-node ConvexHull solid-angle weights.
        'uniform'   — always return equal 1/N weights (bypass ConvexHull).
    balance_weight_cv : float
        CV threshold for the auto mode snap-to-uniform decision (default 0.12).

    Returns
    -------
    weights : ndarray, shape (M,)
        Non-negative weights that sum to 1.0.  Points that were removed by the
        sanitisation pass have weight 0.
    """
    az = source_pos[:, 0]
    el = source_pos[:, 1]

    # ── Cleaning prologue ────────────────────────────────────────────────────
    if advanced_dfeq:
        dominant_r = get_dominant_radius(source_pos)

        # Filter out points that deviate significantly from the dominant radius
        valid_idx = np.abs(source_pos[:, 2] - dominant_r) < 0.05

        # Remove duplicate angular positions
        az_el_rounded = np.round(source_pos[valid_idx][:, 0:2], decimals=3)
        _, unique_idx = np.unique(az_el_rounded, axis=0, return_index=True)

        final_idx = np.where(valid_idx)[0][unique_idx]
        az_clean = az[final_idx]
        el_clean = el[final_idx]
        cart_coords = spherical_to_cartesian(1.0, az_clean, el_clean)
    else:
        cart_coords = spherical_to_cartesian(1.0, az, el)
        final_idx = np.arange(len(source_pos))

    # ── Uniform helper ───────────────────────────────────────────────────────
    def _uniform_weights():
        w = np.zeros(len(source_pos))
        w[final_idx] = 1.0 / len(final_idx)
        return w

    mode = weighting_mode.lower()
    if mode == 'uniform':
        print(f"   [Weighting] Uniform (1/{len(final_idx)})")
        return _uniform_weights()

    # ── Ghost-node ConvexHull solid-angle weights ────────────────────────────
    # Augments the real measurement grid with a dense Fibonacci-sphere cap of
    # dummy points placed below the dataset's lowest real point.  This closes
    # the ConvexHull for open/hemispherical grids (e.g. ARI) so that every
    # real measurement node receives a solid-angle weight proportional to its
    # true coverage area rather than benefiting from the artificial "floor"
    # that the bare-hull algorithm creates.
    try:
        # 1. Dense Fibonacci-sphere dummy points
        samples = 1000
        indices = np.arange(0, samples, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / samples)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        gx = np.cos(theta) * np.sin(phi)
        gy = np.sin(theta) * np.sin(phi)
        gz = np.cos(phi)
        ghost_coords = np.column_stack((gx, gy, gz))

        # 2. Keep only ghosts that fall below the real dataset's lowest point
        min_z = np.min(cart_coords[:, 2])
        ghost_coords = ghost_coords[ghost_coords[:, 2] < (min_z - 0.01)]

        # 3. Build hull over the closed (real + ghost) point set
        cart_coords_closed = np.vstack((cart_coords, ghost_coords))
        hull = ConvexHull(cart_coords_closed)

        weights = np.zeros(len(source_pos))
        temp_weights = np.zeros(len(cart_coords_closed))
        for simplex in hull.simplices:
            A, B, C = cart_coords_closed[simplex]
            cross_prod = np.cross(B - A, C - A)
            area = 0.5 * np.linalg.norm(cross_prod)
            temp_weights[simplex] += area / 3.0

        # 4. Strip ghost-node weights; keep only real-node weights
        real_weights = temp_weights[:len(cart_coords)]

        # 5. Map back and normalise
        weights[final_idx] = real_weights
        weights /= np.sum(weights)

    except Exception as e:
        print(f"[Warning] Weight calculation fallback used: {e}")
        weights = np.ones(len(source_pos)) / len(source_pos)

    if mode == 'geometric':
        return weights

    # ── Auto: snap to uniform when the grid is already well-balanced ─────────
    # If the computed geometric weights have low variance the grid is nearly
    # uniform (e.g. a spherical t-design) and 1/N is both more accurate and
    # more stable than a hull estimate.
    wa = weights[weights > 0]
    cv = float(wa.std() / wa.mean()) if len(wa) > 0 and wa.mean() > 0 else 1.0
    if cv < balance_weight_cv:
        print(f"   [Weighting] Auto -> uniform  "
              f"(geo-weight CV={cv * 100:.1f}% < threshold {balance_weight_cv * 100:.0f}%)")
        return _uniform_weights()
    print(f"   [Weighting] Auto -> geometric (geo-weight CV={cv * 100:.1f}%)")
    return weights
