"""compare_weighting_grids.py

Standalone test utility that generates two weighted-grid comparison plots for a
single SOFA file — one per weighting method — so you can visually inspect the
difference between them.

Usage
-----
    python tests/compare_weighting_grids.py --input path/to/file.sofa
    python tests/compare_weighting_grids.py --input path/to/file.sofa --out-dir path/to/output/

Output
------
    <out_dir>/<stem>_weighting_original.png  — bare ConvexHull (may inflate edge
                                               weights on hemispherical grids)
    <out_dir>/<stem>_weighting_new.png       — ghost-node ConvexHull (current
                                               production method)

Both plots use identical styling (constant node size, viridis colormap, 1st–99th
percentile color range) so that only the weight distribution — not the rendering —
differs between them.
"""

import argparse
import os
import sys
import numpy as np
import sofar as sf
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Match the marker size used in the main app (generate_dfhrtf.py).
NODE_SIZE = 20  # pts² — constant; weight is encoded by color only


# ─────────────────────────── geometry helpers ────────────────────────────────

def spherical_to_cartesian(r, az, el):
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    x = r * np.cos(el_rad) * np.cos(az_rad)
    y = r * np.cos(el_rad) * np.sin(az_rad)
    z = r * np.sin(el_rad)
    return np.column_stack((x, y, z))


def _clean_source_pos(source_pos):
    """Shared data-sanitisation prologue used by both weighting methods.

    Returns
    -------
    cart_coords : (K, 3) array  Cartesian coords of the cleaned point set
    final_idx   : (K,) int array  Indices back into the original source_pos
    """
    az = source_pos[:, 0]
    el = source_pos[:, 1]
    r  = source_pos[:, 2]

    # Find dominant radius and discard outliers
    radii_rounded = np.round(r, decimals=3)
    values, counts = np.unique(radii_rounded, return_counts=True)
    dominant_r = values[np.argmax(counts)]
    valid_idx = np.abs(r - dominant_r) < 0.05

    # Remove duplicate angular coordinates within the valid subset
    az_el_rounded = np.round(source_pos[valid_idx][:, 0:2], decimals=3)
    _, unique_idx = np.unique(az_el_rounded, axis=0, return_index=True)

    final_idx   = np.where(valid_idx)[0][unique_idx]
    cart_coords = spherical_to_cartesian(1.0, az[final_idx], el[final_idx])
    return cart_coords, final_idx


# ─────────────────────────── weighting methods ───────────────────────────────

def geometric_weights_original(source_pos):
    """Original ConvexHull method (currently commented out in generate_dfhrtf.py).

    Builds a ConvexHull directly over the cleaned point set.  Works well for
    full-sphere grids but can over-weight edge nodes on open / hemispherical
    grids because the hull closes across the gap.
    """
    cart_coords, final_idx = _clean_source_pos(source_pos)
    weights = np.zeros(len(source_pos))
    try:
        hull = ConvexHull(cart_coords)
        temp_weights = np.zeros(len(cart_coords))
        for simplex in hull.simplices:
            A, B, C = cart_coords[simplex]
            cross_prod = np.cross(B - A, C - A)
            area = 0.5 * np.linalg.norm(cross_prod)
            temp_weights[simplex] += area / 3.0
        weights[final_idx] = temp_weights
        weights /= np.sum(weights)
    except Exception as e:
        print(f"[Warning] original method fallback: {e}")
        weights = np.ones(len(source_pos)) / len(source_pos)
    return weights


def geometric_weights_new(source_pos):
    """New ghost-node ConvexHull method (active in generate_dfhrtf.py).

    Adds Fibonacci-sphere dummy nodes below the lowest real point so the hull
    closes naturally, computes face-area weights, then strips the ghost weights.
    """
    cart_coords, final_idx = _clean_source_pos(source_pos)
    weights = np.zeros(len(source_pos))
    try:
        # Generate a dense Fibonacci-sphere grid and keep only ghost nodes that
        # fall below the actual dataset's lowest point (with a tiny offset).
        samples = 1000
        indices = np.arange(0, samples, dtype=float) + 0.5
        phi    = np.arccos(1 - 2 * indices / samples)
        theta  = np.pi * (1 + 5**0.5) * indices
        gx = np.cos(theta) * np.sin(phi)
        gy = np.sin(theta) * np.sin(phi)
        gz = np.cos(phi)
        ghost_coords = np.column_stack((gx, gy, gz))

        min_z = np.min(cart_coords[:, 2])
        ghost_coords = ghost_coords[ghost_coords[:, 2] < (min_z - 0.01)]

        cart_coords_closed = np.vstack((cart_coords, ghost_coords))
        hull = ConvexHull(cart_coords_closed)
        temp_weights = np.zeros(len(cart_coords_closed))
        for simplex in hull.simplices:
            A, B, C = cart_coords_closed[simplex]
            cross_prod = np.cross(B - A, C - A)
            area = 0.5 * np.linalg.norm(cross_prod)
            temp_weights[simplex] += area / 3.0

        # Strip ghost node weights (keep only the original nodes)
        real_weights = temp_weights[:len(cart_coords)]
        weights[final_idx] = real_weights
        weights /= np.sum(weights)
    except Exception as e:
        print(f"[Warning] new method fallback: {e}")
        weights = np.ones(len(source_pos)) / len(source_pos)
    return weights


def geometric_weights_uniform(source_pos):
    """Uniform 1/N weighting — correct for balanced grids such as spherical t-designs.

    Uses the same cleaning prologue as the other methods so *N* is the kept
    node count, not the raw measurement count.
    """
    _, final_idx = _clean_source_pos(source_pos)
    weights = np.zeros(len(source_pos))
    weights[final_idx] = 1.0 / len(final_idx)
    return weights


def weight_cv(weights):
    """Coefficient of variation (std/mean) of the active (> 0) weights.

    Returns 1.0 as a safe sentinel when weights are all zero.
    """
    wa = weights[weights > 0]
    if len(wa) == 0 or wa.mean() == 0:
        return 1.0
    return float(wa.std() / wa.mean())


def weights_for_mode(source_pos, mode, balance_weight_cv=0.12):
    """Dispatcher: compute weights according to *mode*.

    Parameters
    ----------
    source_pos        : (M, 3) array  azimuth / elevation / radius
    mode              : 'auto', 'geometric', or 'uniform' (case-insensitive)
    balance_weight_cv : CV threshold used by 'auto' (default 0.12)

    Returns
    -------
    weights       : (M,) ndarray
    resolved_mode : 'geometric' or 'uniform' — the mode actually applied
    cv            : float — geo-weight CV (always computed for the metadata bar)
    """
    mode = mode.lower()
    w_geo = geometric_weights_new(source_pos)
    cv = weight_cv(w_geo)

    if mode == 'uniform':
        return geometric_weights_uniform(source_pos), 'uniform', cv
    if mode == 'geometric':
        return w_geo, 'geometric', cv
    # auto
    if cv < balance_weight_cv:
        return geometric_weights_uniform(source_pos), 'uniform', cv
    return w_geo, 'geometric', cv


# ─────────────────────────── plotting helper ─────────────────────────────────

def plot_weighted_grid(source_pos, weights, title, out_path):
    """Render a weighted-grid scatter plot to *out_path*.

    Styling mirrors the main app's weighted-grid block in generate_dfhrtf.py:
    constant node size, viridis colormap, 1st–99th percentile color range.
    """
    cart = spherical_to_cartesian(1.0, source_pos[:, 0], source_pos[:, 1])

    mask      = weights > 0
    w         = weights[mask]
    cart_clean = cart[mask]

    vmin = np.percentile(w, 1)
    vmax = np.percentile(w, 99)

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(
        cart_clean[:, 0], cart_clean[:, 1], cart_clean[:, 2],
        c=w, s=NODE_SIZE, cmap='viridis', alpha=0.85, vmin=vmin, vmax=vmax
    )
    plt.colorbar(sc, ax=ax, label='Solid-angle weight', shrink=0.6, pad=0.1)
    ax.set_title(f"{title}\n{int(mask.sum())} nodes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [+] Saved: {os.path.basename(out_path)}")


# ─────────────────────────── entry point ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate weighted-grid comparison plots (original vs new method)."
    )
    parser.add_argument("--input",   required=True, help="Path to input SOFA file")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same folder as input)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.input))[0]

    print(f"\n--- Loading: {os.path.basename(args.input)} ---")
    sofa       = sf.read_sofa(args.input)
    source_pos = sofa.SourcePosition
    print(f"    {len(source_pos)} measurement positions")

    print("\n[1/2] Computing weights — Original method (bare ConvexHull)...")
    w_orig = geometric_weights_original(source_pos)
    n_active_orig = int((w_orig > 0).sum())
    print(f"      {n_active_orig} active nodes")
    plot_weighted_grid(
        source_pos, w_orig,
        title="Original weighting (bare ConvexHull)",
        out_path=os.path.join(out_dir, f"{stem}_weighting_original.png"),
    )

    print("\n[2/2] Computing weights — New method (ghost-node ConvexHull)...")
    w_new = geometric_weights_new(source_pos)
    n_active_new = int((w_new > 0).sum())
    print(f"      {n_active_new} active nodes")
    plot_weighted_grid(
        source_pos, w_new,
        title="New weighting (ghost-node ConvexHull)",
        out_path=os.path.join(out_dir, f"{stem}_weighting_new.png"),
    )

    print(f"\nDone. Output folder: {out_dir}")


if __name__ == "__main__":
    main()
