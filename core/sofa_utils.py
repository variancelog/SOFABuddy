import netCDF4 as nc
import numpy as np

from core.grid_info import (  # noqa: F401  (re-exported)
    GridCoverage, grid_coverage, elevation_range, format_elevation_range,
    format_elevation_span, elevation_ring_xyz, boundary_ring_elevations,
    draw_elevation_rings, apply_unit_sphere_axes,
)

def get_sofa_metadata(file_path: str) -> dict:
    """
    Extracts metadata from a SOFA file without loading the entire impulse response array.
    Uses netCDF4 directly for high performance.
    """
    try:
        with nc.Dataset(file_path, 'r') as ds:
            # 1. Samplerate
            sr_var = ds.variables.get('Data.SamplingRate')
            if sr_var is not None:
                samplerate = float(sr_var[0])
            else:
                samplerate = 0.0

            # 2. Points (M) and Length (N)
            # IR is usually (M, R, N)
            ir_var = ds.variables.get('Data.IR')
            if ir_var is not None:
                points = int(ir_var.shape[0])
                length = int(ir_var.shape[2])
            else:
                points = 0
                length = 0

            # 3. Elevation coverage (None if SourcePosition is absent/unreadable)
            el_min = el_max = coverage = None
            try:
                src_var = ds.variables.get('SourcePosition')
                if src_var is not None:
                    src = src_var[:]
                    coverage = grid_coverage(src[:, 0], src[:, 1])
                    if coverage is not None:
                        el_min, el_max = coverage.el_min, coverage.el_max
            except Exception:
                pass

            return {
                "samplerate": samplerate,
                "length": length,
                "points": points,
                "el_min": el_min,
                "el_max": el_max,
                "coverage": coverage,
            }
    except Exception as e:
        print(f"Error reading SOFA metadata for {file_path}: {e}")
        return None

def calculate_grid_radius(file_path: str) -> float:
    """
    Returns the dominant (most common) grid radius in metres from SourcePosition,
    matching the radius reported by the Evaluation Grids plot.
    """
    try:
        with nc.Dataset(file_path, 'r') as ds:
            src_var = ds.variables.get('SourcePosition')
            if src_var is None: return 0.0
            src = src_var[:]              # (M, 3) -> [az, el, radius]
            radii = np.round(src[:, 2], decimals=3)
            vals, counts = np.unique(radii, return_counts=True)
            return float(vals[np.argmax(counts)])
    except Exception as e:
        print(f"Error calculating grid radius for {file_path}: {e}")
        return 0.0

def get_grid_cartesian(file_path: str) -> dict | None:
    """
    Reads SourcePosition from a SOFA file and returns the full measurement
    grid projected onto the unit sphere as cartesian coordinates.

    Returns a dict with:
        'cart'   : np.ndarray (M, 3) — xyz coordinates on the unit sphere
        'radius' : float — dominant measurement radius (metres)
        'n_nodes': int   — number of measurement points
        'el_min', 'el_max' : float | None — raw elevation extent in degrees
        'coverage': GridCoverage | None — pole coverage analysis
    Returns None on failure.
    """
    try:
        with nc.Dataset(file_path, 'r') as ds:
            src_var = ds.variables.get('SourcePosition')
            if src_var is None:
                return None
            src = src_var[:]                     # (M, 3) -> [az, el, radius]
            n_nodes = src.shape[0]

            # Dominant radius
            radii = np.round(src[:, 2], decimals=3)
            vals, counts = np.unique(radii, return_counts=True)
            dominant_r = float(vals[np.argmax(counts)])

            # Project to unit sphere (az/el in degrees)
            az_rad = np.radians(src[:, 0])
            el_rad = np.radians(src[:, 1])
            x = np.cos(el_rad) * np.cos(az_rad)
            y = np.cos(el_rad) * np.sin(az_rad)
            z = np.sin(el_rad)
            cart = np.column_stack((x, y, z))

            coverage = grid_coverage(src[:, 0], src[:, 1])
            el_min, el_max = ((coverage.el_min, coverage.el_max)
                              if coverage is not None else (None, None))
            return {'cart': cart, 'radius': dominant_r, 'n_nodes': n_nodes,
                    'el_min': el_min, 'el_max': el_max, 'coverage': coverage}
    except Exception as e:
        print(f"Error loading grid for {file_path}: {e}")
        return None

def calculate_max_itd(file_path: str) -> float:
    """
    Calculates the maximum ITD (in ms) by finding peak impulse indices across all measurements.
    """
    try:
        with nc.Dataset(file_path, 'r') as ds:
            sr_var = ds.variables.get('Data.SamplingRate')
            if sr_var is None: return 0.0
            samplerate = float(sr_var[0])

            ir_var = ds.variables.get('Data.IR')
            if ir_var is None: return 0.0
            
            # Load IR data (M, R, N)
            ir_data = ir_var[:]
            
            # Find index of maximum absolute value for each measurement and receiver
            # peak_indices shape: (M, R)
            peak_indices = np.argmax(np.abs(ir_data), axis=2)
            
            # Calculate difference between Right (1) and Left (0) for all M
            itd_samples = np.abs(peak_indices[:, 1] - peak_indices[:, 0])
            
            max_itd_samples = np.max(itd_samples)
            return (max_itd_samples / samplerate) * 1000.0
            
    except Exception as e:
        print(f"Error calculating ITD for {file_path}: {e}")
        return 0.0
