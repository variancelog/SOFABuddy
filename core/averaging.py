import os
import numpy as np
import pandas as pd
import concurrent.futures
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from typing import List, Dict
from tslearn.metrics import cdist_soft_dtw, SquaredEuclidean, SoftDTW

# --- Helper Functions for Soft-DTW ---

def _sdtw_worker(Z, X_i, gamma):
    D = SquaredEuclidean(Z, X_i)
    sdtw = SoftDTW(D, gamma=gamma)
    return sdtw.compute(), D.jacobian_product(sdtw.grad())

def _softdtw_func(Z, X, weights, barycenter_shape, gamma, progress_queue=None, state=None, max_iter=25, n_workers=1):
    Z_reshaped = Z.reshape(barycenter_shape)
    G = np.zeros_like(Z_reshaped)
    obj = 0.0

    total_curves = len(X)
    # Bounded worker count: default 1 keeps this in-loop pool from oversubscribing
    # cores when the caller already provides concurrency (chunk subprocesses /
    # the chunk-pool sizing used by the parent's global averaging).
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_workers)) as executor:
        futures = {executor.submit(_sdtw_worker, Z_reshaped, X[i], gamma): i for i in range(total_curves)}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            val, G_tmp = future.result()
            G += weights[i] * G_tmp
            obj += weights[i] * val
            completed += 1
            if progress_queue is not None and state is not None:
                progress_queue.put(("sub_progress", (state['iter'], completed, total_curves, max_iter)))
            
    return obj, G.ravel()

# --- Grid Helpers ---

def _fractional_octave_grid(min_freq, max_freq, fraction):
    """Generate a 1/N-octave log-spaced grid from min_freq up to max_freq.

    Mirrors the generate_fractional_octave_frequencies() logic used by
    generate_dfhrtf.py so that averaging output files share the same frequency
    axis as the per-SOFA text files.
    """
    freqs = []
    f = float(min_freq)
    step = 2.0 ** (1.0 / int(fraction))
    # Small tolerance avoids dropping the last point due to float rounding
    while f <= float(max_freq) + 1e-6:
        freqs.append(f)
        f *= step
    return np.array(freqs)

# --- Main Classes ---

class DataIngestor:
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """
        Loads a CSV file and attempts to identify frequency and magnitude columns.
        Returns a DataFrame with standardized 'frequency' and 'magnitude' columns.
        """
        try:
            # save_text_data() only writes a header row for .csv output --
            # every other delimiter/extension combo is headerless, so the
            # header can't be assumed; letting pandas default to header=0
            # on a headerless file eats the lowest-frequency row as column
            # names. Detect it by checking whether the first row's first
            # field parses as a number.
            with open(filepath, 'r') as fh:
                first_line = fh.readline().strip()
            first_token = first_line
            for sep_char in (',', ';', '\t'):
                if sep_char in first_line:
                    first_token = first_line.split(sep_char)[0]
                    break
            else:
                first_token = first_line.split()[0] if first_line.split() else first_line
            try:
                float(first_token)
                header_arg = None
            except ValueError:
                header_arg = 0

            # Attempt to read CSV with pandas
            df = pd.read_csv(filepath, header=header_arg)

            # Basic validation: need at least 2 columns; comma is pandas'
            # default, so fall back through the other delimiters generate_
            # dfhrtf.py's text_delimiter setting can actually produce.
            if len(df.columns) < 2:
                for sep in (';', r'\s+'):
                    df = pd.read_csv(filepath, sep=sep, header=header_arg)
                    if len(df.columns) >= 2:
                        break
                if len(df.columns) < 2:
                    raise ValueError(f"File {os.path.basename(filepath)} does not have at least two columns.")

            freq_col = None
            mag_col = None
            
            lower_cols = [str(c).lower() for c in df.columns]
            
            for i, col in enumerate(lower_cols):
                if 'freq' in col or 'hz' in col:
                    freq_col = df.columns[i]
                elif 'mag' in col or 'db' in col or 'amp' in col:
                    mag_col = df.columns[i]
                    
            if freq_col is None:
                freq_col = df.columns[0]
            if mag_col is None:
                mag_col = df.columns[1]
                
            result_df = pd.DataFrame({
                'frequency': df[freq_col].astype(float),
                'magnitude': df[mag_col].astype(float)
            })
            
            # Sort by frequency
            result_df = result_df.sort_values('frequency').reset_index(drop=True)
            return result_df
            
        except Exception as e:
            raise RuntimeError(f"Error loading {os.path.basename(filepath)}: {e}")

class CommonAxisInterpolator:
    @staticmethod
    def interpolate_to_common_axis(dataframes: List[pd.DataFrame], num_points: int = 512, octave_fraction: int = None) -> Dict:
        """
        Takes a list of dataframes and maps them to a rigid common frequency axis.
        Normalizes both axes (log-frequency and magnitude) to [0.0, 1.0] before returning.

        When octave_fraction is provided the common axis uses 1/N-octave spacing
        (matching the per-SOFA text file grid); otherwise falls back to a
        num_points log-spaced grid.
        """
        if not dataframes:
            return {}
            
        # Find the overlapping frequency range
        min_freqs = [df['frequency'].min() for df in dataframes]
        max_freqs = [df['frequency'].max() for df in dataframes]
        
        global_min_freq = max(min_freqs)
        global_max_freq = min(max_freqs)
        
        if global_min_freq >= global_max_freq:
             raise ValueError("Dataframes do not have overlapping frequency ranges.")
             
        # Enforce strictly positive bounds for logarithmic space
        if global_min_freq <= 0:
            global_min_freq = 1e-10
            
        # Create common rigid logarithmic frequency grid
        if octave_fraction is not None:
            # Use fractional-octave spacing so the axis matches the per-SOFA text files
            common_freqs = _fractional_octave_grid(global_min_freq, global_max_freq, octave_fraction)
        else:
            common_freqs = np.logspace(np.log10(global_min_freq), np.log10(global_max_freq), num_points)
        
        interpolated_mags = []
        for df in dataframes:
            # Create interpolator for this dataframe
            interpolator = interp1d(df['frequency'], df['magnitude'], kind='linear', bounds_error=False, fill_value="extrapolate")
            mapped_mag = interpolator(common_freqs)
            interpolated_mags.append(mapped_mag)
            
        raw_mags_2d = np.vstack(interpolated_mags)
        
        # --- Normalization Step ---
        
        # 1. Frequency (Log-Scale)
        log_freqs = np.log10(common_freqs)
        log_f_min, log_f_max = np.min(log_freqs), np.max(log_freqs)
        
        if log_f_max > log_f_min:
            norm_freqs = (log_freqs - log_f_min) / (log_f_max - log_f_min)
        else:
            norm_freqs = np.zeros_like(log_freqs)
            
        # 2. Magnitude
        mag_min = np.min(raw_mags_2d)
        mag_max = np.max(raw_mags_2d)
        
        if mag_max > mag_min:
            norm_mags_2d = (raw_mags_2d - mag_min) / (mag_max - mag_min)
        else:
            norm_mags_2d = np.zeros_like(raw_mags_2d)
            
        return {
            'common_freqs': common_freqs,
            'raw_mags_2d': raw_mags_2d,
            'norm_freqs': norm_freqs,
            'norm_mags_2d': norm_mags_2d,
            'mag_min': float(mag_min),
            'mag_max': float(mag_max)
        }

    @staticmethod
    def unnormalize_magnitude(norm_mag: np.ndarray, mag_min: float, mag_max: float) -> np.ndarray:
        """Restores the averaged 0-1 magnitude curve back to original dB scale."""
        return norm_mag * (mag_max - mag_min) + mag_min

class OutlierDetector:
    @staticmethod
    def filter_outliers(norm_mags_2d, method, threshold_std=2.0):
        """
        Calculates pairwise distances between all sequences, computes each sequence's 
        median distance to the rest of the batch, and flags sequences that exceed 
        the threshold (in standard deviations from the overall median).
        Returns: (list_of_kept_indices, list_of_excluded_indices)
        """
        n = len(norm_mags_2d)
        if n < 3:
            # Need at least 3 files to compute a meaningful statistical median/std exclusion
            return list(range(n)), []

        # 1. Compute Pairwise Distance Matrix
        # Euclidean distance for all methods
        dist_matrix = squareform(pdist(norm_mags_2d, metric='euclidean'))

        # 2. Calculate the average distance of each curve to all other curves
        # Summing each row and dividing by (n-1) to exclude the 0 distance to itself
        avg_distances = np.sum(dist_matrix, axis=1) / (n - 1)

        # 3. Calculate batch statistics
        median_dist = np.median(avg_distances)
        std_dist = np.std(avg_distances)

        if std_dist == 0:
            return list(range(n)), []

        # 4. Filter indices
        kept_indices = []
        excluded_indices = []
        
        limit = median_dist + (threshold_std * std_dist)
        
        for i, dist in enumerate(avg_distances):
            if dist <= limit:
                kept_indices.append(i)
            else:
                excluded_indices.append(i)

        # Safety: never exclude everything
        if not kept_indices:
            best_idx = int(np.argmin(avg_distances))
            kept_indices = [best_idx]
            if best_idx in excluded_indices:
                excluded_indices.remove(best_idx)

        return kept_indices, excluded_indices

class CurveAverager:
    @staticmethod
    def arithmetic_mean(norm_mags_2d: np.ndarray) -> np.ndarray:
        """Standard point-by-point mean on normalized data."""
        return np.mean(norm_mags_2d, axis=0)

    @staticmethod
    def geometric_arc_length_blend(norm_freqs: np.ndarray, norm_mags_2d: np.ndarray) -> np.ndarray:
        """Geometric blend treating responses as paths and averaging normalized arc lengths."""
        if len(norm_mags_2d) == 0:
            return np.array([])
        if len(norm_mags_2d) == 1:
            return norm_mags_2d[0]
            
        num_points = len(norm_freqs)
            
        def blend_two(mag_A, mag_B, weight_A, weight_B):
            # Calculate cumulative arc length for both curves based on the pre-normalized axes
            def calc_arc_length(x, y):
                dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
                arc = np.concatenate(([0], np.cumsum(dist)))
                return arc / arc[-1] if arc[-1] > 0 else arc

            arc_A = calc_arc_length(norm_freqs, mag_A)
            arc_B = calc_arc_length(norm_freqs, mag_B)

            # Create a common parametric timeline (t)
            t_common = np.linspace(0, 1, num_points)

            # Interpolate X and Y coordinates along the common timeline
            interp_x_A = interp1d(arc_A, norm_freqs)(t_common)
            interp_y_A = interp1d(arc_A, mag_A)(t_common)
            
            interp_x_B = interp1d(arc_B, norm_freqs)(t_common)
            interp_y_B = interp1d(arc_B, mag_B)(t_common)

            # Geometric Blend: Average the X and Y coordinates using dynamic weights
            blend_x_norm = (interp_x_A * weight_A) + (interp_x_B * weight_B)
            blend_y_norm = (interp_y_A * weight_A) + (interp_y_B * weight_B)

            # Re-interpolate the blended parametric curve back onto the rigid common frequency grid
            final_mags = interp1d(blend_x_norm, blend_y_norm, bounds_error=False, fill_value="extrapolate")(norm_freqs)

            return final_mags

        # Iteratively blend all sequences applying proper running average weights
        current_avg = norm_mags_2d[0]
        for i in range(1, len(norm_mags_2d)):
            weight_B = 1.0 / (i + 1)
            weight_A = 1.0 - weight_B
            current_avg = blend_two(current_avg, norm_mags_2d[i], weight_A, weight_B)
            
        return current_avg

    @staticmethod
    def soft_dtw_barycenter(norm_mags_2d: np.ndarray, gamma: float = 0.01,
                             max_iter: int = 25, tol: float = 1e-3,
                             progress_queue=None, n_workers: int = 1) -> np.ndarray:
        """Smooth shape-preserving mean using Regularized Soft-DTW."""
        if len(norm_mags_2d) == 0:
            return np.array([])

        ts_data = np.expand_dims(norm_mags_2d, axis=-1)
        weights = np.ones(len(ts_data)) / len(ts_data)

        # Initialize with Euclidean barycenter (arithmetic mean)
        init_barycenter = np.mean(ts_data, axis=0)

        state = {'iter': 0}
        total_curves = len(ts_data)

        if progress_queue is not None:
            progress_queue.put(("init_progress", max_iter * total_curves))

        def callback(xk):
            state['iter'] += 1

        res = minimize(
            _softdtw_func,
            init_barycenter.ravel(),
            args=(ts_data, weights, init_barycenter.shape, gamma, progress_queue, state, max_iter, n_workers),
            method="L-BFGS-B",
            jac=True,
            tol=tol,
            callback=callback,
            options=dict(maxiter=max_iter)
        )

        barycenter = res.x.reshape(init_barycenter.shape)
        return barycenter.flatten()
