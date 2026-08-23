"""Headless tests for `core/averaging.py` (the global-averaging pipeline).

Covers the four public classes — `DataIngestor`, `CommonAxisInterpolator`,
`OutlierDetector`, `CurveAverager` — plus the module-private `_softdtw_func`
concurrency contract.

No Qt and no real SOFA files: every fixture is a small synthetic curve.

The highest-value case here is `TestTextRoundTrip`, which exercises the real
production hand-off: `generate_dfhrtf.save_text_data()` writes the per-SOFA
text files that `DataIngestor.load_csv()` later ingests for global averaging.
"""

import concurrent.futures
import inspect

import numpy as np
import pandas as pd
import pytest

from core import averaging
from core.averaging import (
    CommonAxisInterpolator,
    CurveAverager,
    DataIngestor,
    OutlierDetector,
    _softdtw_func,
)


# --------------------------------------------------------------------------- #
# Shared synthetic curves
# --------------------------------------------------------------------------- #
@pytest.fixture
def norm_axis():
    """A normalized [0, 1] frequency axis, as produced by the interpolator."""
    return np.linspace(0.0, 1.0, 64)


@pytest.fixture
def linear_curve(norm_axis):
    """A straight line in normalized space — arc-length blending is exact here."""
    return 0.2 + 0.5 * norm_axis


@pytest.fixture
def wavy_curve(norm_axis):
    """A curved response with genuine peaks/nulls (shape-preservation case)."""
    return 0.5 + 0.4 * np.sin(2 * np.pi * norm_axis * 2)


# --------------------------------------------------------------------------- #
# Priority 1 — the v0.4.1 CPU-oversubscription regression guard
# --------------------------------------------------------------------------- #
class TestSoftDtwWorkerBound:
    """`_softdtw_func` runs *inside* an L-BFGS-B objective that is itself
    already invoked from chunked/parallel batch processing. v0.4.1 bounded its
    inner ThreadPoolExecutor to a single worker by default so it stops
    oversubscribing cores. The file comments that contract; these tests enforce
    it.
    """

    def test_softdtw_func_n_workers_defaults_to_one(self):
        default = inspect.signature(_softdtw_func).parameters["n_workers"].default
        assert default == 1, (
            "core.averaging._softdtw_func n_workers default regressed from 1 to "
            f"{default!r}; this is the v0.4.1 CPU-oversubscription fix."
        )
        assert default is not None

    def test_barycenter_n_workers_defaults_to_one(self):
        default = inspect.signature(
            CurveAverager.soft_dtw_barycenter
        ).parameters["n_workers"].default
        assert default == 1, (
            "CurveAverager.soft_dtw_barycenter n_workers default regressed from "
            f"1 to {default!r}."
        )
        assert default is not None

    @pytest.mark.parametrize("n_workers", [None, 1, 3])
    def test_executor_is_constructed_with_bounded_max_workers(
        self, monkeypatch, wavy_curve, n_workers
    ):
        """Whatever `n_workers` the caller passes (including the default), the
        inner pool must be built with an explicit, positive `max_workers` — never
        `None` (which would let Python size it from the CPU count)."""
        recorded = []
        real_executor = concurrent.futures.ThreadPoolExecutor

        class RecordingExecutor(real_executor):
            def __init__(self, *args, **kwargs):
                recorded.append(kwargs.get("max_workers", args[0] if args else None))
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(
            averaging.concurrent.futures, "ThreadPoolExecutor", RecordingExecutor
        )

        mags = np.vstack([wavy_curve, wavy_curve * 0.9 + 0.05, wavy_curve * 1.1])
        kwargs = {} if n_workers is None else {"n_workers": n_workers}
        CurveAverager.soft_dtw_barycenter(mags, max_iter=3, **kwargs)

        expected = 1 if n_workers is None else n_workers
        assert recorded, "soft_dtw_barycenter never constructed a ThreadPoolExecutor"
        assert all(m is not None for m in recorded), (
            f"unbounded ThreadPoolExecutor(max_workers=None) constructed: {recorded}"
        )
        assert set(recorded) == {expected}


# --------------------------------------------------------------------------- #
# Highest-value integration: generate_dfhrtf text file -> DataIngestor
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_curve_data():
    freqs = np.array([20.0, 100.0, 1000.0, 5000.0, 20000.0])
    mags = np.array([-3.5, 0.0, 1.25, -7.75, -12.0])
    return freqs, mags


class TestTextRoundTrip:
    """`generate_dfhrtf.save_text_data()` -> `DataIngestor.load_csv()`."""

    def test_csv_round_trip_is_lossless(self, tmp_path, sample_curve_data):
        from core.scripts.generate_dfhrtf import save_text_data

        freqs, mags = sample_curve_data
        path = tmp_path / "curve.csv"
        save_text_data(str(path), freqs, mags, delimiter=",")

        df = DataIngestor.load_csv(str(path))

        assert list(df.columns) == ["frequency", "magnitude"]
        assert len(df) == len(freqs)
        np.testing.assert_allclose(df["frequency"].to_numpy(), freqs, rtol=0, atol=1e-6)
        np.testing.assert_allclose(df["magnitude"].to_numpy(), mags, rtol=0, atol=1e-3)

    def test_csv_header_names_are_recognized_not_positional(
        self, tmp_path, sample_curve_data
    ):
        """The header written by save_text_data is 'Frequency (Hz)' /
        'Magnitude (dB)'; the loader's keyword sniffing must match on those and
        not silently fall back to column order."""
        from core.scripts.generate_dfhrtf import save_text_data

        freqs, mags = sample_curve_data
        path = tmp_path / "curve.csv"
        save_text_data(str(path), freqs, mags, delimiter=",")

        raw = pd.read_csv(path)
        assert list(raw.columns) == ["Frequency (Hz)", "Magnitude (dB)"]

        df = DataIngestor.load_csv(str(path))
        np.testing.assert_allclose(df["magnitude"].to_numpy(), mags, atol=1e-3)

    @pytest.mark.parametrize("delimiter", ["\t", " ", ";"])
    def test_headerless_txt_round_trip_keeps_every_point(
        self, tmp_path, sample_curve_data, delimiter
    ):
        from core.scripts.generate_dfhrtf import save_text_data

        freqs, mags = sample_curve_data
        path = tmp_path / "curve.txt"
        save_text_data(str(path), freqs, mags, delimiter=delimiter)

        df = DataIngestor.load_csv(str(path))
        assert len(df) == len(freqs)
        np.testing.assert_allclose(df["frequency"].to_numpy(), freqs, atol=1e-6)

    def test_load_csv_sorts_by_frequency(self, tmp_path):
        path = tmp_path / "unsorted.csv"
        path.write_text(
            "Frequency (Hz),Magnitude (dB)\n1000,1.0\n100,2.0\n10000,3.0\n"
        )
        df = DataIngestor.load_csv(str(path))
        np.testing.assert_allclose(df["frequency"].to_numpy(), [100, 1000, 10000])
        np.testing.assert_allclose(df["magnitude"].to_numpy(), [2.0, 1.0, 3.0])
        assert list(df.index) == [0, 1, 2]

    def test_single_column_file_raises(self, tmp_path):
        path = tmp_path / "onecol.csv"
        # No whitespace anywhere, so the sep=r'\s+' retry also yields one column.
        path.write_text("Frequency\n100\n200\n300\n")
        with pytest.raises(RuntimeError):
            DataIngestor.load_csv(str(path))

    def test_missing_file_raises_runtime_error(self, tmp_path):
        with pytest.raises(RuntimeError):
            DataIngestor.load_csv(str(tmp_path / "nope.csv"))


# --------------------------------------------------------------------------- #
# CommonAxisInterpolator
# --------------------------------------------------------------------------- #
def _df(freqs, mags):
    return pd.DataFrame({"frequency": freqs, "magnitude": mags})


class TestCommonAxisInterpolator:
    def test_round_trip_on_analytic_curve(self):
        """A curve already sampled on the target log grid must survive
        interpolate -> normalize -> unnormalize to 1e-9."""
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 512)
        mags = -6.0 * np.log2(freqs / 1000.0)  # exact -6 dB/oct tilt

        out = CommonAxisInterpolator.interpolate_to_common_axis([_df(freqs, mags)],
                                                                num_points=512)

        np.testing.assert_allclose(out["common_freqs"], freqs, rtol=1e-12)
        np.testing.assert_allclose(out["raw_mags_2d"][0], mags, atol=1e-9)

        restored = CommonAxisInterpolator.unnormalize_magnitude(
            out["norm_mags_2d"][0], out["mag_min"], out["mag_max"]
        )
        np.testing.assert_allclose(restored, mags, atol=1e-9)

    def test_normalized_axes_span_zero_to_one(self):
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 128)
        dfs = [_df(freqs, np.sin(np.log10(freqs)) * 5.0),
               _df(freqs, np.cos(np.log10(freqs)) * 3.0)]
        out = CommonAxisInterpolator.interpolate_to_common_axis(dfs, num_points=128)

        assert out["norm_freqs"][0] == pytest.approx(0.0)
        assert out["norm_freqs"][-1] == pytest.approx(1.0)
        assert out["norm_mags_2d"].min() == pytest.approx(0.0)
        assert out["norm_mags_2d"].max() == pytest.approx(1.0)

    def test_flat_magnitude_zero_division_branch(self):
        """mag_max == mag_min must yield zeros, not NaN/inf."""
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 64)
        dfs = [_df(freqs, np.full(64, -4.0)), _df(freqs, np.full(64, -4.0))]

        out = CommonAxisInterpolator.interpolate_to_common_axis(dfs, num_points=64)

        assert out["mag_min"] == pytest.approx(-4.0)
        assert out["mag_max"] == pytest.approx(-4.0)
        assert np.all(np.isfinite(out["norm_mags_2d"]))
        np.testing.assert_array_equal(out["norm_mags_2d"], np.zeros((2, 64)))

        # unnormalize must still return the original flat level
        restored = CommonAxisInterpolator.unnormalize_magnitude(
            out["norm_mags_2d"][0], out["mag_min"], out["mag_max"]
        )
        np.testing.assert_allclose(restored, -4.0)

    def test_empty_input_returns_empty_dict(self):
        assert CommonAxisInterpolator.interpolate_to_common_axis([]) == {}

    def test_non_overlapping_ranges_raise(self):
        a = _df(np.linspace(20.0, 200.0, 16), np.zeros(16))
        b = _df(np.linspace(5000.0, 20000.0, 16), np.zeros(16))
        with pytest.raises(ValueError, match="overlapping"):
            CommonAxisInterpolator.interpolate_to_common_axis([a, b])

    def test_octave_fraction_grid_matches_generate_dfhrtf(self):
        """The 1/N-octave axis must match generate_dfhrtf's own generator so
        averaged output shares the per-SOFA text-file frequency grid."""
        from core.scripts.generate_dfhrtf import generate_fractional_octave_frequencies

        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 256)
        out = CommonAxisInterpolator.interpolate_to_common_axis(
            [_df(freqs, np.zeros(256))], octave_fraction=6
        )
        expected = generate_fractional_octave_frequencies(
            out["common_freqs"][0], freqs.max(), fraction=6
        )
        np.testing.assert_allclose(out["common_freqs"], expected, rtol=1e-9)


# --------------------------------------------------------------------------- #
# OutlierDetector
# --------------------------------------------------------------------------- #
class TestOutlierDetector:
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_fewer_than_three_curves_keeps_all(self, n):
        mags = np.zeros((n, 16)) if n else np.zeros((0, 16))
        kept, excluded = OutlierDetector.filter_outliers(mags, "euclidean")
        assert kept == list(range(n))
        assert excluded == []

    def test_identical_curves_have_zero_std_and_keep_all(self, wavy_curve):
        mags = np.vstack([wavy_curve] * 5)
        kept, excluded = OutlierDetector.filter_outliers(mags, "euclidean")
        assert kept == [0, 1, 2, 3, 4]
        assert excluded == []

    def test_flags_a_genuine_outlier(self, norm_axis, wavy_curve):
        cluster = [wavy_curve + 0.001 * i for i in range(6)]
        rogue = 1.0 - wavy_curve * 4.0
        mags = np.vstack(cluster + [rogue])

        kept, excluded = OutlierDetector.filter_outliers(mags, "euclidean",
                                                         threshold_std=1.0)
        assert 6 in excluded
        assert set(kept) == {0, 1, 2, 3, 4, 5}

    def test_never_excludes_everything(self, norm_axis):
        """The safety guard at the end of filter_outliers: if the threshold is
        tight enough to exclude every curve, the closest-to-median curve must be
        kept and removed from the exclusion list."""
        rng = np.random.default_rng(0)
        mags = rng.random((6, 32))

        kept, excluded = OutlierDetector.filter_outliers(
            mags, "euclidean", threshold_std=-1e6
        )

        assert len(kept) == 1, "safety guard must keep exactly one survivor"
        assert kept[0] not in excluded
        assert sorted(kept + excluded) == list(range(6))

        # the survivor is the curve with the smallest mean distance to the batch
        from scipy.spatial.distance import pdist, squareform

        avg = squareform(pdist(mags, metric="euclidean")).sum(axis=1) / 5
        assert kept[0] == int(np.argmin(avg))


# --------------------------------------------------------------------------- #
# CurveAverager
# --------------------------------------------------------------------------- #
class TestCurveAveragerIdentity:
    """Feeding N copies of one curve to any averager must return that curve."""

    @pytest.mark.parametrize("n_copies", [1, 2, 5])
    def test_arithmetic_mean_identity(self, wavy_curve, n_copies):
        mags = np.vstack([wavy_curve] * n_copies)
        np.testing.assert_allclose(CurveAverager.arithmetic_mean(mags), wavy_curve,
                                   atol=1e-12)

    @pytest.mark.parametrize("n_copies", [1, 2, 5])
    def test_geometric_blend_identity_exact_on_linear(self, norm_axis, linear_curve,
                                                      n_copies):
        """Arc-length reparameterization is exact for a straight path."""
        mags = np.vstack([linear_curve] * n_copies)
        out = CurveAverager.geometric_arc_length_blend(norm_axis, mags)
        np.testing.assert_allclose(out, linear_curve, atol=1e-12)

    @pytest.mark.parametrize("n_copies", [2, 5])
    def test_geometric_blend_identity_on_curved_input(self, norm_axis, wavy_curve,
                                                      n_copies):
        """On a curved path the re-parameterize/re-interpolate cycle is only
        approximate; it must still reproduce the curve to within 0.03 of the
        normalized [0, 1] magnitude range."""
        mags = np.vstack([wavy_curve] * n_copies)
        out = CurveAverager.geometric_arc_length_blend(norm_axis, mags)
        assert np.max(np.abs(out - wavy_curve)) < 0.03

    @pytest.mark.parametrize("n_copies", [1, 3])
    def test_soft_dtw_identity(self, wavy_curve, n_copies):
        mags = np.vstack([wavy_curve] * n_copies)
        out = CurveAverager.soft_dtw_barycenter(mags)
        assert out.shape == wavy_curve.shape
        assert np.max(np.abs(out - wavy_curve)) < 0.03


class TestCurveAveragerEdgeCases:
    def test_arithmetic_mean_of_two_curves(self, norm_axis):
        a = np.full(16, 0.2)
        b = np.full(16, 0.8)
        np.testing.assert_allclose(
            CurveAverager.arithmetic_mean(np.vstack([a, b])), np.full(16, 0.5)
        )

    def test_geometric_blend_empty_and_single(self, norm_axis, wavy_curve):
        assert CurveAverager.geometric_arc_length_blend(
            norm_axis, np.zeros((0, 64))
        ).size == 0
        np.testing.assert_array_equal(
            CurveAverager.geometric_arc_length_blend(norm_axis, np.vstack([wavy_curve])),
            wavy_curve,
        )

    def test_soft_dtw_empty(self):
        assert CurveAverager.soft_dtw_barycenter(np.zeros((0, 64))).size == 0

    def test_geometric_blend_zero_arc_length_branch(self):
        """`calc_arc_length`'s `arc[-1] > 0` guard: a degenerate path of zero
        total length (constant frequency axis *and* flat magnitudes) must not
        divide by zero. The guard returns the raw all-zero arc, after which
        interp1d rejects the collapsed parametric range with a ValueError --
        pinned here so the branch can never silently start emitting NaNs."""
        flat_axis = np.zeros(8)
        mags = np.vstack([np.zeros(8), np.zeros(8)])

        with pytest.raises(ValueError):
            CurveAverager.geometric_arc_length_blend(flat_axis, mags)

    def test_geometric_blend_handles_flat_magnitudes_on_real_axis(self, norm_axis):
        """A flat magnitude curve still has non-zero arc length along the
        frequency axis, so it must blend normally."""
        mags = np.vstack([np.full(64, 0.5), np.full(64, 0.5)])
        out = CurveAverager.geometric_arc_length_blend(norm_axis, mags)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, 0.5, atol=1e-9)


class TestCurveAveragerAgreement:
    """On a tight cluster of near-identical curves all three methods should
    land in the same place — this is the property the UI's algo dropdown
    implicitly promises."""

    def test_all_methods_agree_on_tight_cluster(self, norm_axis, linear_curve):
        mags = np.vstack([linear_curve + d for d in (-0.01, 0.0, 0.01)])
        mean = CurveAverager.arithmetic_mean(mags)
        geo = CurveAverager.geometric_arc_length_blend(norm_axis, mags)
        sdtw = CurveAverager.soft_dtw_barycenter(mags)

        assert np.max(np.abs(geo - mean)) < 0.03
        assert np.max(np.abs(sdtw - mean)) < 0.03
