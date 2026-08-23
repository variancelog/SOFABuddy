"""Shared pytest fixtures for the SOFA Buddy test suite.

Provides three families of helpers plus two collection hooks:

* **Argparse-derived arg fixtures** (`sofa_args`, `dfhrtf_args`) — build a
  complete `argparse.Namespace` from the backend scripts' real `build_parser()`
  defaults, so tests never hand-copy defaults that can silently drift.
* **Manifest-common fixtures** (`sofa_common`, `dfhrtf_common`) — the same
  defaults as a plain dict shaped for a manifest's `"common"` section, with the
  per-job keys excluded.
* **Synthetic SOFA factories** (`fibonacci_sphere`, `synthetic_sofa`,
  `measured_like_sofa`, `synthetic_sofa_file`, `ild_sofa`) — small in-memory
  fixtures so the suite needs no large binaries.

Hooks: a report header line stating whether real `tests/data/sofa/*.sofa`
fixtures are present, and auto-application of the `qt` marker to any test that
pulls in `qtbot`/`qapp` (so `pytest -m "not qt"` is reliably headless).
"""

import argparse
import glob
import os

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Manifest per-job keys
# --------------------------------------------------------------------------- #
def dfhrtf_manifest_job_keys():
    """generate_dfhrtf.py already owns its per-job key tuple; read it from the
    module (lazily, to keep conftest import cheap) so a newly added per-job key
    is automatically excluded from `dfhrtf_common`."""
    from core.scripts.generate_dfhrtf import _MANIFEST_JOB_OVERRIDE_KEYS
    return _MANIFEST_JOB_OVERRIDE_KEYS


# generate_sofa.py has no equivalent tuple (its run_manifest() reads the job
# keys positionally: job["input"], job["output"], job["target_fs"],
# job.get("dfeq")). Mirrored explicitly here; keep in sync with run_manifest().
SOFA_MANIFEST_JOB_KEYS = ("input", "output", "target_fs", "dfeq")

# `--manifest` itself is never part of a manifest's own "common" section.
_MANIFEST_ONLY_KEYS = ("manifest",)


# --------------------------------------------------------------------------- #
# Argparse-derived defaults
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def _sofa_arg_defaults():
    """Every generate_sofa.py CLI dest -> its argparse default."""
    from core.scripts.generate_sofa import build_parser
    return vars(build_parser().parse_args([]))


@pytest.fixture(scope="session")
def _dfhrtf_arg_defaults():
    """Every generate_dfhrtf.py CLI dest -> its argparse default."""
    from core.scripts.generate_dfhrtf import build_parser
    return vars(build_parser().parse_args([]))


@pytest.fixture
def sofa_args(_sofa_arg_defaults):
    """Factory: `sofa_args(**overrides)` -> full generate_sofa argparse Namespace.

    Raises AttributeError on an override key the parser doesn't define, so a
    renamed CLI flag fails loudly instead of being silently ignored.
    """
    defaults = _sofa_arg_defaults

    def _make(**overrides):
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise AttributeError(f"unknown generate_sofa arg(s): {sorted(unknown)}")
        return argparse.Namespace(**{**defaults, **overrides})

    return _make


@pytest.fixture
def dfhrtf_args(_dfhrtf_arg_defaults):
    """Factory: `dfhrtf_args(**overrides)` -> full generate_dfhrtf argparse Namespace."""
    defaults = _dfhrtf_arg_defaults

    def _make(**overrides):
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise AttributeError(f"unknown generate_dfhrtf arg(s): {sorted(unknown)}")
        return argparse.Namespace(**{**defaults, **overrides})

    return _make


def _common_dict(defaults, job_keys, overrides):
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise AttributeError(f"unknown manifest common key(s): {sorted(unknown)}")
    excluded = set(job_keys) | set(_MANIFEST_ONLY_KEYS)
    base = {k: v for k, v in defaults.items() if k not in excluded}
    base.update(overrides)
    return base


@pytest.fixture
def sofa_common(_sofa_arg_defaults):
    """Factory: manifest `"common"` dict for generate_sofa.run_manifest().

    Argparse defaults minus the per-job keys (input/output/target_fs/dfeq) and
    `manifest`.
    """
    def _make(**overrides):
        return _common_dict(_sofa_arg_defaults, SOFA_MANIFEST_JOB_KEYS, overrides)
    return _make


@pytest.fixture
def dfhrtf_common(_dfhrtf_arg_defaults):
    """Factory: manifest `"common"` dict for generate_dfhrtf.run_manifest().

    Argparse defaults minus `_MANIFEST_JOB_OVERRIDE_KEYS` and `manifest`.
    """
    def _make(**overrides):
        return _common_dict(_dfhrtf_arg_defaults, dfhrtf_manifest_job_keys(), overrides)
    return _make


# --------------------------------------------------------------------------- #
# Grid / synthetic SOFA builders
# --------------------------------------------------------------------------- #
def _fibonacci_sphere(n):
    """n points ~evenly on the unit sphere; returns (az_deg, el_deg)."""
    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)          # colatitude [0, pi]
    theta = np.pi * (1 + 5 ** 0.5) * i      # golden-angle azimuth
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    return az, el


@pytest.fixture(scope="session")
def fibonacci_sphere():
    """The `_fibonacci_sphere(n) -> (az_deg, el_deg)` helper as a fixture."""
    return _fibonacci_sphere


def _shaped_burst(n_taps):
    """Impulse + decaying tail: smooth, strictly positive spectrum."""
    tail = 0.5 ** np.arange(n_taps)
    burst = np.zeros(n_taps)
    burst[0] = 1.0
    burst += 0.4 * tail
    return burst


def _make_synthetic_sofa(radius_m=1.5, fs=48000, n_points=120, n_taps=64,
                         n_samples=256, onset=60):
    """Synthetic SimpleFreeFieldHRIR: full-sphere grid, single-pole shaped
    bursts (positive magnitude across the audio band, so no divide-by-zero in
    ratio tests)."""
    import sofar as sf

    az, el = _fibonacci_sphere(n_points)
    M = len(az)
    burst = _shaped_burst(n_taps)

    ir = np.zeros((M, 2, n_samples), dtype=float)
    for m in range(M):
        # Mild per-ear level trim so L/R aren't bit-identical; still same shape.
        ir[m, 0, onset:onset + n_taps] = 0.30 * burst
        ir[m, 1, onset:onset + n_taps] = 0.28 * burst

    s = sf.Sofa("SimpleFreeFieldHRIR")
    s.Data_IR = ir
    s.Data_SamplingRate = float(fs)
    s.SourcePosition = np.column_stack((az, el, np.full(M, float(radius_m))))
    return s


@pytest.fixture
def synthetic_sofa():
    """Factory: `synthetic_sofa(radius_m=1.5, fs=48000, n_points=120,
    n_taps=64, n_samples=256, onset=60)` -> in-memory sofar Sofa."""
    return _make_synthetic_sofa


def _make_measured_like_sofa(n_points=120, fs=48000, n=256, lf_noise=True, seed=0):
    """Synthetic SimpleFreeFieldHRIR whose LF content is near-zero with chaotic
    phase (like a real measured HRTF), plus a per-position bulk delay + ITD.

    A single-pole broadband burst placed at a known onset gives a well-behaved
    mid/high band; when lf_noise is True the sub-~120 Hz band is replaced by tiny
    random-phase content — the exact condition that baits a DC-anchored unwrap
    slip.
    """
    import scipy.fft as fft
    import sofar as sf

    rng = np.random.default_rng(seed)
    az, el = _fibonacci_sphere(n_points)
    M = len(az)
    freqs = fft.rfftfreq(n, 1.0 / fs)
    lf = freqs < 120.0

    burst = _shaped_burst(64)

    onset = 50
    ir = np.zeros((M, 2, n), dtype=float)
    for m in range(M):
        # small per-position ITD so delays are not identical
        itd = int(round(6 * np.sin(np.radians(az[m])) * np.cos(np.radians(el[m]))))
        for r, lvl in ((0, 0.30), (1, 0.28)):
            o = onset + (itd if r == 0 else -itd)
            ir[m, r, o:o + 64] = lvl * burst
        if lf_noise:
            H = fft.rfft(ir[m], axis=-1)
            # tiny magnitude, random phase in the LF band -> unwrap bait
            amp = 1e-4 * np.max(np.abs(H))
            noise = amp * np.exp(1j * rng.uniform(-np.pi, np.pi, size=(2, lf.sum())))
            H[:, lf] = noise
            ir[m] = fft.irfft(H, n=n, axis=-1)

    s = sf.Sofa("SimpleFreeFieldHRIR")
    s.Data_IR = ir
    s.Data_SamplingRate = float(fs)
    s.SourcePosition = np.column_stack((az, el, np.full(M, 1.5)))
    return s


@pytest.fixture
def measured_like_sofa():
    """Factory: `measured_like_sofa(n_points=120, fs=48000, n=256,
    lf_noise=True, seed=0)` -> in-memory sofar Sofa with measured-like LF."""
    return _make_measured_like_sofa


def _write_synthetic_sofa(path, fs=48000, n_points=96, n_taps=64, n_samples=256,
                          onset=40, radius_m=1.5):
    """Write a tiny synthetic SimpleFreeFieldHRIR SOFA file to *path*; return str."""
    import sofar as sf

    s = _make_synthetic_sofa(radius_m=radius_m, fs=fs, n_points=n_points,
                             n_taps=n_taps, n_samples=n_samples, onset=onset)
    sf.write_sofa(str(path), s)
    return str(path)


@pytest.fixture
def synthetic_sofa_file():
    """Factory: `synthetic_sofa_file(path, ...)` writes a synthetic SOFA to
    *path* and returns the path as a str. Same knobs as `synthetic_sofa`."""
    return _write_synthetic_sofa


# --------------------------------------------------------------------------- #
# Known-ILD synthetic SOFA
# --------------------------------------------------------------------------- #
def _make_ild_sofa(ild_db, fs=44100.0, n_samples=1024, n_points=60,
                   band=None, seed=42):
    """Build a synthetic SOFA with a KNOWN, injected inter-aural level difference.

    The L channel is left as-is and the R channel is attenuated by `ild_db`, so
    `core.ild_audit.diffuse_field_ild()` should report
    `ild_estimate_db ~= +ild_db` (positive = L louder).

    Parameters
    ----------
    ild_db : float
        Target ILD in dB. Positive means L louder than R.
    fs : float
        Sample rate. Default 44100 Hz.
    n_samples : int
        IR length. Default 1024 -> ~43.07 Hz FFT bin spacing, which puts ~9
        bins inside the 100-500 Hz LF band that `ild_audit._BANDS` inspects
        (the old 256-sample fixture gave only ~2 usable bins there).
    n_points : int
        Grid size. Default 60 = a uniform azimuth ring at elevation 0
        (360 deg / 6 deg steps), matching the historical `_FakeSofa` grid.
    band : tuple(float, float) or None
        None (default) injects the ILD flat across all frequencies. A
        (low_hz, high_hz) tuple injects it only inside that band, leaving the
        rest of the spectrum at 0 dB ILD — use this to exercise band-selective
        behavior (e.g. an ILD present at LF but not at HF).
    seed : int
        RNG seed for the white-noise HRIRs (deterministic).

    Returns
    -------
    sofar.Sofa with SimpleFreeFieldHRIR convention.
    """
    import scipy.fft as fft
    import sofar as sf

    rng = np.random.default_rng(seed)
    M, N = int(n_points), int(n_samples)
    ir = rng.standard_normal((M, 2, N))

    gain_r = 10.0 ** (-float(ild_db) / 20.0)
    if band is None:
        ir[:, 1, :] *= gain_r
    else:
        lo, hi = band
        freqs = fft.rfftfreq(N, 1.0 / float(fs))
        mask = (freqs >= lo) & (freqs < hi)
        H = fft.rfft(ir[:, 1, :], axis=-1)
        H[:, mask] *= gain_r
        ir[:, 1, :] = fft.irfft(H, n=N, axis=-1)

    # Uniform azimuth ring at elevation 0, tiled/truncated to M points.
    az = np.linspace(0.0, 360.0, M, endpoint=False)
    el = np.zeros_like(az)
    r = np.ones_like(az)

    s = sf.Sofa("SimpleFreeFieldHRIR")
    s.Data_IR = ir
    s.Data_SamplingRate = float(fs)
    s.SourcePosition = np.column_stack((az, el, r))
    return s


@pytest.fixture
def ild_sofa():
    """Factory: `ild_sofa(ild_db, fs=44100.0, n_samples=1024, n_points=60,
    band=None, seed=42)` -> sofar Sofa with a known injected ILD.

    Positive `ild_db` means L louder. Pass `band=(lo_hz, hi_hz)` for a
    band-limited injection instead of a flat one. See `_make_ild_sofa`.
    """
    return _make_ild_sofa


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #
_SOFA_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "data", "sofa")


def pytest_report_header(config):
    """Report whether real .sofa fixture files are present.

    Globs for `*.sofa` specifically — a non-empty directory holding only, say,
    a README is NOT fixture coverage.
    """
    files = glob.glob(os.path.join(_SOFA_FIXTURE_DIR, "*.sofa"))
    if files:
        return f"sofa fixtures: {len(files)} .sofa file(s) in tests/data/sofa/"
    return "sofa fixtures: none (no *.sofa in tests/data/sofa/) — fixture-backed tests will skip"


def pytest_collection_modifyitems(config, items):
    """Auto-mark any test that pulls in a Qt fixture with `qt`, so
    `pytest -m "not qt"` reliably runs headless."""
    for item in items:
        fixturenames = getattr(item, "fixturenames", ())
        if "qtbot" in fixturenames or "qapp" in fixturenames:
            item.add_marker(pytest.mark.qt)
