# Real SOFA test fixtures

This directory holds the real HRTF `.sofa` files that a handful of regression
tests measure properties of. **The files themselves are not in git** — `*.sofa`
is in the repo's `.gitignore` because these are 1.6-11 MB binaries each
(~25 MB total), and several carry upstream redistribution terms.

Everything else in the suite is synthetic and needs nothing from this
directory. Only the real-corpus tests below depend on it, and they skip (with a
reason, not silently) when it is empty.

## Expected files

| File | Size | Upstream database |
|---|---|---|
| `measured_ARI.sofa` | ~2.8 MB | **ARI** — Acoustics Research Institute (Austrian Academy of Sciences, Vienna) HRTF database. Hemispherical 1550-point grid. |
| `measured_BILI.sofa` | ~11 MB | **BILI** — IRCAM "Base de données d'Individus LIsteners" / BiLi project. 1680 points. Written by the legacy SOFA C++ API 1.x, so it needs `core.sofa_io.read_sofa_robust()`'s `NC_GLOBAL.` prefix repair. |
| `measured_HUTUBS.sofa` | ~1.6 MB | **HUTUBS** — TU Berlin head-and-torso HRTF database (Brinkmann et al.). 440 points. |
| `measured_HUTUBS_2.sofa` | ~1.6 MB | **HUTUBS** — a second subject from the same database. 440 points. |
| `measured_LISTEN.sofa` | ~5.5 MB | **LISTEN** — IRCAM/AKG LISTEN HRTF database. 187 points. |
| `simulated_VARLOG.sofa` | ~2.8 MB | Not from a public database: a locally **simulated** (numerically computed, Mesh2HRTF-style) set on a 1550-point grid, referred to elsewhere in the repo as VARLOG/VL03. Exact provenance beyond "simulated in-house" is **not documented** in this repo. |

Filenames matter: `tests/test_diffuse_ild.py` hardcodes
`{"measured_BILI.sofa", "measured_LISTEN.sofa"}` as `KNOWN_OFFENDERS` — the two
files with a genuine low-frequency inter-aural level imbalance, which the audit
engine is asserted to *flag*. All other `*.sofa` files present are asserted to
be calibration-clean.

## What depends on these files

- `tests/test_diffuse_ild.py::test_healthy_sofa_fixtures`
- `tests/test_diffuse_ild.py::test_known_offenders_are_flagged`

Both are marked `@pytest.mark.needs_sofa_fixtures`.

Grid-geometry-only coverage does **not** need these files: the
`SourcePosition` arrays were extracted into committed `.npy` fixtures under
`tests/data/grids/` (a few KB each) by `tools/extract_grid_fixtures.py`, so
`tests/test_weighting_classifier.py` runs fully on a clean clone.

## Running the full, non-skipped suite

Place the files above in this directory, then:

```bash
python -m pytest tests/ -ra
```

`-ra` prints the skip reasons, so a missing corpus is reported rather than
silently passed over. `pytest` also prints a header line each run stating how
many `*.sofa` fixtures it found (see `pytest_report_header` in
`tests/conftest.py`).

To run only the fixture-backed tests:

```bash
python -m pytest tests/ -m needs_sofa_fixtures -ra
```

To run everything *except* them:

```bash
python -m pytest tests/ -m "not needs_sofa_fixtures"
```

There is no `--require-sofa-fixtures` flag; skips are the mechanism, and `-ra`
is how you see them.

## Regenerating the grid fixtures

After adding or replacing a `.sofa` file here:

```bash
python tools/extract_grid_fixtures.py          # writes tests/data/grids/*.npy
python tools/extract_grid_fixtures.py --check  # dry run, reports shapes only
```
