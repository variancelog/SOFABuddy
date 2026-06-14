# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the application
python main.pyw

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_ui.py::test_flame_toggle_button

# Install dependencies
pip install -r requirements.txt
```

Tests use `pytest-qt` (not in `requirements.txt` but required for UI tests). `tests/test_config.py` runs without Qt.

## Architecture

SOFA Buddy is a PySide6 GUI application for processing SOFA (Spatially Oriented Format for Acoustics) HRTF files. Processing runs in two sequential stages: SOFA mastering → DFHRTF analysis.

### UI Layer (`ui/`)

- **`main_window.py`** — `SofaBuddyMainWindow`: top-level window with a 3-tab nav bar (Files / SOFA / DFHRTF), a resizable console pane at the bottom, and a single "Start Processing" button that drives `ProcessingWorker`. On Start Processing and app close, snapshots the current tab state via `ConfigManager.snapshot()` and saves it atomically with `ConfigManager.save()`.
- **`tabs.py`** — Three tab widgets, each with a `get_config() -> dict` method that serializes the tab's current settings into a flat config dict consumed by the processors:
  - `FilesTab`: manages the input `.sofa` file list; shows per-file metadata (samplerate, length, point count, max ITD, grid radius) on selection. "View Grid" renders the measurement grid as a 3D scatter plot (window titled with the source filename); Calculate buttons compute ITD and grid radius on demand. Output Folder control (Default: per-source-folder outputs + per-folder averages; Custom: single shared root + one global average). Remove Selected button (enabled on selection) + right-click "Remove" context menu; "Clear All" removes all files.
  - `SofaTab`: emits `sofa_toggled(bool)`; controls resampling, LF extension, DFEQ, radius normalization, HRIR length, and output naming template.
  - `DfhrtfTab`: emits `dfhrtf_toggled(bool)`; controls text outputs (Per SOFA L/R, Per SOFA Avg + algo), graph outputs (DFHRTF graphs, Stereo graphs, Weighted Grid + Per-Folder/All-Files scope), smoothing, tilt, LF extension, and global averaging (Average All Ls/Rs + algo + outlier exclusion).
- **`widgets.py`** — Custom "Flame"-style widget set: `FlameToggleButton`, `FlameNumericSlider` (drag up/down to change value, right-click to type, double-click to reset to default), `FlameDropdown`, `FlameLabel`, `FlameLineEdit`.
- **`workers.py`** — `ProcessingWorker(QThread)`: runs SOFA then DFHRTF processing off the main thread. Exposes `progress`, `log`, `error`, `finished` signals and a `stop()` method checked via a `stop_checker` callable passed into each processor.

### Core Layer (`core/`)

- **`sofa_processor.py`** — `process_sofa_exports()`: builds a task list and shells out to `core/scripts/generate_sofa.py` as subprocesses (one per file × samplerate × EQ variant). Outputs to `<output_root>/sofa_mastered/`.
- **`dfhrtf_processor.py`** — `process_dfhrtf_exports()`: resolves which SOFA files to analyze (mastered or raw), calls `core/scripts/generate_dfhrtf.py` per file, then optionally runs global averaging via `run_averaging()` in-process. Outputs to `<output_root>/DFHRTF/`.
- **`averaging.py`** — Four classes for the global averaging pipeline:
  - `DataIngestor`: loads delimiter-auto-detected CSVs into standardized `frequency`/`magnitude` DataFrames.
  - `CommonAxisInterpolator`: maps all curves onto a shared log-spaced grid and normalizes both axes to [0, 1].
  - `OutlierDetector`: filters by pairwise Euclidean or Soft-DTW distance, keeping curves within `median + N*std`.
  - `CurveAverager`: three methods — `arithmetic_mean`, `geometric_arc_length_blend` (iterative parametric blend), `soft_dtw_barycenter` (L-BFGS-B via `tslearn`).
- **`config.py`** — `ConfigManager` singleton with cascaded JSON persistence: `default_settings.json` (read-only factory defaults) ⊕ per-user `user_settings.json` (located via `QStandardPaths` AppConfigLocation). `load_merged` deep-merges user over defaults; `validate` type-coerces, clamps numerics to `[min, max]`, reverts bad choices, and filters invalid list items. Atomic save via `tempfile.mkstemp` + `os.replace`. `manager().reset()` deletes the user file and reloads defaults. Advanced and averaging tunables are JSON-only (`sofa.advanced.*`, `dfhrtf.advanced.*`, `averaging.*`); the `files` list is never persisted. Key `dfhrtf.advanced` tunables: `weighted_grid_scale_mode` (`"auto"` or `"manual"`), `weighted_grid_scale_low` (manual color-range minimum, default 0.001), `weighted_grid_scale_high` (manual color-range maximum, default 0.003) — the scale settings are used by the main app when generating the weighted-grid PNG, and seeded into the `tests/grid_comparator.py` UI (read-only; no main-app UI); `preserve_lr_balance` (bool, default `true`) — when true, both L and R channels are normalized using a single mean-anchored 1 kHz reference so the real inter-aural level difference (ILD) is preserved; set to `false` to revert to the legacy behavior of zeroing each channel independently.
- **`default_settings.json`** — Factory schema alongside `config.py`: all surfaced and advanced defaults. Acts as the `validate` fallback and Reset target.
- **`geometry_weights.py`** — Shared module (imported by both backend scripts): `calculate_geometric_weights()` returns solid-angle weights for spherical DF integration. Auto mode uses `_is_equal_weight_design()` — a provenance-free SH quadrature test — to detect genuine t/n-design grids and snap to 1/N weights; non-designs always receive ConvexHull geometric weights.
- **`sofa_utils.py`** — Lightweight netCDF4 reads called on the main thread from `FilesTab`: `get_sofa_metadata` (samplerate/length/points), `calculate_max_itd`, `calculate_grid_radius` (dominant measurement sphere radius), `get_grid_cartesian` (Cartesian XYZ grid points for 3D visualization).

### Backend Scripts (`core/scripts/`)

Invoked as subprocesses so heavy imports don't block the GUI:

- **`generate_sofa.py`** — Full SOFA mastering pipeline: resample → pad to 2048 samples → global impulse alignment → optional radius harmonization (physics delay + gain) → geometry-weighted diffuse-field response (convex hull solid-angle weights) → optional LF splice (min-phase synthetic signal below crossover) → optional DF EQ (min-phase corrective filter) → crop/window to target length → normalize → write with `sofar`.
- **`generate_dfhrtf.py`** — DFHRTF extraction: geometry-weighted FFT average → 1/N octave smoothing → normalize at 1 kHz (shared L/R reference when `--preserve-lr-balance`, preserving ILD; per-channel otherwise) → optional LF extension → spectral tilt → save text files and/or matplotlib plots (diffuse-field and ±30° stereo graphs).

### Data

- **`data/lf_ref_mag.txt`** — Two-column (Hz, dB) reference curve used by both scripts for LF extension below the crossover frequency.

### Filename Templates

Output filenames are built from templates containing substitution variables. Variables resolve per file; multi-folder batches resolve `{folder}` independently for each source file. Substitution is applied via `.replace("{var}", value)` chains in `sofa_processor.py` and `dfhrtf_processor.py`.

| Variable | Expands to | Available in |
|---|---|---|
| `{filename}` | Source SOFA filename (no extension) | SOFA naming, DFHRTF text, DFHRTF graphs |
| `{folder}` | Source file's parent folder name | SOFA naming, DFHRTF text, DFHRTF graphs |
| `{samplerate}` | Output samplerate (e.g. `48000`) | SOFA naming, DFHRTF text, DFHRTF graphs |
| `{eq}` | EQ variant (`uneq` or `dfeq`) | SOFA naming only |
| `{tilt}` | Spectral tilt value in dB/oct (e.g. `0.0`, `-0.5`) | DFHRTF text, DFHRTF graphs, global-average text |
| `{LR}` | Channel (`L`, `R`, or `Avg`) | DFHRTF text only |
| `{graphtype}` | Plot type (`df`, `stereo`, or `weighted`) | DFHRTF graphs only |
| `{algo}` | Short algo name (`arithmean`, `geoarc`, `softdtw`) | Global-average text files only |

### Processing Flow

```
UI config dict
    └─> ProcessingWorker (QThread)
            ├─> process_sofa_exports()
            │       └─> generate_sofa.py (subprocess × file × SR × EQ)
            │               └─> <output_root>/sofa_mastered/*.sofa
            └─> process_dfhrtf_exports()
                    ├─> generate_dfhrtf.py (subprocess × SOFA file)
                    │       └─> <output_root>/DFHRTF/*.txt / *.png
                    └─> run_averaging() (in-process, post-step)
                            └─> <output_root>/DFHRTF/dfhrtfs_avg_*.txt
```

The config dict passed to both processors has four top-level keys: `files` (list of paths), `output` (`FilesTab.get_output_config()`), `sofa` (`SofaTab.get_config()`), and `dfhrtf` (`DfhrtfTab.get_config()`). When SOFA processing is enabled, DFHRTF analysis sources from the mastered SOFAs in `sofa_mastered/`; otherwise it reads the raw input files directly. On Start Processing and app close, the full config is snapshotted via `ConfigManager.snapshot()` and saved atomically to `user_settings.json`.
