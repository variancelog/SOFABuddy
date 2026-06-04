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

Tests use `pytest-qt` (not in `requirements.txt` but required for UI tests). 

## Architecture

SOFA Buddy is a PySide6 GUI application for processing SOFA (Spatially Oriented Format for Acoustics) HRTF files. Processing runs in two sequential stages: SOFA mastering → DFHRTF analysis.

### UI Layer (`ui/`)

- **`main_window.py`** — `SofaBuddyMainWindow`: top-level window with a 3-tab nav bar (Files / SOFA / DFHRTF), a resizable console pane at the bottom, and a single "Start Processing" button that drives `ProcessingWorker`.
- **`tabs.py`** — Three tab widgets, each with a `get_config() -> dict` method that serializes the tab's current settings into a flat config dict consumed by the processors:
  - `FilesTab`: manages the input `.sofa` file list; shows per-file metadata (samplerate, length, point count, max ITD) on selection.
  - `SofaTab`: emits `sofa_toggled(bool)`; controls resampling, LF extension, DFEQ, radius normalization, HRIR length, and output naming template.
  - `DfhrtfTab`: emits `dfhrtf_toggled(bool)`; controls text/graph outputs, smoothing, tilt, LF extension, and global averaging.
- **`widgets.py`** — Custom "Flame"-style widget set: `FlameToggleButton`, `FlameNumericSlider` (drag up/down to change value, right-click to type, double-click to reset to default), `FlameDropdown`, `FlameLabel`, `FlameLineEdit`.
- **`workers.py`** — `ProcessingWorker(QThread)`: runs SOFA then DFHRTF processing off the main thread. Exposes `progress`, `log`, `error`, `finished` signals and a `stop()` method checked via a `stop_checker` callable passed into each processor.

### Core Layer (`core/`)

- **`sofa_processor.py`** — `process_sofa_exports()`: builds a task list and shells out to `core/scripts/generate_sofa.py` as subprocesses (one per file × samplerate × EQ variant). Outputs to `<input_dir>/sofa_mastered/`.
- **`dfhrtf_processor.py`** — `process_dfhrtf_exports()`: resolves which SOFA files to analyze (mastered or raw), calls `core/scripts/generate_dfhrtf.py` per file, then optionally runs global averaging via `run_averaging()` in-process. Outputs to `<input_dir>/DFHRTF/`.
- **`averaging.py`** — Four classes for the global averaging pipeline:
  - `DataIngestor`: loads delimiter-auto-detected CSVs into standardized `frequency`/`magnitude` DataFrames.
  - `CommonAxisInterpolator`: maps all curves onto a shared log-spaced grid and normalizes both axes to [0, 1].
  - `OutlierDetector`: filters by pairwise Euclidean or Soft-DTW distance, keeping curves within `median + N*std`.
  - `CurveAverager`: three methods — `arithmetic_mean`, `geometric_arc_length_blend` (iterative parametric blend), `soft_dtw_barycenter` (L-BFGS-B via `tslearn`).
- **`sofa_utils.py`** — Lightweight netCDF4 reads for per-file metadata and max ITD; called on the main thread from `FilesTab`.

### Backend Scripts (`core/scripts/`)

Invoked as subprocesses so heavy imports don't block the GUI:

- **`generate_sofa.py`** — Full SOFA mastering pipeline: resample → pad to 2048 samples → global impulse alignment → optional radius harmonization (physics delay + gain) → geometry-weighted diffuse-field response (convex hull solid-angle weights) → optional LF splice (min-phase synthetic signal below crossover) → optional DF EQ (min-phase corrective filter) → crop/window to target length → normalize → write with `sofar`.
- **`generate_dfhrtf.py`** — DFHRTF extraction: geometry-weighted FFT average → 1/N octave smoothing → normalize at 1 kHz → optional LF extension → spectral tilt → save text files and/or matplotlib plots (diffuse-field and ±30° stereo graphs).

### Data

- **`data/lf_ref_mag.txt`** — Two-column (Hz, dB) reference curve used by both scripts for LF extension below the crossover frequency.

### Processing Flow

```
UI config dict
    └─> ProcessingWorker (QThread)
            ├─> process_sofa_exports()
            │       └─> generate_sofa.py (subprocess × file × SR × EQ)
            │               └─> <input_dir>/sofa_mastered/*.sofa
            └─> process_dfhrtf_exports()
                    ├─> generate_dfhrtf.py (subprocess × SOFA file)
                    │       └─> <input_dir>/DFHRTF/*.txt / *.png
                    └─> run_averaging() (in-process, post-step)
                            └─> <input_dir>/DFHRTF/dfhrtfs_avg_*.txt
```

The config dict passed to both processors has three keys: `files` (list of paths), `sofa` (`SofaTab.get_config()`), and `dfhrtf` (`DfhrtfTab.get_config()`). When SOFA processing is enabled, DFHRTF analysis sources from the mastered SOFAs in `sofa_mastered/`; otherwise it reads the raw input files directly.
