## v0.4.0 (2026-07-08)

### Added
- **Parallel + chunked batch processing** for both SOFA mastering and DFHRTF analysis: each stage now builds its full job list up front, groups it into contiguous chunks, and runs chunks concurrently through a bounded thread pool (`core/subprocess_utils.py`: `run_chunks_parallel()`), instead of one subprocess per file run strictly sequentially. Each backend script gained a `--manifest` mode that processes many jobs in one process (amortizing the numpy/scipy/sofar import cost) and prints a `[JOB OK]`/`[JOB FAIL]` sentinel per file so progress and failure reporting stay attributed per file rather than per subprocess. New JSON-only settings `processing.max_workers` (default `0` = auto, half of CPU cores) and `processing.chunk_size` (default `16`).
- **Per-file console/progress feedback** during batch runs: `run_task()` now streams every subprocess line through an optional `line_callback`, and `run_chunks_parallel()` uses it to parse `[JOB OK]`/`[JOB FAIL]` sentinels as they arrive, advancing the progress bar per file across all concurrent chunks instead of only when an entire chunk finishes.

### Changed
- Raised the SOFA/DFHRTF LF-splice crossover default from 400 Hz to 1000 Hz (schema bounds updated to match): matching Brinkmann's HUTUBS reference showed the old crossover forced the phase-blend seam into a band where measured-file phase is unreliable, producing ripple on measured (ARI/BILI/LISTEN) sources at 256-sample output. All four validation datasets now pass cleanly at 256 samples with no algorithm change.
- If SOFA mastering fails for every job in a run, `process_sofa_exports()` now returns failure instead of reporting success — the app stops before running DFHRTF analysis on stale or raw input, rather than logging a misleading "OK" over data that was never mastered.
- If SOFA mastering is enabled but fails for only *some* files, DFHRTF analysis now skips those specific files (with a `[!] Skipping <name>: no mastered SOFA found` log line) instead of silently falling back to analyzing their raw, un-mastered input.
- `tests/` is now tracked in git (previously gitignored despite containing the full test suite); superseded ad-hoc diagnostic scripts were removed as part of the same cleanup.

### Fixed
- `core/sofa_io.py`'s `sanitize_sofa()` now normalizes non-conformant `GLOBAL_RoomType` values (e.g. `"free-field"` → `"free field"`) on load. Some source databases (e.g. RWTH Aachen) declare RoomType with a hyphen/underscore/mixed case, which sofar accepts on read but rejects on write ("must be free field, reverberant, shoebox, dae") — every SOFA mastering job for such a file previously failed.
- Fixed 6 stale test failures and 1 collection error surfaced by the crossover-default change and the tests/ tracking cleanup (hardcoded crossover literals, an ILD test fixture that needed curating, and grid-classifier fixtures referencing directories that were never committed).

## v0.3.0 (2026-07-04)

### Added
- **Variable smoothing** (new default for DFHRTF): frequency-dependent Gaussian smoothing in log-frequency space (`core/smoothing.py`: `SmoothingParams`, `variable_smooth()`), transitioning from 1-octave bandwidth at/below 350 Hz to 1/12-octave at/above 3 kHz. Tunable via `dfhrtf.advanced.variable_smoothing_*` keys (`freq_low`, `freq_hi`, `n_low`, `n_hi`, `gamma`, `mean_mode`). The DFHRTF tab's smoothing control is now a dropdown (`Variable`, `1/3`…`1/48`) replacing the old fixed 1/n slider.
- **ILD correction** on the SOFA tab: a "Balance ILDs" toggle + dB threshold slider estimates diffuse-field inter-aural level difference and applies a symmetric split-level gain correction when it exceeds the threshold. Wired through `core/ild_audit.py` (`diffuse_field_ild(..., estimate_band=...)`) and new `generate_sofa.py` flags (`--ild-fix`, `--ild-threshold`, `--ild-low-freq`, `--ild-high-freq`).
- **Robust SOFA loading** (`core/sofa_io.py`, `read_sofa_robust()`): normalizes legacy `NC_GLOBAL.`-prefixed attributes from old SOFA C++ API 1.x files (e.g. IRCAM BILI), falls back to `verify=False` when a file's non-standard custom entries fail sofar's convention check, and sanitizes known convention violations in-memory (e.g. collapsing multi-emitter positions on `SimpleFreeFieldHRIR` files). Used by both `generate_sofa.py` and `ild_audit.py`.
- New minimum-phase LF splicing algorithm in `generate_sofa.py`: blends minimum-phase and excess-phase components separately, replacing the raw excess phase below crossover with a pure linear delay extrapolated from the crossover's time-of-arrival — fixes comb-filtering and time-dispersion artifacts from the previous complex-domain splice.
- Files tab multi-select: extended selection across the file list, batch "Calculate" for ITD/Grid Radius across all selected files, and a multi-file "View Grid" viewer with ←/→ cycling between grids in one window. The list is now sorted by filename (full path retained for processing/tooltips).
- App icon and splash screen image (`ui/sb_icon.png`, `ui/sb_splash.png`), replacing the placeholder colored-pixmap splash.
- `core/plotting.py`: shared `plot_diffuse_field()` helper so averaged and per-SOFA diffuse-field graphs render with identical styling.
- Global averaging can now interpolate onto the same fractional-octave axis used by the per-SOFA text files (`CommonAxisInterpolator.interpolate_to_common_axis(..., octave_fraction=...)`) instead of always falling back to a fixed-`num_points` log grid.

### Changed
- **Output Folder control moved** from the Files tab to a new row on the main window (above the action bar) — it's an app-level setting, not scoped to a single tab.
- `OutlierDetector.filter_outliers()` now always uses Euclidean pairwise distance for outlier detection (dropped the Soft-DTW-gamma distance-matrix path); Soft-DTW remains available as an averaging *algorithm* downstream.
- New defaults: SOFA `lf_extension` is now `true` (was `false`); `xover` for both SOFA and DFHRTF is `400 Hz` (was `375 Hz`); global averaging `algo` is `Geo-Arc-Length` (was `Soft-DTW`); `averaging.geo_arc_y_scale` is `0.02` (was `0.01`).
- Simplified default filename templates: SOFA naming drops the `-n` suffix (`{filename} {samplerate} {eq}`), DFHRTF graph filename drops `{folder}` (`{graphtype} {filename}`).
- Removed dead code from `generate_sofa.py`: the unused `mesh2hrtf` project import/export helpers and the standalone `merge_sofas` L/R-merge path, along with their now-unused `--left`/`--right`/`--m2h_path`/`--only-48k` CLI flags.
- SOFA/DFHRTF crossover sliders now read their min/max from the JSON schema (`schema_bounds()` in `core/config.py`) instead of hardcoded UI values.
- Updated README screenshots and docs for the new Smoothing dropdown, ILD controls, default templates, and output folder layout.

### Fixed
- `dfhrtf.avg_graph_filename` was missing from `SCHEMA` in `core/config.py`, so it wasn't type-validated/coerced like other filename templates.

## v0.2.1 (2026-06-14)

### Added
- **ILD preservation** (`dfhrtf.advanced.preserve_lr_balance`, default `true`): L and R DFHRTF channels (both diffuse-field and ±30° stereo outputs) are now normalized using a single shared mean 1 kHz reference, preserving the real inter-aural level difference. Set to `false` to restore the legacy per-channel zeroing behavior.
- Console color-coding: diagnostics log path lines appear in blue, `[+]` success lines in teal, "Processing completed." in teal. Added `OK_TEXT` and `SUCCESS_TEXT` constants to `ui/theme.py`.

### Changed
- **Improved auto weighting-mode**: replaced the CV-threshold heuristic with a provenance-free spherical-harmonic quadrature test (`_is_equal_weight_design` in `geometry_weights.py`). Genuine t/n-design grids (SH residual ~1e-11) now receive 1/N weights regardless of grid size or order; non-designs (residual ~1e0) always receive geometric weights. Eliminates false positives on high-order n-designs where the old CV was erratic (8–16% with no clean threshold).
- Removed `balance_weight_cv_threshold` advanced setting and `--balance-weight-cv` CLI flag (superseded by the SH quadrature test).
- Added scipy ≥ 1.15 compatibility shim for the renamed `sph_harm_y` API in `geometry_weights.py`.
- Updated weighting-mode tooltip in the DFHRTF tab to describe the new SH quadrature test.

### Fixed
- Diagnostics log path messages were emitted via `error_callback` (red) in both SOFA and DFHRTF processors; now correctly use `log_callback`.

## v0.2.0 (2026-06-13)

### Added
- Unified subprocess management (`core/subprocess_utils.py`) with real-time console logging and persistent diagnostics logs.
- Advanced configuration schema (`core/default_settings.schema.json`) with automated generation script for developer ergonomics.
- Support for `{tilt}` variable in filename templates across SOFA and DFHRTF exports.
- Automated diagnostics logging: execution logs are now saved to a `logs/` directory relative to the output root.
- Soft-DTW gamma controls are now split into per-SOFA (`softdtw_gamma_persofa`) and global (`softdtw_gamma_global`) settings for finer tuning.
- `IDEA.md` for tracking roadmap and future feature brainstorms.

### Changed
- Refined default naming templates to be more concise (e.g., `s_` for SOFA, `t_` for text, `g_` for graphs).
- Improved SOFA mastering to automatically ensure attribute units (SamplingRate, etc.) are lowercase, fixing compatibility with strict SOFA readers.
- Migrated all internal task execution to use the new `run_task` utility for consistent error reporting.
- Enhanced help text and tooltips in the DFHRTF tab to better explain template variables.
- Improved weighting algorithm for diffuse field calculation: process now correctly weights sliced-off grids (e.g. measurement grids missing lower elevations)

### Fixed
- Fixed SOFA unit casing issues that prevented saving mastered files from certain source datasets.
- Improved exception handling and traceback reporting in CLI scripts to ensure errors are captured in logs even when piped.

## 2026-06-11

### Added
- Configuration persistence (`core/config.py`, `core/default_settings.json`): cascaded defaults ⊕ per-user `user_settings.json` with schema validation (type coercion, range clamping, choice/list sanitizing), atomic save on app close and on Start Processing, and a "Reset Settings" button. The imported files list is never persisted; advanced tunables (`sofa.advanced.*`, `dfhrtf.advanced.*`, `averaging.*`) are JSON-only.
- App-level Output Folder control on the Files tab: Default (per-source-folder outputs + per-folder averages) vs Custom (single shared root + one global average).
- Grid Radius readout with on-demand Calculate, and a "View Grid" button that renders the measurement grid as a 3D scatter plot (window titled with the source filename).
- DFHRTF weighted-grid graph output with Per-Folder / All-Files scope, and Per-SOFA Avg DFHRTF text output with a selectable averaging algorithm.
- `{folder}` filename-template variable (source file's parent folder name), usable in SOFA naming and DFHRTF text/graph filename templates; resolves per file so multi-folder batches name correctly.
- "Remove Selected" action on the Files tab: a 2×2 button grid (`[Add Files][Add Folder]` / `[Remove Selected][Clear All]`) plus a right-click "Remove" context menu.
- `tests/test_config.py` covering config load, merge, validate, and persistence.

### Changed
- Renamed the Files-tab "Clear List" button to "Clear All".
- `core/sofa_utils.py` gained `calculate_grid_radius()` and `get_grid_cartesian()` for the Grid Radius readout and grid visualization.

## 2026-06-04

### Added
- Added `ui/theme.py` as a single source of truth for all UI colors and sizing constants; refactored inline stylesheets across `main_window.py`, `tabs.py`, and `widgets.py` to reference it.
- Bundled Roboto and Roboto Mono font faces in `ui/fonts/`, auto-registered at startup via `QFontDatabase` (`_load_bundled_fonts` in `main.pyw`), with graceful fallback to system fonts when absent.

### Changed
- Migrated the UI framework from PyQt6 to PySide6 (`pyqtSignal` → `Signal`); updated `requirements.txt` accordingly and added the `sofar` dependency.
- Standardized all interactive control heights to 40px via `theme.CONTROL_HEIGHT` for consistent vertical alignment.
- Added a 120px minimum width to the first grid column of the SOFA and DFHRTF tabs to align setting labels.
- Made `FlameLabel` backgrounds transparent and tidied control padding for consistent spacing.

## 2026-06-01

### Added
- Added a draggable vertical `QSplitter` to adjust the size of the bottom console pane.
- Added `{algo}` variable support for text output filename templates, automatically inserting shortened algorithm names (`arithmean`, `geoarc`, `softdtw`) for global averages.

### Changed
- Improved navigation bar file count to update dynamically for all list actions (add, clear, drag-and-drop).
- Unified dimensions of navigation bar buttons for consistent width and height.
- Standardized custom UI widget heights (`FlameToggleButton`, `FlameDropdown`, `FlameNumericSlider`, `FlameLineEdit`) to `32px` to fix horizontal alignment issues.
- Set the console's initial minimum height to 200px and ensured window resizing only expands the console.
- Adjusted default gamma/smoothing weights for Geometric Arc-Length and Soft-DTW averaging algorithms from `0.05` to `0.01`.

## 2026-05-31

Initial version