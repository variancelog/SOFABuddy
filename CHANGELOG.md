## v0.7.0 (2026-11-25)

### Added
- `tools/preflight.py`: pre-push checks for the things that drifted in past releases — app version vs CHANGELOG, a CHANGELOG entry for code changes since the last tag, README image links and unlinked screenshots, forbidden or oversized files in outgoing commits, and the test suite. `--release vX.Y.Z` also checks the tag on GitHub and flags screenshots that may be stale after UI changes; `--fast` skips the tests.
- Test guards in `tests/test_suite_health.py`: the app version matches the top CHANGELOG heading, README image links resolve, and every screenshot in the repo root is linked from the README.

### Changed
- **Weighted Grid** plot redrawn: nodes are coloured by relative weight (weight x node count, 1 = mean) on a fixed 0-2 range; the colorbar marks the grid's min/max (or "all = 1" when uniform); excluded zero-weight nodes are drawn grey; the two-line title shows radius, node count (with "+k excluded"), elevation range, Full/Partial Sphere and a weight summary; axes run exactly +-radius in metres (this replaces the earlier fixed +-1.05 axes for this plot; View Grid is unchanged); 10.5x8 figure with black-edged markers. Drawing lives in the new `core/grid_plot.py`.
- The weighted-grid colour range settings moved to relative units: `dfhrtf.advanced.weighted_grid_scale_low/high` are replaced by `weighted_grid_rel_low/high` (defaults 0.0 and 2.0), and the `--grid-scale-*` CLI defaults now match (manual, 0.0, 2.0). Old saved values are ignored. 'auto' scale uses the 1st-99th percentile of the relative weights.
- DFHRTF export is much faster, with identical output files. The per-SOFA L/R average is now computed only when the average text file is requested (it used to be computed and thrown away, which made every worker load pandas/tslearn/numba). Weighted-grid-only jobs skip the FFT, smoothing and normalisation entirely; jobs that would produce no file (Per Folder grid scope, no other outputs) are no longer sent to a worker. Smoothing is evaluated only at the FFT bins the output grid actually reads, the diffuse-field average runs in bounded blocks (a 41 MB measured file dropped from about 6.4 GB to about 0.3 GB peak memory per worker), and matplotlib, pandas and tslearn are imported only when needed.
- SOFA mastering is faster, with identical output files. The uneq and dfeq variants of the same file and sample rate now share the work up to the LF splice, the grid-weight calculation is vectorised and reused, and a per-position calculation that only depended on the ear is done once.
- The default per-SOFA L/R average algorithm (`dfhrtf.per_sofa_avg_algo`) is now Geo-Arc-Length (was Soft-DTW), which is far cheaper and needs no tslearn. Soft-DTW stays selectable.
- README: View Grid and Weighted Grid screenshots updated to the new plot style.
- The app version now lives only in `core/version.py`; the window title reads it.
- Filename docs: `{samplerate}` is no longer listed for DFHRTF graph filenames, which never substituted it. The README's variable table now also lists where each variable works in the average-file names, and names the three average templates that are only set in `user_settings.json`.

### Removed
- `tools/weighting_lab.py` and `tools/grid_comparator.py`, maintainer scripts used while developing grid weighting, no longer needed.
- `data/lf_ref_mag_00.txt`, an unused backup of the previous LF reference curve.

## v0.6.0 (2026-11-21)

### Added
- **Elevation coverage display** for measurement grids, backed by a new dependency-free (numpy-only) module, `core/grid_info.py` (`GridCoverage`, `grid_coverage()`, `elevation_range()`, `format_elevation_range()`, `elevation_ring_xyz()`, `draw_elevation_rings()`, `apply_unit_sphere_axes()`). `core/sofa_utils.py` re-exports these; the backend scripts import them straight from `core/grid_info.py` so they don't load netCDF4.
  - **Files tab**: new "Elevation range" metadata row, computed instantly from the raw `SourcePosition`, with a "(full sphere)" tag when both poles are covered.
  - **View Grid** and **Weighted Grid** plots: the title gets a second line giving the elevation range. Dashed boundary rings are drawn only at open ends of partial grids. Axes are fixed at ±1.05 so different grids can be compared side by side. The Weighted Grid PNG computes its range from the cleaned (surviving) nodes only.
  - A pole counts as covered when its empty polar cap is no wider than the grid's median nearest-neighbour angular spacing. This means quadrature grids with no node on the pole, such as a t-design whose nodes only reach ±83.4°, still read as full sphere.
  - New tests for the helpers, the Files tab metadata, the axis limits and the weighted-grid PNG.

### Removed
- The dead `--advanced-dfeq` CLI option in `generate_dfhrtf.py`. It was always on, and `dfhrtf_processor.py` always passed `True` for it. Behaviour is unchanged.

### Changed
- README: the File list, View Grid and Weighted Grid sections now describe the elevation coverage display.
- `reference/` (a local maintainer folder) is now gitignored.
- IDEA.md: marked the finished items as done (parallel processing, Mesh2SOFA cleanup, file multiselect, the Output Folder move, elevation coverage, the "Post Processing" row move, and the DF source-of-truth question). Also noted that grid resampling is shelved on the local `wip/grid-resampling` branch, since it showed no DFHRTF benefit over geometric weighting.

## v0.5.0 (2026-08-22)

### Added
- **`core/naming.py`**: single implementation of output-filename template substitution (`render_filename()`), replacing five separate ad hoc `.replace()` chains previously duplicated across `sofa_processor.py` and `dfhrtf_processor.py`.
- **`core/config.py`**: `schema_default(section, key)` reads a single factory default out of `default_settings.json`. `ui/tabs.py` now sources every widget construction default and `set_config()` fallback from it instead of hardcoded literals, so the JSON is the sole source of truth for what the UI shows.
- **`core/geometry_weights.py`**: extracted the real-valued spherical-harmonic construction (with Condon–Shortley sign convention) into `real_sph_harm()`, now the single source of truth shared by the design classifier and its test suite.
- Test suite substantially expanded and reorganized (126 → 353 collected tests): `pytest.ini` (markers `qt`, `needs_sofa_fixtures`, `slow`; `xfail_strict = true`) and `tests/conftest.py` (shared argparse-derived fixtures, synthetic SOFA factories, automatic Qt marking) centralize what was previously duplicated per-test setup. `requirements-dev.txt` split out test-only deps (`pytest`, `pytest-qt`).
- Former ad hoc diagnostic scripts under `tests/` are now proper maintainer tools under `tools/`: `tools/audit_diffuse_ild.py`, `tools/grid_comparator.py`, `tools/weighting_lab.py` (renamed from `_grid_comparator.py` / `compare_weighting_grids.py`), plus new `tools/extract_grid_fixtures.py`.

### Fixed
- **Three-way default-value desync** between the GUI, CLI, and processor fallback paths, all now guarded by `tests/test_defaults_parity.py`: `ui/tabs.py` widget construction literals, `core/sofa_processor.py`'s `dict.get(..., <fallback>)` literals, and `generate_dfhrtf.py`'s `--var-smooth-*` argparse defaults had each drifted independently from `default_settings.json`. Corrected values (JSON is authoritative): SOFA `hrir_length` default is now **256** samples (was 512 in the UI/processor fallback) — spectrally equivalent given the 1 kHz LF crossover, at half the file size; `distance_m` 1.5 m (was 1.2); `ild_threshold` 1.2 dB over a 350–3500 Hz band (was 1.5 dB over 300–6000 Hz); LF-splice `blend_low_oct`/`blend_high_oct` 0.33/1.0 (was 0.50/0.75); variable-smoothing `freq_hi`/`n_low` 6000 Hz/2.0 in the CLI path (was 3000 Hz/1.0, matching what the JSON/GUI path already used — see `core/smoothing.py`'s `SmoothingParams`).
- `process_dfhrtf_exports()` now checks the DFHRTF backend script exists before running, matching the guard already present on the SOFA processor — previously a missing/misnamed `generate_dfhrtf.py` failed asymmetrically compared to a missing `generate_sofa.py`.
- `core/averaging.py`'s `DataIngestor` assumed `header=0` unconditionally when reading averaging-input text files; for the headerless space/tab/semicolon-delimited files `generate_dfhrtf.py`'s text output can produce, this ate the lowest-frequency row as a column name. Now detects header presence by checking whether the first row's first field parses as a number.

## v0.4.1 (2026-07-09)

### Fixed
- **CPU oversubscription during parallel batch runs**: with the new bounded chunk pool (half of CPU cores) the app was still pegging every core at 100%. Root cause: `core/averaging.py`'s Soft-DTW barycenter (`_softdtw_func`) spun up an *unbounded* `ThreadPoolExecutor` (one thread per input curve) on top of the chunk-pool parallelism — nested inside every chunk subprocess's per-SOFA L/R average, and again unguarded (plus BLAS fan-out) in the parent process's global averaging. `soft_dtw_barycenter()`/`_softdtw_func()` now take an `n_workers` parameter (default `1`): the per-SOFA average inside a chunk subprocess passes `n_workers=1` (the chunk pool already provides concurrency), and the parent's global averaging (`dfhrtf_processor.run_averaging`) sizes it to the resolved chunk-pool worker count. The app entry point (`main.pyw`) now pins `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` (via `setdefault`, respecting user values) before any numpy/scipy import so the in-process averaging can't oversubscribe through BLAS. Batch worker subprocesses are also launched at below-normal priority (`BELOW_NORMAL_PRIORITY_CLASS` on Windows, `os.nice(10)` on POSIX) so a full-core run keeps the GUI responsive.

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