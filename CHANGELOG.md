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