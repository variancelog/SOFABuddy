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