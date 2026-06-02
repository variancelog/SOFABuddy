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