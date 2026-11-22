## Features

### General
- **DONE** - Add parallel processing for large file sets
  - Chunked subprocesses on a bounded thread pool (`processing.max_workers` / `processing.chunk_size`), per-file progress.
  - Covers both SOFA mastering and DFHRTF stages (also 2/3 step HRTF inclusive workflows, once added).
- **DONE** - Review and remove vestigial references to Mesh2SOFA-related code not used in the current project. (No Mesh2SOFA references left in the code as of 2026-11-22.)
- Move on/off toggle to tab selector and remove from top of tab. I keep wanting to click on the circle to enable even though I designed it, so I can only imagine what others would do.

- **DONE** - Improve DFHRTF auto-weighting-mode: existing function is brittle for some high-order n-design grids
- **SHELVED** (2026-11-22) - Grid resampling stage (resample every SOFA onto a shared Fibonacci grid, Triangular / SUpDEq). Round-trip tests showed no DFHRTF benefit over geometric weighting of the original grid. Parked on local branch `wip/grid-resampling`; not for main.

### Console
- **DONE** Set notificaton color (e.g. Log file created) to blue or green instead of red

### Add HRTF tab
- Tab for HRTF specific outputs
- Graphs:
  - Magnitude responses for arbitrary elevation/azimuth angles
  - Graphs like ![Magnitude overlay](http://raw.githubusercontent.com/Any2HRTF/Mesh2HRTF/master/docs/figures/graphical_abstract-01.png)
  - Graphs like ![HRTF heatmap](http://mesh2hrtf.org/images/HRTFs.png)
- Text:
  - HRTF magnitude response for arbitrary angles
- Impulses:
  - Arbitrary HRIR impulses
  - "True Stereo" impulses for convolution plugins

## Files tab
- Add file sort options
  - Partial: list is auto-sorted by filename; no other sort modes yet.
- **DONE** - Add file multiselect option
  - Multi-select shows "-" for per-file metadata; "View Grid" opens all selected files and pages through them with ←/→; Remove Selected + right-click Remove.
- Add "Extract grid" function to generate mesh2HRTF-compatible evaluation grids from SOFA files. Usefull to run simulation with same grid as measurements for accurate comparisons.
  - Note: `tools/extract_grid_fixtures.py` only extracts test fixtures (.npy); not the user-facing feature.
- **DONE** - **Important:** Move output folder section from Files tab to Main window so that it persists between tabs.
- **DONE** (2026-11-22) - Show elevation coverage: "Elevation range" metadata row, plus elevation title line, "(full sphere)" tag, boundary rings at open ends and fixed ±1.05 axes on View Grid and Weighted Grid plots.

## DFHRTF tab
- **DONE** Move "Post Processsing" row to below Text outputs filename row
  - Plus add post-proc filenaming row. (Average filenames exist but are JSON-only for now: `dfhrtf.avg_filename`, `dfhrtf.per_sofa_avg_filename`.)
- Should DFHRTF be just DFHRTF Text, while Graphs can go on a separate page.

## Bugs
- **DONE** - DHFRTF text and graph outputs align l/r magnitude at exactly 1kHz, which loses the real l/r magnitude relation.
- **DONE** - Why is dfhrtf_processor.py not referenced by generate_sofa.py? The dfhrtf_processor is supposed to the "source of truth" for dfhrtf determination but they appear to be separate still.
  - Resolved (since 2026-06): `dfhrtf_processor.py` only orchestrates DFHRTF jobs and was never meant to hold DF maths. The shared source of truth is `core/geometry_weights.py` (`calculate_geometric_weights`): SOFA mastering and DFHRTF analysis use the same weighting mode (DFHRTF tab setting) and the same radius/duplicate cleanup, then the identical weighted power average. The dead `--advanced-dfeq` CLI option (always on) was removed 2026-11-22.