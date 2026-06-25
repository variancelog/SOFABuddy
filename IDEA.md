## Features

### General
- Add parallel processing for large file sets
 - Distribute one file per thread a la numcalc manager?
 - Account for multitab projects (SOFA > DFHRTF)  (also 2/3 step HRTF inclusive workflows, once added).
- Review and remove vestigial references to Mesh2SOFA-related code not used in the current project.
- Move on/off toggle to tab selector and remove from top of tab. I keep wanting to click on the circle to enable even though I designed it, so I can only imagine what others would do.

- **DONE** - Improve DFHRTF auto-weighting-mode: existing function is brittle for some high-order n-design grids

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
- Add file multiselect option
  - Must decide interaction with "Calculate" (ITD) button and "View grid" button. Ideally view grid on multiple can use matplotlib standard multi-file support.
- Add "Extract grid" function to generate mesh2HRTF-compatible evaluation grids from SOFA files. Usefull to run simulation with same grid as measurements for accurate comparisons.
- **Important:** Move output folder section from Files tab to Main window so that it persists between tabs.

## DFHRTF tab
- Move "Post Processsing" row to below Text outputs filename row
  - Plus add post-proc filenaming row.
- Should DFHRTF be just DFHRTF Text, while Graphs can go on a separate page.

## Bugs
- **DONE** - DHFRTF text and graph outputs align l/r magnitude at exactly 1kHz, which loses the real l/r magnitude relation.
- Why is dfhrtf_processor.py not referenced by generate_sofa.py? The dfhrtf_processor is supposed to the "source of truth" for dfhrtf determination but they appear to be separate still.