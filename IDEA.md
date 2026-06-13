## Features

### General
- Add parallel processing for large file sets
 - Distribute one file per thread a la numcalc manager?
 - Must factor two-step SOFA > DFHRTF (also 2/3 stept HRTF included, once added) workflows.

### HRTF tab
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
  - Must decide interaction with "Calculate" buttons and "View grid" button.
- Add "Extract grid" to generate mesh2HRTF-compatible evaluation grid. Usefull to run simulation with same grid as measurements for accurate comparisons.

## Bugs
- DHFRTF text and graph outputs align l/r magnitude at exactly 1kHz, which loses the real l/r magnitude relation. 
  - Explore options for median mag alignment (e.g. 1kHz +- 1 octave), and/or aligning left channel and then adjusting right by same amount.