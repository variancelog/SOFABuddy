# Graph Report - .  (2026-06-11)

## Corpus Check
- 25 files · ~50,962 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 327 nodes · 538 edges · 16 communities (14 shown, 2 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 64 edges (avg confidence: 0.89)
- Token cost: 12,500 input · 3,200 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Config & Settings Management|Config & Settings Management]]
- [[_COMMUNITY_DFHRTF Generation Script|DFHRTF Generation Script]]
- [[_COMMUNITY_UI Tab Layer|UI Tab Layer]]
- [[_COMMUNITY_Processing Orchestration|Processing Orchestration]]
- [[_COMMUNITY_DFHRTF Tab Controls|DFHRTF Tab Controls]]
- [[_COMMUNITY_SOFA File Utilities|SOFA File Utilities]]
- [[_COMMUNITY_Curve Averaging Pipeline|Curve Averaging Pipeline]]
- [[_COMMUNITY_Flame Widget Internals|Flame Widget Internals]]
- [[_COMMUNITY_Files Tab UI|Files Tab UI]]
- [[_COMMUNITY_SOFA Tab Controls|SOFA Tab Controls]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]
- [[_COMMUNITY_Grid Visualization|Grid Visualization]]
- [[_COMMUNITY_Spatial Grid Measurements|Spatial Grid Measurements]]
- [[_COMMUNITY_UI Package Init|UI Package Init]]
- [[_COMMUNITY_Main Window Module|Main Window Module]]

## God Nodes (most connected - your core abstractions)
1. `FilesTab` - 31 edges
2. `SofaBuddyMainWindow` - 24 edges
3. `FlameNumericSlider` - 22 edges
4. `SofaTab` - 18 edges
5. `DfhrtfTab` - 18 edges
6. `ConfigManager` - 16 edges
7. `FlameToggleButton` - 15 edges
8. `main()` - 14 edges
9. `FlameLabel` - 13 edges
10. `FlameDropdown` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Spherical HRTF Measurement Grid (Raw)` --semantically_similar_to--> `Spherical HRTF Measurement Grid (Weighted)`  [INFERRED] [semantically similar]
  plot_grid_raw.png → plot_grid_weighted.png
- `get_grid_cartesian()` --semantically_similar_to--> `calculate_geometric_weights()`  [INFERRED] [semantically similar]
  core/sofa_utils.py → core/scripts/generate_dfhrtf.py
- `Raw Measurement Grid Plot` --shares_data_with--> `Weighted Grid Plot`  [INFERRED]
  plot_grid_raw.png → plot_grid_weighted.png
- `1680 Measurement Nodes` --conceptually_related_to--> `1625 Weighted Nodes`  [INFERRED]
  plot_grid_raw.png → plot_grid_weighted.png
- `calculate_geometric_weights()` --semantically_similar_to--> `calculate_geometric_weights()`  [INFERRED] [semantically similar]
  core/scripts/generate_sofa.py → core/scripts/generate_dfhrtf.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **End-to-End Processing Flow: UI Config → Worker → Processors → Scripts** — config_dict_pattern, processing_pipeline, subprocess_isolation, global_averaging_pipeline [EXTRACTED 0.95]
- **LF Extension Reference Data Used by Both Backend Scripts** — data_lf_ref_mag, lf_extension_concept, processing_pipeline [EXTRACTED 0.95]
- **Global Averaging Pipeline (ingest → interpolate → outlier filter → average)** — core_averaging_dataingestor, core_averaging_commonaxisinterpolator, core_averaging_outlierdetector, core_averaging_curveaverager [EXTRACTED 1.00]
- **SOFA Mastering Pipeline (resample → align → optional LF splice → DFEQ → crop → normalize → write)** — scripts_generate_sofa_master_sofa, scripts_generate_sofa_find_global_onset, scripts_generate_sofa_calculate_geometric_weights, scripts_generate_sofa_load_lf_target_base, scripts_generate_sofa_get_blend_weights [EXTRACTED 1.00]
- **DFHRTF Extraction Pipeline (weighted FFT avg → smooth → normalize → optional LF ext → tilt → save)** — scripts_generate_dfhrtf_calculate_geometric_weights, scripts_generate_dfhrtf_apply_smoothing, scripts_generate_dfhrtf_apply_lf_extension, scripts_generate_dfhrtf_apply_spectral_tilt, scripts_generate_dfhrtf_save_text_data [EXTRACTED 1.00]
- **Tab Config Dict Assembly for ProcessingWorker** — ui_tabs_filestab, ui_tabs_sofatab, ui_tabs_dfhrtftab, ui_main_window_sofabuddymainwindow, ui_workers_processingworker [EXTRACTED 1.00]
- **Flame Widget + Theme Rendering System** — ui_widgets_flametogglebutton, ui_widgets_flamenumericslider, ui_widgets_flamedropdown, ui_theme_module [INFERRED 0.95]
- **Settings Save/Restore Across Tabs** — ui_main_window_sofabuddymainwindow, ui_tabs_filestab, ui_tabs_sofatab, ui_tabs_dfhrtftab [EXTRACTED 1.00]

## Communities (16 total, 2 thin omitted)

### Community 0 - "Config & Settings Management"
Cohesion: 0.08
Nodes (29): Cascaded JSON Config (defaults + user overlay), ConfigManager, _deep_merge(), _get_nested(), load_defaults(), load_merged(), manager(), manager (singleton factory) (+21 more)

### Community 1 - "DFHRTF Generation Script"
Cohesion: 0.09
Nodes (35): Diffuse Field HRTF (concept), Geometry-Weighted Solid Angle Integration (concept), LF Extension via Crossfade Splicing (concept), apply_lf_extension(), apply_smoothing(), apply_spectral_tilt(), calculate_geometric_weights(), find_closest_grid_point() (+27 more)

### Community 2 - "UI Tab Layer"
Cohesion: 0.13
Nodes (15): Flame-Style Widget System, Settings Persistence (save/restore), QComboBox, QWidget, DfhrtfTab, Restore all SOFA tab widgets from a saved config dict (surfaced keys only)., Restore all DFHRTF tab widgets from a saved config dict (surfaced keys only)., SofaTab (+7 more)

### Community 3 - "Processing Orchestration"
Cohesion: 0.09
Nodes (14): Config Dict (Processing Configuration), Tab get_config() Protocol, process_dfhrtf_exports(), Main processing logic for DFHRTF exports.     Analyzes SOFA files and generates, process_sofa_exports(), Main processing logic for SOFA exports.     Breaks down the UI config into indi, QMainWindow, QPushButton (+6 more)

### Community 4 - "DFHRTF Tab Controls"
Cohesion: 0.08
Nodes (31): Average All Ls/Rs Toggle, Averaging Method Dropdown (Soft-DTW), Processing Console Log Output, Crossover Hz Control (375), Text Output Delimiter (Tab), Exclude Outliers Toggle, Extend LF Toggle, Text Output Filename Template (gr_{folder}_{filename}_{LR}) (+23 more)

### Community 5 - "SOFA File Utilities"
Cohesion: 0.10
Nodes (10): calculate_grid_radius(), calculate_max_itd(), get_grid_cartesian(), get_sofa_metadata(), Returns the dominant (most common) grid radius in metres from SourcePosition,, Extracts metadata from a SOFA file without loading the entire impulse response a, Reads SourcePosition from a SOFA file and returns the full measurement     grid, Calculates the maximum ITD (in ms) by finding peak impulse indices across all me (+2 more)

### Community 6 - "Curve Averaging Pipeline"
Cohesion: 0.11
Nodes (18): Soft-DTW Barycenter Averaging (concept), CommonAxisInterpolator, CurveAverager, DataIngestor, OutlierDetector, Restores the averaged 0-1 magnitude curve back to original dB scale., Calculates pairwise distances between all sequences, computes each sequence's, Standard point-by-point mean on normalized data. (+10 more)

### Community 7 - "Flame Widget Internals"
Cohesion: 0.14
Nodes (6): QLabel, QLineEdit, QMouseEvent, FlameNumericSlider, FlameSliderLabel, Programmatically set the slider value (clamped to step/allowed_values).

### Community 8 - "Files Tab UI"
Cohesion: 0.12
Nodes (20): Add Files Button, Add Folder Button, Clear All Button, Console Output Pane, DFHRTF Tab Nav Button (disabled indicator), File Info Panel, SOFA File List, Grid Radius Field (1.20 m) (+12 more)

### Community 9 - "SOFA Tab Controls"
Cohesion: 0.14
Nodes (17): Console / Log Pane, Crossover Hz Setting, DF-EQ'd Output Variant, Distance / Normalize Grid Radius Control, Equalization Mode Control, HRIR Length Control, Low Frequency Extension Control, SOFA Output Naming Template (+9 more)

### Community 10 - "Project Documentation"
Cohesion: 0.18
Nodes (12): CLAUDE.md — Project Architecture Guide, UI Config Dict Pattern (files/sofa/dfhrtf keys), LF Reference Magnitude Curve (Hz/dB), Flame-Style Custom Widget Set, Global DFHRTF Averaging Pipeline, LF Extension Below Crossover Frequency, Two-Stage Processing Pipeline (SOFA Mastering → DFHRTF Analysis), PyQt6 to PySide6 Migration (+4 more)

### Community 11 - "Grid Visualization"
Cohesion: 0.24
Nodes (11): Convex Hull Solid Angle Computation, Diffuse-Field Geometry Weighting, Raw Measurement Grid Plot, 1680 Measurement Nodes, 2.06 m Sphere Radius, Spherical HRTF Measurement Grid (Raw), Per-Node Weight Colormap (0.0003 to 0.001), Weighted Grid Plot (+3 more)

### Community 12 - "Spatial Grid Measurements"
Cohesion: 0.33
Nodes (7): Convex Hull Solid-Angle Weighting, Diffuse-Field Response Computation, HRTF Spatial Sampling Points, Grid Original PNG - 3D Spatial Measurement Grid, 1625 Measurement Nodes, 2 Meter Measurement Radius, Spherical Measurement Grid (2m radius, 1625 nodes)

## Knowledge Gaps
- **51 isolated node(s):** `Global DFHRTF Averaging Pipeline`, `manager (singleton factory)`, `SCHEMA (config validation schema)`, `Diffuse Field HRTF (concept)`, `Cascaded JSON Config (defaults + user overlay)` (+46 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `calculate_geometric_weights()` connect `DFHRTF Generation Script` to `SOFA File Utilities`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `get_grid_cartesian()` connect `SOFA File Utilities` to `DFHRTF Generation Script`, `UI Tab Layer`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Why does `FilesTab` connect `SOFA File Utilities` to `UI Tab Layer`, `Processing Orchestration`, `Flame Widget Internals`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `FilesTab` (e.g. with `SofaBuddyMainWindow` and `SofaTab`) actually correct?**
  _`FilesTab` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `SofaBuddyMainWindow` (e.g. with `DfhrtfTab` and `FilesTab`) actually correct?**
  _`SofaBuddyMainWindow` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FlameNumericSlider` (e.g. with `DfhrtfTab` and `FilesTab`) actually correct?**
  _`FlameNumericSlider` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `SofaTab` (e.g. with `SofaBuddyMainWindow` and `FilesTab`) actually correct?**
  _`SofaTab` has 8 INFERRED edges - model-reasoned connections that need verification._