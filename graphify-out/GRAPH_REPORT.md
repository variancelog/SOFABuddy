# Graph Report - .  (2026-06-13)

## Corpus Check
- 29 files · ~57,632 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 662 nodes · 875 edges · 55 communities (51 shown, 4 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 64 edges (avg confidence: 0.79)
- Token cost: 18,500 input · 4,200 output

## Community Hubs (Navigation)
- [[_COMMUNITY_SOFA Utilities & Widget Layer|SOFA Utilities & Widget Layer]]
- [[_COMMUNITY_Pipeline Docs & Averaging Concepts|Pipeline Docs & Averaging Concepts]]
- [[_COMMUNITY_JSON Schema Generic Properties|JSON Schema Generic Properties]]
- [[_COMMUNITY_Schema Validation Constraints|Schema Validation Constraints]]
- [[_COMMUNITY_Config Schema Definitions|Config Schema Definitions]]
- [[_COMMUNITY_ConfigManager Core|ConfigManager Core]]
- [[_COMMUNITY_SOFA Default Settings|SOFA Default Settings]]
- [[_COMMUNITY_DFHRTF Advanced Settings|DFHRTF Advanced Settings]]
- [[_COMMUNITY_DFHRTF Tab Controls|DFHRTF Tab Controls]]
- [[_COMMUNITY_DFHRTF Processing Scripts|DFHRTF Processing Scripts]]
- [[_COMMUNITY_Files Tab UI|Files Tab UI]]
- [[_COMMUNITY_Main Window Shell|Main Window Shell]]
- [[_COMMUNITY_SOFA Tab Controls|SOFA Tab Controls]]
- [[_COMMUNITY_SOFA Mastering Scripts|SOFA Mastering Scripts]]
- [[_COMMUNITY_Measurement Grid Visualization|Measurement Grid Visualization]]
- [[_COMMUNITY_Config Persistence & Save Flow|Config Persistence & Save Flow]]
- [[_COMMUNITY_Weighted Grid Schema Params|Weighted Grid Schema Params]]
- [[_COMMUNITY_Diffuse-Field Geometry|Diffuse-Field Geometry]]
- [[_COMMUNITY_LF Extension Schema|LF Extension Schema]]
- [[_COMMUNITY_blend_high_oct Schema|blend_high_oct Schema]]
- [[_COMMUNITY_blend_low_oct Schema|blend_low_oct Schema]]
- [[_COMMUNITY_lf_blend_high_oct Schema|lf_blend_high_oct Schema]]
- [[_COMMUNITY_lf_blend_low_oct Schema|lf_blend_low_oct Schema]]
- [[_COMMUNITY_n_fft Schema|n_fft Schema]]
- [[_COMMUNITY_norm_freq_hz Schema|norm_freq_hz Schema]]
- [[_COMMUNITY_norm_target_db Schema|norm_target_db Schema]]
- [[_COMMUNITY_onset_buffer Schema|onset_buffer Schema]]
- [[_COMMUNITY_onset_threshold_ratio Schema|onset_threshold_ratio Schema]]
- [[_COMMUNITY_output_octave_fraction Schema|output_octave_fraction Schema]]
- [[_COMMUNITY_processing_length Schema|processing_length Schema]]
- [[_COMMUNITY_stereo_angle_deg Schema|stereo_angle_deg Schema]]
- [[_COMMUNITY_tilt Schema|tilt Schema]]
- [[_COMMUNITY_weighted_grid_scale_high Schema|weighted_grid_scale_high Schema]]
- [[_COMMUNITY_weighted_grid_scale_low Schema|weighted_grid_scale_low Schema]]
- [[_COMMUNITY_algo Schema|algo Schema]]
- [[_COMMUNITY_avg_all Schema|avg_all Schema]]
- [[_COMMUNITY_dfhrtf_graphs Schema|dfhrtf_graphs Schema]]
- [[_COMMUNITY_exclude_outliers Schema|exclude_outliers Schema]]
- [[_COMMUNITY_graph_extension Schema|graph_extension Schema]]
- [[_COMMUNITY_graph_filename Schema|graph_filename Schema]]
- [[_COMMUNITY_per_sofa_avg_algo Schema|per_sofa_avg_algo Schema]]
- [[_COMMUNITY_per_sofa_avg Schema|per_sofa_avg Schema]]
- [[_COMMUNITY_per_sofa_lr Schema|per_sofa_lr Schema]]
- [[_COMMUNITY_smoothing Schema|smoothing Schema]]
- [[_COMMUNITY_text_delimiter Schema|text_delimiter Schema]]
- [[_COMMUNITY_text_extension Schema|text_extension Schema]]
- [[_COMMUNITY_text_filename Schema|text_filename Schema]]
- [[_COMMUNITY_weighted_grid Schema|weighted_grid Schema]]
- [[_COMMUNITY_weighted_grid_scope Schema|weighted_grid_scope Schema]]
- [[_COMMUNITY_DFHRTF Output Dir Constant|DFHRTF Output Dir Constant]]
- [[_COMMUNITY_LF Reference Curve|LF Reference Curve]]
- [[_COMMUNITY_Python Dependencies|Python Dependencies]]
- [[_COMMUNITY_UI Package Init|UI Package Init]]

## God Nodes (most connected - your core abstractions)
1. `FilesTab` - 28 edges
2. `dfhrtf` - 22 edges
3. `FlameNumericSlider` - 22 edges
4. `SofaBuddyMainWindow` - 21 edges
5. `DfhrtfTab` - 15 edges
6. `FlameToggleButton` - 15 edges
7. `SofaTab` - 14 edges
8. `ConfigManager` - 13 edges
9. `sofa` - 13 edges
10. `FlameLabel` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Three Averaging Algorithms (ArithMean, GeoArc, SoftDTW)` --conceptually_related_to--> `run_averaging()`  [INFERRED]
  README.md → core/dfhrtf_processor.py
- `Filename Template Variable System` --conceptually_related_to--> `process_dfhrtf_exports()`  [INFERRED]
  CLAUDE.md → core/dfhrtf_processor.py
- `Two-Stage SOFA→DFHRTF Processing Pipeline` --conceptually_related_to--> `process_dfhrtf_exports()`  [INFERRED]
  CLAUDE.md → core/dfhrtf_processor.py
- `Filename Template Variable System` --conceptually_related_to--> `process_sofa_exports()`  [INFERRED]
  CLAUDE.md → core/sofa_processor.py
- `Two-Stage SOFA→DFHRTF Processing Pipeline` --conceptually_related_to--> `process_sofa_exports()`  [INFERRED]
  CLAUDE.md → core/sofa_processor.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Flame Widget + Theme Rendering System** — ui_widgets_flametogglebutton, ui_widgets_flamenumericslider, ui_widgets_flamedropdown, ui_theme_module [INFERRED 0.95]
- **End-to-End SOFA->DFHRTF Processing Flow** — ui_main_window_on_process_clicked, core_sofa_processor_process_sofa_exports, core_dfhrtf_processor_process_dfhrtf_exports, core_dfhrtf_processor_run_averaging, core_subprocess_utils_run_task [EXTRACTED 0.95]
- **Config Persistence Cycle (snapshot + save + restore)** — core_config_snapshot, core_config_save, ui_main_window_restore_settings, ui_main_window_close_event, ui_main_window_on_process_clicked [EXTRACTED 0.95]
- **Tab get_config() serialization pattern** — ui_tabs_filestab_get_config, ui_tabs_filestab_get_output_config, ui_tabs_sofatab_get_config, ui_tabs_dfhrtftab_get_config [EXTRACTED 0.95]

## Communities (55 total, 4 thin omitted)

### Community 0 - "SOFA Utilities & Widget Layer"
Cohesion: 0.05
Nodes (31): Flame-Style Widget System, calculate_grid_radius(), calculate_max_itd(), get_grid_cartesian(), get_sofa_metadata(), Returns the dominant (most common) grid radius in metres from SourcePosition,, Extracts metadata from a SOFA file without loading the entire impulse response a, Reads SourcePosition from a SOFA file and returns the full measurement     grid (+23 more)

### Community 1 - "Pipeline Docs & Averaging Concepts"
Cohesion: 0.06
Nodes (34): Three Averaging Algorithms (ArithMean, GeoArc, SoftDTW), Filename Template Variable System, Soft-DTW Barycenter Averaging (concept), Two-Stage SOFA→DFHRTF Processing Pipeline, CommonAxisInterpolator, CurveAverager, DataIngestor, OutlierDetector (+26 more)

### Community 2 - "JSON Schema Generic Properties"
Cohesion: 0.04
Nodes (48): additionalProperties, description, type, description, type, description, maximum, minimum (+40 more)

### Community 3 - "Schema Validation Constraints"
Cohesion: 0.04
Nodes (45): additionalProperties, description, properties, type, description, maximum, minimum, type (+37 more)

### Community 4 - "Config Schema Definitions"
Cohesion: 0.05
Nodes (41): SCHEMA validation dict, additionalProperties, description, type, description, additionalProperties, description, type (+33 more)

### Community 5 - "ConfigManager Core"
Cohesion: 0.09
Nodes (27): Cascaded JSON Config: defaults + user overlay, ConfigManager, _deep_merge(), _get_nested(), load_defaults(), load_merged(), manager(), _migrate() (+19 more)

### Community 6 - "SOFA Default Settings"
Cohesion: 0.06
Nodes (33): blend_high_oct, blend_low_oct, norm_target_db, onset_buffer, onset_threshold_ratio, processing_length, averaging, geo_arc_y_scale (+25 more)

### Community 7 - "DFHRTF Advanced Settings"
Cohesion: 0.06
Nodes (32): balance_weight_cv_threshold, lf_blend_high_oct, lf_blend_low_oct, n_fft, norm_freq_hz, output_octave_fraction, stereo_angle_deg, weighted_grid_scale_high (+24 more)

### Community 8 - "DFHRTF Tab Controls"
Cohesion: 0.08
Nodes (31): Average All Ls/Rs Toggle, Averaging Method Dropdown (Soft-DTW), Processing Console Log Output, Crossover Hz Control (375), Text Output Delimiter (Tab), Exclude Outliers Toggle, Extend LF Toggle, Text Output Filename Template (gr_{folder}_{filename}_{LR}) (+23 more)

### Community 9 - "DFHRTF Processing Scripts"
Cohesion: 0.15
Nodes (20): Ghost-Node ConvexHull Solid-Angle Weighting, LF Extension via Cosine Crossfade with Reference Curve, apply_lf_extension(), apply_smoothing(), apply_spectral_tilt(), calculate_geometric_weights(), find_closest_grid_point(), generate_fractional_octave_frequencies() (+12 more)

### Community 10 - "Files Tab UI"
Cohesion: 0.12
Nodes (20): Add Files Button, Add Folder Button, Clear All Button, Console Output Pane, DFHRTF Tab Nav Button (disabled indicator), File Info Panel, SOFA File List, Grid Radius Field (1.20 m) (+12 more)

### Community 11 - "Main Window Shell"
Cohesion: 0.16
Nodes (5): QMainWindow, Load persisted settings and push them into the tabs., Reset all settings to factory defaults and update the tabs., Persist current settings when the window is closed., SofaBuddyMainWindow

### Community 12 - "SOFA Tab Controls"
Cohesion: 0.14
Nodes (17): Console / Log Pane, Crossover Hz Setting, DF-EQ'd Output Variant, Distance / Normalize Grid Radius Control, Equalization Mode Control, HRIR Length Control, Low Frequency Extension Control, SOFA Output Naming Template (+9 more)

### Community 13 - "SOFA Mastering Scripts"
Cohesion: 0.25
Nodes (14): calculate_geometric_weights(), ensure_mesh2hrtf_import(), find_global_onset(), find_sofas_in_project(), get_blend_weights(), get_dominant_radius(), load_lf_target_base(), main() (+6 more)

### Community 14 - "Measurement Grid Visualization"
Cohesion: 0.24
Nodes (11): Convex Hull Solid Angle Computation, Diffuse-Field Geometry Weighting, Raw Measurement Grid Plot, 1680 Measurement Nodes, 2.06 m Sphere Radius, Spherical HRTF Measurement Grid (Raw), Per-Node Weight Colormap (0.0003 to 0.001), Weighted Grid Plot (+3 more)

### Community 15 - "Config Persistence & Save Flow"
Cohesion: 0.25
Nodes (6): Atomic Config Save (tempfile + os.replace), Default vs Custom Output Folder Mode, ConfigManager.save(), ConfigManager.snapshot(), closeEvent(), _on_process_clicked()

### Community 16 - "Weighted Grid Schema Params"
Cohesion: 0.29
Nodes (7): properties, maximum, minimum, type, balance_weight_cv_threshold, weighted_grid_scale_mode, enum

### Community 17 - "Diffuse-Field Geometry"
Cohesion: 0.33
Nodes (7): Convex Hull Solid-Angle Weighting, Diffuse-Field Response Computation, HRTF Spatial Sampling Points, Grid Original PNG - 3D Spatial Measurement Grid, 1625 Measurement Nodes, 2 Meter Measurement Radius, Spherical Measurement Grid (2m radius, 1625 nodes)

### Community 18 - "LF Extension Schema"
Cohesion: 0.33
Nodes (6): properties, description, type, extend_lf, weighting_mode, enum

### Community 19 - "blend_high_oct Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, blend_high_oct

### Community 20 - "blend_low_oct Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, blend_low_oct

### Community 21 - "lf_blend_high_oct Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, lf_blend_high_oct

### Community 22 - "lf_blend_low_oct Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, lf_blend_low_oct

### Community 23 - "n_fft Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, n_fft

### Community 24 - "norm_freq_hz Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, norm_freq_hz

### Community 25 - "norm_target_db Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, norm_target_db

### Community 26 - "onset_buffer Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, onset_buffer

### Community 27 - "onset_threshold_ratio Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, onset_threshold_ratio

### Community 28 - "output_octave_fraction Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, output_octave_fraction

### Community 29 - "processing_length Schema"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, processing_length

### Community 30 - "stereo_angle_deg Schema"
Cohesion: 0.40
Nodes (5): stereo_angle_deg, description, maximum, minimum, type

### Community 31 - "tilt Schema"
Cohesion: 0.40
Nodes (5): tilt, description, maximum, minimum, type

### Community 32 - "weighted_grid_scale_high Schema"
Cohesion: 0.50
Nodes (4): weighted_grid_scale_high, maximum, minimum, type

### Community 33 - "weighted_grid_scale_low Schema"
Cohesion: 0.50
Nodes (4): weighted_grid_scale_low, maximum, minimum, type

### Community 34 - "algo Schema"
Cohesion: 0.67
Nodes (3): description, enum, algo

### Community 35 - "avg_all Schema"
Cohesion: 0.67
Nodes (3): description, type, avg_all

### Community 36 - "dfhrtf_graphs Schema"
Cohesion: 0.67
Nodes (3): description, type, dfhrtf_graphs

### Community 37 - "exclude_outliers Schema"
Cohesion: 0.67
Nodes (3): description, type, exclude_outliers

### Community 38 - "graph_extension Schema"
Cohesion: 0.67
Nodes (3): description, enum, graph_extension

### Community 39 - "graph_filename Schema"
Cohesion: 0.67
Nodes (3): description, type, graph_filename

### Community 40 - "per_sofa_avg_algo Schema"
Cohesion: 0.67
Nodes (3): description, enum, per_sofa_avg_algo

### Community 41 - "per_sofa_avg Schema"
Cohesion: 0.67
Nodes (3): description, type, per_sofa_avg

### Community 42 - "per_sofa_lr Schema"
Cohesion: 0.67
Nodes (3): description, type, per_sofa_lr

### Community 43 - "smoothing Schema"
Cohesion: 0.67
Nodes (3): smoothing, description, enum

### Community 44 - "text_delimiter Schema"
Cohesion: 0.67
Nodes (3): text_delimiter, description, enum

### Community 45 - "text_extension Schema"
Cohesion: 0.67
Nodes (3): text_extension, description, enum

### Community 46 - "text_filename Schema"
Cohesion: 0.67
Nodes (3): text_filename, description, type

### Community 47 - "weighted_grid Schema"
Cohesion: 0.67
Nodes (3): weighted_grid, description, type

### Community 48 - "weighted_grid_scope Schema"
Cohesion: 0.67
Nodes (3): weighted_grid_scope, description, enum

## Knowledge Gaps
- **297 isolated node(s):** `$schema`, `schema_version`, `default`, `folder`, `enabled` (+292 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `properties` connect `Config Schema Definitions` to `Schema Validation Constraints`?**
  _High betweenness centrality (0.423) - this node is a cross-community bridge._
- **Why does `properties` connect `LF Extension Schema` to `JSON Schema Generic Properties`, `avg_all Schema`, `algo Schema`, `exclude_outliers Schema`, `dfhrtf_graphs Schema`, `graph_extension Schema`, `graph_filename Schema`, `per_sofa_avg Schema`, `per_sofa_avg_algo Schema`, `per_sofa_lr Schema`, `smoothing Schema`, `text_delimiter Schema`, `text_extension Schema`, `text_filename Schema`, `weighted_grid Schema`, `weighted_grid_scope Schema`, `Config Schema Definitions`, `tilt Schema`?**
  _High betweenness centrality (0.216) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `FilesTab` (e.g. with `SofaBuddyMainWindow` and `FlameDropdown`) actually correct?**
  _`FilesTab` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FlameNumericSlider` (e.g. with `DfhrtfTab` and `FilesTab`) actually correct?**
  _`FlameNumericSlider` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `SofaBuddyMainWindow` (e.g. with `DfhrtfTab` and `FilesTab`) actually correct?**
  _`SofaBuddyMainWindow` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Loads a CSV file and attempts to identify frequency and magnitude columns.`, `Takes a list of dataframes and maps them to a rigid common frequency axis.`, `Restores the averaged 0-1 magnitude curve back to original dB scale.` to the rest of the system?**
  _357 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `SOFA Utilities & Widget Layer` be split into smaller, more focused modules?**
  _Cohesion score 0.0515406162464986 - nodes in this community are weakly interconnected._