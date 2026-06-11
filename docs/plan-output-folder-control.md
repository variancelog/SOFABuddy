# Plan: Per-Tab Output Folder Control + Multi-Folder Fix

## Context

SOFA Buddy lets users import `.sofa` files from multiple folders into one flat list. Today both processors derive their output location from **only the first file** (`os.path.dirname(files[0])`), with one exception that makes things worse:

- `process_sofa_exports` writes **every** mastered SOFA into `dirname(files[0])/sofa_mastered/` (`core/sofa_processor.py:28-29,51`).
- `process_dfhrtf_exports` looks for each file's mastered SOFA in **that file's own** folder `dirname(raw_path)/sofa_mastered/` (`core/dfhrtf_processor.py:104-105`), so files not in the first folder are never found and silently fall back to **raw/unprocessed** analysis.
- DFHRTF text/graph output and global averaging also use the first folder only (`dfhrtf_processor.py:135-137,253`).

**Goal:** Give each of the SOFA and DFHRTF tabs an "Output Folder:" row with a `Default` / `Select:` choice, and fix the processors so multi-folder imports behave correctly in both modes:

- **Default** (per-source): each file's outputs go to a `sofa_mastered/` or `DFHRTF/` subfolder inside **that file's own** source folder. DFHRTF correctly finds the per-folder mastered SOFA.
- **Select:** all outputs for that tab go **directly** into one chosen folder (no subfolder).
- **Global averaging:** Default mode → a **per-folder** average written into each source folder's `DFHRTF/`. Select mode → one **true global** average across all files into the chosen folder. (Both fall out of grouping generated files by their output directory.)

The two tabs' settings are independent (a row on each).

---

## UI Changes — `ui/tabs.py`

Add a new bottom grid row to **both** `SofaTab` and `DfhrtfTab`. Layout (col 0 label, col 1 HBox):

```
Output Folder:  [Default]  [Select:]  [ <line edit> ]  [ ... ]
```

Widgets per tab (reuse `FlameToggleButton` / `FlameLineEdit` / `FlameLabel`; `QFileDialog`, `QPushButton` already imported):

- `btn_output_default = FlameToggleButton("Default")` — checked initially.
- `btn_output_select = FlameToggleButton("Select:")`.
- `edit_output_folder = FlameLineEdit("")` — disabled initially.
- `btn_output_browse = QPushButton("...")` — disabled initially; opens the dialog.

Placement: SofaTab grid row 6 (before `addLayout(grid)` ~line 310); DfhrtfTab grid row 7 (~line 475).

Mutual exclusion / enable-disable via small handlers using `blockSignals` (no `QButtonGroup` precedent in codebase); browse via `QFileDialog.getExistingDirectory`. `get_config()` gains `output_default` and `output_folder` keys in both tabs.

## Core Changes — `core/sofa_processor.py`
Per-file `out_root`: Default → `dirname(file)/sofa_mastered`; Select → custom dir directly. Validate custom dir non-empty.

## Core Changes — `core/dfhrtf_processor.py`
1. Mastered-SOFA lookup follows the **SOFA tab's** output setting.
2. DFHRTF text/graph `output_root` computed per-file (carry `src_dir` in `resolved_inputs` tuple).
3. Group generated L/R files by output dir; average per group → per-folder averages in Default, single global average in Select.
4. Hoist `delim` above the loop.

## Verification
1. UI row behaves (Default checked, Select enables box+`...`, never both off).
2. Multi-folder Default: each folder gets its own `sofa_mastered/`, `DFHRTF/`, and per-folder average; no raw-fallback warnings.
3. Select mode: all mastered SOFAs in folder C, all DFHRTF in folder D, single global average in D.
4. Single-folder Default regression matches old behavior.
5. `pytest tests/`.
