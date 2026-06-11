from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QPushButton, QGroupBox, QLabel, QGridLayout, QLineEdit, QFileDialog
from PySide6.QtCore import Signal

from ui.widgets import FlameToggleButton, FlameDropdown, FlameNumericSlider, FlameLabel, FlameLineEdit
from ui.theme import BG_WIDGET, BG_CONTROL, ACCENT, FG, CONTROL_HEIGHT
from core.sofa_utils import get_sofa_metadata, calculate_max_itd, calculate_grid_radius

# --- Files Tab Widget ---
class FilesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self._itd_cache = {}    # Cache for calculated ITDs
        self._radius_cache = {} # Cache for calculated grid radii

        # --- Top content: file list + right panel ---
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        # - File list
        self.file_list = QListWidget()
        self.file_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {BG_WIDGET};
                color: {FG};
                border: 1px solid {BG_CONTROL};
                margin-right: 5px;
                margin-top: 10px;
            }}
        """)
        content.addWidget(self.file_list, stretch=2)

        right_panel = QVBoxLayout()

        self.btn_open_files = QPushButton("Add Files")
        self.btn_open_files.setToolTip("Add Files")
        self.btn_open_folder = QPushButton("Add Folder")
        self.btn_open_folder.setToolTip("Add Folder")
        self.btn_clear_list = QPushButton("Clear List")
        self.btn_clear_list.setToolTip("Clear List")

        for btn in [self.btn_open_files, self.btn_open_folder, self.btn_clear_list]:
            btn.setStyleSheet(
                f"background-color: {BG_CONTROL}; color: {FG}; border: none; "
                f"padding: 0px 10px; "
                f"min-height: {CONTROL_HEIGHT}px; max-height: {CONTROL_HEIGHT}px; "
                f"margin-top: 10px; margin-left: 5px;"
            )
            right_panel.addWidget(btn)

        self.btn_open_files.clicked.connect(self._open_files)
        self.btn_open_folder.clicked.connect(self._open_folder)
        self.btn_clear_list.clicked.connect(self._clear_list)

        self.info_group = QGroupBox("File Info")
        self.info_group.setStyleSheet(f"color: {FG}; border: 1px solid {BG_CONTROL}; margin-top: 10px;")
        info_layout = QVBoxLayout(self.info_group)
        self.lbl_samplerate = FlameLabel("Samplerate: -")
        self.lbl_length = FlameLabel("Length: -")
        self.lbl_points = FlameLabel("Points: -")

        # ITD Row
        itd_layout = QHBoxLayout()
        itd_layout.setContentsMargins(0, 0, 0, 0)
        itd_layout.addWidget(FlameLabel("Max ITD: "))
        self.lbl_itd_val = FlameLabel("-")
        self.btn_calc_itd = QPushButton("Calculate")
        self.btn_calc_itd.setToolTip("Calculate ITD")
        self.btn_calc_itd.setFixedWidth(80)
        self.btn_calc_itd.setStyleSheet(
            f"background-color: {ACCENT}; color: {FG}; border: none; font-size: 10px; padding: 2px;"
        )
        self.btn_calc_itd.clicked.connect(self._on_calc_itd_clicked)

        itd_layout.addWidget(self.lbl_itd_val)
        itd_layout.addWidget(self.btn_calc_itd)
        itd_layout.addStretch()

        # Grid Radius Row
        radius_layout = QHBoxLayout()
        radius_layout.setContentsMargins(0, 0, 0, 0)
        radius_layout.addWidget(FlameLabel("Grid Radius: "))
        self.lbl_radius_val = FlameLabel("-")
        self.btn_calc_radius = QPushButton("Calculate")
        self.btn_calc_radius.setToolTip("Calculate Grid Radius")
        self.btn_calc_radius.setFixedWidth(80)
        self.btn_calc_radius.setStyleSheet(
            f"background-color: {ACCENT}; color: {FG}; border: none; font-size: 10px; padding: 2px;"
        )
        self.btn_calc_radius.clicked.connect(self._on_calc_radius_clicked)
        radius_layout.addWidget(self.lbl_radius_val)
        radius_layout.addWidget(self.btn_calc_radius)
        radius_layout.addStretch()

        info_layout.addWidget(self.lbl_samplerate)
        info_layout.addWidget(self.lbl_length)
        info_layout.addWidget(self.lbl_points)
        info_layout.addLayout(itd_layout)
        info_layout.addLayout(radius_layout)

        right_panel.addWidget(self.info_group)
        right_panel.addStretch()

        content.addLayout(right_panel, stretch=1)
        self.layout.addLayout(content)

        self.file_list.itemSelectionChanged.connect(self._on_file_selection_changed)

        # --- Output Folder row (spans full width at bottom) ---
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 6, 0, 0)
        out_row.setSpacing(8)
        out_row.addWidget(FlameLabel("Output:"))

        self.btn_output_default = FlameToggleButton("Default")
        self.btn_output_default.setToolTip(
            "<p>Each file's outputs go to <b>sofa_mastered/</b> and <b>DFHRTF/</b> subfolders "
            "inside its own source folder. Global averaging produces a per-folder average.</p>"
        )
        self.btn_output_default.setChecked(True)
        self.btn_output_custom = FlameToggleButton("Custom:")
        self.btn_output_custom.setToolTip(
            "<p>All outputs go under the chosen folder (with <b>sofa_mastered/</b> and <b>DFHRTF/</b> subfolders). "
            "Global averaging produces a single average across all files.</p>"
        )
        self.edit_output_folder = FlameLineEdit("")
        self.edit_output_folder.setToolTip("<p>Root folder for all outputs (sofa_mastered/ and DFHRTF/ will be created inside it).</p>")
        self.edit_output_folder.setEnabled(False)
        self.btn_output_browse = QPushButton("...")
        self.btn_output_browse.setEnabled(False)
        self.btn_output_browse.setFixedWidth(40)
        self.btn_output_browse.setStyleSheet(
            f"background-color: {BG_CONTROL}; color: {FG}; border: none; "
            f"min-height: {CONTROL_HEIGHT}px; max-height: {CONTROL_HEIGHT}px;"
        )

        self.btn_output_default.toggled.connect(self._on_output_default_toggled)
        self.btn_output_custom.toggled.connect(self._on_output_custom_toggled)
        self.btn_output_browse.clicked.connect(self._browse_output_folder)

        out_row.addWidget(self.btn_output_default)
        out_row.addWidget(self.btn_output_custom)
        out_row.addWidget(self.edit_output_folder, stretch=1)
        out_row.addWidget(self.btn_output_browse)
        self.layout.addLayout(out_row)

    def _set_toggle(self, btn, value):
        btn.blockSignals(True)
        btn.setChecked(value)
        btn.update_style()
        btn.blockSignals(False)

    def _on_output_default_toggled(self, checked):
        if checked:
            self._set_toggle(self.btn_output_custom, False)
        elif not self.btn_output_custom.isChecked():
            self._set_toggle(self.btn_output_default, True)
        self._sync_output_enabled()

    def _on_output_custom_toggled(self, checked):
        if checked:
            self._set_toggle(self.btn_output_default, False)
        elif not self.btn_output_default.isChecked():
            self._set_toggle(self.btn_output_default, True)
        self._sync_output_enabled()

    def _sync_output_enabled(self):
        on = self.btn_output_custom.isChecked()
        self.edit_output_folder.setEnabled(on)
        self.btn_output_browse.setEnabled(on)

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.edit_output_folder.setText(folder)

    def _clear_list(self):
        self.file_list.clear()
        self._itd_cache.clear()
        self._radius_cache.clear()

    def _on_file_selection_changed(self):
        selected = self.file_list.selectedItems()
        if not selected:
            self.lbl_samplerate.setText("Samplerate: -")
            self.lbl_length.setText("Length: -")
            self.lbl_points.setText("Points: -")
            self.lbl_itd_val.setText("-")
            self.btn_calc_itd.hide()
            self.lbl_itd_val.show()
            self.lbl_radius_val.setText("-")
            self.btn_calc_radius.hide()
            self.lbl_radius_val.show()
            return

        file_path = selected[0].text()
        meta = get_sofa_metadata(file_path)

        if meta:
            self.lbl_samplerate.setText(f"Samplerate: {int(meta['samplerate'])} Hz")
            self.lbl_length.setText(f"Length: {meta['length']} samples")
            self.lbl_points.setText(f"Points: {meta['points']}")

            # Update ITD display from cache
            if file_path in self._itd_cache:
                self.lbl_itd_val.setText(f"{self._itd_cache[file_path]:.2f} ms")
                self.btn_calc_itd.hide()
                self.lbl_itd_val.show()
            else:
                self.lbl_itd_val.hide()
                self.btn_calc_itd.setText("Calculate")
                self.btn_calc_itd.setEnabled(True)
                self.btn_calc_itd.show()

            # Update Grid Radius display from cache
            if file_path in self._radius_cache:
                self.lbl_radius_val.setText(f"{self._radius_cache[file_path]:.2f} m")
                self.btn_calc_radius.hide()
                self.lbl_radius_val.show()
            else:
                self.lbl_radius_val.hide()
                self.btn_calc_radius.setText("Calculate")
                self.btn_calc_radius.setEnabled(True)
                self.btn_calc_radius.show()
        else:
            self.lbl_samplerate.setText("Samplerate: ERROR")
            self.lbl_length.setText("Length: -")
            self.lbl_points.setText("Points: -")
            self.lbl_itd_val.setText("-")
            self.btn_calc_itd.hide()
            self.lbl_itd_val.show()
            self.lbl_radius_val.setText("-")
            self.btn_calc_radius.hide()
            self.lbl_radius_val.show()

    def _on_calc_itd_clicked(self):
        selected = self.file_list.selectedItems()
        if not selected: return

        file_path = selected[0].text()
        self.btn_calc_itd.setText("...")
        self.btn_calc_itd.setEnabled(False)
        self.repaint() # Force UI update to show "..."

        itd_ms = calculate_max_itd(file_path)
        self._itd_cache[file_path] = itd_ms

        self.lbl_itd_val.setText(f"{itd_ms:.2f} ms")
        self.btn_calc_itd.hide()
        self.lbl_itd_val.show()

    def _on_calc_radius_clicked(self):
        selected = self.file_list.selectedItems()
        if not selected: return

        file_path = selected[0].text()
        self.btn_calc_radius.setText("...")
        self.btn_calc_radius.setEnabled(False)
        self.repaint() # Force UI update to show "..."

        radius_m = calculate_grid_radius(file_path)
        self._radius_cache[file_path] = radius_m

        self.lbl_radius_val.setText(f"{radius_m:.2f} m")
        self.btn_calc_radius.hide()
        self.lbl_radius_val.show()

    def _open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select SOFA files", "", "SOFA Files (*.sofa *.SOFA)")
        if files:
            self.file_list.addItems(files)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            import os
            for f in os.listdir(folder):
                if f.lower().endswith('.sofa'):
                    self.file_list.addItem(os.path.join(folder, f))

    def get_config(self) -> list:
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]

    def get_output_config(self) -> dict:
        return {
            'default': self.btn_output_default.get_value(),
            'folder': self.edit_output_folder.text().strip()
        }

# --- SOFA Tab Widget
class SofaTab(QWidget):
    sofa_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 10, 0, 0)
        self.layout.setSpacing(0)

        self.btn_enable = FlameToggleButton("SOFA Outputs")
        self.btn_enable.setToolTip("<p>Enables mastered SOFA outputs. At least one samplerate must be selected. Read button tooltips for information about each setting.</p>")
        self.btn_enable.toggled.connect(self._on_enable_toggled)
        self.layout.addWidget(self.btn_enable)

        self.container = QWidget()
        self.container.setEnabled(False)
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 10, 0, 5)
        self.container_layout.setSpacing(25)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10) # Keep some spacing between grid elements for readability
        grid.setColumnMinimumWidth(0, 120)

        # --- sr_layout is definition of samplerates table
        sr_layout = QHBoxLayout()
        sr_layout.setSpacing(10)
        self.btn_sr_44k = FlameToggleButton("44.1k")
        self.btn_sr_44k.setToolTip("<p>Include 44.1kHz SOFA file.</p>")
        self.btn_sr_48k = FlameToggleButton("48k")
        self.btn_sr_48k.setToolTip("<p>Include 48kHz SOFA file.</p>")
        self.btn_sr_48k.setChecked(True)
        self.btn_sr_96k = FlameToggleButton("96k")
        self.btn_sr_96k.setToolTip("<p>Include 96kHz SOFA file. More useful for bat HRTFs than human ones.</p>")
        sr_layout.addWidget(self.btn_sr_44k)
        sr_layout.addWidget(self.btn_sr_48k)
        sr_layout.addWidget(self.btn_sr_96k)

        lf_xover_layout = QHBoxLayout()
        self.btn_lf = FlameToggleButton("Extend LF")
        self.btn_lf.setToolTip("<p>Replaces HRIR magnitudes below the crossover frequency with an 'ideal' frequency response. Recommended for most measured SOFAs.</p>")
        self.slider_xover = FlameNumericSlider("Crossover Hz", 200.0, 500.0, 5.0, 375.0)
        self.slider_xover.setToolTip("<p>Crossover frequency for Extend LF function. 375Hz works best in most cases.</p>")
        self.btn_lf.toggled.connect(self.slider_xover.setEnabled)
        self.slider_xover.setEnabled(False)
        lf_xover_layout.addWidget(self.btn_lf)
        lf_xover_layout.addWidget(self.slider_xover)

        eq_layout = QHBoxLayout()
        eq_layout.setSpacing(10)
        self.btn_uneq = FlameToggleButton("Un-EQ'd")
        self.btn_uneq.setToolTip("<p>Include SOFAs with unequalized frequency response. Useful to obtain HRTF/DFHRTF responses.</p>")
        self.btn_dfeq = FlameToggleButton("DF-EQ'd")
        self.btn_dfeq.setToolTip("<p>Include SOFAs with diffuse-field equalized freqency response. Useful in binaural rendering applications and to view DTFs.</p>")
        self.btn_dfeq.setChecked(True)
        eq_layout.addWidget(self.btn_uneq)
        eq_layout.addWidget(self.btn_dfeq)

        dist_layout = QHBoxLayout()
        self.btn_dist = FlameToggleButton("Normalize Grid Radius")
        self.btn_dist.setToolTip("<p>Normalizes grid radius to a user-defined distance. Useful for comparing HRTFs across labs. Enabling this setting overrides the default peak normalization (useful to prevent clipping in binaural renderers).</p>")
        self.btn_dist.setChecked(True)
        self.slider_dist = FlameNumericSlider("m", 1.0, 2.0, 0.1, 1.2)
        self.slider_dist.setToolTip("<p>Grid radius in metres.</p>")
        self.btn_dist.toggled.connect(self.slider_dist.setEnabled)
        dist_layout.addWidget(self.btn_dist)
        dist_layout.addWidget(self.slider_dist)

        self.slider_length = FlameNumericSlider("Samples", 128.0, 1024.0, 128.0, 512.0)
        self.slider_length.setToolTip("<p>Output HRIR sample length. 512 recommended for most cases.</p>")

        self.lbl_naming = FlameLabel("SOFA Naming:")
        self.edit_naming = FlameLineEdit("{filename}_{samplerate}_{eq}")
        self.edit_naming.setToolTip("<p>Naming of output SOFA files. Can include static text and the following variables: {filename} (original SOFA filename); {samplerate} (samplerate of processed SOFA file); {eq} (uneq or dfeq)</p>")

        # --- Apply configurations from above
        grid.addWidget(FlameLabel("Samplerates:"), 0, 0)
        grid.addLayout(sr_layout, 0, 1)

        grid.addWidget(FlameLabel("Low Frequency:"), 1, 0)
        grid.addLayout(lf_xover_layout, 1, 1)

        grid.addWidget(FlameLabel("Equalization:"), 2, 0)
        grid.addLayout(eq_layout, 2, 1)

        grid.addWidget(FlameLabel("Distance:"), 3, 0)
        grid.addLayout(dist_layout, 3, 1)

        grid.addWidget(FlameLabel("HRIR Length:"), 4, 0)
        grid.addWidget(self.slider_length, 4, 1)
        grid.addWidget(self.lbl_naming, 5, 0)
        grid.addWidget(self.edit_naming, 5, 1)

        self.container_layout.addLayout(grid)
        self.layout.addWidget(self.container)
        self.layout.addStretch()

    def _on_enable_toggled(self, checked):
        self.container.setEnabled(checked)
        self.sofa_toggled.emit(checked)

    def get_config(self) -> dict:
        srs = []
        if self.btn_sr_44k.get_value(): srs.append(44100)
        if self.btn_sr_48k.get_value(): srs.append(48000)
        if self.btn_sr_96k.get_value(): srs.append(96000)

        return {
            'enabled': self.btn_enable.get_value(),
            'samplerates': srs,
            'phase_mode': "Original Phase",
            'lf_extension': self.btn_lf.get_value(),
            'xover': self.slider_xover.get_value(),
            'uneq': self.btn_uneq.get_value(),
            'dfeq': self.btn_dfeq.get_value(),
            'normalize_distance': self.btn_dist.get_value(),
            'distance_m': self.slider_dist.get_value(),
            'hrir_length': self.slider_length.get_value(),
            'naming': self.edit_naming.text()
        }

# --- DFHRTF Tab Widget
class DfhrtfTab(QWidget):
    dfhrtf_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 10, 0, 0)
        self.layout.setSpacing(0)

        self.btn_enable = FlameToggleButton("HRTF Outputs")
        self.btn_enable.setToolTip("<p>Enables HRTF outputs. At least one text output or one graph output must be selected. NOTE: If SOFA outputs are enabled, source SOFA for DFHRTF outpus the processed SOFA files; otherwise, source is imported SOFA files on Files tab.</p>")
        self.btn_enable.toggled.connect(self._on_enable_toggled)
        self.layout.addWidget(self.btn_enable)

        self.container = QWidget()
        self.container.setEnabled(False)
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 10, 0, 5)
        self.container_layout.setSpacing(0)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        grid.setColumnMinimumWidth(0, 120)

        # --- Text Outputs Section ---
        self.btn_per_sofa_lr = FlameToggleButton("Per SOFA L/R DFHRTFs")
        self.btn_per_sofa_lr.setToolTip("<p>Generates separate left ear and right ear DFHRTF magnitute response text files per source SOFA file.</p>")
        self.btn_per_sofa_lr.setChecked(True)
        self.btn_per_sofa_avg = FlameToggleButton("Per SOFA Avg DFHRTFs")
        self.btn_per_sofa_avg.setToolTip("<p>Generates average of left ear + right ear DFHRTF magnitute response text files per source SOFA file.</p>")

        self.edit_text_filename = FlameLineEdit("text_{filename}_{LR}")
        self.edit_text_filename.setToolTip(
            "<p>Naming of generated text files. Supported variables:</p> "
            "<p style='margin-top: 0px; margin-left: 10px;'>"
            "&bull; {filename} source SOFA filename;<br>"
            "&bull; {samplerate} source SOFA samplerate;<br>"
            "&bull; {LR} L/R/Avg.;<br>"
            "&bull; {algo} averaging algorithm (global average only).</p>"
        )
        self.dd_delimiter = FlameDropdown("Delimiter")
        self.dd_delimiter.setToolTip("<p>Text file delimiter.</p>")
        self.dd_delimiter.addItems(["Tab", "Comma", "Semicolon", "Space"])
        self.dd_text_ext = FlameDropdown("Extension")
        self.dd_text_ext.setToolTip("<p>Text file extension</p>")
        self.dd_text_ext.addItems(["txt", "csv"])

        # --- Graph Outputs Section ---
        self.btn_dfhrtf_graphs = FlameToggleButton("DFHRTF Graphs")
        self.btn_dfhrtf_graphs.setToolTip("<p>Enables diffuse field HRTF graph outputs. Left and right ears overlayed on one graph.</p>")
        self.btn_dfhrtf_graphs.setChecked(True)
        self.btn_stereo_graphs = FlameToggleButton("Stereo Graphs")
        self.btn_stereo_graphs.setToolTip("<p>Enables +30/-30 (stereo) graphs outputs. Left ear's left channel HRTF and right ear's right channel HRTF overlayed on one graph.</p>")

        self.edit_graph_filename = FlameLineEdit("graph_{filename}_{graphtype}")
        self.edit_graph_filename.setToolTip(
            "<p>Naming of generated graph files. Supported variables:</p> "
            "<p style='margin-top: 0px; margin-left: 10px;'>"
            "&bull; {filename} source SOFA filename;<br>"
            "&bull; {samplerate} source SOFA samplerate;<br>"
            "&bull; {graphtype} df / stereo.</p>"
        )
        self.dd_graph_ext = FlameDropdown("Extension")
        self.dd_graph_ext.setToolTip("<p>Graph file extension.</p>")
        self.dd_graph_ext.addItems(["png", "jpg"])

        # --- Refinement Section ---
        self.slider_smoothing = FlameNumericSlider("Smooth 1/n Oct", 3, 48, 1, 24, allowed_values=[3, 6, 12, 24, 48])
        self.slider_smoothing.setToolTip("<p>Smoothing 1/n octave. Applies to both text and graph outputs.</p>")
        self.slider_tilt = FlameNumericSlider(" Tilt dB/Oct", -2.0, 1.0, 0.1, 0.0)
        self.slider_tilt.setToolTip("<p>Tilt dB/Oct, center frequency of 1kHz. Applies to both text and graph outputs.</p>")

        self.btn_extend_lf = FlameToggleButton("Extend LF")
        self.btn_extend_lf.setToolTip("<p>Extend and linerizes low frequency (LF) response below the crossover frequency. Not needed for simulated SOFA files or if Extend LF already applied to measured SOFA files.</p>")
        self.slider_xover = FlameNumericSlider("Crossover Hz", 200.0, 500.0, 5.0, 375.0)
        self.slider_xover.setToolTip("<p>Crossover frequency for low frequency extenstion.</p>")

        # --- Post Processing Section ---
        self.btn_avg_all = FlameToggleButton("Average All Ls/Rs")
        self.btn_avg_all.setToolTip("<p>Averages all left ear and all right ear text outputs. Requires 'Per SOFA L/R' to be enabled above.</p>")
        self.dd_algo = FlameDropdown("Algo")
        self.dd_algo.setToolTip("<p>Averaging Algorithm used for averaging frequency responses. See README file for more information.</p>")
        self.dd_algo.addItems(["Arithmetic Mean", "Geo-Arc-Length", "Soft-DTW"])
        self.btn_exclude_outliers = FlameToggleButton("Exclude Outliers")
        self.btn_exclude_outliers.setToolTip("<p>Excludes measureuments that exceed STDEV > 1.0</p>")
        self.btn_exclude_outliers.setChecked(True)

        # --- Connections for Dependency State ---
        for toggle in [self.btn_per_sofa_lr, self.btn_per_sofa_avg, self.btn_dfhrtf_graphs, self.btn_stereo_graphs, self.btn_extend_lf, self.btn_avg_all]:
            toggle.toggled.connect(self._update_dependencies)

        # --- Layout Mapping ---
        grid.addWidget(FlameLabel("Text Outputs:"), 0, 0)
        row0 = QHBoxLayout()
        row0.addWidget(self.btn_per_sofa_lr)
        row0.addWidget(self.btn_per_sofa_avg)
        grid.addLayout(row0, 0, 1)

        row1 = QHBoxLayout()
        row1.addWidget(self.edit_text_filename)
        row1.addWidget(self.dd_delimiter)
        row1.addWidget(self.dd_text_ext)
        grid.addLayout(row1, 1, 1)

        grid.addWidget(FlameLabel("Graph Outputs:"), 2, 0)
        row2 = QHBoxLayout()
        row2.addWidget(self.btn_dfhrtf_graphs)
        row2.addWidget(self.btn_stereo_graphs)
        grid.addLayout(row2, 2, 1)

        row3 = QHBoxLayout()
        row3.addWidget(self.edit_graph_filename)
        row3.addWidget(self.dd_graph_ext)
        grid.addLayout(row3, 3, 1)

        grid.addWidget(FlameLabel("Refinement:"), 4, 0)
        row4 = QHBoxLayout()
        row4.addWidget(self.slider_smoothing)
        row4.addWidget(self.slider_tilt)
        grid.addLayout(row4, 4, 1)

        row5 = QHBoxLayout()
        row5.addWidget(self.btn_extend_lf)
        row5.addWidget(self.slider_xover)
        grid.addLayout(row5, 5, 1)

        grid.addWidget(FlameLabel("Post Processing:"), 6, 0)
        row6 = QHBoxLayout()
        row6.addWidget(self.btn_avg_all)
        row6.addWidget(self.dd_algo)
        row6.addWidget(self.btn_exclude_outliers)
        grid.addLayout(row6, 6, 1)

        self.container_layout.addLayout(grid)
        self.layout.addWidget(self.container)
        self.layout.addStretch()

        self._update_dependencies()

    def _on_enable_toggled(self, checked):
        self.container.setEnabled(checked)
        self.dfhrtf_toggled.emit(checked)

    def _update_dependencies(self):
        any_text = self.btn_per_sofa_lr.isChecked() or self.btn_per_sofa_avg.isChecked()
        any_graph = self.btn_dfhrtf_graphs.isChecked() or self.btn_stereo_graphs.isChecked()
        any_output = any_text or any_graph

        # Text Output Config
        self.edit_text_filename.setEnabled(any_text)
        self.dd_delimiter.setEnabled(any_text)
        self.dd_text_ext.setEnabled(any_text)

        # Graph Output Config
        self.edit_graph_filename.setEnabled(any_graph)
        self.dd_graph_ext.setEnabled(any_graph)

        # Refinement & Post Processing
        self.slider_smoothing.setEnabled(any_output)
        self.slider_tilt.setEnabled(any_output)
        self.btn_extend_lf.setEnabled(any_output)
        self.slider_xover.setEnabled(any_output and self.btn_extend_lf.isChecked())

        self.btn_avg_all.setEnabled(any_output and self.btn_per_sofa_lr.isChecked())
        self.dd_algo.setEnabled(any_output and self.btn_per_sofa_lr.isChecked() and self.btn_avg_all.isChecked())
        self.btn_exclude_outliers.setEnabled(any_output and self.btn_per_sofa_lr.isChecked() and self.btn_avg_all.isChecked())

    def get_config(self) -> dict:
        return {
            'enabled': self.btn_enable.get_value(),
            'per_sofa_lr': self.btn_per_sofa_lr.get_value(),
            'per_sofa_avg': self.btn_per_sofa_avg.get_value(),
            'text_filename': self.edit_text_filename.text(),
            'text_delimiter': self.dd_delimiter.get_value(),
            'text_extension': self.dd_text_ext.get_value(),
            'dfhrtf_graphs': self.btn_dfhrtf_graphs.get_value(),
            'stereo_graphs': self.btn_stereo_graphs.get_value(),
            'graph_filename': self.edit_graph_filename.text(),
            'graph_extension': self.dd_graph_ext.get_value(),
            'smoothing': int(self.slider_smoothing.get_value()),
            'tilt': self.slider_tilt.get_value(),
            'extend_lf': self.btn_extend_lf.get_value(),
            'xover': self.slider_xover.get_value(),
            'avg_all': self.btn_avg_all.get_value(),
            'algo': self.dd_algo.get_value(),
            'exclude_outliers': self.btn_exclude_outliers.get_value()
        }
