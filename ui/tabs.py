import os

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton, QGroupBox, QLabel, QGridLayout, QLineEdit, QFileDialog, QMenu
from PySide6.QtCore import Signal, Qt

from ui.widgets import FlameToggleButton, FlameDropdown, FlameNumericSlider, FlameLabel, FlameLineEdit
from ui.theme import BG_WIDGET, BG_CONTROL, ACCENT, FG, CONTROL_HEIGHT
from core.sofa_utils import (get_sofa_metadata, calculate_max_itd, calculate_grid_radius,
                             get_grid_cartesian, format_elevation_span, format_elevation_range,
                             draw_elevation_rings, apply_unit_sphere_axes)
from core.config import load_defaults, schema_bounds, schema_default


class _FilePathItem(QListWidgetItem):
    """List item that displays the full path but sorts by filename (basename)."""
    def __lt__(self, other):
        return os.path.basename(self.text()).lower() < os.path.basename(other.text()).lower()


# --- Files Tab Widget ---
class FilesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self._itd_cache = {}    # Cache for calculated ITDs
        self._radius_cache = {} # Cache for calculated grid radii
        self._grid_viewers = [] # Keep refs to open grid-viewer figures/callbacks

        # --- Top content: file list + right panel ---
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        # - File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.file_list.setSortingEnabled(True)
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
        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_remove_selected.setToolTip("Remove the selected file from the list")
        self.btn_remove_selected.setEnabled(False)
        self.btn_clear_list = QPushButton("Clear All")
        self.btn_clear_list.setToolTip("Clear all files from the list")

        btn_style = (
            f"background-color: {BG_CONTROL}; color: {FG}; border: none; "
            f"padding: 0px 10px; "
            f"min-height: {CONTROL_HEIGHT}px; max-height: {CONTROL_HEIGHT}px; "
            f"margin-top: 10px; margin-left: 5px;"
        )
        for btn in [self.btn_open_files, self.btn_open_folder,
                    self.btn_remove_selected, self.btn_clear_list]:
            btn.setStyleSheet(btn_style)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(0)
        btn_grid.addWidget(self.btn_open_files,      0, 0)
        btn_grid.addWidget(self.btn_open_folder,     0, 1)
        btn_grid.addWidget(self.btn_remove_selected, 1, 0)
        btn_grid.addWidget(self.btn_clear_list,      1, 1)
        right_panel.addLayout(btn_grid)

        self.btn_open_files.clicked.connect(self._open_files)
        self.btn_open_folder.clicked.connect(self._open_folder)
        self.btn_remove_selected.clicked.connect(self._remove_selected)
        self.btn_clear_list.clicked.connect(self._clear_list)

        # Right-click context menu on the file list
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._show_file_context_menu)

        self.info_group = QGroupBox("File Info")
        self.info_group.setStyleSheet(f"color: {FG}; border: 1px solid {BG_CONTROL}; margin-top: 10px;")
        info_layout = QVBoxLayout(self.info_group)
        self.lbl_samplerate = FlameLabel("Samplerate: -")
        self.lbl_length = FlameLabel("Length: -")
        self.lbl_points = FlameLabel("Points: -")
        self.lbl_elevation = FlameLabel("Elevation range: -")

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

        # View Grid Row
        grid_view_layout = QHBoxLayout()
        grid_view_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_view_grid = QPushButton("View Grid")
        self.btn_view_grid.setToolTip("Plot the measurement grid for the selected SOFA file")
        self.btn_view_grid.setStyleSheet(
            f"background-color: {ACCENT}; color: {FG}; border: none; font-size: 10px; padding: 2px;"
        )
        self.btn_view_grid.setEnabled(False)
        self.btn_view_grid.clicked.connect(self._on_view_grid_clicked)
        grid_view_layout.addWidget(self.btn_view_grid)
        grid_view_layout.addStretch()

        info_layout.addWidget(self.lbl_samplerate)
        info_layout.addWidget(self.lbl_length)
        info_layout.addWidget(self.lbl_points)
        info_layout.addWidget(self.lbl_elevation)
        info_layout.addLayout(itd_layout)
        info_layout.addLayout(radius_layout)
        info_layout.addLayout(grid_view_layout)

        right_panel.addWidget(self.info_group)
        right_panel.addStretch()

        content.addLayout(right_panel, stretch=1)
        self.layout.addLayout(content)

        self.file_list.itemSelectionChanged.connect(self._on_file_selection_changed)

    def add_file(self, path: str) -> None:
        """Add a single path to the (sorted, filename-keyed) file list."""
        self.file_list.addItem(_FilePathItem(path))

    def _clear_list(self):
        self.file_list.clear()
        self._itd_cache.clear()
        self._radius_cache.clear()

    def _remove_selected(self):
        for item in self.file_list.selectedItems():
            path = item.text()
            self._itd_cache.pop(path, None)
            self._radius_cache.pop(path, None)
            self.file_list.takeItem(self.file_list.row(item))

    def _show_file_context_menu(self, pos):
        item = self.file_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        act_remove = menu.addAction("Remove")
        if menu.exec(self.file_list.mapToGlobal(pos)) is act_remove:
            self._remove_selected()

    def _clear_single_file_info(self, samplerate_text="Samplerate: -"):
        self.lbl_samplerate.setText(samplerate_text)
        self.lbl_length.setText("Length: -")
        self.lbl_points.setText("Points: -")
        self.lbl_elevation.setText("Elevation range: -")
        self.lbl_itd_val.setText("-")
        self.btn_calc_itd.hide()
        self.lbl_itd_val.show()
        self.lbl_radius_val.setText("-")
        self.btn_calc_radius.hide()
        self.lbl_radius_val.show()

    def _show_calc_buttons(self):
        """Surface the ITD/radius Calculate buttons (used for multi-select batch calc)."""
        self.lbl_itd_val.hide()
        self.btn_calc_itd.setText("Calculate")
        self.btn_calc_itd.setEnabled(True)
        self.btn_calc_itd.show()
        self.lbl_radius_val.hide()
        self.btn_calc_radius.setText("Calculate")
        self.btn_calc_radius.setEnabled(True)
        self.btn_calc_radius.show()

    def _show_single_file_info(self, file_path):
        meta = get_sofa_metadata(file_path)
        if not meta:
            self._clear_single_file_info("Samplerate: ERROR")
            self.btn_view_grid.setEnabled(False)
            return

        self.lbl_samplerate.setText(f"Samplerate: {int(meta['samplerate'])} Hz")
        self.lbl_length.setText(f"Length: {meta['length']} samples")
        self.lbl_points.setText(f"Points: {meta['points']}")
        cov = meta.get('coverage')
        if cov is not None:
            self.lbl_elevation.setText(
                "Elevation range: "
                + format_elevation_span(cov.el_min, cov.el_max, cov.full_sphere))
        else:
            self.lbl_elevation.setText("Elevation range: -")

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

        self.btn_view_grid.setEnabled(True)

    def _on_file_selection_changed(self):
        selected = self.file_list.selectedItems()

        if not selected:
            self._clear_single_file_info()
            self.btn_view_grid.setEnabled(False)
            self.btn_remove_selected.setEnabled(False)
            return

        self.btn_remove_selected.setEnabled(True)

        if len(selected) == 1:
            self._show_single_file_info(selected[0].text())
            return

        # Multi-select: panel can't show every file. Show a placeholder, but keep
        # the Calculate buttons live so the user can batch-calc all selected files.
        self._clear_single_file_info("Samplerate: Multiple selected")
        self._show_calc_buttons()
        self.btn_view_grid.setEnabled(True)

    def _on_calc_itd_clicked(self):
        selected = self.file_list.selectedItems()
        if not selected:
            return

        self.btn_calc_itd.setText("...")
        self.btn_calc_itd.setEnabled(False)
        self.repaint()  # Force UI update to show "..."

        for item in selected:
            file_path = item.text()
            self._itd_cache[file_path] = calculate_max_itd(file_path)

        if len(selected) == 1:
            self.lbl_itd_val.setText(f"{self._itd_cache[selected[0].text()]:.2f} ms")
            self.btn_calc_itd.hide()
            self.lbl_itd_val.show()
        else:
            self.btn_calc_itd.setText("Calculate")
            self.btn_calc_itd.setEnabled(True)

    def _on_calc_radius_clicked(self):
        selected = self.file_list.selectedItems()
        if not selected:
            return

        self.btn_calc_radius.setText("...")
        self.btn_calc_radius.setEnabled(False)
        self.repaint()  # Force UI update to show "..."

        for item in selected:
            file_path = item.text()
            self._radius_cache[file_path] = calculate_grid_radius(file_path)

        if len(selected) == 1:
            self.lbl_radius_val.setText(f"{self._radius_cache[selected[0].text()]:.2f} m")
            self.btn_calc_radius.hide()
            self.lbl_radius_val.show()
        else:
            self.btn_calc_radius.setText("Calculate")
            self.btn_calc_radius.setEnabled(True)

    def _on_view_grid_clicked(self):
        selected = self.file_list.selectedItems()
        if not selected:
            return

        # Load grids for all selected files (skip any that fail to read).
        grids = []
        for item in selected:
            file_path = item.text()
            grid = get_grid_cartesian(file_path)
            if grid is not None:
                stem = os.path.splitext(os.path.basename(file_path))[0]
                grids.append((stem, grid))
        if not grids:
            return

        import matplotlib
        matplotlib.use('QtAgg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        state = {'idx': 0}

        def draw(i):
            stem, grid = grids[i]
            cart = grid['cart']
            ax.clear()
            ax.scatter(cart[:, 0], cart[:, 1], cart[:, 2], c='black', s=4)
            suffix = f"  ({i + 1}/{len(grids)}  ←/→)" if len(grids) > 1 else ""
            title = f"{grid['radius']:g} m radius, {grid['n_nodes']} nodes{suffix}"
            cov = grid.get('coverage')
            if cov is not None:
                draw_elevation_rings(ax, cov)
                title += "\n" + format_elevation_range(
                    cov.el_min, cov.el_max, cov.full_sphere)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            apply_unit_sphere_axes(ax)
            try:
                fig.canvas.manager.set_window_title(stem)
            except Exception:
                pass  # headless or backend without window manager
            fig.canvas.draw_idle()

        def on_key(event):
            if event.key == 'right':
                state['idx'] = (state['idx'] + 1) % len(grids)
                draw(state['idx'])
            elif event.key == 'left':
                state['idx'] = (state['idx'] - 1) % len(grids)
                draw(state['idx'])

        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        # Keep references alive so the key callback keeps firing.
        self._grid_viewers.append((fig, on_key, cid))
        draw(0)
        plt.show(block=False)

    def _open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select SOFA files", "", "SOFA Files (*.sofa *.SOFA)")
        for f in files:
            self.add_file(f)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            for f in os.listdir(folder):
                if f.lower().endswith('.sofa'):
                    self.add_file(os.path.join(folder, f))

    def get_config(self) -> list:
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]

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
        self.btn_sr_96k = FlameToggleButton("96k")
        self.btn_sr_96k.setToolTip("<p>Include 96kHz SOFA file. More useful for bat HRTFs than human ones.</p>")
        sr_layout.addWidget(self.btn_sr_44k)
        sr_layout.addWidget(self.btn_sr_48k)
        sr_layout.addWidget(self.btn_sr_96k)

        lf_xover_layout = QHBoxLayout()
        self.btn_lf = FlameToggleButton("Extend LF")
        self.btn_lf.setToolTip("<p>Replaces HRIR magnitudes below the crossover frequency with an 'ideal' frequency response. Recommended for most measured SOFAs.</p>")
        _xmin, _xmax = schema_bounds("sofa.xover", (200.0, 500.0))
        self.slider_xover = FlameNumericSlider(
            "Crossover Hz", _xmin, _xmax, 5.0, schema_default("sofa", "xover")
        )
        self.slider_xover.setToolTip("<p>Crossover frequency for Extend LF function. 1000Hz works best in most cases.</p>")
        self.btn_lf.toggled.connect(self.slider_xover.setEnabled)
        # Enable-state is owned entirely by set_config() (called at the end of
        # __init__) — never hand-set here, or it desyncs from lf_extension.
        lf_xover_layout.addWidget(self.btn_lf)
        lf_xover_layout.addWidget(self.slider_xover)

        eq_layout = QHBoxLayout()
        eq_layout.setSpacing(10)
        self.btn_uneq = FlameToggleButton("Un-EQ'd")
        self.btn_uneq.setToolTip("<p>Include SOFAs with unequalized frequency response. Useful to obtain HRTF/DFHRTF responses.</p>")
        self.btn_dfeq = FlameToggleButton("DF-EQ'd")
        self.btn_dfeq.setToolTip("<p>Include SOFAs with diffuse-field equalized freqency response. Useful in binaural rendering applications and to view DTFs.</p>")
        eq_layout.addWidget(self.btn_uneq)
        eq_layout.addWidget(self.btn_dfeq)

        dist_layout = QHBoxLayout()
        self.btn_dist = FlameToggleButton("Normalize Grid Radius")
        self.btn_dist.setToolTip("<p>Normalizes grid radius to a user-defined distance. Useful for comparing HRTFs across labs. Enabling this setting overrides the default peak normalization (useful to prevent clipping in binaural renderers).</p>")
        _dmin, _dmax = schema_bounds("sofa.distance_m", (1.0, 2.0))
        self.slider_dist = FlameNumericSlider(
            "m", _dmin, _dmax, 0.1, schema_default("sofa", "distance_m")
        )
        self.slider_dist.setToolTip("<p>Grid radius in metres.</p>")
        self.btn_dist.toggled.connect(self.slider_dist.setEnabled)
        dist_layout.addWidget(self.btn_dist)
        dist_layout.addWidget(self.slider_dist)

        ild_layout = QHBoxLayout()
        self.btn_ild = FlameToggleButton("Balance ILDs")
        self.btn_ild.setToolTip("<p>Correct left/right diffuse-field level (ILD) calibration offsets in the mastered SOFA. A split-level scalar gain is applied symmetrically to L and R, preserving ITD and per-direction HRIR structure. Only corrects when the estimated ILD exceeds the threshold.</p>")
        _imin, _imax = schema_bounds("sofa.ild_threshold", (0.1, 3.0))
        self.slider_ild = FlameNumericSlider(
            "dB", _imin, _imax, 0.1, schema_default("sofa", "ild_threshold"),
            allowed_values=[
                round(v * 0.1, 1)
                for v in range(int(round(_imin * 10)), int(round(_imax * 10)) + 1)
            ]
        )
        self.slider_ild.setToolTip("<p>ILD correction threshold in dB. Files whose estimated diffuse-field ILD (300–4000 Hz) exceeds this value are corrected. Calibration offsets are typically ≥ 1.5 dB; genuine acoustic ILD is usually &lt; 0.5 dB.</p>")
        self.btn_ild.toggled.connect(self.slider_ild.setEnabled)
        ild_layout.addWidget(self.btn_ild)
        ild_layout.addWidget(self.slider_ild)

        _lmin, _lmax = schema_bounds("sofa.hrir_length", (128.0, 1024.0))
        self.slider_length = FlameNumericSlider(
            "Samples", float(_lmin), float(_lmax), 128.0,
            schema_default("sofa", "hrir_length")
        )
        self.slider_length.setToolTip(
            "<p>Output HRIR sample length. 256 (the default) is recommended for most "
            "cases — with the 1 kHz LF crossover it is spectrally equivalent to 512 "
            "while halving the output size. Longer lengths only help if you need the "
            "extra time-domain tail.</p>"
        )

        self.lbl_naming = FlameLabel("SOFA Naming:")
        self.edit_naming = FlameLineEdit(schema_default("sofa", "naming"))
        self.edit_naming.setToolTip("<p>Naming of output SOFA files. Can include static text and the following variables: {filename} (original SOFA filename); {folder} (source file's parent folder name); {samplerate} (samplerate of processed SOFA file); {eq} (uneq or dfeq)</p>")

        # --- Apply configurations from above
        grid.addWidget(FlameLabel("Samplerates:"), 0, 0)
        grid.addLayout(sr_layout, 0, 1)

        grid.addWidget(FlameLabel("Low Frequency:"), 1, 0)
        grid.addLayout(lf_xover_layout, 1, 1)

        grid.addWidget(FlameLabel("Equalization:"), 2, 0)
        grid.addLayout(eq_layout, 2, 1)

        grid.addWidget(FlameLabel("ILD:"), 3, 0)
        grid.addLayout(ild_layout, 3, 1)

        grid.addWidget(FlameLabel("Distance:"), 4, 0)
        grid.addLayout(dist_layout, 4, 1)

        grid.addWidget(FlameLabel("HRIR Length:"), 5, 0)
        grid.addWidget(self.slider_length, 5, 1)
        grid.addWidget(self.lbl_naming, 6, 0)
        grid.addWidget(self.edit_naming, 6, 1)

        self.container_layout.addLayout(grid)
        self.layout.addWidget(self.container)
        self.layout.addStretch()

        # Belt: the factory defaults always reach the user through set_config(),
        # which is also what handles the dependent enable-state syncing. No
        # construction literal above can be what the user actually sees.
        self.set_config(load_defaults().get("sofa", {}))

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
            'ild_fix': self.btn_ild.get_value(),
            'ild_threshold': self.slider_ild.get_value(),
            'normalize_distance': self.btn_dist.get_value(),
            'distance_m': self.slider_dist.get_value(),
            'hrir_length': self.slider_length.get_value(),
            'naming': self.edit_naming.text()
        }

    def set_config(self, cfg: dict) -> None:
        """Restore all SOFA tab widgets from a saved config dict (surfaced keys only)."""
        srs = cfg.get('samplerates', schema_default('sofa', 'samplerates'))
        self.btn_sr_44k.set_value(44100 in srs)
        self.btn_sr_48k.set_value(48000 in srs)
        self.btn_sr_96k.set_value(96000 in srs)

        lf_extension      = bool(cfg.get('lf_extension', schema_default('sofa', 'lf_extension')))
        ild_fix           = bool(cfg.get('ild_fix', schema_default('sofa', 'ild_fix')))
        normalize_distance = bool(cfg.get('normalize_distance',
                                          schema_default('sofa', 'normalize_distance')))

        self.btn_lf.set_value(lf_extension)
        self.slider_xover.set_value(cfg.get('xover', schema_default('sofa', 'xover')))
        self.btn_uneq.set_value(bool(cfg.get('uneq', schema_default('sofa', 'uneq'))))
        self.btn_dfeq.set_value(bool(cfg.get('dfeq', schema_default('sofa', 'dfeq'))))
        self.btn_ild.set_value(ild_fix)
        self.slider_ild.set_value(cfg.get('ild_threshold', schema_default('sofa', 'ild_threshold')))
        self.btn_dist.set_value(normalize_distance)
        self.slider_dist.set_value(cfg.get('distance_m', schema_default('sofa', 'distance_m')))
        self.slider_length.set_value(cfg.get('hrir_length', schema_default('sofa', 'hrir_length')))
        self.edit_naming.setText(cfg.get('naming', schema_default('sofa', 'naming')))

        # Sync slider enable states
        self.slider_xover.setEnabled(lf_extension)
        self.slider_ild.setEnabled(ild_fix)
        self.slider_dist.setEnabled(normalize_distance)

        # Enable/disable the whole container (fires sofa_toggled signal)
        enabled = bool(cfg.get('enabled', schema_default('sofa', 'enabled')))
        self.btn_enable.set_value(enabled)
        self.container.setEnabled(enabled)
        self.sofa_toggled.emit(enabled)

# --- DFHRTF Tab Widget
class DfhrtfTab(QWidget):
    dfhrtf_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 10, 0, 0)
        self.layout.setSpacing(0)

        self.btn_enable = FlameToggleButton("DHRTF Outputs")
        self.btn_enable.setToolTip("<p>Enables DHRTF outputs. At least one text output or one graph output must be selected. NOTE: If SOFA outputs are enabled, source SOFA for DFHRTF outpus the processed SOFA files; otherwise, source is imported SOFA files on Files tab.</p>")
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
        self.btn_per_sofa_avg = FlameToggleButton("Per SOFA Avg DFHRTFs")
        self.btn_per_sofa_avg.setToolTip("<p>Generates average of left ear + right ear DFHRTF magnitute response text files per source SOFA file.</p>")
        self.dd_per_sofa_avg_algo = FlameDropdown("Avg Algo")
        self.dd_per_sofa_avg_algo.setToolTip("<p>Averaging algorithm used for the per-SOFA L/R average. See README for details.</p>")
        self.dd_per_sofa_avg_algo.addItems(["Arithmetic Mean", "Geo-Arc-Length", "Soft-DTW"])
        self.dd_per_sofa_avg_algo.set_value(schema_default("dfhrtf", "per_sofa_avg_algo"))

        self.edit_text_filename = FlameLineEdit(schema_default("dfhrtf", "text_filename"))
        self.edit_text_filename.setToolTip(
            "<p>Naming of generated text files. Supported variables:</p> "
            "<p style='margin-top: 0px; margin-left: 10px;'>"
            "&bull; {filename} source SOFA filename;<br>"
            "&bull; {folder} source file's parent folder name;<br>"
            "&bull; {samplerate} source SOFA samplerate;<br>"
            "&bull; {tilt} spectral tilt in dB/oct;<br>"
            "&bull; {LR} L/R/Avg.;<br>"
            "&bull; {algo} averaging algorithm (global average only).</p>"
        )
        self.dd_delimiter = FlameDropdown("Delimiter")
        self.dd_delimiter.setToolTip("<p>Text file delimiter.</p>")
        self.dd_delimiter.addItems(["Tab", "Comma", "Semicolon", "Space"])
        self.dd_delimiter.set_value(schema_default("dfhrtf", "text_delimiter"))
        self.dd_text_ext = FlameDropdown("Extension")
        self.dd_text_ext.setToolTip("<p>Text file extension</p>")
        self.dd_text_ext.addItems(["txt", "csv"])
        self.dd_text_ext.set_value(schema_default("dfhrtf", "text_extension"))

        # --- Graph Outputs Section ---
        self.btn_dfhrtf_graphs = FlameToggleButton("DFHRTF Graphs")
        self.btn_dfhrtf_graphs.setToolTip("<p>Enables diffuse field HRTF graph outputs. Left and right ears overlayed on one graph.</p>")

        self.dd_weighting_mode = FlameDropdown("Grid Weighting")
        self.dd_weighting_mode.setToolTip(
            "<p><b>Auto</b>: uses geometric (solid-angle area) weights for irregular or "
            "hemispherical grids; automatically switches to uniform 1/N when the grid is a "
            "genuine spherical t/n-design (verified by a spherical-harmonic quadrature test).<br>"
            "<b>Geometric</b>: always use ConvexHull solid-angle weights.<br>"
            "<b>Uniform</b>: always use equal 1/N weights.</p>"
        )
        self.dd_weighting_mode.addItems(["Auto", "Geometric", "Uniform"])
        self.dd_weighting_mode.set_value(schema_default("dfhrtf", "weighting_mode"))

        self.btn_weighted_grid = FlameToggleButton("Weighted Grid")
        self.btn_weighted_grid.setToolTip(
            "<p>Plots the solid-angle-weighted measurement grid used in the DFHRTF calculation. "
            "Shows the cleaned point set with markers sized and colored by weight.</p>"
        )
        self.dd_weighted_scope = FlameDropdown("Scope")
        self.dd_weighted_scope.setToolTip(
            "<p><b>Per Folder</b>: one representative grid plot per source folder.<br>"
            "<b>All Files</b>: one grid plot per SOFA file.</p>"
        )
        self.dd_weighted_scope.addItems(["Per Folder", "All Files"])
        self.dd_weighted_scope.set_value(schema_default("dfhrtf", "weighted_grid_scope"))

        self.edit_graph_filename = FlameLineEdit(schema_default("dfhrtf", "graph_filename"))
        self.edit_graph_filename.setToolTip(
            "<p>Naming of generated graph files. Supported variables:</p> "
            "<p style='margin-top: 0px; margin-left: 10px;'>"
            "&bull; {filename} source SOFA filename;<br>"
            "&bull; {folder} source file's parent folder name;<br>"
            "&bull; {samplerate} source SOFA samplerate;<br>"
            "&bull; {tilt} spectral tilt in dB/oct;<br>"
            "&bull; {graphtype} df / stereo / weighted.</p>"
        )
        self.dd_graph_ext = FlameDropdown("Extension")
        self.dd_graph_ext.setToolTip("<p>Graph file extension.</p>")
        self.dd_graph_ext.addItems(["png", "jpg"])
        self.dd_graph_ext.set_value(schema_default("dfhrtf", "graph_extension"))

        # --- Refinement Section ---
        self.dd_smoothing = FlameDropdown("Smoothing")
        self.dd_smoothing.addItems(["Variable", "1/3", "1/6", "1/12", "1/24", "1/48"])
        self.dd_smoothing.set_value(schema_default("dfhrtf", "smoothing"))
        self.dd_smoothing.setToolTip(
            "<p>Smoothing applied to both text and graph outputs. "
            "'Variable' uses frequency-dependent smoothing (tunable via the "
            "dfhrtf.advanced.variable_smoothing_* keys in the settings JSON); "
            "'1/n' applies a fixed 1/n octave smoothing.</p>"
        )
        _tmin, _tmax = schema_bounds("dfhrtf.tilt", (-2.0, 1.0))
        self.slider_tilt = FlameNumericSlider(
            " Tilt dB/Oct", _tmin, _tmax, 0.1, schema_default("dfhrtf", "tilt")
        )
        self.slider_tilt.setToolTip("<p>Tilt dB/Oct, center frequency of 1kHz. Applies to both text and graph outputs.</p>")

        self.btn_extend_lf = FlameToggleButton("Extend LF")
        self.btn_extend_lf.setToolTip("<p>Extend and linerizes low frequency (LF) response below the crossover frequency. Not needed for simulated SOFA files or if Extend LF already applied to measured SOFA files.</p>")
        _xmin, _xmax = schema_bounds("dfhrtf.xover", (200.0, 500.0))
        self.slider_xover = FlameNumericSlider(
            "Crossover Hz", _xmin, _xmax, 5.0, schema_default("dfhrtf", "xover")
        )
        self.slider_xover.setToolTip("<p>Crossover frequency for low frequency extenstion.</p>")

        # --- Post Processing Section ---
        self.btn_avg_all = FlameToggleButton("Average All Ls/Rs")
        self.btn_avg_all.setToolTip("<p>Averages all left ear and all right ear text outputs. Requires 'Per SOFA L/R' to be enabled above.</p>")
        self.dd_algo = FlameDropdown("Algo")
        self.dd_algo.setToolTip("<p>Averaging Algorithm used for averaging frequency responses. See README file for more information.</p>")
        self.dd_algo.addItems(["Arithmetic Mean", "Geo-Arc-Length", "Soft-DTW"])
        self.dd_algo.set_value(schema_default("dfhrtf", "algo"))
        self.btn_exclude_outliers = FlameToggleButton("Exclude Outliers")
        self.btn_exclude_outliers.setToolTip("<p>Excludes measureuments that exceed STDEV > 1.0</p>")

        # --- Connections for Dependency State ---
        for toggle in [self.btn_per_sofa_lr, self.btn_per_sofa_avg, self.btn_dfhrtf_graphs, self.btn_weighted_grid, self.btn_extend_lf, self.btn_avg_all]:
            toggle.toggled.connect(self._update_dependencies)

        # --- Layout Mapping ---
        grid.addWidget(FlameLabel("Text Outputs:"), 0, 0)
        row0 = QHBoxLayout()
        row0.addWidget(self.btn_per_sofa_lr)
        row0.addWidget(self.btn_per_sofa_avg)
        row0.addWidget(self.dd_per_sofa_avg_algo)
        grid.addLayout(row0, 0, 1)

        row1 = QHBoxLayout()
        row1.addWidget(self.edit_text_filename)
        row1.addWidget(self.dd_delimiter)
        row1.addWidget(self.dd_text_ext)
        grid.addLayout(row1, 1, 1)

        grid.addWidget(FlameLabel("Graph Outputs:"), 2, 0)
        row2 = QHBoxLayout()
        row2.addWidget(self.btn_dfhrtf_graphs)
        row2.addWidget(self.dd_weighting_mode)
        row2.addWidget(self.btn_weighted_grid)
        row2.addWidget(self.dd_weighted_scope)
        grid.addLayout(row2, 2, 1)

        row3 = QHBoxLayout()
        row3.addWidget(self.edit_graph_filename)
        row3.addWidget(self.dd_graph_ext)
        grid.addLayout(row3, 3, 1)

        grid.addWidget(FlameLabel("Refinement:"), 4, 0)
        row4 = QHBoxLayout()
        row4.addWidget(self.dd_smoothing)
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

        # Belt: the factory defaults always reach the user through set_config(),
        # which also re-runs _update_dependencies() so every dependent
        # enable-state is derived, never hand-set at construction.
        self.set_config(load_defaults().get("dfhrtf", {}))

    def _on_enable_toggled(self, checked):
        self.container.setEnabled(checked)
        self.dfhrtf_toggled.emit(checked)

    def _update_dependencies(self):
        any_text = self.btn_per_sofa_lr.isChecked() or self.btn_per_sofa_avg.isChecked()
        any_graph = self.btn_dfhrtf_graphs.isChecked() or self.btn_weighted_grid.isChecked()
        any_output = any_text or any_graph

        # Text Output Config
        self.edit_text_filename.setEnabled(any_text)
        self.dd_delimiter.setEnabled(any_text)
        self.dd_text_ext.setEnabled(any_text)

        # Graph Output Config
        self.edit_graph_filename.setEnabled(any_graph)
        self.dd_graph_ext.setEnabled(any_graph)
        self.dd_weighted_scope.setEnabled(self.btn_weighted_grid.isChecked())

        # Refinement & Post Processing
        self.dd_weighting_mode.setEnabled(any_output)
        self.dd_smoothing.setEnabled(any_output)
        self.slider_tilt.setEnabled(any_output)
        self.btn_extend_lf.setEnabled(any_output)
        self.slider_xover.setEnabled(any_output and self.btn_extend_lf.isChecked())

        self.dd_per_sofa_avg_algo.setEnabled(self.btn_per_sofa_avg.isChecked())
        self.btn_avg_all.setEnabled(any_output and self.btn_per_sofa_lr.isChecked())
        self.dd_algo.setEnabled(any_output and self.btn_per_sofa_lr.isChecked() and self.btn_avg_all.isChecked())
        self.btn_exclude_outliers.setEnabled(any_output and self.btn_per_sofa_lr.isChecked() and self.btn_avg_all.isChecked())

    def get_config(self) -> dict:
        return {
            'enabled': self.btn_enable.get_value(),
            'per_sofa_lr': self.btn_per_sofa_lr.get_value(),
            'per_sofa_avg': self.btn_per_sofa_avg.get_value(),
            'per_sofa_avg_algo': self.dd_per_sofa_avg_algo.get_value(),
            'text_filename': self.edit_text_filename.text(),
            'text_delimiter': self.dd_delimiter.get_value(),
            'text_extension': self.dd_text_ext.get_value(),
            'dfhrtf_graphs': self.btn_dfhrtf_graphs.get_value(),
            'weighting_mode': self.dd_weighting_mode.get_value(),
            'weighted_grid': self.btn_weighted_grid.get_value(),
            'weighted_grid_scope': self.dd_weighted_scope.get_value(),
            'graph_filename': self.edit_graph_filename.text(),
            'graph_extension': self.dd_graph_ext.get_value(),
            'smoothing': self.dd_smoothing.get_value(),
            'tilt': self.slider_tilt.get_value(),
            'extend_lf': self.btn_extend_lf.get_value(),
            'xover': self.slider_xover.get_value(),
            'avg_all': self.btn_avg_all.get_value(),
            'algo': self.dd_algo.get_value(),
            'exclude_outliers': self.btn_exclude_outliers.get_value()
        }

    def set_config(self, cfg: dict) -> None:
        """Restore all DFHRTF tab widgets from a saved config dict (surfaced keys only)."""
        def _d(key):
            return cfg.get(key, schema_default('dfhrtf', key))

        self.btn_per_sofa_lr.set_value(bool(_d('per_sofa_lr')))
        self.btn_per_sofa_avg.set_value(bool(_d('per_sofa_avg')))
        self.dd_per_sofa_avg_algo.set_value(_d('per_sofa_avg_algo'))
        self.edit_text_filename.setText(_d('text_filename'))
        self.dd_delimiter.set_value(_d('text_delimiter'))
        self.dd_text_ext.set_value(_d('text_extension'))

        self.btn_dfhrtf_graphs.set_value(bool(_d('dfhrtf_graphs')))
        self.dd_weighting_mode.set_value(_d('weighting_mode'))
        self.btn_weighted_grid.set_value(bool(_d('weighted_grid')))
        self.dd_weighted_scope.set_value(_d('weighted_grid_scope'))
        self.edit_graph_filename.setText(_d('graph_filename'))
        self.dd_graph_ext.set_value(_d('graph_extension'))

        self.dd_smoothing.set_value(_d('smoothing'))
        self.slider_tilt.set_value(_d('tilt'))
        self.btn_extend_lf.set_value(bool(_d('extend_lf')))
        self.slider_xover.set_value(_d('xover'))

        self.btn_avg_all.set_value(bool(_d('avg_all')))
        self.dd_algo.set_value(_d('algo'))
        self.btn_exclude_outliers.set_value(bool(_d('exclude_outliers')))

        # Re-run gating once after bulk restore so enabled/disabled state is correct
        self._update_dependencies()

        # Enable/disable the whole container (fires dfhrtf_toggled signal)
        enabled = bool(_d('enabled'))
        self.btn_enable.set_value(enabled)
        self.container.setEnabled(enabled)
        self.dfhrtf_toggled.emit(enabled)
