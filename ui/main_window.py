from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QPlainTextEdit, QSplitter, QFileDialog
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from ui.tabs import FilesTab, SofaTab, DfhrtfTab
from ui.widgets import FlameLabel, FlameToggleButton, FlameLineEdit
from ui.workers import ProcessingWorker
from ui import theme as T
from core import config as C

# Action-bar button styles — named states built from theme constants
_STYLE_BTN_PROCESS = f"background-color: {T.ACCENT}; color: {T.FG}; padding: 12px; font-weight: bold;"
_STYLE_BTN_STOP    = f"background-color: {T.DANGER}; color: {T.FG}; padding: 12px; font-weight: bold;"
_STYLE_BTN_RESET   = (
    f"background-color: {T.BG_CONTROL}; color: {T.FG}; "
    f"padding: 12px; font-weight: bold; min-width: 130px; max-width: 130px;"
)

class SofaBuddyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOFA Buddy v0.2.0")
        self.resize(800, 600)
        self.setStyleSheet(f"background-color: {T.BG_BASE};")

        self.setAcceptDrops(True)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.layout.addWidget(splitter)

        # Top Pane
        top_pane = QWidget()
        top_layout = QVBoxLayout(top_pane)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Navigation Bar
        nav_layout = QHBoxLayout()
        self.btn_nav_files = QPushButton("Files (0)")
        self.btn_nav_sofa = QPushButton("SOFA 🔴")
        self.btn_nav_dfhrtf = QPushButton("DFHRTF 🔴")

        for btn in [self.btn_nav_files, self.btn_nav_sofa, self.btn_nav_dfhrtf]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {T.BG_CONTROL};
                    color: {T.FG};
                    border: none;
                    padding: 0px 10px;
                    min-height: {T.CONTROL_HEIGHT}px;
                    max-height: {T.CONTROL_HEIGHT}px;
                }}
                QPushButton:checked {{
                    background-color: {T.BG_ACTIVE};
                }}
            """)
            btn.setCheckable(True)
            nav_layout.addWidget(btn, stretch=1)

        self.btn_nav_files.setChecked(True)
        top_layout.addLayout(nav_layout)

        # Stacked Widget
        self.stacked_widget = QStackedWidget()
        self.tab_files = FilesTab()
        self.tab_sofa = SofaTab()
        self.tab_dfhrtf = DfhrtfTab()

        self.stacked_widget.addWidget(self.tab_files)
        self.stacked_widget.addWidget(self.tab_sofa)
        self.stacked_widget.addWidget(self.tab_dfhrtf)

        top_layout.addWidget(self.stacked_widget)

        # Output Folder row (global setting — applies to all output)
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 6, 0, 0)
        out_row.setSpacing(8)
        out_row.addWidget(FlameLabel("Output Folder:"))

        self.btn_output_default = FlameToggleButton("Default", padding_h=10)
        self.btn_output_default.setToolTip(
            "<p>Each file's outputs go to <b>sofa_mastered/</b> and <b>DFHRTF/</b> subfolders "
            "inside its own source folder. Global averaging produces a per-folder average.</p>"
        )
        self.btn_output_default.setChecked(True)
        self.btn_output_custom = FlameToggleButton("Custom:", padding_h=10)
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
            f"background-color: {T.BG_CONTROL}; color: {T.FG}; border: none; "
            f"min-height: {T.CONTROL_HEIGHT}px; max-height: {T.CONTROL_HEIGHT}px;"
        )

        self.btn_output_default.toggled.connect(self._on_output_default_toggled)
        self.btn_output_custom.toggled.connect(self._on_output_custom_toggled)
        self.btn_output_browse.clicked.connect(self._browse_output_folder)

        out_row.addWidget(self.btn_output_default)
        out_row.addWidget(self.btn_output_custom)
        out_row.addWidget(self.edit_output_folder, stretch=1)
        out_row.addWidget(self.btn_output_browse)
        top_layout.addLayout(out_row)

        # Action Bar
        action_bar = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Settings")
        self.btn_reset.setStyleSheet(_STYLE_BTN_RESET)
        self.btn_process = QPushButton("Start Processing")
        self.btn_process.setStyleSheet(_STYLE_BTN_PROCESS)
        action_bar.addWidget(self.btn_reset)
        action_bar.addWidget(self.btn_process, stretch=1)
        top_layout.addLayout(action_bar)

        splitter.addWidget(top_pane)

        # Console
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(200)
        self.console.setStyleSheet(
            f"background-color: {T.BG_CONSOLE}; color: {T.FG_CONSOLE}; "
            f"font-family: Roboto Mono; padding: 8px 12px;"
        )
        splitter.addWidget(self.console)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Connections
        self.btn_nav_files.clicked.connect(lambda: self._switch_tab(0))
        self.btn_nav_sofa.clicked.connect(lambda: self._switch_tab(1))
        self.btn_nav_dfhrtf.clicked.connect(lambda: self._switch_tab(2))

        self.tab_sofa.sofa_toggled.connect(self._on_sofa_toggled)
        self.tab_dfhrtf.dfhrtf_toggled.connect(self._on_dfhrtf_toggled)
        self.btn_process.clicked.connect(self._on_process_clicked)
        self.btn_reset.clicked.connect(self._on_reset_clicked)

        # Connect file list model signals to update nav count
        file_list_model = self.tab_files.file_list.model()
        file_list_model.rowsInserted.connect(self._update_file_count)
        file_list_model.rowsRemoved.connect(self._update_file_count)
        file_list_model.modelReset.connect(self._update_file_count)

        # Worker ref
        self.worker = None

        # Restore persisted settings (after all widgets are built)
        self._restore_settings()

    def _restore_settings(self) -> None:
        """Load persisted settings and push them into the tabs."""
        mgr = C.manager()
        cfg = mgr.load()

        # Surface any validation warnings from the user's saved file
        for w in mgr.warnings:
            self.append_error(f"[Config] {w}")

        self.set_output_config(cfg.get('output', {}))
        self.tab_sofa.set_config(cfg.get('sofa', {}))
        self.tab_dfhrtf.set_config(cfg.get('dfhrtf', {}))

    def _on_reset_clicked(self) -> None:
        """Reset all settings to factory defaults and update the tabs."""
        cfg = C.manager().reset()
        self.set_output_config(cfg.get('output', {}))
        self.tab_sofa.set_config(cfg.get('sofa', {}))
        self.tab_dfhrtf.set_config(cfg.get('dfhrtf', {}))
        self.append_log("[Config] Settings reset to defaults.")

    # -- Output Folder row (global setting) --

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

    def get_output_config(self) -> dict:
        return {
            'default': self.btn_output_default.get_value(),
            'folder': self.edit_output_folder.text().strip()
        }

    def set_output_config(self, cfg: dict) -> None:
        """Restore the output-folder row from a saved config dict."""
        use_default = bool(cfg.get('default', True))
        folder = cfg.get('folder', '')

        # Drive through the toggles to keep visual state consistent
        if use_default:
            self._set_toggle(self.btn_output_default, True)
            self._set_toggle(self.btn_output_custom, False)
        else:
            self._set_toggle(self.btn_output_default, False)
            self._set_toggle(self.btn_output_custom, True)
        self.edit_output_folder.setText(folder)
        self._sync_output_enabled()

    def closeEvent(self, event) -> None:
        """Persist current settings when the window is closed."""
        try:
            snap = C.manager().snapshot(self.get_output_config(), self.tab_sofa, self.tab_dfhrtf)
            C.manager().save(snap)
        except Exception as exc:  # noqa: BLE001
            # Never block shutdown due to a config-save failure
            self.append_error(f"[Config] Could not save settings on close: {exc}")
        super().closeEvent(event)

    def _update_file_count(self):
        total = self.tab_files.file_list.count()
        self.btn_nav_files.setText(f"Files ({total})")

    def _switch_tab(self, index):
        self.stacked_widget.setCurrentIndex(index)
        self.btn_nav_files.setChecked(index == 0)
        self.btn_nav_sofa.setChecked(index == 1)
        self.btn_nav_dfhrtf.setChecked(index == 2)

    def _on_sofa_toggled(self, checked):
        state = "🟢" if checked else "🔴"
        self.btn_nav_sofa.setText(f"SOFA {state}")

    def _on_dfhrtf_toggled(self, checked):
        state = "🟢" if checked else "🔴"
        self.btn_nav_dfhrtf.setText(f"DFHRTF {state}")

    def _append_colored(self, text, color):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console.setTextCursor(cursor)
        self.console.setCurrentCharFormat(fmt)
        self.console.appendPlainText(text)
        fmt.setForeground(QColor(T.FG))   # reset so color never bleeds into next line
        self.console.setCurrentCharFormat(fmt)

    def append_error(self, text):
        self._append_colored(text, T.ERROR_TEXT)

    def append_log(self, text):
        s = text.lstrip()
        if s.startswith("[Diagnostics log:"):
            self._append_colored(text, T.OK_TEXT)       # status/info → yellow
        elif "[+]" in text:
            self._append_colored(text, T.SUCCESS_TEXT)  # success → blue
        else:
            self._append_colored(text, T.FG)            # normal → white

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        added = 0
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith('.sofa'):
                self.tab_files.add_file(file_path)
                added += 1
            else:
                self.append_error(f"Error: Skipped invalid file '{file_path}'. Only .sofa files are supported.")

        if added > 0:
            self._switch_tab(0)

    def _on_process_clicked(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.append_log("[!] Stop requested by user. Cleaning up...")
            self.btn_process.setEnabled(False)
            return

        # Snapshot live tab state over cached advanced/averaging sections, then persist.
        # Do this on the main thread before the worker starts (stable dict for the worker).
        mgr = C.manager()
        snap = mgr.snapshot(self.get_output_config(), self.tab_sofa, self.tab_dfhrtf)
        try:
            mgr.save(snap)
        except Exception as exc:  # noqa: BLE001
            self.append_error(f"[Config] Could not save settings: {exc}")

        config = {
            'files':  self.tab_files.get_config(),
            **snap,  # includes output, sofa (with advanced), dfhrtf (with advanced), averaging
        }

        self.btn_process.setText("Stop Processing")
        self.btn_process.setStyleSheet(_STYLE_BTN_STOP)

        self.worker = ProcessingWorker(config)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self.append_error)
        self.worker.start()

    def _on_worker_progress(self, percent, msg):
        self.append_log(f"[{percent}%] {msg}")

    def _on_worker_finished(self, success):
        self.btn_process.setEnabled(True)
        self.btn_process.setText("Start Processing")
        self.btn_process.setStyleSheet(_STYLE_BTN_PROCESS)

        if not success and self.worker and self.worker._is_stopped:
            self.append_log("Processing aborted.")
        else:
            if success:
                self._append_colored("Processing completed.", T.SUCCESS_TEXT)
            else:
                self.append_error("Processing failed.")
