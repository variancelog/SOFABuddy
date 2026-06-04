from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QPlainTextEdit, QSplitter
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from ui.tabs import FilesTab, SofaTab, DfhrtfTab
from ui.workers import ProcessingWorker
from ui import theme as T

# Process button styles — two named states, built from theme constants
_STYLE_BTN_PROCESS = f"background-color: {T.ACCENT}; color: {T.FG}; padding: 12px; font-weight: bold;"
_STYLE_BTN_STOP    = f"background-color: {T.DANGER}; color: {T.FG}; padding: 12px; font-weight: bold;"

class SofaBuddyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOFA Buddy")
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

        # Action Bar
        self.btn_process = QPushButton("Start Processing")
        self.btn_process.setStyleSheet(_STYLE_BTN_PROCESS)
        top_layout.addWidget(self.btn_process)

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

        # Connect file list model signals to update nav count
        file_list_model = self.tab_files.file_list.model()
        file_list_model.rowsInserted.connect(self._update_file_count)
        file_list_model.rowsRemoved.connect(self._update_file_count)
        file_list_model.modelReset.connect(self._update_file_count)

        # Worker ref
        self.worker = None

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

    def append_error(self, text):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(T.ERROR_TEXT))

        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console.setTextCursor(cursor)

        self.console.setCurrentCharFormat(fmt)
        self.console.appendPlainText(text)

        fmt.setForeground(QColor(T.FG))
        self.console.setCurrentCharFormat(fmt)

    def append_log(self, text):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(T.FG))

        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console.setTextCursor(cursor)

        self.console.setCurrentCharFormat(fmt)
        self.console.appendPlainText(text)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        added = 0
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith('.sofa'):
                self.tab_files.file_list.addItem(file_path)
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

        config = {
            'files': self.tab_files.get_config(),
            'sofa': self.tab_sofa.get_config(),
            'dfhrtf': self.tab_dfhrtf.get_config()
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
            self.append_log("Processing completed." if success else "Processing failed.")
