from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QPlainTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from ui.tabs import FilesTab, SofaTab, DfhrtfTab
from ui.workers import ProcessingWorker

class SofaBuddyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOFA Buddy")
        self.resize(800, 600)
        self.setStyleSheet("""
            background-color: #222222;
        """)
        
        self.setAcceptDrops(True)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        
        # Navigation Bar
        nav_layout = QHBoxLayout()
        self.btn_nav_files = QPushButton("Files (0)")
        self.btn_nav_sofa = QPushButton("SOFA 🔴")
        self.btn_nav_dfhrtf = QPushButton("DFHRTF 🔴")
        
        for btn in [self.btn_nav_files, self.btn_nav_sofa, self.btn_nav_dfhrtf]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #444444; 
                    color: white; 
                    border: none; 
                    padding: 10px;
                }
                QPushButton:checked {
                    background-color: #666666;
                }
            """)
            btn.setCheckable(True)
            nav_layout.addWidget(btn)
            
        self.btn_nav_files.setChecked(True)
            
        self.layout.addLayout(nav_layout)
        
        # Stacked Widget
        self.stacked_widget = QStackedWidget()
        self.tab_files = FilesTab()
        self.tab_sofa = SofaTab()
        self.tab_dfhrtf = DfhrtfTab()
        
        self.stacked_widget.addWidget(self.tab_files)
        self.stacked_widget.addWidget(self.tab_sofa)
        self.stacked_widget.addWidget(self.tab_dfhrtf)
        
        self.layout.addWidget(self.stacked_widget, stretch=3)
        
        # Action Bar
        self.btn_process = QPushButton("Start Processing")
        self.btn_process.setStyleSheet("background-color: #286C9E; color: white; padding: 12px; font-weight: bold;")
        self.layout.addWidget(self.btn_process)
        
        # Console
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #111111; color: #EEEEEE; font-family: Roboto Mono; padding: 8px 12px;")
        self.layout.addWidget(self.console, stretch=1)
        
        # Connections
        self.btn_nav_files.clicked.connect(lambda: self._switch_tab(0))
        self.btn_nav_sofa.clicked.connect(lambda: self._switch_tab(1))
        self.btn_nav_dfhrtf.clicked.connect(lambda: self._switch_tab(2))
        
        self.tab_sofa.sofa_toggled.connect(self._on_sofa_toggled)
        self.tab_dfhrtf.dfhrtf_toggled.connect(self._on_dfhrtf_toggled)
        self.btn_process.clicked.connect(self._on_process_clicked)
        
        # Worker ref
        self.worker = None

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
        fmt.setForeground(QColor("#E06666"))
        
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console.setTextCursor(cursor)
        
        self.console.setCurrentCharFormat(fmt)
        self.console.appendPlainText(text)
        
        fmt.setForeground(QColor("white"))
        self.console.setCurrentCharFormat(fmt)
        
    def append_log(self, text):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("white"))
        
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
            total = self.tab_files.file_list.count()
            self.btn_nav_files.setText(f"Files ({total})")
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
        self.btn_process.setStyleSheet("background-color: #D9534F; color: white; padding: 12px; font-weight: bold;")

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
        self.btn_process.setStyleSheet("background-color: #286C9E; color: white; padding: 12px; font-weight: bold;")

        if not success and self.worker and self.worker._is_stopped:
            self.append_log("Processing aborted.")
        else:
            self.append_log("Processing completed." if success else "Processing failed.")