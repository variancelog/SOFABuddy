from PyQt6.QtCore import QThread, pyqtSignal
from core.sofa_processor import process_sofa_exports

class ProcessingWorker(QThread):
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        from core.dfhrtf_processor import process_dfhrtf_exports
        try:
            files = self.config.get('files', [])
            if not files:
                self.error.emit("No files selected for processing.")
                self.finished.emit(False)
                return

            def stop_checker():
                return self._is_stopped

            # 1. Run SOFA Processing
            sofa_success = process_sofa_exports(
                self.config, 
                lambda p, m: self.progress.emit(int(p * 0.5), f"SOFA: {m}"),
                self.log.emit, 
                self.error.emit,
                stop_checker
            )
            
            if self._is_stopped:
                self.progress.emit(0, "Processing stopped by user.")
                self.finished.emit(False)
                return

            if not sofa_success:
                self.finished.emit(False)
                return

            # 2. Run DFHRTF Analysis
            dfhrtf_success = process_dfhrtf_exports(
                self.config,
                lambda p, m: self.progress.emit(50 + int(p * 0.5), f"DFHRTF: {m}"),
                self.log.emit,
                self.error.emit,
                stop_checker
            )
            
            if self._is_stopped:
                self.progress.emit(0, "Processing stopped by user.")
                self.finished.emit(False)
                return

            if dfhrtf_success:
                self.progress.emit(100, "Processing complete.")
            
            self.finished.emit(dfhrtf_success)
            
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False)