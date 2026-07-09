import pytest
from PySide6.QtCore import Qt, QMimeData, QUrl
from PySide6.QtGui import QMouseEvent
from ui.widgets import FlameToggleButton, FlameNumericSlider
from ui.tabs import SofaTab, FilesTab
from ui.main_window import SofaBuddyMainWindow
from ui.workers import ProcessingWorker

def test_flame_toggle_button(qtbot):
    btn = FlameToggleButton("Test")
    qtbot.addWidget(btn)
    
    assert not btn.get_value()
    assert "Test" in btn.text()
    
    qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
    
    assert btn.get_value()
    assert "Test" in btn.text()

def test_flame_numeric_slider(qtbot):
    slider = FlameNumericSlider("Test", 0.0, 100.0, 1.0, 50.0)
    qtbot.addWidget(slider)
    
    assert slider.get_value() == 50.0
    
    # Right click opens edit mode
    qtbot.mouseClick(slider, Qt.MouseButton.RightButton)
    qtbot.keyClicks(slider.edit_widget, "25")
    qtbot.keyClick(slider.edit_widget, Qt.Key.Key_Return)
    
    assert slider.get_value() == 25.0
    
    # Double click resets to default
    qtbot.mouseDClick(slider, Qt.MouseButton.LeftButton)
    assert slider.get_value() == 50.0

def test_flame_numeric_slider_discrete(qtbot):
    # Test discrete snapping (allowed_values)
    allowed = [3, 6, 12, 24, 48]
    slider = FlameNumericSlider("1/n", 3, 48, 1, 24, allowed_values=allowed)
    qtbot.addWidget(slider)
    
    assert slider.get_value() == 24
    
    # Mock a drag movement
    # _anchor_y is 100
    # moving mouse UP to global y = 50 -> dy = 100 - 50 = 50.
    # With sensitivity 0.25, delta_idx = (50/5)*0.25 = 2.5 -> round to 3.
    # current_idx of 24 is 3. 3 + 3 = 6. Clamped to index 4.
    # index 4 is 48.
    slider._anchor_y = 100
    slider._anchor_value = 24
    
    # Create mock event with globalPos at 50
    from PySide6.QtCore import QPointF
    pos = QPointF(0, 50)
    event = QMouseEvent(QMouseEvent.Type.MouseMove, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    
    # We found in debug that event.globalPosition().y() was 1265.0 in the test environment 
    # even with pos=50. Let's force anchor_y to be global_y + 50 to get dy=50.
    slider._anchor_y = event.globalPosition().y() + 50
    slider._anchor_value = 24
    
    slider.mouseMoveEvent(event)
    assert slider.get_value() == 48

def test_sofa_tab_config(qtbot):
    tab = SofaTab()
    qtbot.addWidget(tab)
    
    qtbot.mouseClick(tab.btn_enable, Qt.MouseButton.LeftButton)
    
    config = tab.get_config()
    assert config['enabled'] is True
    assert config['samplerates'] == [48000]
    assert config['phase_mode'] == "Original Phase"
    assert config['lf_extension'] is False
    assert config['xover'] == 1000.0
    assert config['normalize_distance'] is True
    assert config['distance_m'] == 1.2
    assert config['hrir_length'] == 512.0
    assert config['naming'] == "{filename}_{samplerate}_{eq}"

def test_dfhrtf_tab_config(qtbot):
    from ui.tabs import DfhrtfTab
    tab = DfhrtfTab()
    qtbot.addWidget(tab)
    
    qtbot.mouseClick(tab.btn_enable, Qt.MouseButton.LeftButton)
    
    config = tab.get_config()
    assert config['enabled'] is True
    assert config['smoothing'] == "Variable"
    assert config['tilt'] == 0.0
    
def test_files_tab_config(qtbot):
    tab = FilesTab()
    qtbot.addWidget(tab)
    
    tab.file_list.addItem("C:/test/file1.sofa")
    tab.file_list.addItem("C:/test/file2.sofa")
    
    config = tab.get_config()
    assert len(config) == 2
    assert "file1.sofa" in config[0]
    assert "file2.sofa" in config[1]

class MockDragEvent:
    def __init__(self, urls):
        self._mimeData = QMimeData()
        self._mimeData.setUrls([QUrl.fromLocalFile(u) for u in urls])
        self.accepted = False
        
    def mimeData(self):
        return self._mimeData
        
    def acceptProposedAction(self):
        self.accepted = True

def test_main_window_drag_drop_valid(qtbot):
    window = SofaBuddyMainWindow()
    qtbot.addWidget(window)
    
    event = MockDragEvent(["C:/fake/path/test.sofa"])
    window.dragEnterEvent(event)
    assert event.accepted
    
    window.dropEvent(event)
    assert window.tab_files.file_list.count() == 1
    assert "Files (1)" in window.btn_nav_files.text()
    
def test_main_window_drag_drop_invalid(qtbot):
    window = SofaBuddyMainWindow()
    qtbot.addWidget(window)
    
    event = MockDragEvent(["C:/fake/path/test.wav"])
    window.dropEvent(event)
    
    assert window.tab_files.file_list.count() == 0
    assert "Error: Skipped invalid file" in window.console.toPlainText()

def test_processing_worker_signals(qtbot):
    config = {'files': ["test1.sofa"]}
    worker = ProcessingWorker(config)
    
    with qtbot.waitSignals([worker.progress, worker.finished], timeout=2000):
        worker.start()