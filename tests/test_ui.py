import pytest
from PySide6.QtCore import Qt, QMimeData, QUrl
from PySide6.QtGui import QMouseEvent
from ui.widgets import FlameToggleButton, FlameNumericSlider
from ui.tabs import SofaTab, FilesTab
from ui.main_window import SofaBuddyMainWindow

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

def test_sofa_tab_enable_toggle_is_isolated(qtbot):
    """Clicking the master toggle flips only `enabled` and nothing else.

    Behavioral, not value-based: what the individual controls *default* to is
    pinned against default_settings.json in tests/test_widget_defaults.py, so
    duplicating those literals here would just re-create the drift this suite
    is meant to catch.
    """
    tab = SofaTab()
    qtbot.addWidget(tab)

    before = tab.get_config()
    assert before['enabled'] is False
    assert tab.container.isEnabled() is False

    qtbot.mouseClick(tab.btn_enable, Qt.MouseButton.LeftButton)

    after = tab.get_config()
    assert after['enabled'] is True
    assert tab.container.isEnabled() is True
    assert {k: v for k, v in after.items() if k != 'enabled'} == \
           {k: v for k, v in before.items() if k != 'enabled'}

    # ...and back again.
    qtbot.mouseClick(tab.btn_enable, Qt.MouseButton.LeftButton)
    assert tab.get_config() == before
    assert tab.container.isEnabled() is False


def test_sofa_tab_enable_toggle_emits_signal(qtbot):
    tab = SofaTab()
    qtbot.addWidget(tab)

    seen = []
    tab.sofa_toggled.connect(seen.append)

    qtbot.mouseClick(tab.btn_enable, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(tab.btn_enable, Qt.MouseButton.LeftButton)
    assert seen == [True, False]


def test_dfhrtf_tab_enable_toggle_is_isolated(qtbot):
    from ui.tabs import DfhrtfTab
    tab = DfhrtfTab()
    qtbot.addWidget(tab)

    before = tab.get_config()
    assert before['enabled'] is False

    qtbot.mouseClick(tab.btn_enable, Qt.MouseButton.LeftButton)

    after = tab.get_config()
    assert after['enabled'] is True
    assert tab.container.isEnabled() is True
    assert {k: v for k, v in after.items() if k != 'enabled'} == \
           {k: v for k, v in before.items() if k != 'enabled'}

    qtbot.mouseClick(tab.btn_enable, Qt.MouseButton.LeftButton)
    assert tab.get_config() == before


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


def test_files_tab_shows_elevation_range(qtbot, tmp_path, synthetic_sofa):
    import numpy as np
    import sofar as sf

    els = np.arange(-40.0, 81.0, 20.0)
    az = np.tile(np.arange(0.0, 360.0, 30.0), len(els))
    s = synthetic_sofa(n_points=len(az))
    s.SourcePosition = np.column_stack((az, np.repeat(els, 12), np.ones(len(az))))
    path = str(tmp_path / "partial.sofa")
    sf.write_sofa(path, s)

    tab = FilesTab()
    qtbot.addWidget(tab)
    tab._show_single_file_info(path)
    text = tab.lbl_elevation.text()
    assert text.startswith("Elevation range:")
    assert "40" in text and "+80" in text
    assert "full sphere" not in text

    tab._clear_single_file_info()
    assert tab.lbl_elevation.text() == "Elevation range: -"
