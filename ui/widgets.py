from PyQt6.QtWidgets import QPushButton, QComboBox, QWidget, QLabel, QLineEdit, QHBoxLayout
from PyQt6.QtCore import pyqtSignal, Qt, QRect, QVariantAnimation
from PyQt6.QtGui import QMouseEvent, QPainter, QColor

class FlameToggleButton(QPushButton):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self._label = label
        self.update_style()
        self.toggled.connect(self.update_style)

    def update_style(self):
        checked = self.isChecked()
        if checked:
            self.setText(f"{self._label}")
            self.setStyleSheet("""
                QPushButton {
                    background-color: #286C9E;
                    color: white;
                    border-left: 4px #5aacdb;
                    border-style: inset;
                    padding: 8px 16px; /* top left bottom right */
                    min-height: 32px;
                }
                QPushButton:disabled {
                    color: #111111;
                    background-color: #404952;
                    border-left: 4px #404952;
                    border-style: inset;
                    padding: 8px 16px; /* top left bottom right */
                    min-height: 32px;
                }
            """)
        else:
            self.setText(f"{self._label}")
            self.setStyleSheet("""
                QPushButton {
                    background-color: #444444;
                    color: white;
                    border-left: 4px #888888;
                    border-style: inset;
                    padding: 8px 16px; /* top left bottom right */
                    min-height: 32px;
                }
                QPushButton:disabled {
                    color: #111111;
                    border-left: 4px #444444;
                    border-style: inset;
                    padding: 8px 16px; /* top left bottom right */
                    min-height: 32px;
                }
            """)

    def get_value(self) -> bool:
        return self.isChecked()

class FlameDropdown(QComboBox):
    def __init__(self, label: str = None, parent=None):
        super().__init__(parent)
        self._label = label

        self.setStyleSheet("""
            QComboBox {
                background-color: #444444;
                color: white;
                border: none;
                padding: 8px 16px; /* top left bottom right */
                min-height: 32px;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border: none;
            }
            
            QComboBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #5aacdb; 
                width: 0px;
                height: 0px;
                margin-right: 16px; 
            }   
            QComboBox::down-arrow:disabled {
                border-top-color: #111111;
            }
                           
            QComboBox:disabled {
                color: #111111;
            }
        """)

    def get_value(self):
        return self.currentData() if self.currentData() is not None else self.currentText()

class FlameLabel(QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("color: white;")

class FlameLineEdit(QLineEdit):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QLineEdit {
                background-color: #444444; 
                color: white; 
                padding: 8px 16px; /* top left bottom right */
                min-height: 32px;
                border-top: 0px;
                border-left: 0px;
                border-right: 0px;
                border-bottom: 1px inset #666666;
            }
            QLineEdit:disabled {
                color: #111111;
                border-top: 0px;
                border-left: 0px;
                border-right: 0px;
                border-bottom: 0px;
            }
        """)

# --- Custom Label to handle the drawing of the value bar ---
class FlameSliderLabel(QLabel):
    def __init__(self, slider_instance, parent=None):
        super().__init__(parent)
        self.slider = slider_instance
        self._visual_pct = 0.0
        
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(50)
        self.anim.valueChanged.connect(self._on_anim_step)
        
        self.setStyleSheet("""
            QLabel {
                color: white; 
                background-color: transparent; 
                padding: 8px 16px; /* top left bottom right */
                min-height: 32px;
            }
            QLabel:disabled {
                color: #111111;
            }
        """)

    def set_initial_pct(self, pct):
        self._visual_pct = pct
        self.update()

    def _on_anim_step(self, value):
        self._visual_pct = value
        self.update()

    def update_target_pct(self, target_pct):
        self.anim.stop()
        self.anim.setStartValue(self._visual_pct)
        self.anim.setEndValue(target_pct)
        self.anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        
        # 1. Draw the base background
        # Use #444444 like the other widgets
        bg_color = QColor("#444444")
        painter.fillRect(self.rect(), bg_color)

        # 2. Draw the dynamic indicator bar if enabled
        if self.isEnabled():
            bar_height = self.height() # Fill full height
            bar_width = int(self.width() * self._visual_pct)
            
            bar_rect = QRect(0, self.height() - bar_height, bar_width, bar_height)
            
            # Use darker blue #286C9E for background fill to keep text readable
            painter.fillRect(bar_rect, QColor("#355775"))
        
        painter.end()

        # 3. Let the QLabel draw its text ON TOP of our manual background/bar
        super().paintEvent(event)


class FlameNumericSlider(QWidget):
    def __init__(self, label: str, min_val: float, max_val: float, step: float, default_val: float, allowed_values: list = None, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.default_val = default_val
        self.value = default_val
        self.allowed_values = allowed_values
        self._last_y = None
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Replace the standard QLabel with our custom drawn label
        self.label_widget = FlameSliderLabel(self)
        
        self.edit_widget = QLineEdit()
        self.edit_widget.setStyleSheet("color: white; background-color: #286C9E; padding: 8px 16px; min-height: 32px; border-bottom: 1px #111111 inset")
        self.edit_widget.hide()
        
        self.layout.addWidget(self.label_widget)
        self.layout.addWidget(self.edit_widget)
        
        # Set initial visual pct before first update
        rng = self.max_val - self.min_val
        pct = 0.0
        if rng > 0:
            pct = (self.value - self.min_val) / rng
        self.label_widget.set_initial_pct(max(0.0, min(1.0, pct)))

        self.update_display()
        self.edit_widget.editingFinished.connect(self._on_edit_finished)
        
    def _clamp_to_step(self, val):
        if self.allowed_values:
            return min(self.allowed_values, key=lambda x: abs(x - val))
        
        steps = round((val - self.min_val) / self.step)
        clamped = self.min_val + (steps * self.step)
        return max(self.min_val, min(self.max_val, clamped))

    def update_display(self):
        val_str = f"{self.value:.2f}" if isinstance(self.step, float) and self.step < 1 else f"{int(self.value)}"
        self.label_widget.setText(f"{self.label_text}: {val_str}")
        
        # Calculate target pct and update label animation
        rng = self.max_val - self.min_val
        pct = 0.0
        if rng > 0:
            pct = (self.value - self.min_val) / rng
        self.label_widget.update_target_pct(max(0.0, min(1.0, pct)))
        
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._anchor_y = event.globalPosition().y()
            self._anchor_value = self.value
            self._dragged = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.label_widget.hide()
            self.edit_widget.setText(str(self.value))
            self.edit_widget.show()
            self.edit_widget.setFocus()
            self.edit_widget.selectAll()
            
    def mouseMoveEvent(self, event: QMouseEvent):
        if hasattr(self, '_anchor_y') and self._anchor_y is not None:
            dy = self._anchor_y - event.globalPosition().y()
            if abs(dy) > 2:
                self._dragged = True
                modifiers = event.modifiers()
                if modifiers & Qt.KeyboardModifier.ShiftModifier:
                    sensitivity = 0.10
                elif modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
                    sensitivity = 0.50
                else:
                    sensitivity = 0.25
                
                if self.allowed_values:
                    # Drag by index for discrete values
                    try:
                        current_idx = self.allowed_values.index(self._anchor_value)
                    except ValueError:
                        # Fallback if anchor is somehow not in list
                        current_idx = min(range(len(self.allowed_values)), key=lambda i: abs(self.allowed_values[i] - self._anchor_value))
                    
                    delta_idx = (dy / 5.0) * sensitivity
                    new_idx = int(round(current_idx + delta_idx))
                    new_idx = max(0, min(len(self.allowed_values) - 1, new_idx))
                    self.value = self.allowed_values[new_idx]
                else:
                    # Standard continuous dragging
                    delta_steps = (dy / 5.0) * sensitivity
                    new_value = self._anchor_value + (delta_steps * self.step)
                    self.value = self._clamp_to_step(new_value)
                
                self.update_display()
                
    def mouseReleaseEvent(self, event: QMouseEvent):
        self._anchor_y = None
        self._anchor_value = None
        
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.value = self.default_val
            self.update_display()
            self._anchor_y = None
            self._dragged = True

    def _on_edit_finished(self):
        try:
            new_val = float(self.edit_widget.text())
            self.value = self._clamp_to_step(new_val)
        except ValueError:
            pass
        self.edit_widget.hide()
        self.label_widget.show()
        self.update_display()
        
    def get_value(self):
        if isinstance(self.step, float):
            return float(self.value)
        return int(self.value)