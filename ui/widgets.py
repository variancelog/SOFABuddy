from PySide6.QtWidgets import QPushButton, QComboBox, QWidget, QLabel, QLineEdit, QHBoxLayout
from PySide6.QtCore import Qt, QRect, QVariantAnimation
from PySide6.QtGui import QMouseEvent, QPainter, QColor

from ui import theme as T

class FlameToggleButton(QPushButton):
    def __init__(self, label: str, padding_h: int = 0, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self._label = label
        self._padding_h = padding_h
        self.update_style()
        self.toggled.connect(self.update_style)

    def update_style(self):
        checked = self.isChecked()
        bg     = T.ACCENT       if checked else T.BG_CONTROL
        border = T.ACCENT_LIGHT if checked else T.BORDER_UNSEL
        dis_bg = T.DISABLED_SEL_BG if checked else T.DISABLED_UNSEL_BG
        padding = f"padding-left: {self._padding_h}px; padding-right: {self._padding_h}px;" if self._padding_h else ""
        self.setText(self._label)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: {T.FG};
                border-left: 4px {border};
                border-style: inset;
                min-height: {T.CONTROL_HEIGHT}px;
                max-height: {T.CONTROL_HEIGHT}px;
                {padding}
            }}
            QPushButton:disabled {{
                color: {T.FG_DISABLED};
                background-color: {dis_bg};
                border-left: 4px {dis_bg};
                border-style: inset;
                min-height: {T.CONTROL_HEIGHT}px;
                max-height: {T.CONTROL_HEIGHT}px;
                {padding}
            }}
        """)

    def get_value(self) -> bool:
        return self.isChecked()

    def set_value(self, value: bool) -> None:
        """Set the toggle state without firing dependent signals."""
        self.blockSignals(True)
        self.setChecked(bool(value))
        self.update_style()
        self.blockSignals(False)

class FlameDropdown(QComboBox):
    def __init__(self, label: str = None, parent=None):
        super().__init__(parent)
        self._label = label

        self.setStyleSheet(f"""
            QComboBox {{
                background-color: {T.BG_CONTROL};
                color: {T.FG};
                border: none;
                padding: 0px 14px;
                min-height: {T.CONTROL_HEIGHT}px;
                max-height: {T.CONTROL_HEIGHT}px;
            }}

            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border: none;
            }}

            QComboBox::down-arrow {{
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {T.ACCENT_LIGHT};
                width: 0px;
                height: 0px;
                margin-right: 14px;
            }}
            QComboBox::down-arrow:disabled {{
                border-top-color: {T.FG_DISABLED};
            }}

            QComboBox:disabled {{
                color: {T.FG_DISABLED};
            }}
        """)

    def get_value(self):
        return self.currentData() if self.currentData() is not None else self.currentText()

    def set_value(self, value) -> None:
        """Select the item matching *value* (by data first, then text). Falls back to index 0."""
        # Try matching by data
        idx = self.findData(value)
        if idx == -1:
            # Try matching by text
            idx = self.findText(str(value))
        if idx == -1:
            idx = 0
        self.blockSignals(True)
        self.setCurrentIndex(idx)
        self.blockSignals(False)

class FlameLabel(QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"color: {T.FG}; background-color: transparent;")

class FlameLineEdit(QLineEdit):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"""
            QLineEdit {{
                background-color: {T.BG_CONTROL};
                color: {T.FG};
                min-height: {T.CONTROL_HEIGHT}px;
                max-height: {T.CONTROL_HEIGHT}px;
                padding: 0px 14px;
                border-top: 0px;
                border-left: 0px;
                border-right: 0px;
                border-bottom: 1px inset {T.BG_ACTIVE};
            }}
            QLineEdit:disabled {{
                color: {T.FG_DISABLED};
                border-top: 0px;
                border-left: 0px;
                border-right: 0px;
                border-bottom: 0px;
            }}
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

        self.setStyleSheet(f"""
            QLabel {{
                color: {T.FG};
                background-color: transparent;
                padding: 0px 14px;
                min-height: {T.CONTROL_HEIGHT}px;
                max-height: {T.CONTROL_HEIGHT}px;
            }}
            QLabel:disabled {{
                color: {T.FG_DISABLED};
                min-height: {T.CONTROL_HEIGHT}px;
                max-height: {T.CONTROL_HEIGHT}px;
            }}
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
        painter.fillRect(self.rect(), QColor(T.BG_CONTROL))

        # 2. Draw the dynamic indicator bar if enabled
        if self.isEnabled():
            bar_height = self.height()
            bar_width = int(self.width() * self._visual_pct)
            bar_rect = QRect(0, self.height() - bar_height, bar_width, bar_height)
            painter.fillRect(bar_rect, QColor(T.ACCENT_FILL))

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
        self.edit_widget.setStyleSheet(f"""
            color: {T.FG};
            background-color: {T.ACCENT};
            padding: 7px 14px;
            min-height: {T.CONTROL_HEIGHT}px;
            max-height: {T.CONTROL_HEIGHT}px;
            border-bottom: 1px {T.BG_CONSOLE} inset;
        """)
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

    def set_value(self, value) -> None:
        """Programmatically set the slider value (clamped to step/allowed_values)."""
        try:
            self.value = self._clamp_to_step(float(value))
        except (TypeError, ValueError):
            self.value = self.default_val
        self.update_display()
