from PyQt6.QtWidgets import QTableWidget
from PyQt6.QtCore import Qt

class ToggleTableWidget(QTableWidget):
    """QTableWidget supporting ExtendedSelection (Ctrl/Shift multi-select)
    and single-click toggle/deselect when clicking an already selected sole row."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                row = item.row()
                modifiers = event.modifiers()
                has_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

                selected_rows = list(set(it.row() for it in self.selectedItems()))

                if not has_ctrl and not has_shift:
                    if selected_rows == [row]:
                        self.clearSelection()
                        return
                    elif row in selected_rows and len(selected_rows) > 1:
                        self.clearSelection()
                        self.selectRow(row)
                        return

        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        # Prevent Left/Right arrow keys from scrolling columns; route to ResultPanel SD stepper!
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight,
                           Qt.Key.Key_Comma, Qt.Key.Key_Period, Qt.Key.Key_0, Qt.Key.Key_R, Qt.Key.Key_Home, Qt.Key.Key_Space):
            parent = self.parentWidget()
            while parent and not hasattr(parent, 'step_sd'):
                parent = parent.parentWidget()
            if parent and hasattr(parent, 'step_sd'):
                parent.keyPressEvent(event)
                event.accept()
                return
        super().keyPressEvent(event)
