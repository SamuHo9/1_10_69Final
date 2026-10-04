from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel

class ViewToolbar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet("background-color: #6A82D2; border-bottom: 1px solid #5A72C2;")
        self.setFixedHeight(28)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(2)
        
        pin_btn = QPushButton("📌")
        pin_btn.setFixedSize(24, 24)
        pin_btn.setStyleSheet("border: none; color: black; background: transparent;")
        
        num_label = QLabel("1")
        num_label.setStyleSheet("color: black; font-weight: bold; border: none; padding-left: 2px; padding-right: 2px;")
        
        self.center_btn = QPushButton("⌖")
        self.center_btn.setFixedSize(24, 24)
        self.center_btn.setStyleSheet("border: none; color: black; background: transparent; font-size: 18px;")
        self.center_btn.setToolTip("Reset Camera")
        
        box_btn = QPushButton("☐")
        box_btn.setFixedSize(24, 24)
        box_btn.setStyleSheet("border: none; color: black; background: transparent; font-size: 18px; font-weight: bold;")
        
        layout.addWidget(pin_btn)
        layout.addWidget(num_label)
        layout.addWidget(self.center_btn)
        layout.addWidget(box_btn)
        layout.addStretch()
