import html
from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QCheckBox, QPushButton, QTextEdit, QApplication)
from PyQt6.QtCore import pyqtSignal

class ExecutionConsole(QWidget):
    """
    Reusable Execution Console widget with real-time colored log output,
    auto-scrolling, clipboard copy, and expand/collapse controls.
    """
    signal_close_requested = pyqtSignal()
    signal_expand_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header Toolbar
        header = QHBoxLayout()
        header.setContentsMargins(2, 0, 2, 0)
        header.setSpacing(8)

        title = QLabel("Terminal & Execution Console")
        title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        header.addWidget(title)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: #7f8c8d; font-size: 11px; padding-left: 4px;")
        header.addWidget(self.status_lbl)

        header.addStretch()

        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.setStyleSheet("color: #34495e; font-size: 11px;")
        header.addWidget(self.auto_scroll_cb)

        copy_btn = QPushButton("Copy All")
        copy_btn.setFixedHeight(24)
        copy_btn.setStyleSheet("""
            QPushButton {
                background: #f1f2f6;
                color: #2f3542;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 11px;
                padding: 2px 8px;
            }
            QPushButton:hover { background: #e4e7eb; }
        """)
        copy_btn.clicked.connect(self.copy_logs)
        header.addWidget(copy_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(24)
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #f1f2f6;
                color: #2f3542;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 11px;
                padding: 2px 8px;
            }
            QPushButton:hover { background: #e4e7eb; color: #e74c3c; }
        """)
        clear_btn.clicked.connect(self.clear_logs)
        header.addWidget(clear_btn)

        self.expand_btn = QPushButton("Expand")
        self.expand_btn.setFixedHeight(24)
        self.expand_btn.setToolTip("Expand terminal to full screen covering workspace")
        self.expand_btn.setStyleSheet("""
            QPushButton {
                background: #34495e;
                color: white;
                border: 1px solid #2c3e50;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 10px;
            }
            QPushButton:hover { background: #415b76; }
        """)
        self.expand_btn.clicked.connect(self.signal_expand_requested.emit)
        header.addWidget(self.expand_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedHeight(24)
        close_btn.setFixedWidth(24)
        close_btn.setToolTip("Hide Terminal (Show again via View menu)")
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #7f8c8d;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background: #e74c3c; color: white; border-color: #c0392b; }
        """)
        close_btn.clicked.connect(self.signal_close_requested.emit)
        header.addWidget(close_btn)

        layout.addLayout(header)

        # Log Window
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setStyleSheet("""
            QTextEdit {
                background-color: #161b22;
                color: #e6edf3;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.4;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.log_window)

    def log(self, message):
        """Append a colored, timestamped log entry."""
        now_str = datetime.now().strftime("%H:%M:%S")
        escaped_msg = html.escape(str(message))

        if "[ERROR]" in escaped_msg:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #ff7b72; font-weight: bold;">{escaped_msg}</span>'
            self.set_status("Last status: Error", "#ff7b72")
        elif "[WARNING]" in escaped_msg:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #d29922; font-weight: bold;">{escaped_msg}</span>'
            self.set_status("Last status: Warning", "#d29922")
        elif ">>>" in escaped_msg or "Starting" in escaped_msg or "Running" in escaped_msg:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #58a6ff; font-weight: bold;">{escaped_msg}</span>'
            self.set_status("Running...", "#58a6ff")
        elif "SUCCESS" in escaped_msg or "completed successfully" in escaped_msg.lower():
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #3fb950; font-weight: bold;">{escaped_msg}</span>'
            self.set_status("Ready (Success)", "#3fb950")
        elif "[INFO]" in escaped_msg:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #79c0ff;">{escaped_msg}</span>'
        else:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #c9d1d9;">{escaped_msg}</span>'

        self.log_window.append(formatted)

        if self.auto_scroll_cb.isChecked():
            sb = self.log_window.verticalScrollBar()
            sb.setValue(sb.maximum())

    def copy_logs(self):
        """Copy all log messages to clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self.log_window.toPlainText())
            self.set_status("Logs copied to clipboard!", "#2ecc71")

    def clear_logs(self):
        """Clear log messages."""
        self.log_window.clear()
        self.set_status("Console cleared", "#7f8c8d")

    def set_status(self, text: str, color_hex: str = "#7f8c8d"):
        """Update the status text and color."""
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color: {color_hex}; font-size: 11px; padding-left: 4px;")
