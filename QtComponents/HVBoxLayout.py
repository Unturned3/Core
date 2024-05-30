
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QSizePolicy, QLayout
from PySide6.QtCore import Qt

class HVBoxLayout(QVBoxLayout):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_line = QHBoxLayout()
        super().addLayout(self.current_line)

    def addWidget(self, widget, proportion=1):
        #widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.current_line.addWidget(widget, proportion)

    def addLayout(self, layout, proportion=1):
        #layout.setSizeConstraint(QLayout.SetMinimumSize)
        self.current_line.addLayout(layout, proportion)

    def addStretch(self, proportion=1):
        #layout.setSizeConstraint(QLayout.SetMinimumSize)
        self.current_line.addStretch(proportion)

    def newline(self):
        self.current_line = QHBoxLayout()
        super().addLayout(self.current_line)
