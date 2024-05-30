
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Signal, QPoint

class ImagePanel(QLabel):

    clicked = Signal(QPoint)

    def __init__(self, parent=None, size=(640, 480)):
        super().__init__(parent)
        self.setCursor(Qt.CrossCursor)
        self.setMaximumSize(*size)
        self.setAlignment(Qt.AlignmentFlag.AlignTop)

    def mousePressEvent(self, event):
        pos = event.pos()
        if event.button() == Qt.LeftButton:
            self.clicked.emit(pos)
        super().mousePressEvent(event)

