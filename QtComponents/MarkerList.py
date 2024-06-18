from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QLineEdit,
    QWidget,
    QListWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QListWidgetItem,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal, Slot, QTimer, Property


class MarkerList(QListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSelectionMode(QListWidget.SingleSelection)

    def mousePressEvent(self, event):
        pass

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.pos())
        if item is not None:
            self.itemDoubleClicked.emit(item)

    def mouseReleaseEvent(self, event):
        item = self.itemAt(event.pos())
        if item is not None:
            item.setSelected(not item.isSelected())

    def mouseMoveEvent(self, event):
        """
        This is disabled due to the following: The behavior of
        mouseReleaseEvent is very strange. If I press the mouse on a list widget
        item, then drag the mouse (still on the list widget item) without
        releasing the left button, it counts as a release. When I actually
        release the left mouse button after I stop dragging, it counts as
        another release.
        """
        pass
