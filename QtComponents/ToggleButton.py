
from PySide6.QtWidgets import QPushButton

class ToggleButton(QPushButton):
    def __init__(self, parent, unchecked_text, checked_text):
        super().__init__(parent)
        self.unchecked_text = unchecked_text
        self.checked_text = checked_text
        self.setCheckable(True)
        self.setChecked(False)
        self.setText(self.unchecked_text)

        self.clicked.connect(self.update_text)

    def update_text(self):
        if self.isChecked():
            self.setText(self.checked_text)
        else:
            self.setText(self.unchecked_text)
