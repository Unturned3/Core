
from PySide6.QtWidgets import (
    QWidget, QSlider, QSizePolicy, QHBoxLayout, QLineEdit)
from PySide6.QtCore import Qt, Signal


class SliderWithTextBox(QWidget):

    # Set up a signal to emit the current value of the slider
    value = Signal(int)

    def __init__(self, parent, min_value=0, max_value=100, initial_value=0,
                 slider_proportion=90, text_box_proportion=10):

        super().__init__(parent)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setValue(initial_value)

        self.slider.setMinimumHeight(30)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.text_box = QLineEdit()
        self.text_box.setFocusPolicy(Qt.ClickFocus)
        self.text_box.setText(str(initial_value))

        self.slider.valueChanged.connect(self.update_text_box)
        self.text_box.textChanged.connect(self.update_slider)

        layout = QHBoxLayout()
        layout.addWidget(self.slider, slider_proportion)
        layout.addWidget(self.text_box, text_box_proportion)

        self.setLayout(layout)

    def update_text_box(self, value, passive=True):
        self.text_box.setText(str(value))
        if not passive:
            self.value.emit(value)

    def update_slider(self):
        text = self.text_box.text()
        if text.isdigit():
            value = int(text)
            self.slider.setValue(value)

    def update_value(self, value):
        self.update_text_box(value, passive=False)
