
import sys
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget,
    QListWidget, QGridLayout,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal, Slot, QTimer, Property

from QtComponents import ImagePanel, ToggleButton

class VideoVisualizer(QMainWindow):

    frameNumChanged = Signal()

    def __init__(self, video_path):
        super().__init__()

        self.setWindowTitle("Visualization")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.video_is_playing = False

        self._frame_num = 0
        self.frameNumChanged.connect(self.displayFrame)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerTimeout)

        self.initUI()


    def initUI(self):

        self.setFocus()
        self.setFocusPolicy(Qt.StrongFocus)

        self.video_panel = ImagePanel(self)
        self.video_panel.clicked.connect(self.onVideoPanelClick)

        self.frame_num_textbox = QLineEdit(self)
        self.frame_num_textbox.setText(str(self._frame_num))
        self.frame_num_textbox.returnPressed.connect(self.onFrameInputReturn)
        self.frame_num_textbox.setFocusPolicy(Qt.ClickFocus)

        self.play_button = ToggleButton(self, "Play", "Pause")
        self.play_button.clicked.connect(self.toggle_play_video)

        self.points_list = QListWidget(self)

        grid = QGridLayout()
        grid.addWidget(self.video_panel, 0, 0, 1, 2)
        grid.addWidget(self.frame_num_textbox, 1, 0)
        grid.addWidget(self.play_button, 1, 1)
        grid.addWidget(self.points_list, 0, 2)

        container = QWidget()
        container.setLayout(grid)
        self.setCentralWidget(container)

        self.displayFrame()

    def getFrameNum(self):
        return self._frame_num

    def setFrameNum(self, frame_num):
        try:
            frame_num = int(frame_num)

            if frame_num < 0:
                frame_num = self.n_frames - 1
            elif frame_num >= self.n_frames:
                frame_num = 0

            self.frame_num_textbox.setStyleSheet("")

            # Only seek if necessary (it's expensive)
            if self._frame_num != frame_num - 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            self._frame_num = frame_num
            self.frameNumChanged.emit()
            self.frame_num_textbox.setText(str(self._frame_num))

        except ValueError:
            self.frame_num_textbox.setStyleSheet("background-color: pink;")

    frame_num = Property(int, getFrameNum, setFrameNum, notify=frameNumChanged)

    def keyPressEvent(self, event):
        match event.key():
            case Qt.Key_Right:
                self.frame_num += 1
            case Qt.Key_Left:
                self.frame_num -= 1
            case Qt.Key_Space:
                self.toggle_play_video()

    @Slot()
    def displayFrame(self):
        ret, frame = self.cap.read()
        if ret:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_panel.setPixmap(pixmap)

    @Slot()
    def onFrameInputReturn(self):
        self.setFocus()
        self.frame_num = self.frame_num_textbox.text()

    @Slot()
    def toggle_play_video(self):
        if self.video_is_playing:
            self.timer.stop()
        else:
            self.timer.start(33)
        self.video_is_playing = not self.video_is_playing

    @Slot()
    def timerTimeout(self):
        self.frame_num += 1

    @Slot()
    def onVideoPanelClick(self, pos):
        x = pos.x()
        y = pos.y()
        self.points_list.addItem(f"({x}, {y})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoVisualizer(sys.argv[1])
    window.show()
    sys.exit(app.exec())
