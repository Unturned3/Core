
import sys
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QWidget,
    QListWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal, Slot, QTimer, Property

from QtComponents import ImagePanel, ToggleButton, HVBoxLayout

import numpy as np
import utils

class VideoVisualizer(QMainWindow):

    frameNumChanged = Signal()

    def __init__(self, video_path):
        super().__init__()

        self.setWindowTitle("Visualization")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_scale = 640 / self.vid_w
        self.scaled_size = (640, int(self.vid_h * self.vid_scale))

        self.cpe_orig, self.cpg_orig = utils.load_est_gt_poses(video_path, self.vid_w)
        self.cpe, self.cpg = utils.set_ref_cam(0, self.cpe_orig, self.cpg_orig)

        self.video_is_playing = False

        self._frame_num = 0
        self.frameNumChanged.connect(self.readNewFrame)
        self.frameNumChanged.connect(self.drawFrame)
        self.frameNumChanged.connect(self.updateFrameNumTextbox)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerTimeout)

        self.initUI()


    def initUI(self):

        self.setFocus()
        self.setFocusPolicy(Qt.StrongFocus)

        self.video_panel = ImagePanel(self, self.scaled_size)
        self.video_panel.clicked.connect(self.onVideoPanelClick)

        self.frame_num_textbox = QLineEdit(self)
        self.frame_num_textbox.setText(str(self._frame_num))
        self.frame_num_textbox.returnPressed.connect(self.onFrameNumTextboxReturn)
        self.frame_num_textbox.setFocusPolicy(Qt.ClickFocus)

        #self.play_button = ToggleButton(self, "Play", "Pause")
        #self.play_button.clicked.connect(self.toggle_play_video)

        self.ref_frame_textbox = QLineEdit(self)
        self.ref_frame_textbox.setFocusPolicy(Qt.ClickFocus)
        self.ref_frame_textbox.setText("0")
        self.ref_frame_textbox.returnPressed.connect(self.onRefFrameTextboxReturn)

        self.points_list = QListWidget(self)

        self.container = QWidget(self)
        self.setCentralWidget(self.container)

        box = HVBoxLayout(self.container)
        box.addWidget(self.video_panel)
        box.addWidget(self.points_list)
        box.newline()
        box.addWidget(QLabel('Frame'), 0)
        box.addWidget(self.frame_num_textbox)
        box.addWidget(QLabel(f'out of {self.n_frames-1}'), 0)
        box.addStretch(10)
        box.addWidget(QLabel('Pose reference frame:'), 0)
        box.addWidget(self.ref_frame_textbox)
        #box.addWidget(self.play_button, 12)

        self.readNewFrame()
        self.drawFrame()

    def getFrameNum(self):
        return self._frame_num

    def setFrameNum(self, frame_num):
        frame_num = int(frame_num)

        if frame_num < 0:
            frame_num = self.n_frames - 1
        elif frame_num >= self.n_frames:
            frame_num = 0

        # Only seek if necessary (it's expensive)
        if self._frame_num != frame_num - 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        self._frame_num = frame_num
        self.frameNumChanged.emit()

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
    def readNewFrame(self):
        ret, frame = self.cap.read()
        if ret:
            self.cur_frame = frame

    @Slot()
    def drawFrame(self):
        frame = self.cur_frame.copy()
        idx = self.frame_num
        h2, w2 = self.vid_h / 2, self.vid_w / 2

        pts = np.array([
            [w2 - 20, h2], [w2 + 20, h2], [w2, h2 - 20], [w2, h2 + 20]
        ]).astype(int)

        if idx in self.cpe:
            H = utils.H_between_frames(self.cpe[idx], self.cpg[idx], self.vid_w, self.vid_h)
            pts2 = utils.project_points(H, pts).astype(int)
            for i in range(0, 4, 2):
                cv2.line(frame, tuple(pts2[i]), tuple(pts2[i+1]), (0, 0, 255), 2)

        for i in range(0, 4, 2):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 0), 2)

        frame = cv2.resize(frame, self.scaled_size)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_panel.setPixmap(pixmap)

    @Slot()
    def updateFrameNumTextbox(self):
        self.frame_num_textbox.setStyleSheet("")
        self.frame_num_textbox.setText(str(self._frame_num))

    @Slot()
    def onFrameNumTextboxReturn(self):
        self.setFocus()
        try:
            self.frame_num = self.frame_num_textbox.text()
            self.frame_num_textbox.setStyleSheet("")
        except:
            self.frame_num_textbox.setStyleSheet("background-color: pink;")

    @Slot()
    def onRefFrameTextboxReturn(self):
        self.setFocus()
        try:
            ref_idx = int(self.ref_frame_textbox.text())
            self.cpe, self.cpg = utils.set_ref_cam(ref_idx, self.cpe_orig, self.cpg_orig)
            self.ref_frame_textbox.setStyleSheet("")
        except:
            self.ref_frame_textbox.setStyleSheet("background-color: pink;")
            return
        self.drawFrame()

    @Slot()
    def toggle_play_video(self):
        if self.video_is_playing:
            self.timer.stop()
        else:
            self.timer.start(int(1000 / self.vid_fps))
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
