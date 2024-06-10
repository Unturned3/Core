
import sys, traceback, copy
from dataclasses import dataclass
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QWidget,
    QListWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton,
    QListWidgetItem,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal, Slot, QTimer, Property

from QtComponents import ImagePanel, ToggleButton, HVBoxLayout, MarkerList

import numpy as np
from numpy.typing import NDArray
import utils

from PolyRenderer import PolyMarker, PolyRenderer3D


np.set_printoptions(precision=4, suppress=True)

@dataclass
class Marker(QListWidgetItem):
    uid: int
    frame_num: int
    screen_xy: NDArray[np.float64]
    world_xyz: NDArray[np.float64]

    def __post_init__(self):
        super().__init__()
        self.updateText()

    def updateText(self):
        fmt_arr = np.array2string(
            self.world_xyz,
            formatter={'float_kind': lambda x: f"{x:.4f}"}
        )
        self.setText(f'{self.uid}, {self.frame_num}, {fmt_arr}')


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
        self.gt_available = self.cpg_orig is not None

        if self.gt_available:
            self.cpe, self.cpg = utils.set_ref_cam(0, self.cpe_orig, self.cpg_orig)
        else:
            self.cpe = copy.deepcopy(self.cpe_orig)

        Xs = [p['R'][:, 0] for p in self.cpe.values()]
        up_vec = utils.compute_up_vector(Xs)

        from scipy.spatial.transform import Rotation as Rot
        R = Rot.align_vectors([[0, 1, 0]], [up_vec])[0].as_matrix()

        for p in self.cpe.values():
            p['R'] = R @ p['R']

        if self.gt_available:
            for p in self.cpg.values():
                p['R'] = R @ p['R']

        self.video_is_playing = False
        self.selectedMarker = None
        self.marker_count = 0

        self.poly_count = 0
        self.poly_renderer = PolyRenderer3D(self.vid_w, self.vid_h)

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

        self.create_poly_textbox = QLineEdit(self)
        self.create_poly_textbox.returnPressed.connect(self.onCreatePolyTextboxReturn)
        self.create_poly_textbox.setFocusPolicy(Qt.ClickFocus)

        self.ref_frame_textbox = QLineEdit(self)
        self.ref_frame_textbox.setFocusPolicy(Qt.ClickFocus)
        self.ref_frame_textbox.setText("0")
        self.ref_frame_textbox.returnPressed.connect(self.onRefFrameTextboxReturn)
        # Disabled for now because we're using a precomputed ground plane
        self.ref_frame_textbox.setEnabled(False)
        self.ref_frame_textbox.setText("(n/a)")

        self.marker_list: QListWidget = MarkerList(self)
        self.marker_list.itemSelectionChanged.connect(self.onMarkerListSelectionChange)
        self.marker_list.itemDoubleClicked.connect(self.onMarkerListItemDoubleClick)

        self.container = QWidget(self)
        self.setCentralWidget(self.container)

        box = HVBoxLayout(self.container)

        box.addWidget(self.video_panel)
        box.addWidget(self.marker_list)
        box.newline()

        box.addWidget(QLabel('Frame'), 0)
        box.addWidget(self.frame_num_textbox)
        box.addWidget(QLabel(f'out of {self.n_frames-1}'), 0)
        box.addStretch(5)
        box.addWidget(QLabel(f'Create poly:'), 0)
        box.addWidget(self.create_poly_textbox, 5)
        box.addWidget(QLabel('Pose reference frame:'), 0)
        box.addWidget(self.ref_frame_textbox)

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

        if self.gt_available:
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

        lookat = -self.cpe[idx]['R'][:, 2]
        up = self.cpe[idx]['R'][:, 1]
        hfov = self.cpe[idx]['hfov']
        self.poly_renderer.set_cam_pose(lookat, up, hfov)
        overlay = self.poly_renderer.render()

        #overlay[:, :, 3] = 255
        #print("min/max:", overlay[:,:,3].min(), overlay[:,:,3].max())
        frame = utils.alpha_blend(overlay, frame)
        #frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

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
        if self.video_is_playing:
            return

        x = pos.x()
        y = pos.y()
        screen_xy = np.array([[x, y]])
        M = utils.view_to_world(self.cpe[self.frame_num], self.vid_w, self.vid_h)

        # NOTE: using M to "un-project" the screen points into the world only
        # preserves the direction information. Since we set the homogeneous
        # component of the vector to 1 by convention, this component being
        # interpreted as the z-coordinate of the world point results in the
        # point landing _behind_ the camera. Therefore we negate the vector.
        world_xyz = utils.project_points(
            M, np.array([[x, y]]),
            keep_z=True) * -1

        if world_xyz[0, 1] >= 0:
            return  # Not a valid point on the virtual ground plane

        # Extend the vector to the ground plane (y = -1)
        world_xyz /= -world_xyz[0, 1]

        if self.selectedMarker is not None:
            m = self.selectedMarker
            m.frame_num = self.frame_num
            m.screen_xy = screen_xy
            m.world_xyz = world_xyz
            m.updateText()
        else:
            self.marker_count += 1
            m = Marker(self.marker_count, self.frame_num, screen_xy, world_xyz)
            self.marker_list.addItem(m)
        self.poly_renderer.verts[m.uid] = world_xyz.ravel()

        self.drawFrame()

    @Slot()
    def onMarkerListSelectionChange(self):
        length = len(self.marker_list.selectedItems())
        assert length in (0, 1)
        self.selectedMarker = self.marker_list.selectedItems()[0] if length == 1 else None

    @Slot()
    def onMarkerListItemDoubleClick(self):
        self.frame_num = self.selectedMarker.frame_num

    @Slot()
    def onCreatePolyTextboxReturn(self):
        try:
            self.setFocus()
            ids = [int(i) for i in self.create_poly_textbox.text().split()]
            world_xyzs = []
            for i in range(self.marker_list.count()):
                m = self.marker_list.item(i)
                if m.uid in ids:
                    world_xyzs.append(m.world_xyz)
            self.poly_count += 1
            self.poly_renderer.create_poly(
                self.poly_count,
                (0.0, 1.0, 0.0, 0.5),
                np.array(world_xyzs),
            )
            self.create_poly_textbox.setText("")
            self.create_poly_textbox.setStyleSheet("")
            self.drawFrame()
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.create_poly_textbox.setStyleSheet("background-color: pink;")
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoVisualizer(sys.argv[1])
    window.show()
    sys.exit(app.exec())
