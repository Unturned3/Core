
import sys, traceback, copy
from dataclasses import dataclass
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QWidget,
    QListWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton,
    QListWidgetItem, QSlider,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal, Slot, QTimer, Property

from QtComponents import ImagePanel, ToggleButton, HVBoxLayout, MarkerList

import numpy as np
from numpy.typing import NDArray
import utils

from PolyRenderer import PolyRenderer3D


np.set_printoptions(precision=4, suppress=True)

@dataclass
class PointMarker(QListWidgetItem):
    uid: int
    frame_num: int
    vid_xy: NDArray[np.float64]
    world_xyz: NDArray[np.float64]
    map_xy: NDArray[np.float64] = None

    def __post_init__(self):
        super().__init__()
        self.updateText()

    def updateText(self):
        self.setText(f'{self.uid}')

@dataclass
class PolyMarker(QListWidgetItem):
    uid: int
    points: list[PointMarker]
    H: NDArray[np.float64] = None

    def __post_init__(self):
        super().__init__()
        self.updateText()

    def updateText(self):
        self.setText(f'{self.uid}: {[p.uid for p in self.points]}')

    def compute_H(self):
        A = np.array(
            [p.world_xyz for p in self.points]
        ).reshape(-1, 3)[:, [0, 2]]
        B = np.array(
            [p.map_xy for p in self.points]
        ).reshape(-1, 2)
        self.H, _ = cv2.findHomography(A, B, cv2.RANSAC, 1)

class VideoVisualizer(QMainWindow):

    frameNumChanged = Signal()

    def __init__(self, video_path, map_path=None):
        super().__init__()

        self.setWindowTitle("Visualization")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.real_vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.real_vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_disp_scale = 640 / self.real_vid_w

        self.cpe_orig, self.cpg_orig = utils.load_est_gt_poses(video_path, self.real_vid_w)
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


        self.map_path = map_path

        if self.map_path is not None:
            image = cv2.imread(self.map_path)
            h, *_ = image.shape
            scale = self._vid_disp_size()[1] / h
            self.map_image = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            self.map_image = None


        self.showPolyMarkers = True
        self.showPointMarkers = True
        self.showPoseCross = True

        self.video_is_playing = False
        self.selectedMarker = None
        self.marker_count = 0

        self.poly_count = 0
        self.poly_renderer = PolyRenderer3D(self.real_vid_w, self.real_vid_h)

        self._frame_num = 0
        self.frameNumChanged.connect(self.readNewFrame)
        self.frameNumChanged.connect(self.drawFrame)
        self.frameNumChanged.connect(self.updateFrameNumTextbox)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerTimeout)

        self.initUI()

    def _get_QListWidget_items(self, list_widget):
        return [list_widget.item(i) for i in range(list_widget.count())]

    def _vid_disp_size(self):
        return (
            int(self.real_vid_w * self.vid_disp_scale),
            int(self.real_vid_h * self.vid_disp_scale)
        )

    def _map_disp_size(self):
        if self.map_image is None:
            return (200, self._vid_disp_size()[1])
        return self.map_image.shape[:2][::-1]

    def initUI(self):

        self.setFocus()
        self.setFocusPolicy(Qt.StrongFocus)

        self.video_panel = ImagePanel(self, self._vid_disp_size())
        self.video_panel.clicked.connect(self.onVideoPanelClick)

        self.map_panel = ImagePanel(self, self._map_disp_size())
        self.map_panel.clicked.connect(self.onMapPanelClick)

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

        self.points_list: QListWidget = MarkerList(self)
        self.points_list.itemSelectionChanged.connect(self.onMarkerListSelectionChange)
        self.points_list.itemDoubleClicked.connect(self.onMarkerListItemDoubleClick)

        self.polys_list: QListWidget = MarkerList(self)
        #self.points_list.itemSelectionChanged.connect(self.onMarkerListSelectionChange)
        #self.points_list.itemDoubleClicked.connect(self.onMarkerListItemDoubleClick)

        self.container = QWidget(self)
        self.setCentralWidget(self.container)

        ### UI layout specification ###

        v1 = QVBoxLayout(self.container)

        r1 = QHBoxLayout()
        v1.addLayout(r1)
        r1.addWidget(self.video_panel)
        r1.addWidget(self.map_panel)

        v2 = QVBoxLayout()
        r1.addLayout(v2)
        v2.addWidget(QLabel('Points:'))
        v2.addWidget(self.points_list)
        v2.addWidget(QLabel('Areas:'))
        v2.addWidget(self.polys_list)

        r2 = QHBoxLayout()
        v1.addLayout(r2)
        r2.addWidget(QLabel('Frame'), 0)
        r2.addWidget(self.frame_num_textbox)
        r2.addWidget(QLabel(f'out of {self.n_frames-1}'), 0)
        r2.addStretch(5)
        r2.addWidget(QLabel(f'Create poly:'), 0)
        r2.addWidget(self.create_poly_textbox, 5)
        r2.addWidget(QLabel('Pose reference frame:'), 0)
        r2.addWidget(self.ref_frame_textbox)

        self.readNewFrame()
        self.drawFrame()
        self.drawMap()

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
        h2, w2 = self.real_vid_h / 2, self.real_vid_w / 2

        if self.gt_available:
            pts = np.array([
                [w2 - 20, h2], [w2 + 20, h2], [w2, h2 - 20], [w2, h2 + 20]
            ]).astype(int)

            if idx in self.cpe:
                H = utils.H_between_frames(self.cpe[idx], self.cpg[idx], self.real_vid_w, self.real_vid_h)
                pts2 = utils.project_points(H, pts).astype(int)
                if self.showPoseCross:
                    for i in range(0, 4, 2):
                        cv2.line(frame, tuple(pts2[i]), tuple(pts2[i+1]), (0, 0, 255), 2)

            if self.showPoseCross:
                for i in range(0, 4, 2):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 0), 2)

        lookat = -self.cpe[idx]['R'][:, 2]
        up = self.cpe[idx]['R'][:, 1]
        hfov = self.cpe[idx]['hfov']
        self.poly_renderer.set_cam_pose(lookat, up, hfov)
        overlay = self.poly_renderer.render(self.showPolyMarkers, self.showPointMarkers)

        frame = utils.alpha_blend(overlay, frame)

        frame = cv2.resize(frame, None, fx=self.vid_disp_scale, fy=self.vid_disp_scale)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_panel.setPixmap(pixmap)

    @Slot()
    def drawMap(self):
        if self.map_image is None:
            return
        img  = self.map_image.copy()

        for m in self._get_QListWidget_items(self.points_list):
            if m.map_xy is not None:
                x, y = m.map_xy[0]
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
                cv2.putText(img, str(m.uid), (x+2, y-1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.map_panel.setPixmap(pixmap)

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

        x = pos.x() / self.vid_disp_scale
        y = pos.y() / self.vid_disp_scale
        vid_xy = np.array([[x, y]])
        M = utils.view_to_world(self.cpe[self.frame_num], self.real_vid_w, self.real_vid_h)

        # NOTE: using M to "un-project" the screen points into the world only
        # preserves the direction information. Since we set the homogeneous
        # component of the vector to 1 by convention, this component being
        # interpreted as the z-coordinate of the world point results in the
        # point landing _behind_ the camera. Therefore we negate the vector.
        world_xyz = utils.project_points(
            M, vid_xy,
            keep_z=True) * -1

        if world_xyz[0, 1] >= 0:
            return  # Not a valid point on the virtual ground plane

        # Extend the vector to the ground plane (y = -1)
        world_xyz /= -world_xyz[0, 1]

        if self.selectedMarker is not None:
            m = self.selectedMarker
            m.frame_num = self.frame_num
            m.vid_xy = vid_xy
            m.world_xyz = world_xyz
            m.updateText()
        else:
            self.marker_count += 1
            m = PointMarker(self.marker_count, self.frame_num, vid_xy, world_xyz)
            self.points_list.addItem(m)
        self.poly_renderer.verts[m.uid] = world_xyz.ravel()

        self.drawFrame()

    @Slot()
    def onMapPanelClick(self, pos):
        if not self.selectedMarker:
            return
        x = pos.x()# / self.?_disp_scale
        y = pos.y()# / self.?_disp_scale
        map_xy = np.array([[x, y]])
        self.selectedMarker.map_xy = map_xy
        self.drawMap()

    @Slot()
    def onMarkerListSelectionChange(self):
        length = len(self.points_list.selectedItems())
        assert length in (0, 1)
        self.selectedMarker = self.points_list.selectedItems()[0] if length == 1 else None
        self.setFocus()

    @Slot()
    def onMarkerListItemDoubleClick(self):
        self.frame_num = self.selectedMarker.frame_num
        self.setFocus()

    @Slot()
    def onCreatePolyTextboxReturn(self):
        try:
            self.setFocus()
            ids = [int(i) for i in self.create_poly_textbox.text().split()]
            world_xyzs = []
            pms = {p.uid: p for p in self._get_QListWidget_items(self.points_list)}
            for i in ids:
                if i not in pms.keys():
                    raise ValueError(f"Point {i} not found!")
                world_xyzs.append(pms[i].world_xyz)
            self.poly_count += 1
            self.poly_renderer.create_poly(
                self.poly_count,
                (0.0, 1.0, 0.0, 0.35),
                np.array(world_xyzs),
            )
            self.polys_list.addItem(
                PolyMarker(self.poly_count, [pms[i] for i in ids]))
            self.create_poly_textbox.setText("")
            self.create_poly_textbox.setStyleSheet("")
            self.drawFrame()
        except Exception as e:
            print(e)
            #traceback.print_exc()
            self.create_poly_textbox.setStyleSheet("background-color: pink;")
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setDoubleClickInterval(250)
    window = VideoVisualizer(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
    window.move(100, 100)
    window.show()
    sys.exit(app.exec())
