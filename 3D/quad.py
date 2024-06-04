import trimesh
import pyrender
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from scipy.spatial.transform import Rotation as R
from timer import Timer

cap = cv2.VideoCapture(sys.argv[1])
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

scene = pyrender.Scene(bg_color=[0, 0, 0, 255])

# Step 1: Create a mesh for the rectangle
vertices = np.array([
    [-1, -1, -2],
    [ 1, -1, -2],
    [ 1,  1, -2],
    [-1,  1, -2],
], dtype=np.float64)
faces = np.array([[0, 1, 2], [0, 2, 3]])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.visual.vertex_colors = [0, 255, 0, 255]
scene.add(pyrender.Mesh.from_trimesh(mesh), pose=np.eye(4))

# Step 4: Render the scene
camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)
cam_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=cam_pose)

flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA
r = pyrender.OffscreenRenderer(640, 480)

for i in range(900):

    with Timer():

        #cam_pose[:3, :3] = R.from_euler('YXZ', [i % 30, 0, 0], degrees=True).as_matrix()
        #scene.set_pose(scene.main_camera_node, cam_pose)

        color, depth = r.render(scene, flags)
        #color = cv2.addWeighted(frame, 0.7, color, 0.3, 0.0)

        cv2.imshow('Window', color)
        key = cv2.waitKey(1)
        if key == 27:
            break
