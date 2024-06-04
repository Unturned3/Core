import trimesh
import pyrender
import matplotlib.pyplot as plt
import numpy as np

scene = pyrender.Scene(bg_color=[0, 0, 0, 0])

# Step 1: Create a mesh for the rectangle
vertices = np.array([
    [-1, -1, -2],
    [ 1, -1, -2],
    [ 1,  1, -2],
    [-1,  1, -2]
])
faces = np.array([[0, 1, 2], [0, 2, 3]])
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Step 2: Create line segments for the edges of the quad
edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
lines = pyrender.LineSegments(positions=vertices, indices=edges, mode=pyrender.LineSegmentsMode.LINES)

# Step 3: Set the color of the lines to green
material = pyrender.MetallicRoughnessMaterial(emissive=[0, 1, 0])
lines.material = material

# Step 4: Add the lines to the scene
scene.add(lines)

# Step 5: Render the scene
camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)
cam_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=cam_pose)

r = pyrender.OffscreenRenderer(640, 480)
color, depth = r.render(scene)

# Step 6: Display the image using matplotlib
plt.imshow(color)
plt.show()