import moderngl
import numpy as np
import cv2
import time
from timer import Timer

# Define the viewport size
width, height = 640, 480

# Create a context with moderngl
ctx = moderngl.create_standalone_context()

# Define vertex and fragment shaders
vertex_shader = '''
#version 330

in vec3 in_vert;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(in_vert, 1.0);
}
'''

fragment_shader = '''
#version 330

out vec4 fragColor;

void main() {
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
'''

# Compile shaders and create a program
prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

# Define the quad vertices
vertices = np.array([
    [-0.1, -0.1, -0.5],
    [ 0.1, -0.1, -0.5],
    [ 0.1,  0.1, -0.5],
    [-0.1,  0.1, -0.5],
], dtype='f4')

# Create a Vertex Buffer Object (VBO)
vbo = ctx.buffer(vertices.tobytes())

# Define the indices for the wireframe (GL_LINES)
indices = np.array([
    0, 1,
    1, 2,
    2, 3,
    3, 0
], dtype='i4')

# Create an Index Buffer Object (IBO)
ibo = ctx.buffer(indices.tobytes())

# Create a Vertex Array Object (VAO)
vao = ctx.vertex_array(prog, [(vbo, '3f', 'in_vert')], index_buffer=ibo)

def perspective(hfov_deg, aspect, near, far):
    # Convert hfov from degrees to radians
    hfov_rad = np.radians(hfov_deg)

    # Compute the dimensions of the near clipping plane
    w = np.tan(hfov_rad / 2.0) * near
    h = w / aspect

    # Create the perspective projection matrix
    mat = np.zeros((4, 4), dtype='f4')
    mat[0, 0] = near / w
    mat[1, 1] = near / h
    mat[2, 2] = -(far + near) / (far - near)
    mat[2, 3] = -2.0 * far * near / (far - near)
    mat[3, 2] = -1.0
    return mat

# Projection and model matrix
#projection = np.eye(4, dtype='f4')
projection = perspective(90.0, width / height, 0.1, 100.0)
view = np.eye(4, dtype='f4')

prog['projection'].write(projection.tobytes())
prog['view'].write(view.tobytes())

# Create a framebuffer to render to
fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4)])
fbo.use()

# Render loop
while True:

    ctx.clear(0.0, 0.0, 0.0, 0.0)
    vao.render(mode=moderngl.LINES)

    # Read the framebuffer into a numpy array
    data = np.frombuffer(fbo.read(components=3), dtype=np.uint8)
    data = data.reshape((height, width, 3))
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    # Display the image using OpenCV
    cv2.imshow('3D Wireframe', data)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
