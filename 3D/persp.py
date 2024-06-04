
import numpy as np
import moderngl
import cv2
from timer import Timer

class QuadRenderer():

    def __init__(self, width, height):

        self.width, self.height = width, height
        self.aspect_ratio = self.width / self.height

        self.ctx = moderngl.create_standalone_context()

        with open('vert_shader.glsl', 'r') as f:
            self.vert_shader_src = f.read()

        with open('frag_shader.glsl', 'r') as f:
            self.frag_shader_src = f.read()

        self.prog = self.ctx.program(
            vertex_shader=self.vert_shader_src,
            fragment_shader=self.frag_shader_src,
        )

        self.prog['color'].value = (1.0, 0.0, 0.0, 1.0)

        self.prog['z_near'].value = 0.1
        self.prog['z_far'].value = 1000.0
        self.prog['ratio'].value = self.aspect_ratio
        self.prog['fovy'].value = 90

        self.prog['eye'].value = (0, 0, 0)
        self.prog['center'].value = (0, 0, -1)
        self.prog['up'].value = (0, 1, 0)

        self.fbo = self.ctx.framebuffer(color_attachments=[
            self.ctx.texture((self.width, self.height), 4)])
        self.fbo.use()

        self.quads = []

    def _hfov_to_vfov(hfov, aspect_ratio):
        hfov_rad = np.radians(hfov)
        vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) / aspect_ratio)
        vfov_deg = np.degrees(vfov_rad)
        return vfov_deg

    def create_quad(self, vertices, color):
        vertices = np.array(vertices, dtype='f4')
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.vertex_array(
            self.prog, [(vbo, '3f', 'vert')]
        )
        self.quads.append((vao, color))  # Add VAO and color to the list

    def render(self):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        for vao, color in self.quads:
            self.prog['color'].value = color
            vao.render(moderngl.TRIANGLE_STRIP)
        data = np.frombuffer(self.fbo.read(components=3), dtype=np.uint8)
        data = data.reshape((self.height, self.width, 3))
        data = np.flipud(data)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        return data

if __name__ == '__main__':
    cv2.namedWindow('Window')
    cv2.moveWindow('Window', 50, 50)

    p = QuadRenderer()

    verts = [
        [-0.5, -0.5, -1.0],
        [ 0.5, -0.5, -1.0],
        [-0.5,  0.5, -1.0],
        [ 0.5,  0.5, -1.0],
    ]
    p.create_quad(verts, (0.0, 1.0, 0.0, 1.0))

    while True:
        with Timer():
            data = p.render()
            cv2.imshow('Window', data)
            if cv2.waitKey(30) & 0xff == ord('q'):
                break
    cv2.destroyAllWindows()
