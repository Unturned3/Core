
import numpy as np
from numpy.typing import NDArray
import moderngl
import cv2

from dataclasses import dataclass

from pyrr import Matrix44 as M44

@dataclass
class PolyMarker:
    uid: int
    color: tuple[float, float, float, float]
    verts: NDArray[np.float32]
    vbo: moderngl.Buffer
    vao: moderngl.VertexArray

class PolyRenderer3D():

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

        self.fbo = self.ctx.framebuffer(color_attachments=[
            self.ctx.texture((self.width, self.height), 4)])
        self.fbo.use()

        self.polys: dict[int, PolyMarker] = {}

    def _hfov_to_vfov(self, hfov):
        hfov_rad = np.radians(hfov)
        vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) / self.aspect_ratio)
        vfov_deg = np.degrees(vfov_rad)
        return vfov_deg

    def create_poly(self, uid, color, verts):
        verts = np.array(verts, dtype='f4')
        vbo = self.ctx.buffer(verts)
        vao = self.ctx.vertex_array(
            self.prog, [(vbo, '3f', 'vert')]
        )
        m = PolyMarker(uid, color, verts, vbo, vao)
        self.polys[uid] = m

    def render(self, center, up, hfov):
        vfov = self._hfov_to_vfov(hfov)
        self.prog['proj'].write(
            M44.perspective_projection(
                vfov, self.aspect_ratio, 0.1, 1000, dtype='f4'
            ).tobytes()
        )
        self.prog['view'].write(
            M44.look_at((0, 0, 0), center, up, dtype='f4').tobytes()
        )

        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        for p in self.polys.values():
            self.prog['color'].value = p.color
            p.vao.render(moderngl.TRIANGLE_FAN)

        data = np.frombuffer(self.fbo.read(components=3), dtype=np.uint8)
        data = data.reshape((self.height, self.width, 3))
        data = np.flipud(data)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        return data

if __name__ == '__main__':
    cv2.namedWindow('Window')
    cv2.moveWindow('Window', 50, 50)

    p = PolyRenderer3D(640, 480)

    # Define a hexagon, with z = -2
    verts = np.array([
        0, 0, -2,
        1, 0, -2,
        0, 1, -2,
    ]).reshape(-1, 3)
    p.create_poly(1, (0.0, 1.0, 0.0, 1.0), verts)

    while True:
        data = p.render((0, 0, -1), (0, 1, 0), 90)
        cv2.imshow('Window', data)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()
