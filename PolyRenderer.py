
import numpy as np
from numpy.typing import NDArray
import moderngl
import cv2

from dataclasses import dataclass

from pyrr import Matrix44 as M44

@dataclass
class _Polygon:
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

        with open('shaders/vert.glsl', 'r') as f:
            self.vert_shader_src = f.read()

        with open('shaders/frag.glsl', 'r') as f:
            self.frag_shader_src = f.read()

        self.prog = self.ctx.program(
            vertex_shader=self.vert_shader_src,
            fragment_shader=self.frag_shader_src,
        )

        self.fbo = self.ctx.framebuffer(color_attachments=[
            self.ctx.texture((self.width, self.height), 4)])
        self.fbo.use()

        self.polys: dict[int, _Polygon] = {}
        self.verts: dict[int, NDArray[np.float32]] = {}

        self.M_proj = None
        self.M_view = None

    def _hfov_to_vfov(self, hfov):
        hfov_rad = np.radians(hfov)
        vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) / self.aspect_ratio)
        vfov_deg = np.degrees(vfov_rad)
        return vfov_deg

    def _world_to_screen(self, verts: dict) -> dict:
        out = {}
        for uid, v in verts.items():
            # Transform vertex to clip space
            # NOTE: check why is the transpose needed. I think it's because
            # the matrices that pyrr create for OpenGL are col-major.
            v_clip = self.M_proj.T @ self.M_view.T @ np.array([*v, 1.0])
            v_clip = np.array(v_clip)
            # Perspective division to normalized device coordinates
            ndc = v_clip / v_clip[3]
            # Viewport transform
            x = int((ndc[0] * 0.5 + 0.5) * self.width)
            y = int((1 - (ndc[1] * 0.5 + 0.5)) * self.height)
            out[uid] = (x, y, ndc[2], ndc[3])
        return out

    def _is_visible(self, vert):
        x, y, z, _ = vert
        return 0 <= x < self.width and \
               0 <= y < self.height and \
               z >= -1 and z <= 1

    def create_poly(self, uid, color, verts):
        verts = np.array(verts, dtype='f4')
        vbo = self.ctx.buffer(verts)
        vao = self.ctx.vertex_array(
            self.prog, [(vbo, '3f', 'vert')]
        )
        m = _Polygon(uid, color, verts, vbo, vao)
        self.polys[uid] = m

    def set_cam_pose(self, lookat, up, hfov):
        vfov = self._hfov_to_vfov(hfov)
        self.M_proj = M44.perspective_projection(vfov, self.aspect_ratio,
                                               0.1, 1000, dtype='f4')
        self.M_view = M44.look_at((0, 0, 0), lookat, up, dtype='f4')
        self.prog['proj'].write(self.M_proj.tobytes())
        self.prog['view'].write(self.M_view.tobytes())

    def render(self, showPolys, showMarkers):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        if showPolys:
            for p in self.polys.values():
                self.prog['color'].value = p.color
                p.vao.render(moderngl.TRIANGLE_FAN)

        frame = np.frombuffer(self.fbo.read(components=4), dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 4))
        frame = np.flipud(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)

        if showMarkers:
            for uid, v in self._world_to_screen(self.verts).items():
                if self._is_visible(v):
                    cv2.circle(frame, (v[0], v[1]), 4, (0, 0, 255, 255), -1)
                    cv2.putText(frame, f'{uid}', (v[0] + 5, v[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 2)

        return frame

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
