"""clipdistance — animated plane and sphere clipping a dragon.

Port of ``src/clipdistance/clipdistance.cpp``. Loads
``dragon.sbm`` and renders it with two ``gl_ClipDistance`` outputs
that carve away parts of the geometry: one against an animated
plane, one against an animated sphere. The vertex shader writes both
clip distances; the fragment stage runs only on the surviving
fragments.

Reads its shaders from ``media/shaders/clipdistance/``.

Imgui controls: Pause checkbox, Reload-shaders button.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "clipdistance", name)


class ClipdistanceApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_program: int = 0
        self.paused: bool = False
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Clip Distance"

    @override
    def startup(self) -> None:
        self.dragon: sb6m.Object = sb6m.Object()
        self.dragon.load(
            os.path.join(common.require_media(), "objects", "dragon.sbm")
        )
        self.load_shaders()

    def load_shaders(self) -> None:
        if self.render_program:
            GL.glDeleteProgram(self.render_program)
        self.render_program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )
        self.uniform_proj: int = GL.glGetUniformLocation(
            self.render_program, "proj_matrix"
        )
        self.uniform_mv: int = GL.glGetUniformLocation(
            self.render_program, "mv_matrix"
        )
        self.uniform_plane: int = GL.glGetUniformLocation(
            self.render_program, "clip_plane"
        )
        self.uniform_sphere: int = GL.glGetUniformLocation(
            self.render_program, "clip_sphere"
        )

    @override
    def render(self, current_time: float) -> None:
        import math

        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        one: list[float] = [1.0]

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        f: float = float(self._total_time)

        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glUseProgram(self.render_program)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -15.0)
            @ vmath.rotate(f * 0.34, 0.0, 1.0, 0.0)
            @ vmath.translate(0.0, -4.0, 0.0)
        )

        # Animate a clip plane via a rotating frame; row 0 of the
        # rotation matrix is the plane normal, w=0 to make it a pure
        # plane-through-origin.
        plane_rot: np.ndarray = vmath.rotate(
            f * 6.0, 1.0, 0.0, 0.0
        ) @ vmath.rotate(f * 7.3, 0.0, 1.0, 0.0)
        plane: np.ndarray = plane_rot[0].copy()
        plane[3] = 0.0
        n: np.float32 = (plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2) ** 0.5
        if n > 0:
            plane[:3] /= n

        clip_sphere: tuple[float, float, float, float] = (
            math.sin(f * 0.7) * 3.0,
            math.cos(f * 1.9) * 3.0,
            math.sin(f * 0.1) * 3.0,
            math.cos(f * 1.7) + 2.5,
        )

        vmath.upload(self.uniform_proj, proj)
        vmath.upload(self.uniform_mv, mv)
        GL.glUniform4fv(self.uniform_plane, 1, plane)
        GL.glUniform4fv(self.uniform_sphere, 1, clip_sphere)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CLIP_DISTANCE0)
        GL.glEnable(GL.GL_CLIP_DISTANCE1)

        self.dragon.render()

    @override
    def draw_controls(self) -> None:
        _, self.paused = imgui.checkbox("Pause animation", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.dragon.free()
        GL.glDeleteProgram(self.render_program)


if __name__ == "__main__":
    application.run(ClipdistanceApp)
