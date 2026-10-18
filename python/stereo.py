"""stereo — two-eye stereo rendering of dragon/sphere/cube/torus.

Port of ``src/stereo/stereo.cpp``.

The C++ original asks for a quad-buffered stereo GL context
(``info.flags.stereo = 1``) and then renders into ``GL_BACK_LEFT`` /
``GL_BACK_RIGHT`` for shutter-glasses or HMDs. Most desktops don't
expose stereo framebuffers, so this port defaults to **side-by-side**
mode: split the window into a left half and a right half and draw the
two camera views into each. An imgui radio offers the true
quad-buffered path as well — it succeeds where the GL context is
stereo-capable and silently no-ops the right-eye draw otherwise.

Reads its shaders from ``media/shaders/stereo/``.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "stereo", name)


def _model_path(name: str) -> str:
    return os.path.join(common.require_media(), "objects", name)


OBJECT_NAMES: list[str] = ["dragon.sbm", "sphere.sbm", "cube.sbm", "torus.sbm"]
DIFFUSE_COLORS: np.ndarray = np.array(
    [
        [1.0, 0.6, 0.3],
        [0.2, 0.8, 0.9],
        [0.3, 0.9, 0.4],
        [0.5, 0.2, 1.0],
    ],
    dtype=np.float32,
)


class StereoApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.view_program: int = 0
        self.objects: list[sb6m.Object] = []
        self.model_matrices: list[np.ndarray] = [
            vmath.identity() for _ in range(4)
        ]
        # 0 = side-by-side (default, works on any context).
        # 1 = quad-buffered BACK_LEFT/BACK_RIGHT (needs stereo context).
        self.stereo_mode: int = 0
        self.full_shading: bool = True
        self.paused: bool = False
        self.separation: float = 2.0
        self.total_time: float = 0.0
        self._last_time: float | None = None

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Stereo"

    @override
    def startup(self) -> None:
        self.load_shaders()

        for name in OBJECT_NAMES:
            obj: sb6m.Object = sb6m.Object()
            obj.load(_model_path(name))
            self.objects.append(obj)

        self.quad_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.quad_vao)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def load_shaders(self) -> None:
        if self.view_program:
            GL.glDeleteProgram(self.view_program)
        self.view_program = common.compile_program_from_files(
            vs=_shader_path("stereo-render.vs.glsl"),
            fs=_shader_path("stereo-render.fs.glsl"),
        )
        self.u_proj: int = GL.glGetUniformLocation(
            self.view_program, "proj_matrix"
        )
        self.u_mv: int = GL.glGetUniformLocation(self.view_program, "mv_matrix")
        self.u_full: int = GL.glGetUniformLocation(
            self.view_program, "full_shading"
        )
        self.u_diffuse: int = GL.glGetUniformLocation(
            self.view_program, "diffuse_albedo"
        )

    @override
    def render(self, current_time: float) -> None:
        if self._last_time is None:
            self._last_time = current_time
        if not self.paused:
            self.total_time += current_time - self._last_time
        self._last_time = current_time

        f: float = float(self.total_time) + 30.0

        view_position: np.ndarray = np.array([0.0, 0.0, 40.0], dtype=np.float32)
        target: np.ndarray = np.array([0.0, 0.0, -50.0], dtype=np.float32)
        up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        offset: np.ndarray = np.array(
            [self.separation, 0.0, 0.0], dtype=np.float32
        )

        w: int = max(self.info.window_width, 1)
        h: int = max(self.info.window_height, 1)
        # In side-by-side, each eye occupies half the width.
        eye_aspect: float = (w / 2.0) / h if self.stereo_mode == 0 else w / h
        proj: np.ndarray = vmath.perspective(50.0, eye_aspect, 1.0, 200.0)

        view_l: np.ndarray = vmath.lookat(view_position - offset, target, up)
        view_r: np.ndarray = vmath.lookat(view_position + offset, target, up)

        self.model_matrices[0] = (
            vmath.rotate(f * 14.5, 0, 1, 0)
            @ vmath.rotate(20.0, 1, 0, 0)
            @ vmath.translate(0.0, -4.0, 0.0)
        )
        self.model_matrices[1] = (
            vmath.rotate(f * 3.7, 0, 1, 0)
            @ vmath.translate(
                math.sin(f * 0.37) * 12.0, math.cos(f * 0.37) * 12.0, 0.0
            )
            @ vmath.scale(2.0, 2.0, 2.0)
        )
        self.model_matrices[2] = (
            vmath.rotate(f * 6.45, 0, 1, 0)
            @ vmath.translate(
                math.sin(f * 0.25) * 10.0, math.cos(f * 0.25) * 10.0, 0.0
            )
            @ vmath.rotate(f * 99.0, 0, 0, 1)
            @ vmath.scale(2.0, 2.0, 2.0)
        )
        self.model_matrices[3] = (
            vmath.rotate(f * 5.25, 0, 1, 0)
            @ vmath.translate(
                math.sin(f * 0.51) * 14.0, math.cos(f * 0.51) * 14.0, 0.0
            )
            @ vmath.rotate(f * 120.3, 0.707106, 0.0, 0.707106)
            @ vmath.scale(2.0, 2.0, 2.0)
        )

        gray: list[float] = [0.1, 0.1, 0.1, 0.0]
        ones: list[float] = [1.0]

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.view_program)

        if self.stereo_mode == 1:
            # Quad-buffered stereo. Falls back gracefully if the
            # context isn't stereo-capable: GL_BACK_LEFT becomes the
            # only valid back buffer and GL_BACK_RIGHT raises
            # GL_INVALID_OPERATION which we swallow.
            GL.glViewport(0, 0, w, h)
            self._draw_eye(view_l, proj, gray, ones, GL.GL_BACK_LEFT)
            try:
                self._draw_eye(view_r, proj, gray, ones, GL.GL_BACK_RIGHT)
            except GL.error.GLError:
                pass
        else:
            # Side-by-side: clear once, then two viewports.
            GL.glViewport(0, 0, w, h)
            GL.glDrawBuffer(GL.GL_BACK)
            GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
            GL.glClearBufferfv(GL.GL_DEPTH, 0, ones)
            GL.glViewport(0, 0, w // 2, h)
            self._draw_eye(view_l, proj, None, None, None)
            GL.glViewport(w // 2, 0, w - w // 2, h)
            self._draw_eye(view_r, proj, None, None, None)
            GL.glViewport(0, 0, w, h)

    @override
    def draw_controls(self) -> None:
        imgui.text("Stereo mode")
        if imgui.radio_button("Side-by-side", self.stereo_mode == 0):
            self.stereo_mode = 0
        imgui.same_line()
        if imgui.radio_button(
            "Quad-buffered (BACK_LEFT/RIGHT)", self.stereo_mode == 1
        ):
            self.stereo_mode = 1
        _, self.full_shading = imgui.checkbox("Full shading", self.full_shading)
        _, self.paused = imgui.checkbox("Paused", self.paused)
        _, self.separation = imgui.slider_float(
            "Eye separation", self.separation, 0.0, 6.0
        )
        if imgui.button("Reload shaders"):
            self.load_shaders()

    def _draw_eye(
        self,
        view_matrix: np.ndarray,
        proj: np.ndarray,
        clear_color: list[float] | None,
        clear_depth: list[float] | None,
        drawbuf,
    ) -> None:
        if drawbuf is not None:
            GL.glDrawBuffer(drawbuf)
            GL.glClearBufferfv(GL.GL_COLOR, 0, clear_color)
            GL.glClearBufferfv(GL.GL_DEPTH, 0, clear_depth)
        GL.glUniformMatrix4fv(self.u_proj, 1, GL.GL_FALSE, proj.T)
        GL.glUniform1i(self.u_full, 1 if self.full_shading else 0)
        for i, obj in enumerate(self.objects):
            mv: np.ndarray = view_matrix @ self.model_matrices[i]
            GL.glUniformMatrix4fv(self.u_mv, 1, GL.GL_FALSE, mv.T)
            GL.glUniform3fv(self.u_diffuse, 1, DIFFUSE_COLORS[i])
            obj.render()

    @override
    def shutdown(self) -> None:
        for obj in self.objects:
            obj.free()
        GL.glDeleteProgram(self.view_program)
        GL.glDeleteVertexArrays(1, [self.quad_vao])


if __name__ == "__main__":
    application.run(StereoApp)
