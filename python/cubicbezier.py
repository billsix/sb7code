"""cubicbezier — animated 4×4 Bezier patch tessellated on the GPU.

Port of ``src/cubicbezier/cubicbezier.cpp``. A 16-control-
point Bezier patch with z-coordinates animated by sinusoids. The TCS
sets a fixed tessellation level; the TES evaluates the cubic Bernstein
basis against the patch. A second program draws the control points
and the cage of edges between them, as toggleable overlays.

Reads its shaders from ``media/shaders/cubicbezier/``.

Imgui controls: Wireframe / Show points / Show cage / Pause checkboxes
+ Reload button.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "cubicbezier", name)


PATCH_INITIALIZER = np.array(
    [
        -1.0,
        -1.0,
        0.0,
        -0.33,
        -1.0,
        0.0,
        0.33,
        -1.0,
        0.0,
        1.0,
        -1.0,
        0.0,
        -1.0,
        -0.33,
        0.0,
        -0.33,
        -0.33,
        0.0,
        0.33,
        -0.33,
        0.0,
        1.0,
        -0.33,
        0.0,
        -1.0,
        0.33,
        0.0,
        -0.33,
        0.33,
        0.0,
        0.33,
        0.33,
        0.0,
        1.0,
        0.33,
        0.0,
        -1.0,
        1.0,
        0.0,
        -0.33,
        1.0,
        0.0,
        0.33,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
    ],
    dtype=np.float32,
).reshape(16, 3)

CAGE_INDICES: np.ndarray = np.array(
    [
        0,
        1,
        1,
        2,
        2,
        3,
        4,
        5,
        5,
        6,
        6,
        7,
        8,
        9,
        9,
        10,
        10,
        11,
        12,
        13,
        13,
        14,
        14,
        15,
        0,
        4,
        4,
        8,
        8,
        12,
        1,
        5,
        5,
        9,
        9,
        13,
        2,
        6,
        6,
        10,
        10,
        14,
        3,
        7,
        7,
        11,
        11,
        15,
    ],
    dtype=np.uint16,
)


class CubicbezierApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.show_points: bool = False
        self.show_cage: bool = False
        self.wireframe: bool = False
        self.paused: bool = False
        self._last_time: float = 0.0
        self._total_time: float = 0.0
        self.tess_program: int = 0
        self.draw_cp_program: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Cubic Bezier Patch"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.patch_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.patch_vao)

        self.patch_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.patch_buffer)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            PATCH_INITIALIZER.nbytes,
            None,
            GL.GL_DYNAMIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        self.cage_indices: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.cage_indices)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            CAGE_INDICES.nbytes,
            CAGE_INDICES,
            GL.GL_STATIC_DRAW,
        )

    def load_shaders(self) -> None:
        if self.tess_program:
            GL.glDeleteProgram(self.tess_program)
        self.tess_program = common.compile_program_from_files(
            vs=_shader_path("cubicbezier.vs.glsl"),
            tcs=_shader_path("cubicbezier.tcs.glsl"),
            tes=_shader_path("cubicbezier.tes.glsl"),
            fs=_shader_path("cubicbezier.fs.glsl"),
        )
        self.uniform_patch_mv: int = GL.glGetUniformLocation(
            self.tess_program, "mv_matrix"
        )
        self.uniform_patch_proj: int = GL.glGetUniformLocation(
            self.tess_program, "proj_matrix"
        )
        self.uniform_patch_mvp: int = GL.glGetUniformLocation(
            self.tess_program, "mvp"
        )

        if self.draw_cp_program:
            GL.glDeleteProgram(self.draw_cp_program)
        self.draw_cp_program = common.compile_program_from_files(
            vs=_shader_path("draw-control-points.vs.glsl"),
            fs=_shader_path("draw-control-points.fs.glsl"),
        )
        self.uniform_cp_color: int = GL.glGetUniformLocation(
            self.draw_cp_program, "draw_color"
        )
        self.uniform_cp_mvp: int = GL.glGetUniformLocation(
            self.draw_cp_program, "mvp"
        )

    @override
    def render(self, current_time: float) -> None:
        gray: list[float] = [0.1, 0.1, 0.1, 0.0]
        one: list[float] = [1.0]

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        t = self._total_time

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Animate z components.
        patch = PATCH_INITIALIZER.copy()
        for i in range(16):
            fi = i / 16.0
            patch[i, 2] = math.sin(t * (0.2 + fi * 0.3))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.patch_buffer)
        GL.glBufferSubData(
            GL.GL_ARRAY_BUFFER, 0, patch.nbytes, np.ascontiguousarray(patch)
        )

        GL.glBindVertexArray(self.patch_vao)
        GL.glUseProgram(self.tess_program)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 1.0, 1000.0)
        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -4.0)
            @ vmath.rotate(t * 10.0, 0.0, 1.0, 0.0)
            @ vmath.rotate(t * 17.0, 1.0, 0.0, 0.0)
        )
        mvp = proj @ mv

        vmath.upload(self.uniform_patch_mv, mv)
        vmath.upload(self.uniform_patch_proj, proj)
        vmath.upload(self.uniform_patch_mvp, mvp)

        GL.glPolygonMode(
            GL.GL_FRONT_AND_BACK, GL.GL_LINE if self.wireframe else GL.GL_FILL
        )
        GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 16)
        GL.glDrawArrays(GL.GL_PATCHES, 0, 16)

        GL.glUseProgram(self.draw_cp_program)
        vmath.upload(self.uniform_cp_mvp, mvp)

        if self.show_points:
            GL.glPointSize(9.0)
            GL.glUniform4fv(self.uniform_cp_color, 1, [0.2, 0.7, 0.9, 1.0])
            GL.glDrawArrays(GL.GL_POINTS, 0, 16)
        if self.show_cage:
            GL.glUniform4fv(self.uniform_cp_color, 1, [0.7, 0.9, 0.2, 1.0])
            GL.glDrawElements(GL.GL_LINES, 48, GL.GL_UNSIGNED_SHORT, None)

    @override
    def draw_controls(self) -> None:
        _, self.wireframe = imgui.checkbox("Wireframe", self.wireframe)
        _, self.show_cage = imgui.checkbox("Show control cage", self.show_cage)
        _, self.show_points = imgui.checkbox(
            "Show control points", self.show_points
        )
        _, self.paused = imgui.checkbox("Pause animation", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.tess_program)
        GL.glDeleteProgram(self.draw_cp_program)
        GL.glDeleteVertexArrays(1, [self.patch_vao])
        GL.glDeleteBuffers(1, [self.patch_buffer])
        GL.glDeleteBuffers(1, [self.cage_indices])


if __name__ == "__main__":
    application.run(CubicbezierApp)
