"""phonglighting — sphere with per-vertex vs per-fragment Phong lighting.

Port of ``src/phonglighting/phonglighting.cpp``. Loads
``sphere.sbm`` and renders it with two shader programs: one that does
the Phong calculation per vertex (Gouraud highlights), the other per
fragment (proper Phong highlights). The imgui radio group toggles
between the two so students can see why per-fragment is the right
default.

Reads its shaders from ``media/shaders/phonglighting/``.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(
        common.require_media(), "shaders", "phonglighting", name
    )


# uniforms_block in the C++ source: 3 mat4s = 192 bytes.
UNIFORMS_BLOCK_SIZE: int = 3 * 64


class PhonglightingApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.per_fragment_program: int = 0
        self.per_vertex_program: int = 0
        self.per_vertex: bool = False

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Phong Lighting"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.uniforms_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.uniforms_buffer)
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER, UNIFORMS_BLOCK_SIZE, None, GL.GL_DYNAMIC_DRAW
        )

        self.sphere: sb6m.Object = sb6m.Object()
        self.sphere.load(
            os.path.join(common.require_media(), "objects", "sphere.sbm")
        )

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

    def load_shaders(self) -> None:
        if self.per_fragment_program:
            GL.glDeleteProgram(self.per_fragment_program)
        if self.per_vertex_program:
            GL.glDeleteProgram(self.per_vertex_program)
        self.per_fragment_program = common.compile_program_from_files(
            vs=_shader_path("per-fragment-phong.vs.glsl"),
            fs=_shader_path("per-fragment-phong.fs.glsl"),
        )
        self.per_vertex_program = common.compile_program_from_files(
            vs=_shader_path("per-vertex-phong.vs.glsl"),
            fs=_shader_path("per-vertex-phong.fs.glsl"),
        )

        # Two programs share the same uniform names; query each.
        self.specular_power_loc: list[int] = [
            GL.glGetUniformLocation(p, "specular_power")
            for p in (self.per_fragment_program, self.per_vertex_program)
        ]
        self.specular_albedo_loc: list[int] = [
            GL.glGetUniformLocation(p, "specular_albedo")
            for p in (self.per_fragment_program, self.per_vertex_program)
        ]

    @override
    def render(self, current_time: float) -> None:
        gray: list[float] = [0.1, 0.1, 0.1, 0.0]
        ones: list[float] = [1.0]

        prog_idx: int = 1 if self.per_vertex else 0
        prog: int = (
            self.per_vertex_program
            if self.per_vertex
            else self.per_fragment_program
        )
        GL.glUseProgram(prog)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, ones)

        view: np.ndarray = vmath.lookat(
            (0.0, 0.0, 50.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        )
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        model: np.ndarray = vmath.scale(7.0)

        # Build the std140 block: mv_matrix, view_matrix, proj_matrix.
        # GLSL std140 mat4 is column-major; transpose row-major numpy
        # before upload.
        block: np.ndarray = np.empty((3, 4, 4), dtype=np.float32)
        block[0] = (view @ model).T
        block[1] = view.T
        block[2] = proj.T

        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.uniforms_buffer)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER, 0, block.nbytes, np.ascontiguousarray(block)
        )

        GL.glUniform1f(self.specular_power_loc[prog_idx], 30.0)
        GL.glUniform3fv(self.specular_albedo_loc[prog_idx], 1, [1.0, 1.0, 1.0])

        self.sphere.render()

    @override
    def draw_controls(self) -> None:
        imgui.text("Lighting model:")
        if imgui.radio_button("Per-fragment Phong", not self.per_vertex):
            self.per_vertex = False
        if imgui.radio_button("Per-vertex (Gouraud)", self.per_vertex):
            self.per_vertex = True
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.sphere.free()
        GL.glDeleteProgram(self.per_fragment_program)
        GL.glDeleteProgram(self.per_vertex_program)
        GL.glDeleteBuffers(1, [self.uniforms_buffer])


if __name__ == "__main__":
    application.run(PhonglightingApp)
