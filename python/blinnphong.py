"""blinnphong — sphere with Blinn-Phong shading.

Port of ``src/blinnphong/blinnphong.cpp``. Loads
``sphere.sbm`` and renders it with a Blinn-Phong shader (half-vector
specular instead of reflection-dot-view). The C++ source has a
commented-out "MANY_OBJECTS" branch — we keep the single-sphere
default.

Reads its shaders from ``media/shaders/blinnphong/``.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "blinnphong", name)


UNIFORMS_BLOCK_SIZE: int = 3 * 64


class BlinnphongApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.per_fragment_program: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Blinn-Phong Shading"

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
        self.per_fragment_program = common.compile_program_from_files(
            vs=_shader_path("blinnphong.vs.glsl"),
            fs=_shader_path("blinnphong.fs.glsl"),
        )
        self.specular_power_loc: int = GL.glGetUniformLocation(
            self.per_fragment_program, "specular_power"
        )
        self.specular_albedo_loc: int = GL.glGetUniformLocation(
            self.per_fragment_program, "specular_albedo"
        )

    @override
    def render(self, current_time: float) -> None:
        gray: list[float] = [0.1, 0.1, 0.1, 0.0]
        ones: list[float] = [1.0]

        GL.glUseProgram(self.per_fragment_program)
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

        block: np.ndarray = np.empty((3, 4, 4), dtype=np.float32)
        block[0] = (view @ model).T
        block[1] = view.T
        block[2] = proj.T

        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.uniforms_buffer)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER, 0, block.nbytes, np.ascontiguousarray(block)
        )

        GL.glUniform1f(self.specular_power_loc, 30.0)
        GL.glUniform3fv(self.specular_albedo_loc, 1, [1.0, 1.0, 1.0])

        self.sphere.render()

    @override
    def draw_controls(self) -> None:
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.sphere.free()
        GL.glDeleteProgram(self.per_fragment_program)
        GL.glDeleteBuffers(1, [self.uniforms_buffer])


if __name__ == "__main__":
    application.run(BlinnphongApp)
