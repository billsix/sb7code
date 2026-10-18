"""bumpmapping — ladybug with diffuse + tangent-space normal map.

Port of ``src/bumpmapping/bumpmapping.cpp``. Loads
``ladybug.sbm`` plus two textures: ``ladybug_co.ktx`` (diffuse color)
and ``ladybug_nm.ktx`` (tangent-space normal map). The shaders read
both maps to produce per-pixel detail lighting that wouldn't be
present from the model's vertex normals alone.

The C++ source has a TGA-screenshot helper (S key) we drop —
imgui_bundle has its own screenshot path if needed.

Reads its shaders from ``media/shaders/bumpmapping/``.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, ktx, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "bumpmapping", name)


class BumpmappingApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.program: int = 0
        self.paused: bool = False
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Bump Mapping"

    @override
    def startup(self) -> None:
        self.load_shaders()

        media: str = common.require_media()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        self.tex_color: int = ktx.load(
            os.path.join(media, "textures", "ladybug_co.ktx")
        )
        GL.glActiveTexture(GL.GL_TEXTURE1)
        self.tex_normals: int = ktx.load(
            os.path.join(media, "textures", "ladybug_nm.ktx")
        )
        if not self.tex_color or not self.tex_normals:
            raise RuntimeError("failed to load ladybug textures")

        self.ladybug: sb6m.Object = sb6m.Object()
        self.ladybug.load(os.path.join(media, "objects", "ladybug.sbm"))

    def load_shaders(self) -> None:
        if self.program:
            GL.glDeleteProgram(self.program)
        self.program = common.compile_program_from_files(
            vs=_shader_path("bumpmapping.vs.glsl"),
            fs=_shader_path("bumpmapping.fs.glsl"),
        )
        self.uniform_mv: int = GL.glGetUniformLocation(
            self.program, "mv_matrix"
        )
        self.uniform_proj: int = GL.glGetUniformLocation(
            self.program, "proj_matrix"
        )
        self.uniform_light: int = GL.glGetUniformLocation(
            self.program, "light_pos"
        )

    @override
    def render(self, current_time: float) -> None:
        gray: list[float] = [0.1, 0.1, 0.1, 0.0]
        ones: list[float] = [1.0]

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        f: float = float(self._total_time)

        GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, ones)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glUseProgram(self.program)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        vmath.upload(self.uniform_proj, proj)

        mv: np.ndarray = (
            vmath.translate(0.0, -0.2, -5.5)
            @ vmath.rotate(14.5, 1.0, 0.0, 0.0)
            @ vmath.rotate(-20.0, 0.0, 1.0, 0.0)
        )
        vmath.upload(self.uniform_mv, mv)

        GL.glUniform3fv(
            self.uniform_light,
            1,
            [40.0 * math.sin(f), 30.0 + 20.0 * math.cos(f), 40.0],
        )

        self.ladybug.render()

    @override
    def draw_controls(self) -> None:
        _, self.paused = imgui.checkbox("Pause light", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.ladybug.free()
        GL.glDeleteProgram(self.program)
        GL.glDeleteTextures(2, [self.tex_color, self.tex_normals])


if __name__ == "__main__":
    application.run(BumpmappingApp)
