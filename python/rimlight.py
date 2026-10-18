"""rimlight — dragon with adjustable rim-light shading.

Port of ``src/rimlight/rimlight.cpp``. Loads
``dragon.sbm`` and shades it with a rim-light term: brightness rises
near silhouette edges (where the surface normal is ~perpendicular to
the view vector), giving a back-lit halo effect. The imgui panel
exposes the rim color (RGB) and the power exponent.

Reads its shaders from ``media/shaders/rimlight/``.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "rimlight", name)


class RimlightApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.program: int = 0
        self.paused: bool = False
        self.rim_color: list[float] = [0.3, 0.3, 0.3]
        self.rim_power: float = 2.5
        self.rim_enable: bool = True
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Rim Lighting"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.dragon: sb6m.Object = sb6m.Object()
        self.dragon.load(
            os.path.join(common.require_media(), "objects", "dragon.sbm")
        )

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

    def load_shaders(self) -> None:
        if self.program:
            GL.glDeleteProgram(self.program)
        self.program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )
        self.uniform_mv: int = GL.glGetUniformLocation(
            self.program, "mv_matrix"
        )
        self.uniform_proj: int = GL.glGetUniformLocation(
            self.program, "proj_matrix"
        )
        self.uniform_rim_color: int = GL.glGetUniformLocation(
            self.program, "rim_color"
        )
        self.uniform_rim_power: int = GL.glGetUniformLocation(
            self.program, "rim_power"
        )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        one: list[float] = [1.0]

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        f: float = float(self._total_time)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glUseProgram(self.program)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        vmath.upload(self.uniform_proj, proj)

        mv: np.ndarray = vmath.translate(0.0, -5.0, -20.0) @ vmath.rotate(
            f * 5.0, 0.0, 1.0, 0.0
        )
        vmath.upload(self.uniform_mv, mv)

        rim_color = self.rim_color if self.rim_enable else [0.0, 0.0, 0.0]
        GL.glUniform3fv(self.uniform_rim_color, 1, rim_color)
        GL.glUniform1f(self.uniform_rim_power, self.rim_power)

        self.dragon.render()

    @override
    def draw_controls(self) -> None:
        _, self.rim_enable = imgui.checkbox("Rim light", self.rim_enable)
        if self.rim_enable:
            _, self.rim_color = imgui.color_edit3("Rim color", self.rim_color)
            _, self.rim_power = imgui.slider_float(
                "Rim power",
                self.rim_power,
                0.1,
                10.0,
                flags=imgui.SliderFlags_.logarithmic.value,
            )
        _, self.paused = imgui.checkbox("Pause rotation", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.dragon.free()
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(RimlightApp)
