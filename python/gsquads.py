"""gsquads — render quads two ways: as triangle fans, or as
GL_LINES_ADJACENCY routed through a geometry shader.

Port of ``src/gsquads/gsquads.cpp``. Demonstrates how
``GL_QUADS`` was replaced in core profile: either decompose into a
triangle fan client-side, or feed four vertices as ``GL_LINES_ADJACENCY``
and have a geometry shader emit a quad-as-triangle-strip with proper
per-corner color interpolation.

The C++ side uses ``sb7::text_overlay`` and key shortcuts. We use an
imgui control panel: radio buttons for the mode, a Pause checkbox, a
slider for the vertex-ID offset, and a Reload button.
"""

import os
from typing import override

import numpy as np
import OpenGL.GL as GL

from sb7 import application, common, vmath
from imgui_bundle import imgui  # noqa: E402


def _shader_path(name: str) -> str:
    # Resolve at call time so the missing-media error fires inside
    # startup() with a clear message, not at import time.
    return os.path.join(common.require_media(), "shaders", "gsquads", name)


class GsquadsApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.program_fans: int = 0
        self.program_linesadjacency: int = 0
        self.mode: int = 0
        self.paused: bool = False
        self.vid_offset: int = 0
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Quad Rendering"

    @override
    def startup(self) -> None:
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self.load_shaders()

    def load_shaders(self) -> None:
        if self.program_fans:
            GL.glDeleteProgram(self.program_fans)
        self.program_fans = common.compile_program_from_files(
            vs=_shader_path("quadsasfans.vs.glsl"),
            fs=_shader_path("quadsasfans.fs.glsl"),
        )
        self.mvp_loc_fans: int = GL.glGetUniformLocation(
            self.program_fans, "mvp"
        )
        self.vid_offset_loc_fans: int = GL.glGetUniformLocation(
            self.program_fans, "vid_offset"
        )

        if self.program_linesadjacency:
            GL.glDeleteProgram(self.program_linesadjacency)
        self.program_linesadjacency = common.compile_program_from_files(
            vs=_shader_path("quadsaslinesadj.vs.glsl"),
            gs=_shader_path("quadsaslinesadj.gs.glsl"),
            fs=_shader_path("quadsaslinesadj.fs.glsl"),
        )
        self.mvp_loc_linesadj: int = GL.glGetUniformLocation(
            self.program_linesadjacency, "mvp"
        )
        self.vid_offset_loc_linesadj: int = GL.glGetUniformLocation(
            self.program_linesadjacency, "vid_offset"
        )

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.program_fans)
        GL.glDeleteProgram(self.program_linesadjacency)
        GL.glDeleteVertexArrays(1, [self.vao])

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.25, 0.0, 1.0]
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        t: float = float(self._total_time)

        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -2.0)
            @ vmath.rotate(t * 5.0, 0.0, 0.0, 1.0)
            @ vmath.rotate(t * 30.0, 1.0, 0.0, 0.0)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        mvp = proj @ mv

        if self.mode == 0:
            GL.glUseProgram(self.program_fans)
            vmath.upload(self.mvp_loc_fans, mvp)
            GL.glUniform1i(self.vid_offset_loc_fans, self.vid_offset)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        else:
            GL.glUseProgram(self.program_linesadjacency)
            vmath.upload(self.mvp_loc_linesadj, mvp)
            GL.glUniform1i(self.vid_offset_loc_linesadj, self.vid_offset)
            GL.glDrawArrays(GL.GL_LINES_ADJACENCY, 0, 4)

    # Replaces ``sb7::text_overlay`` from the C++ source.
    @override
    def draw_controls(self) -> None:
        imgui.text("Quad rendering mode:")
        if imgui.radio_button("GL_TRIANGLE_FAN", self.mode == 0):
            self.mode = 0
        if imgui.radio_button(
            "GL_LINES_ADJACENCY (geometry shader)", self.mode == 1
        ):
            self.mode = 1
        imgui.separator()
        _, self.paused = imgui.checkbox("Pause rotation", self.paused)
        _, self.vid_offset = imgui.slider_int(
            "Vertex ID offset", self.vid_offset, -8, 8
        )
        if imgui.button("Reload shaders"):
            self.load_shaders()


if __name__ == "__main__":
    application.run(GsquadsApp)
