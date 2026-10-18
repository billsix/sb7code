"""hqfilter — bicubic texture filtering done in the fragment shader.

Port of ``src/hqfilter/hqfilter.cpp``. Renders a baboon
KTX texture with a custom bicubic-interpolation fragment shader rather
than relying on the hardware's built-in linear filter. The result is
visibly higher-quality magnification at the cost of more shader work.

Reads its shaders from ``media/shaders/hqfilter/``.
"""

import os
from typing import override

import glfw  # noqa: F401
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, ktx


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "hqfilter", name)


class HqfilterApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.draw_tex_program: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - High Quality Texture Filtering"

    @override
    def startup(self) -> None:
        self.load_shaders()
        self.dummy_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.dummy_vao)
        media: str = common.require_media()
        self.source_tex: int = ktx.load(
            os.path.join(media, "textures", "baboon.ktx")
        )
        if not self.source_tex:
            raise RuntimeError("failed to load baboon.ktx")

    def load_shaders(self) -> None:
        if self.draw_tex_program:
            GL.glDeleteProgram(self.draw_tex_program)
        self.draw_tex_program = common.compile_program_from_files(
            vs=_shader_path("hqfilter.vs.glsl"),
            fs=_shader_path("hqfilter.fs.glsl"),
        )

    @override
    def render(self, current_time: float) -> None:
        gray: list[float] = [0.1, 0.1, 0.1, 0.0]
        one: list[float] = [1.0]

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glUseProgram(self.draw_tex_program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.source_tex)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        if imgui.button("Reload shaders"):
            self.load_shaders()


if __name__ == "__main__":
    application.run(HqfilterApp)
