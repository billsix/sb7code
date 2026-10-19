"""mirrorclampedge — compare GL_CLAMP_TO_BORDER vs. GL_MIRROR_CLAMP_TO_EDGE.

Port of ``src/mirrorclampedge/mirrorclampedge.cpp``. A
single quad samples a 2D KTX texture (``flare.ktx``) with texture
coords spilling outside [0, 1]. The imgui panel toggles between the
two boundary modes so you can see the difference between "clip
beyond edge" (border color) and "mirror beyond edge".

Reads its shaders from ``media/shaders/mirrorclampedge/``.
"""

import os
from typing import override

import glfw  # noqa: F401
import OpenGL.GL as GL
from OpenGL.constant import Constant
from imgui_bundle import imgui

from sb7 import application, colors, common, ktx


def _shader_path(name: str) -> str:
    return os.path.join(
        common.require_media(), "shaders", "mirrorclampedge", name
    )


CLAMP_TO_BORDER: int = 0
MIRROR_CLAMP_TO_EDGE: int = 1
MODE_NAMES: list[str] = ["GL_CLAMP_TO_BORDER", "GL_MIRROR_CLAMP_TO_EDGE"]


class MirrorclampedgeApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.display_mode: int = CLAMP_TO_BORDER
        self.render_program: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - GL_MIRROR_CLAMP_TO_EDGE"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        self.input_texture: int = ktx.load(
            os.path.join(media, "textures", "flare.ktx")
        )
        if not self.input_texture:
            raise RuntimeError("failed to load flare.ktx")

        self.load_shaders()
        self.dummy_vao: int = GL.glGenVertexArrays(1)

    def load_shaders(self) -> None:
        if self.render_program:
            GL.glDeleteProgram(self.render_program)
        self.render_program = common.compile_program_from_files(
            vs=_shader_path("drawquad.vs.glsl"),
            fs=_shader_path("drawquad.fs.glsl"),
        )

    @override
    def render(self, current_time: float) -> None:
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, colors.Color.Black)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.input_texture)

        GL.glBindVertexArray(self.dummy_vao)
        GL.glUseProgram(self.render_program)

        if self.display_mode == CLAMP_TO_BORDER:
            wrap: int | Constant = GL.GL_CLAMP_TO_BORDER
        else:
            wrap = GL.GL_MIRROR_CLAMP_TO_EDGE
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap)

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        for i, name in enumerate(MODE_NAMES):
            if imgui.radio_button(name, self.display_mode == i):
                self.display_mode = i
        if imgui.button("Reload shaders"):
            self.load_shaders()


if __name__ == "__main__":
    application.run(MirrorclampedgeApp)
