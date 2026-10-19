"""wrapmodes — visualize the four GL_TEXTURE_WRAP modes side-by-side.

Port of ``src/wrapmodes/wrapmodes.cpp``. A 2D texture
(``rightarrows.ktx``) is sampled with texture coordinates that go
outside [0, 1], drawing one quad per wrap mode in a 2×2 grid:

  GL_CLAMP_TO_EDGE       — last row of texels stretches to the edge
  GL_REPEAT              — texture tiles
  GL_CLAMP_TO_BORDER     — outside-[0,1] reads the border color (yellow)
  GL_MIRRORED_REPEAT     — texture tiles with every other tile mirrored
"""

import os
from typing import override

import glfw  # noqa: F401
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401
from OpenGL.constant import Constant

from sb7 import application, common, ktx

VS_SOURCE: str = """
#version 410 core

uniform vec2 offset;
out vec2 tex_coord;

void main(void)
{
    const vec4 vertices[] = vec4[](vec4(-0.45, -0.45, 0.5, 1.0),
                                   vec4( 0.45, -0.45, 0.5, 1.0),
                                   vec4(-0.45,  0.45, 0.5, 1.0),
                                   vec4( 0.45,  0.45, 0.5, 1.0));

    gl_Position = vertices[gl_VertexID] + vec4(offset, 0.0, 0.0);
    tex_coord = vertices[gl_VertexID].xy * 3.0 + vec2(0.45 * 3);
}
"""

FS_SOURCE: str = """
#version 410 core

uniform sampler2D s;
out vec4 color;
in vec2 tex_coord;

void main(void)
{
    color = texture(s, tex_coord);
}
"""

WRAP_MODES: list[tuple[Constant, str]] = [
    (GL.GL_CLAMP_TO_EDGE, "GL_CLAMP_TO_EDGE"),
    (GL.GL_REPEAT, "GL_REPEAT"),
    (GL.GL_CLAMP_TO_BORDER, "GL_CLAMP_TO_BORDER"),
    (GL.GL_MIRRORED_REPEAT, "GL_MIRRORED_REPEAT"),
]

OFFSETS: list[tuple[float, float]] = [
    (-0.5, -0.5),
    (0.5, -0.5),
    (-0.5, 0.5),
    (0.5, 0.5),
]


class WrapmodesApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Texture Wrap Modes"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        self.texture: int = ktx.load(
            os.path.join(media, "textures", "rightarrows.ktx")
        )
        if not self.texture:
            raise RuntimeError("failed to load rightarrows.ktx")
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        self.program: int = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

    @override
    def render(self, current_time: float) -> None:
        green: list[float] = [0.0, 0.1, 0.0, 1.0]
        yellow: list[float] = [0.4, 0.4, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, green)

        GL.glUseProgram(self.program)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)

        GL.glTexParameterfv(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, yellow
        )

        for i, (mode, _) in enumerate(WRAP_MODES):
            GL.glUniform2fv(0, 1, OFFSETS[i])
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, mode)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # Label the quadrants so students know which is which.

    @override
    def draw_controls(self) -> None:
        imgui.text("Quadrant order (LL, LR, UL, UR):")
        for _, name in WRAP_MODES:
            imgui.bullet_text(name)

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.program)
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteTextures(1, [self.texture])


if __name__ == "__main__":
    application.run(WrapmodesApp)
