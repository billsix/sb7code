"""simpletexture — first textured demo.

Port of ``src/simpletexture/simpletexture.cpp``. Generates
a 256×256 RGBA32F texture procedurally (with the classic
``(x&y), (x|y), (x^y)`` pattern from old demoscene days), uploads it,
and samples it across a single triangle. The fragment shader uses
``gl_FragCoord / textureSize`` so the texture maps directly to screen
pixels — moving the triangle around the screen exposes different
parts of the texture.
"""

from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

from sb7 import application, common

VS_SOURCE: str = """
#version 420 core

void main(void)
{
    const vec4 vertices[] = vec4[](vec4( 0.75, -0.75, 0.5, 1.0),
                                   vec4(-0.75, -0.75, 0.5, 1.0),
                                   vec4( 0.75,  0.75, 0.5, 1.0));

    gl_Position = vertices[gl_VertexID];
}
"""

FS_SOURCE: str = """
#version 430 core

uniform sampler2D s;

out vec4 color;

void main(void)
{
    color = texture(s, gl_FragCoord.xy / textureSize(s, 0));
}
"""


def _generate_texture(w: int, h: int) -> np.ndarray:
    """Build the procedural ``(x&y, x|y, x^y)`` texture as a
    contiguous float32 RGBA array. Vectorized via numpy meshgrid so
    we don't pay 256² = 65K Python iterations at startup."""
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    out: np.ndarray = np.empty((h, w, 4), dtype=np.float32)
    out[..., 0] = ((x & y) & 0xFF) / 255.0
    out[..., 1] = ((x | y) & 0xFF) / 255.0
    out[..., 2] = ((x ^ y) & 0xFF) / 255.0
    out[..., 3] = 1.0
    return out


class SimpletextureApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Simple Texturing"

    @override
    def startup(self) -> None:
        self.texture: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 8, GL.GL_RGBA32F, 256, 256)

        data: np.ndarray = _generate_texture(256, 256)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, 256, 256, GL.GL_RGBA, GL.GL_FLOAT, data
        )

        self.program: int = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

    @override
    def render(self, current_time: float) -> None:
        green: list[float] = [0.0, 0.25, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, green)

        GL.glUseProgram(self.program)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.program)
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteTextures(1, [self.texture])


if __name__ == "__main__":
    application.run(SimpletextureApp)
