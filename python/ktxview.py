"""ktxview — small KTX texture browser with exposure control.

Port of ``src/ktxview/ktxview.cpp``. Loads a tree
texture and draws it to a fullscreen quad with an exposure
multiplier. The C++ source modulates exposure by ``sin(t) * 16 + 16``;
we offer the same animation, plus an imgui slider so students can
override the value manually.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, ktx

VS_SOURCE: str = """
#version 420 core

void main(void)
{
    const vec4 vertices[] = vec4[](vec4(-1.0, -1.0, 0.5, 1.0),
                                   vec4( 1.0, -1.0, 0.5, 1.0),
                                   vec4(-1.0,  1.0, 0.5, 1.0),
                                   vec4( 1.0,  1.0, 0.5, 1.0));
    gl_Position = vertices[gl_VertexID];
}
"""

FS_SOURCE: str = """
#version 430 core

uniform sampler2D s;
uniform float exposure;
out vec4 color;

void main(void)
{
    color = texture(s, gl_FragCoord.xy / textureSize(s, 0)) * exposure;
}
"""


class KtxviewApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.auto_exposure: bool = True
        self.manual_exposure: float = 16.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - KTX Viewer"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        self.texture: int = ktx.load(
            os.path.join(media, "textures", "tree.ktx")
        )
        if not self.texture:
            raise RuntimeError("failed to load tree.ktx")
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        self.program: int = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)
        # The C++ source hardcodes ``glUniform1f(0, ...)`` for the
        # exposure uniform; Mesa won't assign location 0 to it
        # automatically, so query.
        self.exposure_loc: int = GL.glGetUniformLocation(
            self.program, "exposure"
        )
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

    @override
    def render(self, current_time: float) -> None:
        green: list[float] = [0.0, 0.25, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, green)

        GL.glUseProgram(self.program)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)

        if self.auto_exposure:
            exposure = math.sin(current_time) * 16.0 + 16.0
        else:
            exposure = self.manual_exposure
        GL.glUniform1f(self.exposure_loc, exposure)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        _, self.auto_exposure = imgui.checkbox(
            "Auto-animate exposure", self.auto_exposure
        )
        if not self.auto_exposure:
            _, self.manual_exposure = imgui.slider_float(
                "Exposure", self.manual_exposure, 0.0, 32.0
            )

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.program)
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteTextures(1, [self.texture])


if __name__ == "__main__":
    application.run(KtxviewApp)
