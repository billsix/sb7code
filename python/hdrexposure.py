"""hdrexposure — Reinhard-style tone mapping with adjustable exposure.

Port of ``src/hdrexposure/hdrexposure.cpp``. Loads an HDR
KTX texture (``treelights_2k.ktx``) and applies the simple
``1 - exp(-color * exposure)`` tone-map curve in the fragment shader.

Imgui control: Exposure slider (logarithmic).
"""

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
    vec4 c = texture(s, 2.0 * gl_FragCoord.xy / textureSize(s, 0));
    c.xyz = vec3(1.0) - exp(-c.xyz * exposure);
    color = c;
}
"""


class HdrexposureApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.exposure: float = 1.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - HDR Exposure"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        self.texture: int = ktx.load(
            os.path.join(media, "textures", "treelights_2k.ktx")
        )
        if not self.texture:
            raise RuntimeError("failed to load treelights_2k.ktx")
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        self.program: int = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)
        # The C++ source hardcodes ``glUniform1f(0, ...)`` assuming
        # the driver assigns location 0 to ``exposure``. Mesa
        # doesn't, so query the location instead.
        self.exposure_loc: int = GL.glGetUniformLocation(
            self.program, "exposure"
        )
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

    @override
    def render(self, current_time: float) -> None:
        green: list[float] = [0.0, 0.25, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, green)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glUseProgram(self.program)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glUniform1f(self.exposure_loc, self.exposure)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        _, self.exposure = imgui.slider_float(
            "Exposure",
            self.exposure,
            0.01,
            32.0,
            flags=imgui.SliderFlags_.logarithmic.value,
        )

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.program)
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteTextures(1, [self.texture])


if __name__ == "__main__":
    application.run(HdrexposureApp)
