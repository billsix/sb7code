"""noperspective — show the difference between perspective-correct
and screen-space (linear) interpolation.

Port of ``src/noperspective/noperspective.cpp``. A
checkerboard quad rotates in 3D. The vertex shader writes the same
texture coordinates twice — once with default perspective-correct
interpolation, once with the ``noperspective`` qualifier. The fragment
shader picks one or the other based on a uniform; press M to toggle.

When perspective-correct is on, the checker pattern follows the
foreshortening of the plane — squares stay square in 3D. When
``noperspective`` is on, the pattern interpolates linearly across
screen space and looks distorted near the horizon. Classic teaching
demo for what perspective-correct interpolation actually buys you.

Imgui controls: a checkbox for the perspective qualifier and a Pause
checkbox.
"""

from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, vmath

VS_SOURCE: str = """
#version 410 core

out VS_OUT
{
    vec2 tc;
    noperspective vec2 tc_np;
} vs_out;

uniform mat4 mvp;

void main(void)
{
    const vec4 vertices[] = vec4[](vec4(-0.5, -0.5, 0.0, 1.0),
                                   vec4( 0.5, -0.5, 0.0, 1.0),
                                   vec4(-0.5,  0.5, 0.0, 1.0),
                                   vec4( 0.5,  0.5, 0.0, 1.0));

    vec2 tc = (vertices[gl_VertexID].xy + vec2(0.5));
    vs_out.tc = tc;
    vs_out.tc_np = tc;
    gl_Position = mvp * vertices[gl_VertexID];
}
"""

FS_SOURCE: str = """
#version 410 core

out vec4 color;

uniform sampler2D tex_checker;
uniform bool use_perspective = true;

in VS_OUT
{
    vec2 tc;
    noperspective vec2 tc_np;
} fs_in;

void main(void)
{
    vec2 tc = mix(fs_in.tc_np, fs_in.tc, bvec2(use_perspective));
    color = texture(tex_checker, tc).rrrr;
}
"""

# 8x8 checkerboard, 1 byte per texel.
CHECKER_DATA: np.ndarray = np.array(
    [
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
        0xFF,
        0x00,
    ],
    dtype=np.uint8,
)


class NoperspectiveApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.paused: bool = False
        self.use_perspective: bool = True
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Perspective"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)
        self.mvp_loc: int = GL.glGetUniformLocation(self.program, "mvp")
        self.use_perspective_loc: int = GL.glGetUniformLocation(
            self.program, "use_perspective"
        )

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.tex_checker: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_checker)
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_R8, 8, 8)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            8,
            8,
            GL.GL_RED,
            GL.GL_UNSIGNED_BYTE,
            CHECKER_DATA,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        one: list[float] = [1.0]

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        t: float = self._total_time * 14.3

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        mv: np.ndarray = vmath.translate(0.0, 0.0, -1.5) @ vmath.rotate(
            t, 0.0, 1.0, 0.0
        )
        proj: np.ndarray = vmath.perspective(60.0, aspect, 0.1, 1000.0)

        GL.glUseProgram(self.program)
        vmath.upload(self.mvp_loc, proj @ mv)
        GL.glUniform1i(self.use_perspective_loc, int(self.use_perspective))

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        _, self.use_perspective = imgui.checkbox(
            "Perspective-correct interpolation", self.use_perspective
        )
        _, self.paused = imgui.checkbox("Pause rotation", self.paused)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)
        GL.glDeleteTextures(1, [self.tex_checker])


if __name__ == "__main__":
    application.run(NoperspectiveApp)
