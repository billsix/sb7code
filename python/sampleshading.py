"""sampleshading — toggle GL_SAMPLE_SHADING with MSAA 8x.

Port of ``src/sampleshading/sampleshading.cpp``. A
high-frequency shader pattern (``fract(val) >= 0.5``) is rendered onto
24 cubes with an 8-sample MSAA framebuffer. Pressing S toggles
``GL_SAMPLE_SHADING`` — when on, the fragment shader runs once per
sample (not just once per pixel) so the high-frequency pattern is
properly antialiased; when off, you see aliasing despite the MSAA.

Demonstrates the difference between geometric MSAA (always on at
samples=8) and shading-rate MSAA (only with sample shading).

Imgui control: Sample shading checkbox.
"""

import math
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, vmath

MANY_CUBES: bool = True


VS_SOURCE: str = """
#version 420 core

in vec4 position;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

out VS_OUT
{
    vec2 tc;
} vs_out;

void main(void)
{
    gl_Position = proj_matrix * mv_matrix * position;
    vs_out.tc = position.xy;
}
"""

FS_SOURCE: str = """
#version 420 core

out vec4 color;

in VS_OUT
{
    vec2 tc;
} fs_in;

void main(void)
{
    float val = abs(fs_in.tc.x + fs_in.tc.y) * 30.0f;
    color = vec4(fract(val) >= 0.5 ? 1.0 : 0.2);
}
"""


VERTEX_INDICES: np.ndarray = np.array(
    [
        0,
        1,
        2,
        2,
        1,
        3,
        2,
        3,
        4,
        4,
        3,
        5,
        4,
        5,
        6,
        6,
        5,
        7,
        6,
        7,
        0,
        0,
        7,
        1,
        6,
        0,
        2,
        2,
        4,
        6,
        7,
        5,
        3,
        7,
        3,
        1,
    ],
    dtype=np.uint16,
)

VERTEX_POSITIONS: np.ndarray = np.array(
    [
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
    ],
    dtype=np.float32,
)


class SampleshadingApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.sample_shading: bool = False

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Sample Rate Shading"
        self.info.samples = 8

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)
        self.mv_location: int = GL.glGetUniformLocation(
            self.program, "mv_matrix"
        )
        self.proj_location: int = GL.glGetUniformLocation(
            self.program, "proj_matrix"
        )

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.position_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.position_buffer)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            VERTEX_POSITIONS.nbytes,
            VERTEX_POSITIONS,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        self.index_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            VERTEX_INDICES.nbytes,
            VERTEX_INDICES,
            GL.GL_STATIC_DRAW,
        )

        GL.glEnable(GL.GL_CULL_FACE)

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 1.0]

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        GL.glUseProgram(self.program)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        vmath.upload(self.proj_location, proj)

        if self.sample_shading:
            GL.glEnable(GL.GL_SAMPLE_SHADING)
            GL.glMinSampleShading(1.0)
        else:
            GL.glDisable(GL.GL_SAMPLE_SHADING)

        if MANY_CUBES:
            for i in range(24):
                f: float = float(i) + current_time * 0.3
                mv: np.ndarray = (
                    vmath.translate(0.0, 0.0, -20.0)
                    @ vmath.rotate(current_time * 45.0, 0.0, 1.0, 0.0)
                    @ vmath.rotate(current_time * 21.0, 1.0, 0.0, 0.0)
                    @ vmath.translate(
                        math.sin(2.1 * f) * 2.0,
                        math.cos(1.7 * f) * 2.0,
                        math.sin(1.3 * f) * math.cos(1.5 * f) * 2.0,
                    )
                )
                vmath.upload(self.mv_location, mv)
                GL.glDrawElements(
                    GL.GL_TRIANGLES, 36, GL.GL_UNSIGNED_SHORT, None
                )
        else:
            mv: np.ndarray = (
                vmath.translate(0.0, 0.0, -2.0)
                @ vmath.rotate(current_time * 45.0, 0.0, 1.0, 0.0)
                @ vmath.rotate(current_time * 81.0, 1.0, 0.0, 0.0)
            )
            vmath.upload(self.mv_location, mv)
            GL.glDrawElements(GL.GL_TRIANGLES, 36, GL.GL_UNSIGNED_SHORT, None)

    @override
    def draw_controls(self) -> None:
        _, self.sample_shading = imgui.checkbox(
            "Sample shading (per-sample FS)", self.sample_shading
        )

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)
        GL.glDeleteBuffers(1, [self.position_buffer])
        GL.glDeleteBuffers(1, [self.index_buffer])


if __name__ == "__main__":
    application.run(SampleshadingApp)
