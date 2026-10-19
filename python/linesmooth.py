"""linesmooth — wireframe cubes with GL_LINE_SMOOTH antialiasing.

Port of ``src/linesmooth/linesmooth.cpp``. Same indexed-
cube setup as ``indexedcube``, but rendered as wireframe (``glPolygonMode
GL_LINE``) with ``GL_LINE_SMOOTH`` enabled and alpha blending — the
GL line-AA path. The result is much smoother edges than aliased lines.
"""

import math
from typing import override

import numpy as np
import OpenGL.GL as GL

from sb7 import application, common, vmath

MANY_CUBES: bool = True


VS_SOURCE: str = """
#version 420 core

in vec4 position;

out VS_OUT
{
    vec4 color;
} vs_out;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

void main(void)
{
    gl_Position = proj_matrix * mv_matrix * position;
    vs_out.color = position * 2.0 + vec4(0.5, 0.5, 0.5, 0.0);
}
"""

FS_SOURCE: str = """
#version 420 core

out vec4 color;

in VS_OUT
{
    vec4 color;
} fs_in;

void main(void)
{
    color = fs_in.color;
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


class LinesmoothApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Line Smooth"

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

        # Wireframe + alpha-blended + GL_LINE_SMOOTH = smoothed lines.
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LINE_SMOOTH)

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
            mv = (
                vmath.translate(0.0, 0.0, -4.0)
                @ vmath.rotate(current_time * 45.0, 0.0, 1.0, 0.0)
                @ vmath.rotate(current_time * 81.0, 1.0, 0.0, 0.0)
            )
            vmath.upload(self.mv_location, mv)
            GL.glDrawElements(GL.GL_TRIANGLES, 36, GL.GL_UNSIGNED_SHORT, None)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)
        GL.glDeleteBuffers(1, [self.position_buffer])
        GL.glDeleteBuffers(1, [self.index_buffer])


if __name__ == "__main__":
    application.run(LinesmoothApp)
