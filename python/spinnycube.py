"""spinnycube — a single rotating cube (or 24, with MANY_CUBES on).

Port of ``src/spinnycube/spinnycube.cpp``. First demo
that exercises ``vmath``: builds an MV matrix from translate + rotate
each frame, projection from ``perspective``, uploads both as
``mat4`` uniforms.

The cube geometry is the same hand-rolled 36-vertex CW-front-face cube
as in the C++ source — kept verbatim so students can compare the two
files line-for-line.

Set ``MANY_CUBES = True`` to draw 24 of them (the ``#define MANY_CUBES``
branch in the original). Default is the single-cube version.
"""

import math
from typing import override

import numpy as np
import OpenGL.GL as GL

from sb7 import application, common, vmath

MANY_CUBES: bool = False


VS_SOURCE: str = """
#version 410 core

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
#version 410 core

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


VERTEX_POSITIONS: np.ndarray = np.array(
    [
        -0.25,
        0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
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
        0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
    ],
    dtype=np.float32,
)


class SpinnycubeApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Spinny Cube"

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

        self.buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            VERTEX_POSITIONS.nbytes,
            VERTEX_POSITIONS,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glFrontFace(GL.GL_CW)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

        self.proj_matrix: np.ndarray = vmath.identity()
        self.aspect: float = 1.0
        self.on_resize(self.info.window_width, self.info.window_height)

    @override
    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.aspect = float(width) / float(max(1, height))
        self.proj_matrix = vmath.perspective(50.0, self.aspect, 0.1, 1000.0)

    @override
    def render(self, current_time: float) -> None:
        green: list[float] = [0.0, 0.25, 0.0, 1.0]
        one: list[float] = [1.0]
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, green)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glUseProgram(self.program)
        vmath.upload(self.proj_location, self.proj_matrix)

        if MANY_CUBES:
            for i in range(24):
                f: float = float(i) + current_time * 0.3
                mv: np.ndarray = (
                    vmath.translate(0.0, 0.0, -6.0)
                    @ vmath.rotate(current_time * 45.0, 0.0, 1.0, 0.0)
                    @ vmath.rotate(current_time * 21.0, 1.0, 0.0, 0.0)
                    @ vmath.translate(
                        math.sin(2.1 * f) * 2.0,
                        math.cos(1.7 * f) * 2.0,
                        math.sin(1.3 * f) * math.cos(1.5 * f) * 2.0,
                    )
                )
                vmath.upload(self.mv_location, mv)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)
        else:
            f: float = current_time * 0.3
            mv: np.ndarray = (
                vmath.translate(0.0, 0.0, -4.0)
                @ vmath.translate(
                    math.sin(2.1 * f) * 0.5,
                    math.cos(1.7 * f) * 0.5,
                    math.sin(1.3 * f) * math.cos(1.5 * f) * 2.0,
                )
                @ vmath.rotate(current_time * 45.0, 0.0, 1.0, 0.0)
                @ vmath.rotate(current_time * 81.0, 1.0, 0.0, 0.0)
            )
            vmath.upload(self.mv_location, mv)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)
        GL.glDeleteBuffers(1, [self.buffer])


if __name__ == "__main__":
    application.run(SpinnycubeApp)
