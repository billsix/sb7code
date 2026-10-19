"""instancedattribs — instanced rendering via per-instance vertex
attributes (``glVertexAttribDivisor``).

Port of ``src/instancedattribs/instancedattribs.cpp``. A
single 4-vertex square is drawn 4 times via ``glDrawArraysInstanced``.
Per-vertex data is the unit square; per-instance data are the corners'
positions and colors. ``glVertexAttribDivisor(loc, 1)`` tells GL "this
attribute advances once per instance, not per vertex" — the instancing
mechanism in core profile.
"""

from typing import override

import glfw  # noqa: F401  (must import before imgui_bundle on some setups)
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

from sb7 import application, common

VS_SOURCE: str = """
#version 410 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 instance_color;
layout (location = 2) in vec4 instance_position;

out Fragment
{
    vec4 color;
} fragment;

void main(void)
{
    gl_Position = (position + instance_position) * vec4(0.25, 0.25, 1.0, 1.0);
    fragment.color = instance_color;
}
"""

FS_SOURCE: str = """
#version 410 core
precision highp float;

in Fragment
{
    vec4 color;
} fragment;

out vec4 color;

void main(void)
{
    color = fragment.color;
}
"""

SQUARE_VERTICES: np.ndarray = np.array(
    [
        -1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        -1.0,
        1.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)

INSTANCE_COLORS: np.ndarray = np.array(
    [
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)

INSTANCE_POSITIONS: np.ndarray = np.array(
    [
        -2.0,
        -2.0,
        0.0,
        0.0,
        2.0,
        -2.0,
        0.0,
        0.0,
        2.0,
        2.0,
        0.0,
        0.0,
        -2.0,
        2.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


class InstancedattribsApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Instanced Attributes"

    @override
    def startup(self) -> None:
        self.square_vao: int = GL.glGenVertexArrays(1)
        self.square_buffer: int = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.square_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.square_buffer)

        total: int = (
            SQUARE_VERTICES.nbytes
            + INSTANCE_COLORS.nbytes
            + INSTANCE_POSITIONS.nbytes
        )
        GL.glBufferData(GL.GL_ARRAY_BUFFER, total, None, GL.GL_STATIC_DRAW)
        offset: int = 0
        GL.glBufferSubData(
            GL.GL_ARRAY_BUFFER, offset, SQUARE_VERTICES.nbytes, SQUARE_VERTICES
        )
        offset += SQUARE_VERTICES.nbytes
        GL.glBufferSubData(
            GL.GL_ARRAY_BUFFER, offset, INSTANCE_COLORS.nbytes, INSTANCE_COLORS
        )
        offset += INSTANCE_COLORS.nbytes
        GL.glBufferSubData(
            GL.GL_ARRAY_BUFFER,
            offset,
            INSTANCE_POSITIONS.nbytes,
            INSTANCE_POSITIONS,
        )

        # Layout: square_vertices (16 floats) | instance_colors (16) | instance_positions (16)
        GL.glVertexAttribPointer(0, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glVertexAttribPointer(
            1,
            4,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            0,
            GL.GLvoidp(SQUARE_VERTICES.nbytes),
        )
        GL.glVertexAttribPointer(
            2,
            4,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            0,
            GL.GLvoidp(SQUARE_VERTICES.nbytes + INSTANCE_COLORS.nbytes),
        )

        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(1)
        GL.glEnableVertexAttribArray(2)

        # Attributes 1 and 2 advance once per *instance*, not per vertex.
        GL.glVertexAttribDivisor(1, 1)
        GL.glVertexAttribDivisor(2, 1)

        self.square_program: int = common.compile_program(
            vs=VS_SOURCE, fs=FS_SOURCE
        )

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.square_program)
        GL.glDeleteBuffers(1, [self.square_buffer])
        GL.glDeleteVertexArrays(1, [self.square_vao])

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        GL.glUseProgram(self.square_program)
        GL.glBindVertexArray(self.square_vao)
        GL.glDrawArraysInstanced(GL.GL_TRIANGLE_FAN, 0, 4, 4)


if __name__ == "__main__":
    application.run(InstancedattribsApp)
