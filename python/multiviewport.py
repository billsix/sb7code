"""multiviewport — same cube drawn into 4 viewports via geometry shader.

Port of ``src/multiviewport/multiviewport.cpp``. A single
draw call emits the same cube to 4 viewports, with the geometry shader
choosing the destination viewport via ``gl_ViewportIndex`` (one
invocation per viewport, ``layout (triangles, invocations = 4)``). Each
viewport gets its own MVP from a 4-element UBO. Demonstrates:

  - ``glViewportIndexedf`` for per-output viewport rectangles
  - geometry shader instancing (``invocations = 4``)
  - ``GL_UNIFORM_BUFFER`` (UBO) with ``glBindBufferBase`` + ``glMapBufferRange``
"""

from typing import override

import numpy as np
import OpenGL.GL as GL

from sb7 import application, common, vmath

VS_SOURCE: str = """
#version 420 core

in vec4 position;

out VS_OUT
{
    vec4 color;
} vs_out;

void main(void)
{
    gl_Position = position;
    vs_out.color = position * 2.0 + vec4(0.5, 0.5, 0.5, 0.0);
}
"""

GS_SOURCE: str = """
#version 420 core

layout (triangles, invocations = 4) in;
layout (triangle_strip, max_vertices = 3) out;

layout (std140, binding = 0) uniform transform_block
{
    mat4 mvp_matrix[4];
};

in VS_OUT
{
    vec4 color;
} gs_in[];

out GS_OUT
{
    vec4 color;
} gs_out;

void main(void)
{
    for (int i = 0; i < gl_in.length(); i++)
    {
        gs_out.color = gs_in[i].color;
        gl_Position = mvp_matrix[gl_InvocationID] * gl_in[i].gl_Position;
        gl_ViewportIndex = gl_InvocationID;
        EmitVertex();
    }
    EndPrimitive();
}
"""

FS_SOURCE: str = """
#version 420 core

out vec4 color;

in GS_OUT
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


class MultiviewportApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Multiple Viewports"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(
            vs=VS_SOURCE, gs=GS_SOURCE, fs=FS_SOURCE
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

        self.uniform_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.uniform_buffer)
        # 4 mat4s = 4 * 64 bytes = 256 bytes.
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 4 * 64, None, GL.GL_DYNAMIC_DRAW)

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 1.0]
        one: list[float] = [1.0]

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        # Four 7/16-of-screen viewports, one per corner.
        vw: float = (7 * self.info.window_width) / 16.0
        vh: float = (7 * self.info.window_height) / 16.0
        GL.glViewportIndexedf(0, 0, 0, vw, vh)
        GL.glViewportIndexedf(1, self.info.window_width - vw, 0, vw, vh)
        GL.glViewportIndexedf(2, 0, self.info.window_height - vh, vw, vh)
        GL.glViewportIndexedf(
            3, self.info.window_width - vw, self.info.window_height - vh, vw, vh
        )

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)

        # Build 4 MVPs, upload as a column-major contiguous block. We
        # could glMapBufferRange like the C++ side, but glBufferSubData
        # is simpler in Python and just as fast for 256 bytes.
        block: np.ndarray = np.empty((4, 4, 4), dtype=np.float32)
        for i in range(4):
            mv: np.ndarray = (
                vmath.translate(0.0, 0.0, -2.0)
                @ vmath.rotate(current_time * 45.0 * (i + 1), 0.0, 1.0, 0.0)
                @ vmath.rotate(current_time * 81.0 * (i + 1), 1.0, 0.0, 0.0)
            )
            mvp: np.ndarray = proj @ mv
            # Transpose: GLSL std140 mat4 is column-major; our vmath
            # matrices are row-major.
            block[i] = mvp.T

        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.uniform_buffer)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER, 0, block.nbytes, np.ascontiguousarray(block)
        )

        GL.glUseProgram(self.program)
        GL.glDrawElements(GL.GL_TRIANGLES, 36, GL.GL_UNSIGNED_SHORT, None)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)
        GL.glDeleteBuffers(1, [self.position_buffer])
        GL.glDeleteBuffers(1, [self.index_buffer])
        GL.glDeleteBuffers(1, [self.uniform_buffer])


if __name__ == "__main__":
    application.run(MultiviewportApp)
