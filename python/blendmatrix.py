"""blendmatrix — 19x19 grid of cubes, every blend mode crossed with
every other.

Port of ``src/blendmatrix/blendmatrix.cpp``. For each cell
(j, i) in the grid, ``glBlendFunc(blend_func[i], blend_func[j])`` is
set and a cube is drawn — so the picture is a complete table of every
SRC/DST factor combination, including the dual-source-blend variants
(``GL_SRC1_COLOR`` etc.). The cube vertex shader emits two color
outputs, one for ``src``, one for ``src1``, exercising dual-source blend.
"""

from typing import override

import numpy as np
import OpenGL.GL as GL

from sb7 import application, common, vmath

VS_SOURCE: str = """
#version 410 core

in vec4 position;

out VS_OUT
{
    vec4 color0;
    vec4 color1;
} vs_out;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

void main(void)
{
    gl_Position = proj_matrix * mv_matrix * position;
    vs_out.color0 = position * 2.0 + vec4(0.5, 0.5, 0.5, 0.0);
    vs_out.color1 = vec4(0.5, 0.5, 0.5, 0.0) - position * 2.0;
}
"""

FS_SOURCE: str = """
#version 410 core

layout (location = 0, index = 0) out vec4 color0;
layout (location = 0, index = 1) out vec4 color1;

in VS_OUT
{
    vec4 color0;
    vec4 color1;
} fs_in;

void main(void)
{
    color0 = vec4(fs_in.color0.xyz, 1.0);
    color1 = vec4(fs_in.color0.xyz, 1.0);
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


class BlendmatrixApp(application.Application):
    BLEND_FUNCS = [
        GL.GL_ZERO,
        GL.GL_ONE,
        GL.GL_SRC_COLOR,
        GL.GL_ONE_MINUS_SRC_COLOR,
        GL.GL_DST_COLOR,
        GL.GL_ONE_MINUS_DST_COLOR,
        GL.GL_SRC_ALPHA,
        GL.GL_ONE_MINUS_SRC_ALPHA,
        GL.GL_DST_ALPHA,
        GL.GL_ONE_MINUS_DST_ALPHA,
        GL.GL_CONSTANT_COLOR,
        GL.GL_ONE_MINUS_CONSTANT_COLOR,
        GL.GL_CONSTANT_ALPHA,
        GL.GL_ONE_MINUS_CONSTANT_ALPHA,
        GL.GL_SRC_ALPHA_SATURATE,
        GL.GL_SRC1_COLOR,
        GL.GL_ONE_MINUS_SRC1_COLOR,
        GL.GL_SRC1_ALPHA,
        GL.GL_ONE_MINUS_SRC1_ALPHA,
    ]

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Blending Functions"

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
        orange: list[float] = [0.6, 0.4, 0.1, 1.0]
        one: list[float] = [1.0]

        n: int = len(self.BLEND_FUNCS)
        x_scale = 20.0 / n
        y_scale = 16.0 / n
        t: float = float(current_time)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, orange)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glUseProgram(self.program)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        vmath.upload(self.proj_location, proj)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendColor(0.2, 0.5, 0.7, 0.5)
        for j in range(n):
            for i in range(n):
                mv: np.ndarray = (
                    vmath.translate(9.5 - x_scale * i, 7.5 - y_scale * j, -18.0)
                    @ vmath.rotate(t * -45.0, 0.0, 1.0, 0.0)
                    @ vmath.rotate(t * -21.0, 1.0, 0.0, 0.0)
                )
                vmath.upload(self.mv_location, mv)
                GL.glBlendFunc(self.BLEND_FUNCS[i], self.BLEND_FUNCS[j])
                GL.glDrawElements(
                    GL.GL_TRIANGLES, 36, GL.GL_UNSIGNED_SHORT, None
                )

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)
        GL.glDeleteBuffers(1, [self.position_buffer])
        GL.glDeleteBuffers(1, [self.index_buffer])


if __name__ == "__main__":
    application.run(BlendmatrixApp)
