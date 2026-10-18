"""tessellatedcube — cube faces drawn as 4-vertex tessellation patches.

Port of ``src/tessellatedcube/tessellatedcube.cpp``. Each
of the cube's 6 faces is fed as a 4-vertex patch (``glPatchParameteri(
GL_PATCH_VERTICES, 4)``). The TCS computes a *per-edge* tessellation
factor based on each edge's screen-space length after MVP — so faces
near the camera get many subdivisions, distant faces stay coarse.
The TES interpolates the patch coordinate as a quad and pushes the
position out by 0.25× to give each cube some volume; the FS shades by
absolute world-space normal.

The C++ source defaults to ``MANY_CUBES`` (100 cubes flying around) —
same default here.

Imgui controls: Wireframe checkbox.
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

void main(void)
{
    gl_Position = position;
}
"""

TCS_SOURCE: str = """
#version 420 core

layout (vertices = 4) out;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

void main(void)
{
    vec4 pos[4];
    float tf[4];
    float t = 0.0;

    int i;

    if (gl_InvocationID == 0)
    {
        for (i = 0; i < 4; i++)
        {
            pos[i] = proj_matrix * mv_matrix * gl_in[i].gl_Position;
        }

        tf[0] = max(2.0, distance(pos[0].xy / pos[0].w,
                                  pos[1].xy / pos[1].w) * 16.0);
        tf[1] = max(2.0, distance(pos[1].xy / pos[1].w,
                                  pos[3].xy / pos[3].w) * 16.0);
        tf[2] = max(2.0, distance(pos[2].xy / pos[2].w,
                                  pos[3].xy / pos[3].w) * 16.0);
        tf[3] = max(2.0, distance(pos[2].xy / pos[2].w,
                                  pos[0].xy / pos[0].w) * 16.0);
        for (i = 0; i < 4; i++)
        {
            t = max(t, tf[i]);
        }

        gl_TessLevelInner[0] = t;
        gl_TessLevelInner[1] = t;
        gl_TessLevelOuter[0] = tf[0];
        gl_TessLevelOuter[1] = tf[1];
        gl_TessLevelOuter[2] = tf[2];
        gl_TessLevelOuter[3] = tf[3];
    }

    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
"""

TES_SOURCE: str = """
#version 420 core

layout (quads, fractional_odd_spacing, ccw) in;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

out vec3 normal;

void main(void)
{
    vec4 mid1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
    vec4 mid2 = mix(gl_in[2].gl_Position, gl_in[3].gl_Position, gl_TessCoord.x);
    vec4 pos = mix(mid1, mid2, gl_TessCoord.y);
    pos.xyz = pos.xyz * 0.25;
    normal = normalize(mat3(mv_matrix) * pos.xyz);
    gl_Position = proj_matrix * mv_matrix * pos;
}
"""

FS_SOURCE: str = """
#version 420 core

out vec4 color;

in vec3 normal;

void main(void)
{
    color = vec4(abs(normal), 1.0);
}
"""


VERTEX_INDICES: np.ndarray = np.array(
    [
        0,
        1,
        2,
        3,
        2,
        3,
        4,
        5,
        4,
        5,
        6,
        7,
        6,
        7,
        0,
        1,
        0,
        2,
        6,
        4,
        1,
        7,
        3,
        5,
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


class TessellatedcubeApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.wireframe_mode: bool = False

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Tessellated Cube"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(
            vs=VS_SOURCE, tcs=TCS_SOURCE, tes=TES_SOURCE, fs=FS_SOURCE
        )
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
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

    @override
    def render(self, current_time: float) -> None:
        green: list[float] = [0.0, 0.25, 0.0, 1.0]
        one: list[float] = [1.0]

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, green)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glUseProgram(self.program)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        vmath.upload(self.proj_location, proj)
        GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 4)

        GL.glPolygonMode(
            GL.GL_FRONT_AND_BACK,
            GL.GL_LINE if self.wireframe_mode else GL.GL_FILL,
        )

        if MANY_CUBES:
            for i in range(100):
                f: float = float(i) + current_time * 0.03
                mv: np.ndarray = (
                    vmath.translate(0.0, 0.0, -10.0)
                    @ vmath.translate(
                        math.sin(2.1 * f) * 4.0,
                        math.cos(1.7 * f) * 4.0,
                        math.sin(4.3 * f) * math.cos(3.5 * f) * 30.0,
                    )
                    @ vmath.rotate(current_time * 3.0, 1.0, 0.0, 0.0)
                    @ vmath.rotate(current_time * 5.0, 0.0, 1.0, 0.0)
                )
                vmath.upload(self.mv_location, mv)
                GL.glDrawElements(GL.GL_PATCHES, 24, GL.GL_UNSIGNED_SHORT, None)
        else:
            f: float = current_time * 0.3
            mv: np.ndarray = (
                vmath.translate(0.0, 0.0, -3.0)
                @ vmath.translate(
                    0.0, 0.0, math.sin(1.3 * f) * math.cos(1.5 * f) * 15.0
                )
                @ vmath.rotate(current_time * 81.0, 1.0, 0.0, 0.0)
            )
            vmath.upload(self.mv_location, mv)
            GL.glDrawElements(GL.GL_PATCHES, 24, GL.GL_UNSIGNED_SHORT, None)

    @override
    def draw_controls(self) -> None:
        _, self.wireframe_mode = imgui.checkbox(
            "Wireframe", self.wireframe_mode
        )

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)
        GL.glDeleteBuffers(1, [self.position_buffer])
        GL.glDeleteBuffers(1, [self.index_buffer])


if __name__ == "__main__":
    application.run(TessellatedcubeApp)
