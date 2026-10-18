"""gstessellate — geometry-shader-based tetrahedron tessellation.

Port of ``src/gstessellate/gstessellate.cpp``. A
4-triangle tetrahedron mesh is fed to a geometry shader that splits
each triangle into 4 sub-triangles using a stretchable mid-edge offset.
The ``stretch`` uniform animates so the shape "blooms" in and out.
Geometry-shader-based tessellation predates the dedicated TCS/TES
stages; this is the textbook cautionary example of why GS for
tessellation is awkward (max_vertices = 12 hardcoded, color computed
per-emit, etc.).
"""

import math
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

from sb7 import application, common, vmath

VS_SOURCE: str = """
#version 410 core

in vec4 vVertex;

void main(void)
{
    gl_Position = vVertex;
}
"""

GS_SOURCE: str = """
#version 410 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 12) out;

uniform float stretch = 0.7;

flat out vec4 color;

uniform mat4 mvpMatrix;
uniform mat4 mvMatrix;

void make_face(vec3 a, vec3 b, vec3 c)
{
    vec3 face_normal = normalize(cross(c - a, c - b));
    vec4 face_color = vec4(1.0, 0.4, 0.7, 1.0) * (mat3(mvMatrix) * face_normal).z;
    gl_Position = mvpMatrix * vec4(a, 1.0);
    color = face_color;
    EmitVertex();

    gl_Position = mvpMatrix * vec4(b, 1.0);
    color = face_color;
    EmitVertex();

    gl_Position = mvpMatrix * vec4(c, 1.0);
    color = face_color;
    EmitVertex();

    EndPrimitive();
}

void main(void)
{
    vec3 a = gl_in[0].gl_Position.xyz;
    vec3 b = gl_in[1].gl_Position.xyz;
    vec3 c = gl_in[2].gl_Position.xyz;

    vec3 d = (a + b) * stretch;
    vec3 e = (b + c) * stretch;
    vec3 f = (c + a) * stretch;

    a *= (2.0 - stretch);
    b *= (2.0 - stretch);
    c *= (2.0 - stretch);

    make_face(a, d, f);
    make_face(d, b, e);
    make_face(e, c, f);
    make_face(d, e, f);

    EndPrimitive();
}
"""

FS_SOURCE: str = """
#version 410 core

flat in vec4 color;
out vec4 output_color;

void main(void)
{
    output_color = color;
}
"""


TETRA_VERTS: np.ndarray = np.array(
    [
        0.000,
        0.000,
        1.000,
        0.943,
        0.000,
        -0.333,
        -0.471,
        0.816,
        -0.333,
        -0.471,
        -0.816,
        -0.333,
    ],
    dtype=np.float32,
)

TETRA_INDICES: np.ndarray = np.array(
    [
        0,
        1,
        2,
        0,
        2,
        3,
        0,
        3,
        1,
        3,
        2,
        1,
    ],
    dtype=np.uint16,
)


class GstessellateApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Geometry Shader Tessellation"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(
            vs=VS_SOURCE, gs=GS_SOURCE, fs=FS_SOURCE
        )
        self.mv_location: int = GL.glGetUniformLocation(
            self.program, "mvMatrix"
        )
        self.mvp_location: int = GL.glGetUniformLocation(
            self.program, "mvpMatrix"
        )
        self.stretch_location: int = GL.glGetUniformLocation(
            self.program, "stretch"
        )

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # The C++ source packs indices and vertices into one ELEMENT
        # array buffer; we follow the same trick. Buffer is bound as
        # both GL_ELEMENT_ARRAY_BUFFER (for the indices) and
        # GL_ARRAY_BUFFER (for the vertex positions, with offset).
        self.buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffer)
        total = TETRA_VERTS.nbytes + TETRA_INDICES.nbytes
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER, total, None, GL.GL_STATIC_DRAW
        )
        GL.glBufferSubData(
            GL.GL_ELEMENT_ARRAY_BUFFER, 0, TETRA_INDICES.nbytes, TETRA_INDICES
        )
        GL.glBufferSubData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            TETRA_INDICES.nbytes,
            TETRA_VERTS.nbytes,
            TETRA_VERTS,
        )

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.GLvoidp(TETRA_INDICES.nbytes)
        )
        GL.glEnableVertexAttribArray(0)

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 1.0]
        one: list[float] = [1.0]
        f: float = float(current_time)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glUseProgram(self.program)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -8.0)
            @ vmath.rotate(current_time * 71.0, 0.0, 1.0, 0.0)
            @ vmath.rotate(current_time * 10.0, 1.0, 0.0, 0.0)
        )

        vmath.upload(self.mvp_location, proj @ mv)
        vmath.upload(self.mv_location, mv)
        GL.glUniform1f(self.stretch_location, math.sin(f * 4.0) * 0.75 + 1.0)

        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, 12, GL.GL_UNSIGNED_SHORT, None)

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.program)
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.buffer])


if __name__ == "__main__":
    application.run(GstessellateApp)
