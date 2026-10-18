"""normalviewer — visualize per-vertex and face normals on a torus.

Port of ``src/normalviewer/normalviewer.cpp``. Loads
``torus.sbm`` and routes each triangle through a geometry shader that
emits *two* line-strip primitives per triangle: one from the vertex
position outward along the per-vertex normal, and one from the
triangle centroid outward along the face normal. Both lines have
their length animated by sin/cos of time.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL

from sb7 import application, common, sb6m, vmath

VS_SOURCE: str = """
#version 410 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec3 normal;

out VS_OUT
{
    vec3 normal;
    vec4 color;
} vs_out;

void main(void)
{
    gl_Position = position;
    vs_out.color = position * 2.0 + vec4(0.5, 0.5, 0.5, 0.0);
    vs_out.normal = normalize(normal);
}
"""

GS_SOURCE: str = """
#version 410 core

layout (triangles) in;
layout (line_strip, max_vertices = 4) out;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

in VS_OUT
{
    vec3 normal;
    vec4 color;
} gs_in[];

out GS_OUT
{
    vec3 normal;
    vec4 color;
} gs_out;

uniform float normal_length = 0.2;

void main(void)
{
    mat4 mvp = proj_matrix * mv_matrix;
    vec3 ab = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
    vec3 ac = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;
    vec3 face_normal = normalize(cross(ab, ac));

    vec4 tri_centroid = (gl_in[0].gl_Position +
                         gl_in[1].gl_Position +
                         gl_in[2].gl_Position) / 3.0;

    gl_Position = mvp * tri_centroid;
    gs_out.normal = gs_in[0].normal;
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (tri_centroid + vec4(face_normal * normal_length, 0.0));
    gs_out.normal = gs_in[0].normal;
    gs_out.color = gs_in[0].color;
    EmitVertex();
    EndPrimitive();

    gl_Position = mvp * gl_in[0].gl_Position;
    gs_out.normal = gs_in[0].normal;
    gs_out.color = gs_in[0].color;
    EmitVertex();

    gl_Position = mvp * (gl_in[0].gl_Position +
                         vec4(gs_in[0].normal * normal_length, 0.0));
    gs_out.normal = gs_in[0].normal;
    gs_out.color = gs_in[0].color;
    EmitVertex();
    EndPrimitive();
}
"""

FS_SOURCE: str = """
#version 410 core

out vec4 color;

in GS_OUT
{
    vec3 normal;
    vec4 color;
} fs_in;

void main(void)
{
    color = vec4(1.0) * abs(normalize(fs_in.normal).z);
}
"""


class NormalviewerApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Normal Viewer"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(
            vs=VS_SOURCE, gs=GS_SOURCE, fs=FS_SOURCE
        )
        self.uniform_mv: int = GL.glGetUniformLocation(
            self.program, "mv_matrix"
        )
        self.uniform_proj: int = GL.glGetUniformLocation(
            self.program, "proj_matrix"
        )
        self.uniform_nlen: int = GL.glGetUniformLocation(
            self.program, "normal_length"
        )

        self.torus: sb6m.Object = sb6m.Object()
        self.torus.load(
            os.path.join(common.require_media(), "objects", "torus.sbm")
        )

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
            vmath.translate(0.0, 0.0, -3.0)
            @ vmath.rotate(f * 15.0, 0.0, 1.0, 0.0)
            @ vmath.rotate(f * 21.0, 1.0, 0.0, 0.0)
        )
        vmath.upload(self.uniform_proj, proj)
        vmath.upload(self.uniform_mv, mv)
        GL.glUniform1f(
            self.uniform_nlen,
            math.sin(f * 8.0) * math.cos(f * 6.0) * 0.03 + 0.05,
        )

        self.torus.render()

    @override
    def shutdown(self) -> None:
        self.torus.free()
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(NormalviewerApp)
