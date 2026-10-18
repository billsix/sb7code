"""gsculling — geometry shader does back-face culling against a moving viewpoint.

Port of ``src/gsculling/gsculling.cpp``. Loads
``dragon.sbm`` and routes every triangle through a geometry shader that
discards the triangle if its face normal points away from a
time-varying ``viewpoint`` uniform. The viewpoint orbits the model on
its own arbitrary path, so triangles get culled differently from what
``glEnable(GL_CULL_FACE)`` would do — you can see the dragon get
"hollowed out" from the perspective of the moving viewpoint, even
though the camera itself doesn't move.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

from sb7 import application, common, sb6m, vmath

VS_SOURCE: str = """
#version 410 core

layout (location = 0) in vec4 vVertex;
layout (location = 1) in vec3 vNormal;

out Vertex
{
    vec3 normal;
    vec4 color;
} vertex;

uniform vec3 vLightPosition = vec3(-10.0, 40.0, 200.0);
uniform mat4 mvMatrix;

void main(void)
{
    vec3 vEyeNormal = mat3(mvMatrix) * normalize(vNormal);
    vec4 vPosition4 = mvMatrix * vVertex;
    vec3 vPosition3 = vPosition4.xyz / vPosition4.w;
    vec3 vLightDir = normalize(vLightPosition - vPosition3);
    vertex.color = vec4(0.7, 0.6, 1.0, 1.0) * abs(dot(vEyeNormal, vLightDir));
    gl_Position = vVertex;
    vertex.normal = vNormal;
}
"""

GS_SOURCE: str = """
#version 410 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in Vertex
{
    vec3 normal;
    vec4 color;
} vertex[];

out vec4 color;

uniform mat4 mvpMatrix;
uniform mat4 mvMatrix;
uniform vec3 viewpoint;

void main(void)
{
    vec3 ab = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
    vec3 ac = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;
    vec3 normal = normalize(cross(ab, ac));
    vec4 worldspace = gl_in[0].gl_Position;
    vec3 vt = normalize(viewpoint - worldspace.xyz);

    if (dot(normal, vt) > 0.0) {
        for (int n = 0; n < 3; n++) {
            gl_Position = mvpMatrix * gl_in[n].gl_Position;
            color = vertex[n].color;
            EmitVertex();
        }
        EndPrimitive();
    }
}
"""

FS_SOURCE: str = """
#version 410 core

in vec4 color;
out vec4 output_color;

void main(void)
{
    output_color = color;
}
"""


class GscullingApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Geometry Shader Culling"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(
            vs=VS_SOURCE, gs=GS_SOURCE, fs=FS_SOURCE
        )
        self.uniform_mv: int = GL.glGetUniformLocation(self.program, "mvMatrix")
        self.uniform_mvp: int = GL.glGetUniformLocation(
            self.program, "mvpMatrix"
        )
        self.uniform_vp: int = GL.glGetUniformLocation(
            self.program, "viewpoint"
        )

        self.dragon: sb6m.Object = sb6m.Object()
        self.dragon.load(
            os.path.join(common.require_media(), "objects", "dragon.sbm")
        )

        GL.glDisable(GL.GL_CULL_FACE)
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
            vmath.translate(0.0, 0.0, -20.0)
            @ vmath.rotate(f * 5.0, 0.0, 1.0, 0.0)
            @ vmath.rotate(f * 100.0, 1.0, 0.0, 0.0)
        )

        vmath.upload(self.uniform_mvp, proj @ mv)
        vmath.upload(self.uniform_mv, mv)
        viewpoint = (
            math.sin(f * 2.1) * 70.0,
            math.cos(f * 1.4) * 70.0,
            math.sin(f * 0.7) * 70.0,
        )
        GL.glUniform3fv(self.uniform_vp, 1, viewpoint)

        self.dragon.render()

    @override
    def shutdown(self) -> None:
        self.dragon.free()
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(GscullingApp)
