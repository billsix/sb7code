"""depthclamp — visualize GL_DEPTH_CLAMP on a stretched dragon.

Port of ``src/depthclamp/depthclamp.cpp``. Loads
``dragon.sbm`` and animates a per-vertex ``explode_factor`` that
inflates/deflates the model along its position vector. Without
``glEnable(GL_DEPTH_CLAMP)``, vertices that scale past the near plane
get clipped; with it on, they're clamped to the near depth instead
and stay visible.

First SBM-using demo — smoke test for ``sb6m.Object``.
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

layout (location = 0) in vec4 position;
layout (location = 1) in vec3 normal;

out VS_OUT
{
    vec3 normal;
    vec4 color;
} vs_out;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
uniform float explode_factor;

void main(void)
{
    gl_Position = proj_matrix * mv_matrix * position * vec4(vec3(explode_factor), 1.0);
    vs_out.color = position * 2.0 + vec4(0.5, 0.5, 0.5, 0.0);
    vs_out.normal = normalize(mat3(mv_matrix) * normal);
}
"""

FS_SOURCE: str = """
#version 410 core

out vec4 color;

in VS_OUT
{
    vec3 normal;
    vec4 color;
} fs_in;

void main(void)
{
    color = vec4(1.0) * abs(normalize(fs_in.normal).z);
}
"""


class DepthclampApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Depth Clamping"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)
        self.uniform_mv: int = GL.glGetUniformLocation(
            self.program, "mv_matrix"
        )
        self.uniform_proj: int = GL.glGetUniformLocation(
            self.program, "proj_matrix"
        )
        self.uniform_explode: int = GL.glGetUniformLocation(
            self.program, "explode_factor"
        )

        self.dragon: sb6m.Object = sb6m.Object()
        self.dragon.load(
            os.path.join(common.require_media(), "objects", "dragon.sbm")
        )

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
        proj: np.ndarray = vmath.perspective(50.0, aspect, 1.8, 1000.0)
        vmath.upload(self.uniform_proj, proj)

        GL.glEnable(GL.GL_DEPTH_CLAMP)

        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -10.0)
            @ vmath.rotate(f * 45.0, 0.0, 1.0, 0.0)
            @ vmath.rotate(f * 81.0, 1.0, 0.0, 0.0)
        )
        vmath.upload(self.uniform_mv, mv)

        explode: float = (
            math.sin(current_time * 3.0) * math.cos(current_time * 4.0) * 0.7
            + 1.1
        )
        GL.glUniform1f(self.uniform_explode, explode)

        self.dragon.render()

    @override
    def shutdown(self) -> None:
        self.dragon.free()
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(DepthclampApp)
