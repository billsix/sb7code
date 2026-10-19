"""grass — 1M instanced grass blades on a wind-blown plane.

Port of ``src/grass/grass.cpp``. A 6-vertex triangle
strip representing one curved grass blade is drawn as 1024×1024
instances. Each blade samples 4 KTX maps in the vertex shader to pick
its length, rotation, color, and per-blade bend amount, plus an
in-shader hash function for randomized color jitter.

The shader expects a ``grasspalette_texture`` at binding 0 too; the
C++ source never binds anything to texture unit 0, so reads return 0
and the blades fall back to the random-jittered base color. We
preserve that behavior — same as the upstream demo.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

from sb7 import application, common, ktx, vmath

VS_SOURCE: str = """
#version 420 core

in vec4 vVertex;
out vec4 color;

uniform mat4 mvpMatrix;

layout (binding = 0) uniform sampler1D grasspalette_texture;
layout (binding = 1) uniform sampler2D length_texture;
layout (binding = 2) uniform sampler2D orientation_texture;
layout (binding = 3) uniform sampler2D grasscolor_texture;
layout (binding = 4) uniform sampler2D bend_texture;

int random(int seed, int iterations)
{
    int value = seed;
    for (int n = 0; n < iterations; n++) {
        value = ((value >> 7) ^ (value << 9)) * 15485863;
    }
    return value;
}

vec4 random_vector(int seed)
{
    int r = random(gl_InstanceID, 4);
    int g = random(r, 2);
    int b = random(g, 2);
    int a = random(b, 2);

    return vec4(float(r & 0x3FF) / 1024.0,
                float(g & 0x3FF) / 1024.0,
                float(b & 0x3FF) / 1024.0,
                float(a & 0x3FF) / 1024.0);
}

mat4 construct_rotation_matrix(float angle)
{
    float st = sin(angle);
    float ct = cos(angle);
    return mat4(vec4(ct, 0.0, st, 0.0),
                vec4(0.0, 1.0, 0.0, 0.0),
                vec4(-st, 0.0, ct, 0.0),
                vec4(0.0, 0.0, 0.0, 1.0));
}

void main(void)
{
    vec4 offset = vec4(float(gl_InstanceID >> 10) - 512.0,
                       0.0,
                       float(gl_InstanceID & 0x3FF) - 512.0,
                       0.0);
    int number1 = random(gl_InstanceID, 3);
    int number2 = random(number1, 2);
    offset += vec4(float(number1 & 0xFF) / 256.0,
                   0.0,
                   float(number2 & 0xFF) / 256.0,
                   0.0);

    vec2 texcoord = offset.xz / 1024.0 + vec2(0.5);

    float bend_factor = texture(bend_texture, texcoord).r * 2.0;
    float bend_amount = cos(vVertex.y);

    float angle = texture(orientation_texture, texcoord).r * 2.0 * 3.141592;
    mat4 rot = construct_rotation_matrix(angle);
    vec4 position = (rot * (vVertex + vec4(0.0, 0.0, bend_amount * bend_factor, 0.0))) + offset;

    position *= vec4(1.0, texture(length_texture, texcoord).r * 0.9 + 0.3, 1.0, 1.0);

    gl_Position = mvpMatrix * position;
    color = texture(grasspalette_texture, texture(grasscolor_texture, texcoord).r) +
            vec4(random_vector(gl_InstanceID).xyz * vec3(0.1, 0.5, 0.1), 1.0);
}
"""

FS_SOURCE: str = """
#version 420 core

in vec4 color;
out vec4 output_color;

void main(void)
{
    output_color = color;
}
"""

GRASS_BLADE: np.ndarray = np.array(
    [
        -0.30,
        0.0,
        0.30,
        0.0,
        -0.20,
        1.0,
        0.10,
        1.3,
        -0.05,
        2.3,
        0.00,
        3.3,
    ],
    dtype=np.float32,
)


class GrassApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Grass"

    @override
    def startup(self) -> None:
        self.grass_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.grass_buffer)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            GRASS_BLADE.nbytes,
            GRASS_BLADE,
            GL.GL_STATIC_DRAW,
        )

        self.grass_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.grass_vao)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        self.grass_program: int = common.compile_program(
            vs=VS_SOURCE, fs=FS_SOURCE
        )
        self.uniform_mvp: int = GL.glGetUniformLocation(
            self.grass_program, "mvpMatrix"
        )

        media: str = common.require_media()
        # Load each KTX into its own active-texture-unit slot.
        for unit, name in [
            (1, "grass_length.ktx"),
            (2, "grass_orientation.ktx"),
            (3, "grass_color.ktx"),
            (4, "grass_bend.ktx"),
        ]:
            # PyOpenGL constants are int subclasses at runtime, but the
            # stubs' `Constant` declares no `__add__`.
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)  # ty: ignore[unsupported-operator]
            tex: int = ktx.load(os.path.join(media, "textures", name))
            if not tex:
                raise RuntimeError(f"failed to load {name}")
            setattr(self, f"tex_{name.split('.')[0]}", tex)
        GL.glActiveTexture(GL.GL_TEXTURE0)

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 1.0]
        one: list[float] = [1.0]
        t: float = float(current_time) * 0.02
        r: float = 550.0

        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        eye: tuple[float, float, float] = (
            math.sin(t) * r,
            25.0,
            math.cos(t) * r,
        )
        mv: np.ndarray = vmath.lookat(eye, (0.0, -50.0, 0.0), (0.0, 1.0, 0.0))
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(45.0, aspect, 0.1, 1000.0)

        GL.glUseProgram(self.grass_program)
        vmath.upload(self.uniform_mvp, proj @ mv)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glBindVertexArray(self.grass_vao)
        GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 6, 1024 * 1024)

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.grass_program)
        GL.glDeleteBuffers(1, [self.grass_buffer])
        GL.glDeleteVertexArrays(1, [self.grass_vao])


if __name__ == "__main__":
    application.run(GrassApp)
