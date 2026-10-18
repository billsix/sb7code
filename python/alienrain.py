"""alienrain — 256 falling alien sprites driven by a UBO.

Port of ``src/alienrain/alienrain.cpp``. ``aliens.ktx``
is a 2D texture array of 64 alien sprites; 256 instances are drawn,
each picking one alien at ``alien_index % 64``. Per-instance
position/rotation comes from a 256-element ``droplet_t`` UBO that's
remapped each frame.

Validates the ``GL_TEXTURE_2D_ARRAY`` path through ``ktx.load`` —
the file's ``arrayelements`` field is non-zero so the loader picks
the array target automatically.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

from sb7 import application, common, ktx

VS_SOURCE: str = """
#version 410 core

layout (location = 0) in int alien_index;

out VS_OUT
{
    flat int alien;
    vec2 tc;
} vs_out;

struct droplet_t
{
    float x_offset;
    float y_offset;
    float orientation;
    float unused;
};

layout (std140) uniform droplets
{
    droplet_t droplet[256];
};

void main(void)
{
    const vec2[4] position = vec2[4](vec2(-0.5, -0.5),
                                     vec2( 0.5, -0.5),
                                     vec2(-0.5,  0.5),
                                     vec2( 0.5,  0.5));
    vs_out.tc = position[gl_VertexID].xy + vec2(0.5);
    float co = cos(droplet[alien_index].orientation);
    float so = sin(droplet[alien_index].orientation);
    mat2 rot = mat2(vec2(co, so),
                    vec2(-so, co));
    vec2 pos = 0.25 * rot * position[gl_VertexID];
    gl_Position = vec4(pos.x + droplet[alien_index].x_offset,
                       pos.y + droplet[alien_index].y_offset,
                       0.5, 1.0);
    vs_out.alien = alien_index % 64;
}
"""

FS_SOURCE: str = """
#version 410 core

layout (location = 0) out vec4 color;

in VS_OUT
{
    flat int alien;
    vec2 tc;
} fs_in;

uniform sampler2DArray tex_aliens;

void main(void)
{
    color = texture(tex_aliens, vec3(fs_in.tc, float(fs_in.alien)));
}
"""


# Same LCG as starfield/shapedpoints (seed 0x13371337 × 16807 each step).
class _Random:
    def __init__(self, seed: int = 0x13371337) -> None:
        self.seed: int = seed

    def next_float(self) -> float:
        self.seed = (self.seed * 16807) & 0xFFFFFFFF
        tmp = self.seed ^ (self.seed >> 4) ^ ((self.seed << 15) & 0xFFFFFFFF)
        bits = (tmp >> 9) | 0x3F800000
        f = np.frombuffer(
            np.uint32(bits & 0xFFFFFFFF).tobytes(), dtype=np.float32
        )[0]
        return float(f - 1.0)


class AlienrainApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Alien Rain"

    @override
    def startup(self) -> None:
        self.render_prog: int = common.compile_program(
            vs=VS_SOURCE, fs=FS_SOURCE
        )

        self.render_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.render_vao)

        media: str = common.require_media()
        self.tex_alien_array: int = ktx.load(
            os.path.join(media, "textures", "aliens.ktx")
        )
        if not self.tex_alien_array:
            raise RuntimeError("failed to load aliens.ktx")
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.tex_alien_array)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D_ARRAY,
            GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_LINEAR_MIPMAP_LINEAR,
        )

        # 256 vec4 droplets, one per instance.
        self.rain_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.rain_buffer)
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER, 256 * 16, None, GL.GL_DYNAMIC_DRAW
        )

        rng = _Random()
        self.droplet_x_offset: np.ndarray = np.empty(256, dtype=np.float32)
        self.droplet_rot_speed: np.ndarray = np.empty(256, dtype=np.float32)
        self.droplet_fall_speed: np.ndarray = np.empty(256, dtype=np.float32)
        for i in range(256):
            self.droplet_x_offset[i] = rng.next_float() * 2.0 - 1.0
            self.droplet_rot_speed[i] = (rng.next_float() + 0.5) * (
                -3.0 if (i & 1) else 3.0
            )
            self.droplet_fall_speed[i] = rng.next_float() + 0.2

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        t: float = float(current_time)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        GL.glUseProgram(self.render_prog)

        # Build the 256 droplet vec4s.
        droplets: np.ndarray = np.empty((256, 4), dtype=np.float32)
        droplets[:, 0] = self.droplet_x_offset
        i_arr: np.ndarray = np.arange(256, dtype=np.float32)
        droplets[:, 1] = 2.0 - np.mod(
            (t + i_arr) * self.droplet_fall_speed, 4.31
        )
        droplets[:, 2] = t * self.droplet_rot_speed
        droplets[:, 3] = 0.0

        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.rain_buffer)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER,
            0,
            droplets.nbytes,
            np.ascontiguousarray(droplets),
        )

        for alien_index in range(256):
            GL.glVertexAttribI1i(0, alien_index)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.render_prog)
        GL.glDeleteVertexArrays(1, [self.render_vao])
        GL.glDeleteTextures(1, [self.tex_alien_array])
        GL.glDeleteBuffers(1, [self.rain_buffer])


if __name__ == "__main__":
    application.run(AlienrainApp)
