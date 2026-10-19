"""starfield — 2000 textured points scrolling toward the viewer.

Port of ``src/starfield/starfield.cpp``. Each star is a
GL_POINTS vertex with random position and color; ``time`` uniform
shifts ``newVertex.z`` cyclically so points appear far away, grow as
they approach, and wrap. The fragment shader multiplies the per-star
color by ``texture(tex_star, gl_PointCoord)``, sampling the round star
sprite from a 2D KTX texture (``media/textures/star.ktx``).

First demo to load a KTX texture via ``ktx.load`` — uses the new
loader unconditionally.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

from sb7 import application, common, ktx, vmath

NUM_STARS: int = 2000


VS_SOURCE: str = """
#version 410 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 color;

uniform float time;
uniform mat4 proj_matrix;

flat out vec4 starColor;

void main(void)
{
    vec4 newVertex = position;

    newVertex.z += time;
    newVertex.z = fract(newVertex.z);

    float size = (20.0 * newVertex.z * newVertex.z);
    starColor = smoothstep(1.0, 7.0, size) * color;

    newVertex.z = (999.9 * newVertex.z) - 1000.0;
    gl_Position = proj_matrix * newVertex;
    gl_PointSize = size;
}
"""

FS_SOURCE: str = """
#version 410 core

layout (location = 0) out vec4 color;

uniform sampler2D tex_star;
flat in vec4 starColor;

void main(void)
{
    color = starColor * texture(tex_star, gl_PointCoord);
}
"""


# Same LCG random-number generator as the C++ source (seed = 0x13371337,
# multiplied by 16807 each step, then twiddled into the [0,1) range).
class _Random:
    def __init__(self, seed: int = 0x13371337) -> None:
        self.seed: int = seed

    def next_float(self) -> float:
        self.seed = (self.seed * 16807) & 0xFFFFFFFF
        tmp: int = (
            self.seed ^ (self.seed >> 4) ^ ((self.seed << 15) & 0xFFFFFFFF)
        )
        # Same trick: take top 23 bits, OR with exponent for [1.0, 2.0)
        # then subtract 1 to get [0.0, 1.0).
        bits: int = (tmp >> 9) | 0x3F800000
        f: np.float32 = np.frombuffer(
            np.uint32(bits & 0xFFFFFFFF).tobytes(), dtype=np.float32
        )[0]
        return float(f - 1.0)


def _build_stars() -> np.ndarray:
    """Construct the per-star (position vec3, color vec3) interleaved
    buffer. The C++ code only fills the first 1000 entries (a bug, or
    perhaps intentional sparseness?). We follow the same — leaves the
    other 1000 zeroed at origin."""
    rng: _Random = _Random(0x13371337)
    # Each star: vec3 position + vec3 color. Layout 6 floats per star,
    # but VertexAttribPointer stride is sizeof(star_t) which the C++
    # struct gives as 6 * 4 = 24. Match exactly.
    arr: np.ndarray = np.zeros((NUM_STARS, 6), dtype=np.float32)
    for i in range(1000):
        arr[i, 0] = (rng.next_float() * 2.0 - 1.0) * 100.0
        arr[i, 1] = (rng.next_float() * 2.0 - 1.0) * 100.0
        arr[i, 2] = rng.next_float()
        arr[i, 3] = 0.8 + rng.next_float() * 0.2
        arr[i, 4] = 0.8 + rng.next_float() * 0.2
        arr[i, 5] = 0.8 + rng.next_float() * 0.2
    return arr


class StarfieldApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Starfield"

    @override
    def startup(self) -> None:
        self.render_prog: int = common.compile_program(
            vs=VS_SOURCE, fs=FS_SOURCE
        )
        self.time_loc: int = GL.glGetUniformLocation(self.render_prog, "time")
        self.proj_loc: int = GL.glGetUniformLocation(
            self.render_prog, "proj_matrix"
        )

        # Load the round-star sprite.
        media: str = common.require_media()
        self.star_texture: int = ktx.load(
            os.path.join(media, "textures", "star.ktx")
        )
        if not self.star_texture:
            raise RuntimeError("failed to load star.ktx")

        self.star_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.star_vao)

        stars: np.ndarray = _build_stars()
        self.star_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.star_buffer)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, stars.nbytes, stars, GL.GL_STATIC_DRAW
        )

        # Layout: 3 floats position + 3 floats color = 24 bytes/star.
        # The C++ source actually uses ``vmath::vec3 position; vmath::vec3 color;``
        # but the shader declares ``vec4 position`` and ``vec4 color``;
        # GL pulls the missing components from the default
        # (0, 0, 0, 1). Both attributes use stride = 24 (= 6 floats).
        stride: int = 6 * 4
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glVertexAttribPointer(
            1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.GLvoidp(3 * 4)
        )
        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(1)

    @override
    def render(self, current_time: float) -> None:
        import math

        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        one: list[float] = [1.0]

        t: float = float(current_time) * 0.1
        t -= math.floor(t)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glUseProgram(self.render_prog)
        GL.glUniform1f(self.time_loc, t)
        vmath.upload(self.proj_loc, proj)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)

        GL.glBindVertexArray(self.star_vao)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glDrawArrays(GL.GL_POINTS, 0, NUM_STARS)

    @override
    def shutdown(self) -> None:
        GL.glDeleteTextures(1, [self.star_texture])
        GL.glDeleteVertexArrays(1, [self.star_vao])
        GL.glDeleteBuffers(1, [self.star_buffer])
        GL.glDeleteProgram(self.render_prog)


if __name__ == "__main__":
    application.run(StarfieldApp)
