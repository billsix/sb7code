"""ompparticles — 2048-particle gravitational N-body sim on the CPU.

Port of ``src/ompparticles/ompparticles.cpp``. Each frame
recomputes the velocity contribution from every other particle on
every particle (O(N²) = ~4.2 million pair operations) and uploads the
new positions into a persistent-mapped vertex buffer.

The C++ original toggles between a serial loop and an OpenMP
parallel-for via ``#pragma omp parallel``. In Python we use
**numpy broadcasting** for the same computation; it's effectively
parallel and there's no slow path to compare against, so the OMP
toggle is gone. The on-screen panel just shows the particle count and
the FPS.
"""

from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

from sb7 import application, common

PARTICLE_COUNT: int = 2048


VS_SOURCE: str = """
#version 440 core
layout (location = 0) in vec3 position;
out vec4 particle_color;
void main(void)
{
    particle_color = vec4(0.6, 0.8, 0.8, 1.0) * (smoothstep(-10.0, 10.0, position.z) * 0.6 + 0.4);
    gl_Position = vec4(position * 0.2, 1.0);
}
"""

FS_SOURCE: str = """
#version 440 core
layout (location = 0) out vec4 o_color;
in vec4 particle_color;
void main(void)
{
    o_color = particle_color;
}
"""


# Same LCG as the rest of the v7 demos.
class _Random:
    def __init__(self, seed: int = 0x13371337) -> None:
        self.seed: int = seed

    def next_float(self) -> float:
        self.seed = (self.seed * 16807) & 0xFFFFFFFF
        tmp: int = (
            self.seed ^ (self.seed >> 4) ^ ((self.seed << 15) & 0xFFFFFFFF)
        )
        bits: int = (tmp >> 9) | 0x3F800000
        f: np.float32 = np.frombuffer(
            np.uint32(bits & 0xFFFFFFFF).tobytes(), dtype=np.float32
        )[0]
        return float(f - 1.0)


def _initial_particles() -> tuple[np.ndarray, np.ndarray]:
    rng: _Random = _Random()
    pos: np.ndarray = np.empty((PARTICLE_COUNT, 3), dtype=np.float32)
    for i in range(PARTICLE_COUNT):
        pos[i, 0] = rng.next_float() * 6.0 - 3.0
        pos[i, 1] = rng.next_float() * 6.0 - 3.0
        pos[i, 2] = rng.next_float() * 6.0 - 3.0
    vel: np.ndarray = pos * 0.001
    return pos, vel


class OmpparticlesApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self._previous_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Parallel Particles"

    @override
    def startup(self) -> None:
        # We use a regular GL_DYNAMIC_DRAW buffer here, not the C++
        # source's persistent-mapped buffer — glBufferSubData each
        # frame is fine for 2048 vec3s and avoids the GL 4.4
        # persistent-mapped buffer dance.
        self.particle_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.particle_buffer)
        # Each vertex is 6 floats (position + velocity), but the
        # shader only sees the first 3.
        stride: int = 6 * 4
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            PARTICLE_COUNT * stride,
            None,
            GL.GL_DYNAMIC_DRAW,
        )

        self.position, self.velocity = _initial_particles()

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glEnableVertexAttribArray(0)

        self.draw_program: int = common.compile_program(
            vs=VS_SOURCE, fs=FS_SOURCE
        )

        # Pre-allocated working storage to avoid per-frame allocs.
        self._delta_v: np.ndarray = np.empty(
            (PARTICLE_COUNT, 3), dtype=np.float32
        )

    def _update_particles(self, dt: float) -> None:
        """O(N²) gravitational pull, fully vectorized.

        For each pair (i, j) with i != j:
          delta_pos = pos[j] - pos[i]
          delta_dir = delta_pos / |delta_pos|
          delta_v[i] += delta_dir / max(|delta_pos|, 0.005)²
        """
        p: np.ndarray = self.position  # (N, 3)
        # Pairwise displacement: (N, N, 3).
        delta_pos: np.ndarray = p[None, :, :] - p[:, None, :]
        # Pairwise distance, clamped to 0.005.
        dist: np.ndarray = np.maximum(np.linalg.norm(delta_pos, axis=2), 0.005)
        # delta_dir = delta_pos / dist; force = delta_dir / dist²
        # → contribution = delta_pos / dist³
        # Mask the i==j entries to zero so they don't NaN.
        np.fill_diagonal(dist, 1.0)  # avoid 0/0; the contrib is 0 anyway
        contrib: np.ndarray = delta_pos / (dist**3)[..., None]
        # zero out the diagonal contributions
        idx: np.ndarray = np.arange(PARTICLE_COUNT)
        contrib[idx, idx] = 0.0
        self._delta_v[:] = contrib.sum(axis=1)

        # Integrate.
        self.position = self.position + self.velocity
        self.velocity = self.velocity + self._delta_v * dt * 0.01

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]

        dt: float = float(current_time - self._previous_time)
        self._previous_time = current_time

        self._update_particles(dt * 0.001)

        # Build interleaved (pos, vel) and upload. Shader only reads
        # pos so the velocity field is dead weight on the GPU side
        # but this matches the C++ struct layout.
        upload: np.ndarray = np.empty((PARTICLE_COUNT, 6), dtype=np.float32)
        upload[:, 0:3] = self.position
        upload[:, 3:6] = self.velocity
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.particle_buffer)
        GL.glBufferSubData(
            GL.GL_ARRAY_BUFFER, 0, upload.nbytes, np.ascontiguousarray(upload)
        )

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glBindVertexArray(self.vao)
        GL.glPointSize(3.0)
        GL.glUseProgram(self.draw_program)
        GL.glDrawArrays(GL.GL_POINTS, 0, PARTICLE_COUNT)

    @override
    def draw_controls(self) -> None:
        imgui.text(f"{PARTICLE_COUNT} particles, O(N²) numpy")
        imgui.text(f"FPS: {imgui.get_io().framerate:.1f}")

    @override
    def shutdown(self) -> None:
        GL.glDeleteBuffers(1, [self.particle_buffer])
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.draw_program)


if __name__ == "__main__":
    application.run(OmpparticlesApp)
