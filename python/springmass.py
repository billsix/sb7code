"""springmass — 50×50 cloth-like spring/mass simulator on the GPU.

Port of ``src/springmass/springmass.cpp``. A grid of 2500
mass points with 4-way spring connections, simulated entirely by a
vertex shader writing out new positions/velocities via transform
feedback. The fragment stage is disabled (``GL_RASTERIZER_DISCARD``)
during the update step. After ``iterations_per_frame`` updates, the
render program draws the points and the connecting line segments.

Reads its shaders from ``media/shaders/springmass/``.

Demonstrates:
  - transform feedback with separate varyings
  - texture buffer objects (TBOs) for sampling neighbor positions
  - ping-pong double-buffering between two VAOs

Imgui controls: Lines / Points checkboxes, Iterations-per-frame slider,
Reload-shaders button.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common

POINTS_X: int = 50
POINTS_Y: int = 50
POINTS_TOTAL = POINTS_X * POINTS_Y
CONNECTIONS_TOTAL = (POINTS_X - 1) * POINTS_Y + (POINTS_Y - 1) * POINTS_X

# Indices into the m_vbo[] array — same enum as the C++ source.
POSITION_A: int = 0
POSITION_B: int = 1
VELOCITY_A: int = 2
VELOCITY_B: int = 3
CONNECTION: int = 4


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "springmass", name)


def _build_initial_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the initial position/velocity/connection arrays the
    same way the C++ source's nested loops do."""
    positions: np.ndarray = np.zeros((POINTS_TOTAL, 4), dtype=np.float32)
    velocities: np.ndarray = np.zeros((POINTS_TOTAL, 3), dtype=np.float32)
    connections: np.ndarray = np.full((POINTS_TOTAL, 4), -1, dtype=np.int32)

    n: int = 0
    for j in range(POINTS_Y):
        fj = j / POINTS_Y
        for i in range(POINTS_X):
            fi = i / POINTS_X
            positions[n] = [
                (fi - 0.5) * POINTS_X,
                (fj - 0.5) * POINTS_Y,
                0.6 * math.sin(fi) * math.cos(fj),
                1.0,
            ]
            if j != (POINTS_Y - 1):
                if i != 0:
                    connections[n, 0] = n - 1
                if j != 0:
                    connections[n, 1] = n - POINTS_X
                if i != (POINTS_X - 1):
                    connections[n, 2] = n + 1
                if j != (POINTS_Y - 1):
                    connections[n, 3] = n + POINTS_X
            n += 1
    return positions, velocities, connections


def _build_line_indices() -> np.ndarray:
    """The line indices for drawing horizontal and vertical springs.
    Layout follows the C++ source's two nested loops."""
    out: np.ndarray = np.empty(CONNECTIONS_TOTAL * 2, dtype=np.int32)
    e: int = 0
    for j in range(POINTS_Y):
        for i in range(POINTS_X - 1):
            out[e] = i + j * POINTS_X
            out[e + 1] = i + 1 + j * POINTS_X
            e += 2
    for i in range(POINTS_X):
        for j in range(POINTS_Y - 1):
            out[e] = i + j * POINTS_X
            out[e + 1] = POINTS_X + i + j * POINTS_X
            e += 2
    return out


class SpringmassApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.iteration_index: int = 0
        self.update_program: int = 0
        self.render_program: int = 0
        self.draw_points: bool = True
        self.draw_lines: bool = True
        self.iterations_per_frame: int = 16

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Spring-Mass Simulator"

    @override
    def startup(self) -> None:
        self.load_shaders()

        positions, velocities, connections = _build_initial_data()

        self.vao = GL.glGenVertexArrays(2)
        self.vbo = GL.glGenBuffers(5)

        for k in range(2):
            GL.glBindVertexArray(self.vao[k])

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo[POSITION_A + k])
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                positions.nbytes,
                positions,
                GL.GL_DYNAMIC_COPY,
            )
            GL.glVertexAttribPointer(0, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(0)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo[VELOCITY_A + k])
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                velocities.nbytes,
                velocities,
                GL.GL_DYNAMIC_COPY,
            )
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(1)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo[CONNECTION])
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                connections.nbytes,
                connections,
                GL.GL_STATIC_DRAW,
            )
            GL.glVertexAttribIPointer(2, 4, GL.GL_INT, 0, None)
            GL.glEnableVertexAttribArray(2)

        # TBO: a buffer texture wrapping each position VBO so the
        # update shader can sample neighbors via texelFetch.
        self.pos_tbo = GL.glGenTextures(2)
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, self.pos_tbo[0])
        GL.glTexBuffer(
            GL.GL_TEXTURE_BUFFER, GL.GL_RGBA32F, self.vbo[POSITION_A]
        )
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, self.pos_tbo[1])
        GL.glTexBuffer(
            GL.GL_TEXTURE_BUFFER, GL.GL_RGBA32F, self.vbo[POSITION_B]
        )

        # Line index buffer.
        line_indices: np.ndarray = _build_line_indices()
        self.index_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            line_indices.nbytes,
            line_indices,
            GL.GL_STATIC_DRAW,
        )

    def load_shaders(self) -> None:
        # Update program: VS only, with transform feedback varyings.
        if self.update_program:
            GL.glDeleteProgram(self.update_program)
        self.update_program = common.compile_program_from_files(
            vs=_shader_path("update.vs.glsl"),
            tf_varyings=["tf_position_mass", "tf_velocity"],
        )

        # Render program: standard vs+fs.
        if self.render_program:
            GL.glDeleteProgram(self.render_program)
        self.render_program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )

    @override
    def render(self, current_time: float) -> None:
        # Update step: ping-pong N times. Rasterizer is disabled so
        # fragments never run; transform feedback captures the new
        # position/velocity into the OTHER buffer.
        GL.glUseProgram(self.update_program)
        GL.glEnable(GL.GL_RASTERIZER_DISCARD)

        for _ in range(self.iterations_per_frame):
            GL.glBindVertexArray(self.vao[self.iteration_index & 1])
            GL.glBindTexture(
                GL.GL_TEXTURE_BUFFER, self.pos_tbo[self.iteration_index & 1]
            )
            self.iteration_index += 1
            GL.glBindBufferBase(
                GL.GL_TRANSFORM_FEEDBACK_BUFFER,
                0,
                self.vbo[POSITION_A + (self.iteration_index & 1)],
            )
            GL.glBindBufferBase(
                GL.GL_TRANSFORM_FEEDBACK_BUFFER,
                1,
                self.vbo[VELOCITY_A + (self.iteration_index & 1)],
            )
            GL.glBeginTransformFeedback(GL.GL_POINTS)
            GL.glDrawArrays(GL.GL_POINTS, 0, POINTS_TOTAL)
            GL.glEndTransformFeedback()

        GL.glDisable(GL.GL_RASTERIZER_DISCARD)

        # Render step: draw the current grid.
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        GL.glUseProgram(self.render_program)

        if self.draw_points:
            GL.glPointSize(4.0)
            GL.glDrawArrays(GL.GL_POINTS, 0, POINTS_TOTAL)

        if self.draw_lines:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
            GL.glDrawElements(
                GL.GL_LINES, CONNECTIONS_TOTAL * 2, GL.GL_UNSIGNED_INT, None
            )

    @override
    def draw_controls(self) -> None:
        _, self.draw_points = imgui.checkbox("Draw points", self.draw_points)
        _, self.draw_lines = imgui.checkbox(
            "Draw spring lines", self.draw_lines
        )
        _, self.iterations_per_frame = imgui.slider_int(
            "Iterations per frame", self.iterations_per_frame, 0, 64
        )
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.update_program)
        GL.glDeleteProgram(self.render_program)
        GL.glDeleteBuffers(5, self.vbo)
        GL.glDeleteVertexArrays(2, self.vao)


if __name__ == "__main__":
    application.run(SpringmassApp)
