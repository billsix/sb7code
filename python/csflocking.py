"""csflocking — compute-shader flocking simulation.

Port of ``src/csflocking/csflocking.cpp``.

16 384 paper airplanes (NUM_WORKGROUPS=64 × WORKGROUP_SIZE=256) flock
toward a moving goal. Each frame:

  Pass 1 (compute): read positions/velocities from SSBO[frame], apply
         flocking forces, write to SSBO[frame ^ 1]. Two SSBOs are
         double-buffered each frame.

  Pass 2 (render): instance-draw a 6-tri paper-airplane mesh, with
         per-instance position/velocity attributes pulled directly
         from the SSBO via glVertexAttribPointer. Velocity is used in
         the vertex shader to orient each plane along its motion.

Reads its shaders from ``media/shaders/flocking/``.
"""

import ctypes
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "flocking", name)


WORKGROUP_SIZE: int = 256  # local_size_x in flocking.cs.glsl
MAX_WORKGROUPS: int = 64
MAX_FLOCK_SIZE: int = MAX_WORKGROUPS * WORKGROUP_SIZE
# Each member is two vec4s (vec3 + 4 bytes padding × 2) = 32 bytes.
FLOCK_MEMBER_BYTES: int = 32

# Paper-airplane mesh: 8 positions then 8 normals, packed as 12-byte
# vec3s (no padding) — matches the C++ layout exactly.
GEOMETRY: np.ndarray = np.array(
    [
        -5.0,
        1.0,
        0.0,
        -1.0,
        1.5,
        0.0,
        -1.0,
        1.5,
        7.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        10.0,
        1.0,
        1.5,
        0.0,
        1.0,
        1.5,
        7.0,
        5.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.107,
        -0.859,
        0.0,
        0.832,
        0.554,
        0.0,
        -0.59,
        -0.395,
        0.0,
        -0.832,
        0.554,
        0.0,
        0.295,
        -0.196,
        0.0,
        0.124,
        0.992,
        0.0,
    ],
    dtype=np.float32,
)


class CsflockingApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.flock_update_program: int = 0
        self.flock_render_program: int = 0
        self.frame_index: int = 0
        # Live flock size is adjustable via imgui in 256-bird steps
        # (one workgroup per step). SSBOs are pre-sized for the max.
        self.num_workgroups: int = MAX_WORKGROUPS

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Compute Shader Flocking"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.flock_buffer: np.ndarray = GL.glGenBuffers(2)
        for i in range(2):
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.flock_buffer[i])
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER,
                MAX_FLOCK_SIZE * FLOCK_MEMBER_BYTES,
                None,
                GL.GL_DYNAMIC_COPY,
            )

        self.geometry_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.geometry_buffer)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            GEOMETRY.nbytes,
            np.ascontiguousarray(GEOMETRY),
            GL.GL_STATIC_DRAW,
        )

        self.flock_render_vao: np.ndarray = GL.glGenVertexArrays(2)
        for i in range(2):
            GL.glBindVertexArray(self.flock_render_vao[i])
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.geometry_buffer)
            # Geometry: 8 positions (12B each) followed by 8 normals.
            GL.glVertexAttribPointer(
                0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0)
            )
            GL.glVertexAttribPointer(
                1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(8 * 12)
            )
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.flock_buffer[i])
            # Per-instance: position at offset 0, velocity at offset
            # 16 (one vec4 in). Stride 32 (two vec4s).
            GL.glVertexAttribPointer(
                2,
                3,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                FLOCK_MEMBER_BYTES,
                ctypes.c_void_p(0),
            )
            GL.glVertexAttribPointer(
                3,
                3,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                FLOCK_MEMBER_BYTES,
                ctypes.c_void_p(16),
            )
            GL.glVertexAttribDivisor(2, 1)
            GL.glVertexAttribDivisor(3, 1)
            for loc in (0, 1, 2, 3):
                GL.glEnableVertexAttribArray(loc)

        # Initial flock state: random positions in ±150, random unit
        # velocities. Layout matches flock_member exactly.
        rng: np.random.Generator = np.random.default_rng(0xC0FFEE)
        init: np.ndarray = np.zeros((MAX_FLOCK_SIZE, 8), dtype=np.float32)
        init[:, 0:3] = (
            rng.random((MAX_FLOCK_SIZE, 3), dtype=np.float32) - 0.5
        ) * 300.0
        init[:, 4:7] = rng.random((MAX_FLOCK_SIZE, 3), dtype=np.float32) - 0.5
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.flock_buffer[0])
        GL.glBufferSubData(
            GL.GL_ARRAY_BUFFER, 0, init.nbytes, np.ascontiguousarray(init)
        )

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

    def load_shaders(self) -> None:
        if self.flock_update_program:
            GL.glDeleteProgram(self.flock_update_program)
        if self.flock_render_program:
            GL.glDeleteProgram(self.flock_render_program)
        self.flock_update_program = common.compile_program_from_files(
            cs=_shader_path("flocking.cs.glsl")
        )
        self.flock_render_program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )
        self.u_goal: int = GL.glGetUniformLocation(
            self.flock_update_program, "goal"
        )
        self.u_mvp: int = GL.glGetUniformLocation(
            self.flock_render_program, "mvp"
        )

    @override
    def render(self, current_time: float) -> None:
        t: float = float(current_time)

        # Compute pass: SSBO[frame] → SSBO[frame^1].
        GL.glUseProgram(self.flock_update_program)
        goal: np.ndarray = np.array(
            [
                np.sin(t * 0.34) * 35.0,
                np.cos(t * 0.29) * 25.0,
                np.sin(t * 0.12) * np.cos(t * 0.5) * 60.0,
            ],
            dtype=np.float32,
        )
        GL.glUniform3fv(self.u_goal, 1, goal)
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER, 0, self.flock_buffer[self.frame_index]
        )
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER,
            1,
            self.flock_buffer[self.frame_index ^ 1],
        )
        GL.glDispatchCompute(self.num_workgroups, 1, 1)

        # Render pass.
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, [0.0, 0.0, 0.0, 1.0])
        GL.glClearBufferfv(GL.GL_DEPTH, 0, [1.0])
        GL.glUseProgram(self.flock_render_program)

        view: np.ndarray = vmath.lookat(
            np.array([0.0, 0.0, -400.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        proj: np.ndarray = vmath.perspective(
            60.0,
            max(self.info.window_width, 1) / max(self.info.window_height, 1),
            0.1,
            3000.0,
        )
        mvp: np.ndarray = proj @ view
        GL.glUniformMatrix4fv(self.u_mvp, 1, GL.GL_FALSE, mvp.T)

        live_flock: int = self.num_workgroups * WORKGROUP_SIZE
        GL.glBindVertexArray(self.flock_render_vao[self.frame_index])
        GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 8, live_flock)

        self.frame_index ^= 1

    @override
    def draw_controls(self) -> None:
        imgui.text("Paper airplanes flocking toward a moving goal")
        imgui.text(
            "Workgroup size = %d, dispatched workgroups = %d"
            % (WORKGROUP_SIZE, self.num_workgroups)
        )
        live_flock: int = self.num_workgroups * WORKGROUP_SIZE
        imgui.text(
            "Live flock size: %d (max %d)" % (live_flock, MAX_FLOCK_SIZE)
        )
        _, self.num_workgroups = imgui.slider_int(
            "Workgroups", self.num_workgroups, 1, MAX_WORKGROUPS
        )
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.flock_update_program)
        GL.glDeleteProgram(self.flock_render_program)
        GL.glDeleteBuffers(2, self.flock_buffer)
        GL.glDeleteBuffers(1, [self.geometry_buffer])
        GL.glDeleteVertexArrays(2, self.flock_render_vao)


if __name__ == "__main__":
    application.run(CsflockingApp)
