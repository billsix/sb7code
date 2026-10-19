"""multimaterial — benchmark 5 ways to draw 256 cubes with per-instance
matrices.

Port of ``src/multimaterial/multimaterial.cpp``. The same
visual (256 jittering cubes) is rendered 200 passes/frame in five
modes that progressively use cheaper draw paths:

  SIMPLE_UNIFORM           — one glUniformMatrix4fv + glDrawArrays per cube
  BIG_UBO_WITH_UNIFORM     — UBO holds all 256 transforms; uniform int picks
                             which entry the shader uses
  BIG_UBO_WITH_BASEVERTEX  — same, but use ``baseInstance`` as the index
                             (no per-cube uniform write)
  BIG_UBO_WITH_INSTANCING  — single instanced draw call for all 256
  BIG_UBO_INDIRECT         — same as instancing, but commands come from
                             a GL_DRAW_INDIRECT_BUFFER

The clear color changes per mode so you can also see at a glance
which path is active.

Reads its shaders from ``media/shaders/multimaterial/``.

Imgui controls: a radio-button group for the 5 modes, plus
checkboxes for triangle/point primitive and Pause.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from OpenGL.constant import Constant
from imgui_bundle import imgui

from sb7 import application, common, vmath

NUM_PASSES: int = 200
NUM_CUBES: int = 256

(
    SIMPLE_UNIFORM,
    BIG_UBO_WITH_UNIFORM,
    BIG_UBO_WITH_BASEVERTEX,
    BIG_UBO_WITH_INSTANCING,
    BIG_UBO_INDIRECT,
) = range(5)
MODE_NAMES: list[str] = [
    "SIMPLE_UNIFORM",
    "BIG_UBO_WITH_UNIFORM",
    "BIG_UBO_WITH_BASEVERTEX",
    "BIG_UBO_WITH_INSTANCING",
    "BIG_UBO_INDIRECT",
]


def _shader_path(name: str) -> str:
    return os.path.join(
        common.require_media(), "shaders", "multimaterial", name
    )


VERTEX_POSITIONS: np.ndarray = np.array(
    [
        -0.25,
        0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        -0.25,
    ],
    dtype=np.float32,
)


# Each transform_t is mv (mat4) + proj (mat4) = 128 bytes. For the
# UBO we'll pack 256 transforms = 32 KB.
TRANSFORM_SIZE: int = 128
UBO_SIZE: int = NUM_CUBES * TRANSFORM_SIZE


class MultimaterialApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.mode: int = BIG_UBO_INDIRECT
        self.draw_triangles: bool = True
        self.paused: bool = False
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Multi-Material Drawing"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            VERTEX_POSITIONS.nbytes,
            VERTEX_POSITIONS,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glFrontFace(GL.GL_CW)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

        # UBO holding 256 transform_t's.
        self.transform_ubo: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.transform_ubo)
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER, UBO_SIZE, None, GL.GL_DYNAMIC_DRAW
        )

        # Indirect-draw command buffer: 256 cubes, each (count=36,
        # primCount=1, first=0, baseInstance=i). Layout matches GL's
        # DrawArraysIndirectCommand struct: 4 uint32s.
        self.indirect_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self.indirect_buffer)
        cmds: np.ndarray = np.empty((NUM_CUBES, 4), dtype=np.uint32)
        for i in range(NUM_CUBES):
            cmds[i] = (36, 1, 0, i)
        GL.glBufferData(
            GL.GL_DRAW_INDIRECT_BUFFER, cmds.nbytes, cmds, GL.GL_STATIC_DRAW
        )

    def load_shaders(self) -> None:
        self.simple_uniform_program: int = common.compile_program_from_files(
            vs=_shader_path("simpleuniforms.vs.glsl"),
            fs=_shader_path("simpleuniforms.fs.glsl"),
        )
        self.simple_mv_loc: int = GL.glGetUniformLocation(
            self.simple_uniform_program, "mv_matrix"
        )
        self.simple_proj_loc: int = GL.glGetUniformLocation(
            self.simple_uniform_program, "proj_matrix"
        )

        self.ubo_plus_uniform_program: int = common.compile_program_from_files(
            vs=_shader_path("ubo-plus-uniform.vs.glsl"),
            fs=_shader_path("ubo-plus-uniform.fs.glsl"),
        )
        self.transform_index_loc: int = GL.glGetUniformLocation(
            self.ubo_plus_uniform_program, "transform_index"
        )

        self.ubo_plus_base_instance_program: int = (
            common.compile_program_from_files(
                vs=_shader_path("ubo-plus-base-instance.vs.glsl"),
                fs=_shader_path("ubo-plus-base-instance.fs.glsl"),
            )
        )

    def _build_transforms(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute the 256 (proj, mv) matrix pairs for the current
        time. Returns a contiguous (NUM_CUBES, 32) float32 buffer
        suitable for glBufferSubData (mv first, then proj — matches
        the C++ ``transform_t`` struct layout)."""
        f: float = float(t) * 0.3
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        block: np.ndarray = np.empty((NUM_CUBES, 2, 4, 4), dtype=np.float32)
        for i in range(NUM_CUBES):
            mv: np.ndarray = (
                vmath.translate(0.0, 0.0, -4.0)
                @ vmath.translate(
                    math.sin(5.1 * f) * 1.0,
                    math.cos(7.7 * f) * 1.0,
                    math.sin(6.3 * f) * math.cos(1.5 * f) * 2.0,
                )
                @ vmath.rotate(f * 45.0, 0.0, 1.0, 0.0)
                @ vmath.rotate(f * 81.0, 1.0, 0.0, 0.0)
            )
            # Transpose: GLSL std140 mat4 is column-major.
            block[i, 0] = mv.T
            block[i, 1] = proj.T
        return block, proj

    @override
    def render(self, current_time: float) -> None:
        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        t: float = self._total_time

        prim: int | Constant = (
            GL.GL_TRIANGLES if self.draw_triangles else GL.GL_POINTS
        )

        if self.mode == SIMPLE_UNIFORM:
            self._render_simple_uniform(t, prim)
        elif self.mode == BIG_UBO_WITH_UNIFORM:
            self._render_ubo_uniform(t, prim)
        elif self.mode == BIG_UBO_WITH_BASEVERTEX:
            self._render_ubo_baseinstance(t, prim, indirect=False)
        elif self.mode == BIG_UBO_WITH_INSTANCING:
            self._render_ubo_instanced(t, prim)
        elif self.mode == BIG_UBO_INDIRECT:
            self._render_ubo_indirect(t, prim)

    @override
    def draw_controls(self) -> None:
        imgui.text("Render mode:")
        for i, name in enumerate(MODE_NAMES):
            if imgui.radio_button(name, self.mode == i):
                self.mode = i
        imgui.separator()
        _, self.draw_triangles = imgui.checkbox(
            "Draw triangles (else points)", self.draw_triangles
        )
        _, self.paused = imgui.checkbox("Pause animation", self.paused)

    def _clear(self, color: list[float]) -> None:
        one: list[float] = [1.0]
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, color)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

    def _render_simple_uniform(self, t: float, prim: int | Constant) -> None:
        self._clear([0.0, 0.25, 0.0, 1.0])
        block, proj = self._build_transforms(t)
        GL.glUseProgram(self.simple_uniform_program)
        for _ in range(NUM_PASSES):
            for i in range(NUM_CUBES):
                # Upload row-major mv/proj using transpose=GL_TRUE so
                # GLSL receives column-major mat4.
                GL.glUniformMatrix4fv(
                    self.simple_proj_loc,
                    1,
                    GL.GL_FALSE,
                    np.ascontiguousarray(block[i, 1]),
                )
                GL.glUniformMatrix4fv(
                    self.simple_mv_loc,
                    1,
                    GL.GL_FALSE,
                    np.ascontiguousarray(block[i, 0]),
                )
                GL.glDrawArraysInstancedBaseInstance(prim, 0, 36, 1, 0)

    def _upload_ubo(self, t: float) -> None:
        block, _ = self._build_transforms(t)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.transform_ubo)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER, 0, block.nbytes, np.ascontiguousarray(block)
        )

    def _render_ubo_uniform(self, t: float, prim: int | Constant) -> None:
        self._clear([0.0, 0.0, 0.25, 1.0])
        GL.glUseProgram(self.ubo_plus_uniform_program)
        self._upload_ubo(t)
        for _ in range(NUM_PASSES):
            for i in range(NUM_CUBES):
                GL.glUniform1i(self.transform_index_loc, i)
                GL.glDrawArraysInstancedBaseInstance(prim, 0, 36, 1, 0)

    def _render_ubo_baseinstance(
        self, t: float, prim: int | Constant, indirect: bool = False
    ) -> None:
        self._clear([0.25, 0.0, 0.0, 1.0])
        GL.glUseProgram(self.ubo_plus_base_instance_program)
        self._upload_ubo(t)
        for _ in range(NUM_PASSES):
            for i in range(NUM_CUBES):
                GL.glDrawArraysInstancedBaseInstance(prim, 0, 36, 1, i)

    def _render_ubo_instanced(self, t: float, prim: int | Constant) -> None:
        self._clear([0.25, 0.25, 0.0, 1.0])
        GL.glUseProgram(self.ubo_plus_base_instance_program)
        self._upload_ubo(t)
        for _ in range(NUM_PASSES):
            GL.glDrawArraysInstancedBaseInstance(prim, 0, 36, NUM_CUBES, 0)

    def _render_ubo_indirect(self, t: float, prim: int | Constant) -> None:
        self._clear([0.25, 0.0, 0.25, 1.0])
        GL.glUseProgram(self.ubo_plus_base_instance_program)
        self._upload_ubo(t)
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self.indirect_buffer)
        for _ in range(NUM_PASSES):
            GL.glMultiDrawArraysIndirect(prim, None, NUM_CUBES, 0)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.simple_uniform_program)
        GL.glDeleteProgram(self.ubo_plus_uniform_program)
        GL.glDeleteProgram(self.ubo_plus_base_instance_program)
        GL.glDeleteBuffers(1, [self.buffer])
        GL.glDeleteBuffers(1, [self.transform_ubo])
        GL.glDeleteBuffers(1, [self.indirect_buffer])


if __name__ == "__main__":
    application.run(MultimaterialApp)
