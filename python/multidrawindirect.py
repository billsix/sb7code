"""multidrawindirect — 50,000 asteroids drawn via glMultiDrawArraysIndirect.

Port of ``src/multidrawindirect/multidrawindirect.cpp``.
``asteroids.sbm`` contains many sub-meshes (different asteroid shapes).
We pre-build 50,000 indirect-draw commands, each picking
``sub_object[i % count]`` and using ``baseInstance = i`` so the vertex
shader can derive a per-asteroid transform from a draw-index attribute.

Two modes via the imgui radio: a single ``glMultiDrawArraysIndirect``
call (fast path) vs. 50,000 individual ``glDrawArraysInstancedBaseInstance``
calls (slow path, for benchmarking).

Reads its shaders from ``media/shaders/multidrawindirect/``.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(
        common.require_media(), "shaders", "multidrawindirect", name
    )


NUM_DRAWS: int = 50000

MODE_MULTIDRAW: int = 0
MODE_SEPARATE_DRAWS: int = 1


class MultidrawindirectApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_program: int = 0
        self.mode: int = MODE_MULTIDRAW
        self.paused: bool = False
        self.vsync: bool = False
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Asteroids"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.asteroids: sb6m.Object = sb6m.Object()
        self.asteroids.load(
            os.path.join(common.require_media(), "objects", "asteroids.sbm")
        )

        # Build 50K indirect-draw commands. Layout matches GL's
        # DrawArraysIndirectCommand: 4 uint32s per command.
        self.indirect_draw_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self.indirect_draw_buffer)
        cmds: np.ndarray = np.empty((NUM_DRAWS, 4), dtype=np.uint32)
        n_subs: int = self.asteroids.get_sub_object_count()
        for i in range(NUM_DRAWS):
            first, count = self.asteroids.get_sub_object_info(i % n_subs)
            cmds[i] = (count, 1, first, i)
        GL.glBufferData(
            GL.GL_DRAW_INDIRECT_BUFFER, cmds.nbytes, cmds, GL.GL_STATIC_DRAW
        )

        # Per-instance "draw_index" attribute at location 10. The
        # shader reads it via gl_BaseInstance equivalent — see the
        # render.vs.glsl in the media shader directory.
        GL.glBindVertexArray(self.asteroids.vao)
        self.draw_index_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.draw_index_buffer)
        draw_index: np.ndarray = np.arange(NUM_DRAWS, dtype=np.uint32)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, draw_index.nbytes, draw_index, GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribIPointer(10, 1, GL.GL_UNSIGNED_INT, 0, None)
        GL.glVertexAttribDivisor(10, 1)
        GL.glEnableVertexAttribArray(10)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_CULL_FACE)

    def load_shaders(self) -> None:
        if self.render_program:
            GL.glDeleteProgram(self.render_program)
        self.render_program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )
        self.uniform_time: int = GL.glGetUniformLocation(
            self.render_program, "time"
        )
        self.uniform_view: int = GL.glGetUniformLocation(
            self.render_program, "view_matrix"
        )
        self.uniform_proj: int = GL.glGetUniformLocation(
            self.render_program, "proj_matrix"
        )
        self.uniform_vp: int = GL.glGetUniformLocation(
            self.render_program, "viewproj_matrix"
        )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        one: list[float] = [1.0]

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        t: float = float(self._total_time)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        eye: tuple[float, float, float] = (
            100.0 * math.cos(t * 0.023),
            100.0 * math.cos(t * 0.023),
            300.0 * math.sin(t * 0.037) - 600.0,
        )
        # The C++ source uses normalize(vec3(0.1 - cos(t*0.1)*0.3, 1, 0))
        # for the up vector — keep the same.
        ux: float = 0.1 - math.cos(t * 0.1) * 0.3
        un = (ux**2 + 1.0) ** 0.5
        up: tuple[float, float, float] = (ux / un, 1.0 / un, 0.0)
        view: np.ndarray = vmath.lookat(eye, (0.0, 0.0, 260.0), up)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 1.0, 2000.0)

        GL.glUseProgram(self.render_program)
        GL.glUniform1f(self.uniform_time, t)
        vmath.upload(self.uniform_view, view)
        vmath.upload(self.uniform_proj, proj)
        vmath.upload(self.uniform_vp, proj @ view)

        GL.glBindVertexArray(self.asteroids.vao)

        if self.mode == MODE_MULTIDRAW:
            GL.glBindBuffer(
                GL.GL_DRAW_INDIRECT_BUFFER, self.indirect_draw_buffer
            )
            GL.glMultiDrawArraysIndirect(GL.GL_TRIANGLES, None, NUM_DRAWS, 0)
        else:
            n_subs = self.asteroids.get_sub_object_count()
            for j in range(NUM_DRAWS):
                first, count = self.asteroids.get_sub_object_info(j % n_subs)
                GL.glDrawArraysInstancedBaseInstance(
                    GL.GL_TRIANGLES, first, count, 1, j
                )

    @override
    def draw_controls(self) -> None:
        imgui.text(f"Drawing {NUM_DRAWS} asteroids")
        imgui.text("Mode:")
        if imgui.radio_button(
            "glMultiDrawArraysIndirect (fast)", self.mode == MODE_MULTIDRAW
        ):
            self.mode = MODE_MULTIDRAW
        if imgui.radio_button(
            "Separate draws (slow)", self.mode == MODE_SEPARATE_DRAWS
        ):
            self.mode = MODE_SEPARATE_DRAWS
        _, self.paused = imgui.checkbox("Pause camera", self.paused)
        v_changed, self.vsync = imgui.checkbox("Vsync", self.vsync)
        if v_changed:
            self.set_vsync(self.vsync)
        if imgui.button("Reload shaders"):
            self.load_shaders()
        imgui.text(f"FPS: {imgui.get_io().framerate:.1f}")

    @override
    def shutdown(self) -> None:
        self.asteroids.free()
        GL.glDeleteProgram(self.render_program)
        GL.glDeleteBuffers(1, [self.indirect_draw_buffer])
        GL.glDeleteBuffers(1, [self.draw_index_buffer])


if __name__ == "__main__":
    application.run(MultidrawindirectApp)
