"""indirectmaterial — 16384 asteroids each with a per-instance material.

Port of ``src/indirectmaterial/indirectmaterial.cpp``.
``asteroids.sbm`` is rendered NUM_DRAWS=16384 times via
``glMultiDrawArraysIndirect``. Each instance picks one of 100
materials (ambient/diffuse/specular/specular_power) via
``baseInstance``; the vertex shader uses that to index a material
UBO and a model-matrix SSBO.

Three GPU buffers:

  - ``frame_uniforms_buffer``  — a single FrameUniforms (3 mat4s) per
    frame, bound at UBO slot 0.
  - ``material_buffer``        — 100 MaterialProperties at UBO slot 2.
    Built once at startup with sin-pattern colors.
  - ``transform_buffer``       — NUM_DRAWS mat4s as a SHADER_STORAGE_BUFFER.
    Refilled each frame with the current asteroid transforms.

Reads its shaders from ``media/shaders/indirectmaterial/``. C++ has a
text overlay; we use imgui for FPS + draw count.
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
        common.require_media(), "shaders", "indirectmaterial", name
    )


NUM_MATERIALS: int = 100
NUM_DRAWS: int = 16384


def _build_materials() -> np.ndarray:
    """100 MaterialProperties: ambient(vec4) + diffuse(vec4) +
    specular(vec3) + specular_power(float). std140 packs vec3+float
    as a single vec4 — total 12 floats per material, 4800 bytes."""
    out: np.ndarray = np.empty((NUM_MATERIALS, 12), dtype=np.float32)
    for i in range(NUM_MATERIALS):
        f: float = i / NUM_MATERIALS
        out[i, 0] = (math.sin(f * 3.7) + 1.0) * 0.1
        out[i, 1] = (math.sin(f * 5.7 + 3.0) + 1.0) * 0.1
        out[i, 2] = (math.sin(f * 4.3 + 2.0) + 2.0) * 0.1
        out[i, 3] = 0.1
        out[i, 4] = (math.sin(f * 9.9 + 6.0) + 1.0) * 0.4
        out[i, 5] = (math.sin(f * 3.1 + 2.5) + 2.0) * 0.4
        out[i, 6] = (math.sin(f * 7.2 + 9.0) + 2.0) * 0.4
        out[i, 7] = 0.4
        out[i, 8] = (math.sin(f * 1.6 + 4.0) + 19.0) * 0.6
        out[i, 9] = (math.sin(f * 0.8 + 2.7) + 19.0) * 0.6
        out[i, 10] = (math.sin(f * 5.2 + 8.0) + 19.0) * 0.6
        out[i, 11] = 200.0 + math.sin(f) * 50.0
    return out


class IndirectmaterialApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_program: int = 0
        self.frame_count: int = 0
        self.draws_per_frame: int = NUM_DRAWS

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Indirect Materials"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.asteroids: sb6m.Object = sb6m.Object()
        self.asteroids.load(
            os.path.join(common.require_media(), "objects", "asteroids.sbm")
        )

        # Indirect-draw command buffer: one entry per draw, picking a
        # sub-mesh (i % num_objects) and a material (i % NUM_MATERIALS
        # via baseInstance).
        n_subs: int = self.asteroids.get_sub_object_count()
        cmds: np.ndarray = np.empty((NUM_DRAWS, 4), dtype=np.uint32)
        for i in range(NUM_DRAWS):
            first, count = self.asteroids.get_sub_object_info(i % n_subs)
            cmds[i] = (count, 1, first, i % NUM_MATERIALS)
        self.indirect_draw_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self.indirect_draw_buffer)
        GL.glBufferData(
            GL.GL_DRAW_INDIRECT_BUFFER, cmds.nbytes, cmds, GL.GL_STATIC_DRAW
        )

        # Per-instance model-matrix SSBO.
        self.transform_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.transform_buffer)
        GL.glBufferData(
            GL.GL_SHADER_STORAGE_BUFFER,
            NUM_DRAWS * 64,
            None,
            GL.GL_DYNAMIC_DRAW,
        )

        # FrameUniforms UBO (3 mat4s, 192 bytes).
        self.frame_uniforms_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.frame_uniforms_buffer)
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 3 * 64, None, GL.GL_DYNAMIC_DRAW)

        # Materials UBO, populated once.
        materials: np.ndarray = _build_materials()
        self.material_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.material_buffer)
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER, materials.nbytes, materials, GL.GL_STATIC_DRAW
        )

        GL.glBindVertexArray(self.asteroids.vao)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_CULL_FACE)

    def load_shaders(self) -> None:
        if self.render_program:
            GL.glDeleteProgram(self.render_program)
        self.render_program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        white: list[float] = [1.0]
        t: float = float(current_time)

        # Camera follows the same lissajous path as multidrawindirect's
        # asteroids so the field stays visible.
        eye: tuple[float, float, float] = (
            30.0 * math.cos(t * 0.023),
            30.0 * math.cos(t * 0.023),
            30.0 * math.sin(t * 0.037) - 200.0,
        )
        ux: float = 0.1 - math.cos(t * 0.1) * 0.3
        un: float = (ux**2 + 1.0) ** 0.5
        view: np.ndarray = vmath.lookat(
            eye, (0.0, 0.0, 0.0), (ux / un, 1.0 / un, 0.0)
        )
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 1.0, 2000.0)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, white)

        # FrameUniforms — std140 mat4 is column-major.
        frame: np.ndarray = np.empty((3, 4, 4), dtype=np.float32)
        frame[0] = view.T
        frame[1] = proj.T
        frame[2] = (view @ proj).T  # C++ source has it backwards; preserve
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.frame_uniforms_buffer)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER, 0, frame.nbytes, np.ascontiguousarray(frame)
        )

        # Per-instance model matrices into the SSBO.
        f: float = t * 0.1
        models: np.ndarray = np.empty(
            (self.draws_per_frame, 4, 4), dtype=np.float32
        )
        for i in range(self.draws_per_frame):
            m: np.ndarray = vmath.translate(
                math.sin(f * 7.3) * 70.0,
                math.sin(f * 3.7 + 2.0) * 70.0,
                math.sin(f * 2.9 + 8.0) * 70.0,
            ) @ vmath.rotate_xyz(f * 330.0, f * 490.0, f * 250.0)
            models[i] = m.T
            f += 3.1
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER, 0, self.transform_buffer
        )
        GL.glBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER,
            0,
            models.nbytes,
            np.ascontiguousarray(models),
        )

        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 2, self.material_buffer)

        GL.glUseProgram(self.render_program)
        GL.glBindVertexArray(self.asteroids.vao)
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self.indirect_draw_buffer)
        GL.glMultiDrawArraysIndirect(
            GL.GL_TRIANGLES, None, self.draws_per_frame, 0
        )
        self.frame_count += 1

        # Imgui status panel — replaces the C++ text overlay.

    @override
    def draw_controls(self) -> None:
        imgui.text(f"Time: {glfw.get_time():.2f}s")
        imgui.text(f"Frames: {self.frame_count}")
        imgui.text(f"Draws/frame: {self.draws_per_frame}")
        imgui.text(f"FPS: {imgui.get_io().framerate:.1f}")
        _, self.draws_per_frame = imgui.slider_int(
            "Draws",
            self.draws_per_frame,
            100,
            NUM_DRAWS,
            flags=imgui.SliderFlags_.logarithmic.value,
        )
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.asteroids.free()
        GL.glDeleteProgram(self.render_program)
        for b in (
            self.indirect_draw_buffer,
            self.transform_buffer,
            self.frame_uniforms_buffer,
            self.material_buffer,
        ):
            GL.glDeleteBuffers(1, [b])


if __name__ == "__main__":
    application.run(IndirectmaterialApp)
