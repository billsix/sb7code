"""cullindirect — GPU-driven culling + ``glMultiDrawArraysIndirectCount``.

Port of ``src/cullindirect/cullindirect.cpp``.

Pipeline:

  CPU each frame: write 1024 model matrices into a UBO, plus a
       view/proj/view-proj UBO.

  Compute pass (``cull.cs``): each thread reads one CandidateDraw
       record (sphereCenter + radius + firstVertex + vertexCount),
       projects the sphere against the view frustum, and either
       appends a DrawArraysIndirectCommand to a draw-command SSBO
       (atomically incrementing the count in a parameter buffer) or
       skips the candidate.

  Render pass: ``glMultiDrawArraysIndirectCount`` reads the count
       from the parameter buffer and the commands from the draw-
       command SSBO and issues all surviving draws in one call.

This needs OpenGL 4.6 (or ``GL_ARB_indirect_parameters``) for
``GL_PARAMETER_BUFFER`` and ``glMultiDrawArraysIndirectCount``.

Reads its shaders from ``media/shaders/cullindirect/``, the model from
``media/objects/asteroids.sbm``, and ``media/textures/rocks.ktx``.
"""

import ctypes
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, ktx, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "cullindirect", name)


def _media(name: str) -> str:
    return os.path.join(common.require_media(), name)


def _upload_mapped(target: common.GLenum, data: np.ndarray) -> None:
    """Write ``data`` into an immutable ``GL_MAP_WRITE_BIT`` buffer bound
    to ``target``, matching the C++ ``glMapBufferRange`` path
    (cullindirect.cpp maps these UBOs, it does not use glBufferSubData).
    These buffers are created with ``glBufferStorage`` *without*
    ``GL_DYNAMIC_STORAGE_BIT``, so ``glBufferSubData`` on them raises
    ``GL_INVALID_OPERATION`` (1282) — you must map, write, unmap."""
    src = np.ascontiguousarray(data)
    ptr = GL.glMapBufferRange(
        target,
        0,
        src.nbytes,
        # ty: PyOpenGL Constant lacks __or__ in the stubs; the runtime
        # bitwise-or of two GL_MAP_* flags is a plain int mask.
        GL.GL_MAP_WRITE_BIT | GL.GL_MAP_INVALIDATE_BUFFER_BIT,  # ty: ignore[unsupported-operator]
    )
    if not ptr:
        raise RuntimeError("glMapBufferRange returned NULL")
    ctypes.memmove(int(ptr), src.ctypes.data, src.nbytes)
    GL.glUnmapBuffer(target)


CANDIDATE_COUNT: int = 1024
# CandidateDraw struct is 32 bytes: vec3 + float + uint + uint + 8 pad.
CANDIDATE_BYTES: int = 32
# DrawArraysIndirectCommand is 16 bytes: 4 GLuints.
DRAWCMD_BYTES: int = 16
# vmath::mat4 = 64 bytes.
MAT4_BYTES: int = 64
# TransformBuffer = 3 mat4 = 192 bytes.
TRANSFORM_BYTES: int = 192
NUM_INSTANCES: int = 1024


class CullindirectApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.cull_program: int = 0
        self.draw_program: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Indirect Culling"

    @override
    def startup(self) -> None:
        self.object: sb6m.Object = sb6m.Object()
        self.object.load(_media("objects/asteroids.sbm"))

        # Parameter buffer: holds the count consumed by
        # glMultiDrawArraysIndirectCount.
        self.param_buf: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PARAMETER_BUFFER, self.param_buf)
        GL.glBufferStorage(GL.GL_PARAMETER_BUFFER, 256, None, 0)

        # CandidateDraw SSBO: built once, immutable.
        self.candidates_buf: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.candidates_buf)
        sub_count = self.object.get_sub_object_count()
        candidates: np.ndarray = np.zeros(
            CANDIDATE_COUNT,
            dtype=[
                ("center", np.float32, 3),
                ("radius", np.float32),
                ("first", np.uint32),
                ("count", np.uint32),
                ("pad0", np.uint32),
                ("pad1", np.uint32),
            ],
        )
        for i in range(CANDIDATE_COUNT):
            first, count = self.object.get_sub_object_info(i % sub_count)
            candidates[i]["center"] = (0.0, 0.0, 0.0)
            candidates[i]["radius"] = 4.0
            candidates[i]["first"] = first
            candidates[i]["count"] = count
        GL.glBufferStorage(
            GL.GL_SHADER_STORAGE_BUFFER,
            candidates.nbytes,
            np.ascontiguousarray(candidates),
            0,
        )

        # Draw-command SSBO: written by compute, read as
        # GL_DRAW_INDIRECT_BUFFER by the draw call.
        self.drawcmd_buf: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.drawcmd_buf)
        GL.glBufferStorage(
            GL.GL_SHADER_STORAGE_BUFFER,
            CANDIDATE_COUNT * DRAWCMD_BYTES,
            None,
            GL.GL_MAP_READ_BIT,
        )

        # Model-matrix UBO (1024 mat4) + transforms UBO.
        self.model_ubo: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.model_ubo)
        GL.glBufferStorage(
            GL.GL_UNIFORM_BUFFER,
            NUM_INSTANCES * MAT4_BYTES,
            None,
            GL.GL_MAP_WRITE_BIT,
        )

        self.transforms_ubo: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.transforms_ubo)
        GL.glBufferStorage(
            GL.GL_UNIFORM_BUFFER, TRANSFORM_BYTES, None, GL.GL_MAP_WRITE_BIT
        )

        self.texture: int = ktx.load(_media("textures/rocks.ktx"))

        # glMultiDrawArraysIndirectCount was promoted to GL 4.6 core, but
        # PyOpenGL cannot bind the un-suffixed entry point on Mesa
        # (NullFunctionError); radeonsi exports the ARB-extension spelling.
        # Resolve once, here in startup where a context is current, using
        # PyOpenGL's own bool() probe (the very check its error suggests).
        self._multi_draw_count = GL.glMultiDrawArraysIndirectCount
        if not bool(self._multi_draw_count):
            from OpenGL.GL.ARB.indirect_parameters import (
                glMultiDrawArraysIndirectCountARB,
            )

            self._multi_draw_count = glMultiDrawArraysIndirectCountARB

        self.load_shaders()

    def load_shaders(self) -> None:
        if self.cull_program:
            GL.glDeleteProgram(self.cull_program)
        if self.draw_program:
            GL.glDeleteProgram(self.draw_program)
        self.cull_program = common.compile_program_from_files(
            cs=_shader_path("cull.cs.glsl")
        )
        self.draw_program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )

    @override
    def render(self, current_time: float) -> None:
        t: float = float(current_time)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, [0.0, 0.0, 0.0, 1.0])
        GL.glClearBufferfv(GL.GL_DEPTH, 0, [1.0])

        # Reset the indirect-count atomic to zero each frame.
        GL.glBindBufferBase(GL.GL_ATOMIC_COUNTER_BUFFER, 0, self.param_buf)
        zero: np.ndarray = np.zeros(1, dtype=np.uint32)
        GL.glClearBufferSubData(
            GL.GL_ATOMIC_COUNTER_BUFFER,
            GL.GL_R32UI,
            0,
            4,
            GL.GL_RED_INTEGER,
            GL.GL_UNSIGNED_INT,
            zero,
        )

        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self.candidates_buf)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, self.drawcmd_buf)

        # Fill model-matrix UBO with 1024 random rotating positions.
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.model_ubo)
        models: np.ndarray = np.empty((NUM_INSTANCES, 4, 4), dtype=np.float32)
        for i in range(NUM_INSTANCES):
            f = i / 127.0 + t * 0.025
            g = i / 127.0
            pos = 70.0 * np.array(
                [np.sin(f * 3.0), np.cos(f * 5.0), np.cos(f * 9.0)],
                dtype=np.float32,
            )
            axis: np.ndarray = np.array(
                [np.sin(g * 35.0), np.cos(g * 75.0), np.cos(g * 39.0)],
                dtype=np.float32,
            )
            axis /= np.linalg.norm(axis) or 1.0
            mat: np.ndarray = vmath.translate(
                pos[0], pos[1], pos[2]
            ) @ vmath.rotate(t * 140.0, axis[0], axis[1], axis[2])
            models[i] = mat.T  # std140 column-major.
        _upload_mapped(GL.GL_UNIFORM_BUFFER, models)

        # Fill transforms UBO.
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 1, self.transforms_ubo)
        ct = t * 0.1
        eye: np.ndarray = np.array(
            [150.0 * np.cos(ct), 0.0, 150.0 * np.sin(ct)], dtype=np.float32
        )
        view: np.ndarray = vmath.lookat(
            eye,
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        proj: np.ndarray = vmath.perspective(
            50.0,
            max(self.info.window_width, 1) / max(self.info.window_height, 1),
            1.0,
            2000.0,
        )
        vp = proj @ view
        transforms = np.stack([view.T, proj.T, vp.T])
        _upload_mapped(GL.GL_UNIFORM_BUFFER, transforms)

        # Cull pass.
        GL.glUseProgram(self.cull_program)
        GL.glDispatchCompute(CANDIDATE_COUNT // 16, 1, 1)
        GL.glMemoryBarrier(GL.GL_COMMAND_BARRIER_BIT)

        # Draw pass.
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        # REAL BUG, found by ty and deliberately left unfixed pending
        # Was ``self.object.get_vao()`` — the C++ accessor's name; the
        # Python ``sb6m.Object`` exposes the plain ``.vao`` field
        # (as multidrawindirect uses). The method call raised
        # AttributeError the moment the draw pass ran. Found by the ty
        # pass, fixed 2026-07-30 — re-verify on hardware.
        GL.glBindVertexArray(self.object.vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self.drawcmd_buf)
        GL.glBindBuffer(GL.GL_PARAMETER_BUFFER, self.param_buf)
        GL.glUseProgram(self.draw_program)
        self._multi_draw_count(GL.GL_TRIANGLES, None, 0, CANDIDATE_COUNT, 0)
        GL.glDisable(GL.GL_CULL_FACE)

    @override
    def draw_controls(self) -> None:
        imgui.text("1024 candidates → compute culls → indirect-count draw")
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.cull_program)
        GL.glDeleteProgram(self.draw_program)
        GL.glDeleteBuffers(1, [self.param_buf])
        GL.glDeleteBuffers(1, [self.candidates_buf])
        GL.glDeleteBuffers(1, [self.drawcmd_buf])
        GL.glDeleteBuffers(1, [self.model_ubo])
        GL.glDeleteBuffers(1, [self.transforms_ubo])
        GL.glDeleteTextures(1, [self.texture])
        self.object.free()


if __name__ == "__main__":
    application.run(CullindirectApp)
