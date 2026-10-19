"""pmbstreaming — uniform streaming via persistent mapped buffer + fences.

Port of ``src/pmbstreaming/pmbstreaming.cpp``.

Streams a fresh mv/proj matrix pair into a UBO every frame, then
draws a textured torus. The point of the demo is to compare four sync
strategies:

  - **NO_SYNC**: just write and draw. Fastest, but the CPU may
    overwrite memory the GPU is still reading from a prior frame.
  - **FINISH**: ``glFinish()`` after every draw. Stalls hard but is
    always safe.
  - **ONE_SYNC**: a single fence — wait on it before reusing the
    buffer. Stalls if the GPU hasn't caught up.
  - **RINGED_SYNC**: rotate through ``CHUNK_COUNT`` (=4) regions of
    the buffer so the CPU and GPU rarely touch the same chunk; only
    block when wrapping around. The recommended pattern.

Mode is selected via imgui radio buttons (the C++ uses ``M`` to
cycle).

Reads its shaders from ``media/shaders/pmbstreaming/``, the model from
``media/objects/torus_nrms_tc.sbm``, and the texture from
``media/textures/pattern1.ktx``.
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
    return os.path.join(common.require_media(), "shaders", "pmbstreaming", name)


def _media(name: str) -> str:
    return os.path.join(common.require_media(), name)


CHUNK_COUNT: int = 4
# Each MATRICES struct: 2 mat4s = 128 bytes. The C++ defines
# CHUNK_SIZE = 3 * sizeof(mat4) = 192 but writes vs_uniforms[i] with
# struct-stride (128). We mirror the C++: total buffer is
# CHUNK_SIZE * CHUNK_COUNT, but the per-index stride is 128 bytes.
CHUNK_SIZE: int = 192
MATRICES_BYTES: int = 128
BUFFER_SIZE: int = CHUNK_SIZE * CHUNK_COUNT

NO_SYNC, FINISH, ONE_SYNC, RINGED_SYNC = 0, 1, 2, 3
MODE_NAMES: list[str] = ["NO_SYNC", "FINISH", "ONE_SYNC", "RINGED_SYNC"]


class PmbstreamingApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.program: int = 0
        self.mode: int = NO_SYNC
        self.sync_index: int = 0
        self.stalled: bool = False
        self.fence = [None] * CHUNK_COUNT

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Constant Streaming"

    @override
    def startup(self) -> None:
        self.buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.buffer)
        # PyOpenGL constants are int subclasses at runtime, but the
        # stubs' `Constant` declares no `__or__`, so OR-ing two masks
        # trips ty.
        flags: int = (
            GL.GL_MAP_WRITE_BIT  # ty: ignore[unsupported-operator]
            | GL.GL_MAP_PERSISTENT_BIT
            | GL.GL_MAP_COHERENT_BIT
        )
        GL.glBufferStorage(GL.GL_UNIFORM_BUFFER, BUFFER_SIZE, None, flags)
        ptr: int = GL.glMapBufferRange(
            GL.GL_UNIFORM_BUFFER, 0, BUFFER_SIZE, flags
        )
        # Wrap as a (CHUNK_COUNT, 2, 4, 4) float32 view — each entry
        # is a (modelview, projection) mat4 pair. Stride 128 bytes.
        c_arr: ctypes.Array[ctypes.c_float] = (
            ctypes.c_float * (BUFFER_SIZE // 4)
        ).from_address(int(ptr))
        # The mapped region as raw float32 (BUFFER_SIZE/4 floats).
        # We index in element-units: each MATRICES is 32 floats.
        self._mapped_floats: np.ndarray | None = np.frombuffer(
            c_arr, dtype=np.float32
        )

        self.program = common.compile_program_from_files(
            vs=_shader_path("pmbstreaming.vs.glsl"),
            fs=_shader_path("pmbstreaming.fs.glsl"),
        )

        self.object: sb6m.Object = sb6m.Object()
        self.object.load(_media("objects/torus_nrms_tc.sbm"))

        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.buffer)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        self.texture: int = ktx.load(_media("textures/pattern1.ktx"))

    @override
    def render(self, current_time: float) -> None:
        t: float = float(current_time)
        proj: np.ndarray = vmath.perspective(
            60.0,
            max(self.info.window_width, 1) / max(self.info.window_height, 1),
            0.1,
            1800.0,
        )
        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -3.0)
            @ vmath.rotate(t * 43.75, 0.0, 1.0, 0.0)
            @ vmath.rotate(t * 17.75, 0.0, 0.0, 1.0)
            @ vmath.rotate(t * 35.3, 1.0, 0.0, 0.0)
        )

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, [0.0, 0.0, 0.0, 0.0])
        GL.glClearBufferfv(GL.GL_DEPTH, 0, [1.0])
        GL.glUseProgram(self.program)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        # Pre-write sync (ONE_SYNC and RINGED_SYNC).
        self.stalled = False
        if self.mode == ONE_SYNC and self.fence[0] is not None:
            status: np.ndarray = GL.glGetSynciv(
                self.fence[0], GL.GL_SYNC_STATUS, 4
            )
            self.stalled = status[1] == GL.GL_UNSIGNALED
            GL.glClientWaitSync(self.fence[0], 0, GL.GL_TIMEOUT_IGNORED)
            GL.glDeleteSync(self.fence[0])
            self.fence[0] = None
        elif (
            self.mode == RINGED_SYNC and self.fence[self.sync_index] is not None
        ):
            status = GL.glGetSynciv(
                self.fence[self.sync_index], GL.GL_SYNC_STATUS, 4
            )
            self.stalled = status[1] == GL.GL_UNSIGNALED
            GL.glClientWaitSync(
                self.fence[self.sync_index], 0, GL.GL_TIMEOUT_IGNORED
            )
            GL.glDeleteSync(self.fence[self.sync_index])
            self.fence[self.sync_index] = None

        # Write into the persistent UBO at the current chunk slot.
        # 32 floats per slot (2 mat4); stride is 32 floats since the
        # C++ does pointer arithmetic by sizeof(MATRICES)=128.
        slot_off: int = self.sync_index * (MATRICES_BYTES // 4)
        # `_mapped_floats` is None only outside the startup()..
        # shutdown() window, and render() runs strictly inside it; ty
        # can't see that lifecycle ordering.
        self._mapped_floats[slot_off + 0 : slot_off + 16] = mv.T.ravel()  # ty: ignore[invalid-assignment]
        self._mapped_floats[slot_off + 16 : slot_off + 32] = proj.T.ravel()  # ty: ignore[invalid-assignment]

        if self.mode == RINGED_SYNC:
            self.object.render(1, self.sync_index)
        else:
            self.object.render(1, 0)

        # Post-draw sync.
        if self.mode == FINISH:
            GL.glFinish()
            self.stalled = True
        elif self.mode == ONE_SYNC:
            self.fence[0] = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        elif self.mode == RINGED_SYNC:
            self.fence[self.sync_index] = GL.glFenceSync(
                GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0
            )

        self.sync_index = (self.sync_index + 1) % CHUNK_COUNT

    @override
    def draw_controls(self) -> None:
        imgui.text("Sync mode")
        for i, name in enumerate(MODE_NAMES):
            if imgui.radio_button(name, self.mode == i):
                self.mode = i
        imgui.text("STALLED" if self.stalled else "")

    @override
    def shutdown(self) -> None:
        for f in self.fence:
            if f is not None:
                GL.glDeleteSync(f)
        self._mapped_floats = None
        self.object.free()
        GL.glDeleteProgram(self.program)
        GL.glDeleteBuffers(1, [self.buffer])
        GL.glDeleteTextures(1, [self.texture])


if __name__ == "__main__":
    application.run(PmbstreamingApp)
