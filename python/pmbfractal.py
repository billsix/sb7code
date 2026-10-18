"""pmbfractal — Julia fractal computed CPU-side, streamed via PMB.

Port of ``src/pmbfractal/pmbfractal.cpp``.

Demonstrates GL 4.4 **persistent mapped buffers**: ``glBufferStorage``
with ``GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT`` lets the driver
keep a CPU-visible pointer alive across draw calls. The CPU updates
the fractal pixels directly through that pointer; the GPU consumes
them as a ``GL_PIXEL_UNPACK_BUFFER`` source for ``glTexSubImage2D``.

The C++ uses OpenMP across rows. Python's numpy vectorization plays
the same role — we generate the fractal as one big numpy expression
and copy the result into the persistently-mapped region.

Reads its shaders from ``media/shaders/fsq/``.
"""

import ctypes
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "fsq", name)


FRACTAL_WIDTH: int = 512
FRACTAL_HEIGHT: int = 512
BUFFER_SIZE = FRACTAL_WIDTH * FRACTAL_HEIGHT


class PmbfractalApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.program: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Persistent Mapped Fractal"

    @override
    def startup(self) -> None:
        # Allocate a persistently-mapped pixel-unpack buffer.
        self.buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.buffer)
        # PyOpenGL constants are int subclasses at runtime, but the
        # stubs' `Constant` declares no `__or__`, so OR-ing two masks
        # trips ty.
        flags = (
            GL.GL_MAP_WRITE_BIT  # ty: ignore[unsupported-operator]
            | GL.GL_MAP_PERSISTENT_BIT
            | GL.GL_MAP_COHERENT_BIT
        )
        GL.glBufferStorage(GL.GL_PIXEL_UNPACK_BUFFER, BUFFER_SIZE, None, flags)
        ptr = GL.glMapBufferRange(
            GL.GL_PIXEL_UNPACK_BUFFER, 0, BUFFER_SIZE, flags
        )
        # Wrap the raw GL pointer as a numpy array view. The
        # underlying memory stays alive for the lifetime of the
        # buffer thanks to GL_MAP_PERSISTENT_BIT.
        c_arr = (ctypes.c_uint8 * BUFFER_SIZE).from_address(int(ptr))
        self.mapped: np.ndarray | None = np.frombuffer(
            c_arr, dtype=np.uint8
        ).reshape(FRACTAL_HEIGHT, FRACTAL_WIDTH)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        self.program = common.compile_program_from_files(
            vs=_shader_path("fsq.vs.glsl"), fs=_shader_path("fsq.fs.glsl")
        )

        self.texture: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D, 1, GL.GL_R8, FRACTAL_WIDTH, FRACTAL_HEIGHT
        )

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Pixel-grid coordinates as a persistent (W,H) mesh — reused
        # every frame. Cuts allocation cost out of the hot loop.
        xs = np.arange(FRACTAL_WIDTH, dtype=np.float32) / FRACTAL_WIDTH - 0.5
        ys = np.arange(FRACTAL_HEIGHT, dtype=np.float32) / FRACTAL_HEIGHT - 0.5
        self._gridx, self._gridy = np.meshgrid(xs, ys)

    def _update_fractal(self, t: float) -> None:
        cx = (1.5 - np.cos(t * 0.4) * 0.5) * 0.3
        cy = (1.5 + np.cos(t * 0.5) * 0.5) * 0.3
        ox = np.cos(t * 0.14) * 0.25
        oy = np.cos(t * 0.25) * 0.25
        zoom = (np.sin(t) + 1.3) * 0.7

        zx = (zoom * self._gridx + ox).astype(np.float32)
        zy = (zoom * self._gridy + oy).astype(np.float32)
        thresh_squared = np.float32(256.0)
        cx = np.float32(cx)
        cy = np.float32(cy)

        # Iteration count per pixel — Julia escape time. Vectorized;
        # tracks `it` per pixel via a mask.
        it: np.ndarray = np.zeros_like(zx, dtype=np.int32)
        live: np.ndarray = np.ones_like(zx, dtype=bool)
        for n in range(256):
            zx2 = zx * zx - zy * zy + cx
            zy2 = 2.0 * zx * zy + cy
            zx, zy = zx2, zy2
            mag2 = zx * zx + zy * zy
            escaped = mag2 > thresh_squared
            new_escape = live & escaped
            it[new_escape] = n
            live &= ~escaped
            if not live.any():
                break
        # Pixels that never escaped get the maximum (255).
        it[live] = 255
        # `mapped` is None only outside the startup()..shutdown()
        # window, and render() (this method's only caller) runs strictly
        # inside it; ty can't see that lifecycle ordering.
        self.mapped[:] = it.astype(np.uint8)  # ty: ignore[invalid-assignment]

    @override
    def render(self, current_time: float) -> None:
        self._update_fractal(float(current_time))

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glUseProgram(self.program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.buffer)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            FRACTAL_WIDTH,
            FRACTAL_HEIGHT,
            GL.GL_RED,
            GL.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        imgui.text("Julia fractal via persistent mapped pixel-unpack buffer")
        imgui.text("Resolution: %d×%d" % (FRACTAL_WIDTH, FRACTAL_HEIGHT))

    @override
    def shutdown(self) -> None:
        # Drop the numpy view first so we don't keep a dangling
        # reference into the GL-owned pointer after the buffer is
        # deleted.
        self.mapped = None
        GL.glDeleteProgram(self.program)
        GL.glDeleteTextures(1, [self.texture])
        GL.glDeleteBuffers(1, [self.buffer])
        GL.glDeleteVertexArrays(1, [self.vao])


if __name__ == "__main__":
    application.run(PmbfractalApp)
