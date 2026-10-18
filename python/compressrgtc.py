"""compressrgtc — compress a 2D texture to RGTC1 in a compute shader.

Port of ``src/compressrgtc/compressrgtc.cpp``.

Pipeline per frame:

  1. Bind the input KTX texture (``gllogodistsm.ktx``, 512×512 R8) to
     texture unit 0, and the output texture buffer (sized for the
     RGTC1 result, GL_RG32UI = 8 bytes per 4×4 block) as image unit 0.

  2. Dispatch the compress compute shader (TEXTURE_WIDTH/4 ×
     TEXTURE_HEIGHT/4 workgroups). Each invocation writes one 8-byte
     compressed block.

  3. Use the texture buffer as a PIXEL_UNPACK source and call
     ``glCompressedTexImage2D`` to package those bytes as a real
     ``GL_COMPRESSED_RED_RGTC1`` 2D texture.

  4. Draw a fullscreen quad sampling either the input or the
     RGTC-decompressed output (imgui radio).

Reads its shaders from ``media/shaders/rgtc/`` and the source texture
from ``media/textures/gllogodistsm.ktx``.
"""

import ctypes
import os
from typing import override

import glfw  # noqa: F401
import OpenGL.GL as GL
from imgui_bundle import imgui
from OpenGL.raw.GL.VERSION import (
    GL_1_3,  # raw glCompressedTexImage2D (see render)
)

from sb7 import application, common, ktx


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "rgtc", name)


def _texture_path(name: str) -> str:
    return os.path.join(common.require_media(), "textures", name)


TEXTURE_WIDTH: int = 512
TEXTURE_HEIGHT: int = 512


class CompressrgtcApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.compress_program: int = 0
        self.render_program: int = 0
        # 0 = input, 1 = RGTC-compressed output.
        self.display_mode: int = 1

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - RGTC Compressor"
        self.info.window_width = 512
        self.info.window_height = 512

    @override
    def startup(self) -> None:
        self.input_texture: int = ktx.load(_texture_path("gllogodistsm.ktx"))

        # The output 2D texture: receives the decompressed RGTC1
        # result via glCompressedTexImage2D.
        self.output_texture: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.output_texture)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )

        # Output buffer: holds the raw compressed bytes (one
        # GL_RG32UI = 8 bytes per 4×4 block).
        self.output_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, self.output_buffer)
        GL.glBufferStorage(
            GL.GL_TEXTURE_BUFFER,
            TEXTURE_WIDTH * TEXTURE_HEIGHT // 2,
            None,
            GL.GL_MAP_READ_BIT,
        )

        # Texture-buffer view onto that buffer (RG32UI for image
        # writes from the compute shader).
        self.output_buffer_texture: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, self.output_buffer_texture)
        GL.glTexBuffer(GL.GL_TEXTURE_BUFFER, GL.GL_RG32UI, self.output_buffer)

        self.dummy_vao: int = GL.glGenVertexArrays(1)
        self.load_shaders()

    def load_shaders(self) -> None:
        if self.compress_program:
            GL.glDeleteProgram(self.compress_program)
        if self.render_program:
            GL.glDeleteProgram(self.render_program)
        self.compress_program = common.compile_program_from_files(
            cs=_shader_path("rgtccompress.cs.glsl")
        )
        self.render_program = common.compile_program_from_files(
            vs=_shader_path("drawquad.vs.glsl"),
            fs=_shader_path("drawquad.fs.glsl"),
        )

    @override
    def render(self, current_time: float) -> None:
        GL.glClearBufferfv(GL.GL_COLOR, 0, [0.0, 0.0, 0.0, 1.0])

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.input_texture)
        GL.glBindImageTexture(
            0,
            self.output_buffer_texture,
            0,
            GL.GL_FALSE,
            0,
            GL.GL_WRITE_ONLY,
            GL.GL_RG32UI,
        )

        GL.glUseProgram(self.compress_program)
        GL.glDispatchCompute(TEXTURE_WIDTH // 4, TEXTURE_HEIGHT // 4, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Re-package the texture-buffer's raw bytes as a real
        # RGTC1-compressed 2D texture.
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.output_buffer)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.output_texture)
        # The RAW entry point, on purpose. PyOpenGL's wrapped
        # glCompressedTexImage2D drops the C `imageSize` parameter and
        # infers it from the data array -- so the 8-argument C call shifts
        # imageSize into the data slot and dies in NumberHandler
        # ("no attribute arrayByteCount"). With a PIXEL_UNPACK buffer
        # bound there is no array to infer from: data is a byte offset,
        # exactly as the C++ passes NULL, so the size must be explicit.
        GL_1_3.glCompressedTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_COMPRESSED_RED_RGTC1,
            TEXTURE_WIDTH,
            TEXTURE_HEIGHT,
            0,
            TEXTURE_WIDTH * TEXTURE_HEIGHT // 2,
            ctypes.c_void_p(0),
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glBindVertexArray(self.dummy_vao)
        GL.glUseProgram(self.render_program)
        if self.display_mode == 0:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.input_texture)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.output_texture)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        imgui.text("Display:")
        if imgui.radio_button("Input (R8 KTX)", self.display_mode == 0):
            self.display_mode = 0
        imgui.same_line()
        if imgui.radio_button("RGTC-compressed output", self.display_mode == 1):
            self.display_mode = 1
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.compress_program)
        GL.glDeleteProgram(self.render_program)
        GL.glDeleteTextures(1, [self.input_texture])
        GL.glDeleteTextures(1, [self.output_texture])
        GL.glDeleteTextures(1, [self.output_buffer_texture])
        GL.glDeleteBuffers(1, [self.output_buffer])
        GL.glDeleteVertexArrays(1, [self.dummy_vao])


if __name__ == "__main__":
    application.run(CompressrgtcApp)
