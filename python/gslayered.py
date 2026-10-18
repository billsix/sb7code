"""gslayered — geometry shader writes 16 layers of a 2D-array FBO.

Port of ``src/gslayered/gslayered.cpp``. Two passes:

  Pass 1: render the torus once, but the geometry shader runs 16
          times (one invocation per layer) and assigns
          ``gl_Layer = invocation_index``. Each layer gets a
          different MV matrix from a UBO array, producing 16
          slightly-different views in a single draw call. Output is
          a 256×256×16 GL_TEXTURE_2D_ARRAY.

  Pass 2: a fullscreen-quad-per-layer (16 instances of a quad)
          samples the array texture and tiles the 16 layers across
          the screen.

Reads its shaders from ``media/shaders/gslayers/`` (note the upstream
folder is ``gslayers`` without the ``ed``).
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    # Note: upstream folder is "gslayers" not "gslayered".
    return os.path.join(common.require_media(), "shaders", "gslayers", name)


LAYER_TEX_SIZE: int = 256
NUM_LAYERS: int = 16


class GslayeredApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.program_gslayers: int = 0
        self.program_showlayers: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Layered Rendering"

    @override
    def startup(self) -> None:
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.load_shaders()

        self.torus: sb6m.Object = sb6m.Object()
        self.torus.load(
            os.path.join(common.require_media(), "objects", "torus.sbm")
        )

        # Transform UBO: 1 proj_matrix + 16 mv_matrices = 17 mat4s.
        self.transform_ubo: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.transform_ubo)
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 17 * 64, None, GL.GL_DYNAMIC_DRAW)

        # 256×256×16 array texture for color, plus matching depth.
        self.array_texture: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.array_texture)
        GL.glTexStorage3D(
            GL.GL_TEXTURE_2D_ARRAY,
            1,
            GL.GL_RGBA8,
            LAYER_TEX_SIZE,
            LAYER_TEX_SIZE,
            NUM_LAYERS,
        )

        self.array_depth: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.array_depth)
        GL.glTexStorage3D(
            GL.GL_TEXTURE_2D_ARRAY,
            1,
            GL.GL_DEPTH_COMPONENT32,
            LAYER_TEX_SIZE,
            LAYER_TEX_SIZE,
            NUM_LAYERS,
        )

        self.layered_fbo: int = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.layered_fbo)
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.array_texture, 0
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.array_depth, 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def load_shaders(self) -> None:
        if self.program_showlayers:
            GL.glDeleteProgram(self.program_showlayers)
        if self.program_gslayers:
            GL.glDeleteProgram(self.program_gslayers)
        self.program_showlayers = common.compile_program_from_files(
            vs=_shader_path("showlayers.vs.glsl"),
            fs=_shader_path("showlayers.fs.glsl"),
        )
        self.program_gslayers = common.compile_program_from_files(
            vs=_shader_path("gslayers.vs.glsl"),
            gs=_shader_path("gslayers.gs.glsl"),
            fs=_shader_path("gslayers.fs.glsl"),
        )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 1.0]
        gray: list[float] = [0.1, 0.1, 0.1, 1.0]
        one: list[float] = [1.0]
        t: float = float(current_time)

        # Build the transform UBO: 1 proj + 16 mv matrices.
        # std140 mat4 = column-major; transpose on upload.
        proj: np.ndarray = vmath.perspective(50.0, 1.0, 0.1, 1000.0)
        ubo: np.ndarray = np.empty((17, 4, 4), dtype=np.float32)
        ubo[0] = proj.T
        for i in range(NUM_LAYERS):
            fi = (i + 12) / 16.0
            mv: np.ndarray = (
                vmath.translate(0.0, 0.0, -4.0)
                @ vmath.rotate(t * 25.0 * fi, 0.0, 0.0, 1.0)
                @ vmath.rotate(t * 30.0 * fi, 1.0, 0.0, 0.0)
            )
            ubo[1 + i] = mv.T
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.transform_ubo)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER, 0, ubo.nbytes, np.ascontiguousarray(ubo)
        )

        # Pass 1: render once, GS emits 16 layers via gl_Layer.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.layered_fbo)
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
        GL.glViewport(0, 0, LAYER_TEX_SIZE, LAYER_TEX_SIZE)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glUseProgram(self.program_gslayers)
        self.torus.render()

        # Pass 2: fullscreen quad-per-layer tiles the 16 layers.
        # Reset viewport to window — the layered_fbo pass set it to
        # 256×256.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glDrawBuffer(GL.GL_BACK)
        GL.glUseProgram(self.program_showlayers)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.array_texture)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArraysInstanced(GL.GL_TRIANGLE_FAN, 0, 4, NUM_LAYERS)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)

    @override
    def draw_controls(self) -> None:
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.torus.free()
        GL.glDeleteProgram(self.program_showlayers)
        GL.glDeleteProgram(self.program_gslayers)
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteTextures(1, [self.array_texture])
        GL.glDeleteTextures(1, [self.array_depth])
        GL.glDeleteFramebuffers(1, [self.layered_fbo])
        GL.glDeleteBuffers(1, [self.transform_ubo])


if __name__ == "__main__":
    application.run(GslayeredApp)
