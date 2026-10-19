"""toonshading — quantized lighting via a 4-step 1D color ramp.

Port of ``src/toonshading/toonshading.cpp``. Loads
``torus_nrms_tc.sbm`` (a torus with normals and texcoords) and shades
it by sampling a tiny 4-texel 1D ``GL_TEXTURE_1D`` ramp keyed on
N·L. The hard transitions between levels give the classic cel-shaded
look.

The ramp is built from inline byte data (4 dark-red values) — no
external KTX needed despite the C++ source ``#include <sb7ktx.h>``.

Reads its shaders from ``media/shaders/toonshading/``.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "toonshading", name)


# 4 RGBA texels, R from dark to bright, others zero — same data as the
# C++ source.
TOON_TEX_DATA: np.ndarray = np.array(
    [
        0x44,
        0x00,
        0x00,
        0x00,
        0x88,
        0x00,
        0x00,
        0x00,
        0xCC,
        0x00,
        0x00,
        0x00,
        0xFF,
        0x00,
        0x00,
        0x00,
    ],
    dtype=np.uint8,
)


class ToonshadingApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_prog: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Toon Shading"

    @override
    def startup(self) -> None:
        self.tex_toon: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_1D, self.tex_toon)
        GL.glTexStorage1D(GL.GL_TEXTURE_1D, 1, GL.GL_RGB8, 4)
        GL.glTexSubImage1D(
            GL.GL_TEXTURE_1D,
            0,
            0,
            4,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            TOON_TEX_DATA,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )

        self.object: sb6m.Object = sb6m.Object()
        self.object.load(
            os.path.join(common.require_media(), "objects", "torus_nrms_tc.sbm")
        )

        self.load_shaders()

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

    def load_shaders(self) -> None:
        if self.render_prog:
            GL.glDeleteProgram(self.render_prog)
        self.render_prog = common.compile_program_from_files(
            vs=_shader_path("toonshading.vs.glsl"),
            fs=_shader_path("toonshading.fs.glsl"),
        )
        self.uniform_mv: int = GL.glGetUniformLocation(
            self.render_prog, "mv_matrix"
        )
        self.uniform_proj: int = GL.glGetUniformLocation(
            self.render_prog, "proj_matrix"
        )

    @override
    def render(self, current_time: float) -> None:
        gray: list[float] = [0.1, 0.1, 0.1, 1.0]
        ones: list[float] = [1.0]

        GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, ones)

        GL.glBindTexture(GL.GL_TEXTURE_1D, self.tex_toon)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glUseProgram(self.render_prog)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(60.0, aspect, 0.1, 1000.0)
        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -3.0)
            @ vmath.rotate(current_time * 13.75, 0.0, 1.0, 0.0)
            @ vmath.rotate(current_time * 7.75, 0.0, 0.0, 1.0)
            @ vmath.rotate(current_time * 15.3, 1.0, 0.0, 0.0)
        )
        vmath.upload(self.uniform_mv, mv)
        vmath.upload(self.uniform_proj, proj)

        self.object.render()

    @override
    def draw_controls(self) -> None:
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.object.free()
        GL.glDeleteProgram(self.render_prog)
        GL.glDeleteTextures(1, [self.tex_toon])


if __name__ == "__main__":
    application.run(ToonshadingApp)
