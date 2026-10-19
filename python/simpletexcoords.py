"""simpletexcoords — torus with selectable texture (checker / pattern1).

Port of ``src/simpletexcoords/simpletexcoords.cpp``. Loads
``torus_nrms_tc.sbm`` and two textures: a 16×16 procedural checker
built inline and ``pattern1.ktx`` from the media pack. The imgui
panel toggles between the two — useful to see how a model's UVs map
across both a synthetic and a "real" texture.

Reads its shaders from ``media/shaders/simpletexcoords/``.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, ktx, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(
        common.require_media(), "shaders", "simpletexcoords", name
    )


def _build_checker(size: int = 16) -> np.ndarray:
    """16x16 black/white checker with alpha=0xFF — same data layout as
    the C++ source's static array (RGBA bytes, alternating black/white per
    row + alternating per col)."""
    black: tuple[int, int, int, int] = (0x00, 0x00, 0x00, 0x00)
    white: tuple[int, int, int, int] = (0xFF, 0xFF, 0xFF, 0xFF)
    arr: np.ndarray = np.empty((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            arr[y, x] = black if ((x + y) & 1) else white
    return arr


class SimpletexcoordsApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_prog: int = 0
        self.tex_index: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Texture Coordinates"

    @override
    def startup(self) -> None:
        media: str = common.require_media()

        # Texture 0: 16x16 procedural checker.
        self.tex_object: list[int] = [GL.glGenTextures(1), 0]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_object[0])
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGB8, 16, 16)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            16,
            16,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            _build_checker(),
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )

        # Texture 1: pattern1.ktx
        self.tex_object[1] = ktx.load(
            os.path.join(media, "textures", "pattern1.ktx")
        )
        if not self.tex_object[1]:
            raise RuntimeError("failed to load pattern1.ktx")

        self.torus: sb6m.Object = sb6m.Object()
        self.torus.load(os.path.join(media, "objects", "torus_nrms_tc.sbm"))

        self.load_shaders()

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

    def load_shaders(self) -> None:
        if self.render_prog:
            GL.glDeleteProgram(self.render_prog)
        self.render_prog = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )
        self.uniform_mv: int = GL.glGetUniformLocation(
            self.render_prog, "mv_matrix"
        )
        self.uniform_proj: int = GL.glGetUniformLocation(
            self.render_prog, "proj_matrix"
        )

    @override
    def render(self, current_time: float) -> None:
        gray: list[float] = [0.2, 0.2, 0.2, 1.0]
        ones: list[float] = [1.0]

        GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, ones)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_object[self.tex_index])
        GL.glUseProgram(self.render_prog)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(60.0, aspect, 0.1, 1000.0)
        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -3.0)
            @ vmath.rotate(current_time * 19.3, 0.0, 1.0, 0.0)
            @ vmath.rotate(current_time * 21.1, 0.0, 0.0, 1.0)
        )
        vmath.upload(self.uniform_mv, mv)
        vmath.upload(self.uniform_proj, proj)

        self.torus.render()

    @override
    def draw_controls(self) -> None:
        imgui.text("Texture:")
        if imgui.radio_button("Checker (procedural)", self.tex_index == 0):
            self.tex_index = 0
        if imgui.radio_button("pattern1.ktx", self.tex_index == 1):
            self.tex_index = 1
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.torus.free()
        GL.glDeleteProgram(self.render_prog)
        GL.glDeleteTextures(2, self.tex_object)


if __name__ == "__main__":
    application.run(SimpletexcoordsApp)
