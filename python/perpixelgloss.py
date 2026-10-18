"""perpixelgloss — torus reflecting a 3D environment + 2D gloss mask.

Port of ``src/perpixelgloss/perpixelgloss.cpp``. Loads
``mountains3d.ktx`` (a 3D environment volume) into texture unit 0 and
``pattern1.ktx`` (gloss/mask) into unit 1, plus ``torus_nrms_tc.sbm``.
The fragment shader samples the 3D environment using the reflected
view direction, then modulates by the gloss mask.

Validates ``ktx.load`` against the ``GL_TEXTURE_3D`` target — the
loader picks it because the file's ``pixeldepth`` is non-zero.

Reads its shaders from ``media/shaders/perpixelgloss/``.
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
        common.require_media(), "shaders", "perpixelgloss", name
    )


class PerpixelglossApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_prog: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Per-Pixel Gloss"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        self.tex_envmap: int = ktx.load(
            os.path.join(media, "textures", "envmaps", "mountains3d.ktx")
        )
        if not self.tex_envmap:
            raise RuntimeError("failed to load mountains3d.ktx")
        GL.glTexParameteri(
            GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE
        )

        GL.glActiveTexture(GL.GL_TEXTURE1)
        self.tex_glossmap: int = ktx.load(
            os.path.join(media, "textures", "pattern1.ktx")
        )
        if not self.tex_glossmap:
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
            vs=_shader_path("perpixelgloss.vs.glsl"),
            fs=_shader_path("perpixelgloss.fs.glsl"),
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

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self.tex_envmap)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_glossmap)

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

        self.torus.render()

    @override
    def draw_controls(self) -> None:
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.torus.free()
        GL.glDeleteProgram(self.render_prog)
        GL.glDeleteTextures(2, [self.tex_envmap, self.tex_glossmap])


if __name__ == "__main__":
    application.run(PerpixelglossApp)
