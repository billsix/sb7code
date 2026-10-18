"""equirectangular — dragon reflecting an equirectangular HDR map.

Port of ``src/equirectangular/equirectangular.cpp``.
Loads ``equirectangularmap1.ktx`` (a 2:1 equirectangular projection
of an environment) and ``dragon.sbm``. The fragment shader converts
the reflected view direction into spherical (theta, phi) → 2D UV
into the equirectangular map.

Reads its shaders from ``media/shaders/equirectangular/``.
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
        common.require_media(), "shaders", "equirectangular", name
    )


class EquirectangularApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_prog: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Equirectangular Environment Map"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        self.tex_envmap: int = ktx.load(
            os.path.join(
                media, "textures", "envmaps", "equirectangularmap1.ktx"
            )
        )
        if not self.tex_envmap:
            raise RuntimeError("failed to load equirectangularmap1.ktx")
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

        self.dragon: sb6m.Object = sb6m.Object()
        self.dragon.load(os.path.join(media, "objects", "dragon.sbm"))

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
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_envmap)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glUseProgram(self.render_prog)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(60.0, aspect, 0.1, 1000.0)
        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -15.0)
            @ vmath.rotate(current_time * (180.0 / 3.14159265), 1.0, 0.0, 0.0)
            @ vmath.rotate(
                current_time * 1.1 * (180.0 / 3.14159265), 0.0, 1.0, 0.0
            )
            @ vmath.translate(0.0, -4.0, 0.0)
        )

        vmath.upload(self.uniform_mv, mv)
        vmath.upload(self.uniform_proj, proj)

        self.dragon.render()

    @override
    def draw_controls(self) -> None:
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.dragon.free()
        GL.glDeleteProgram(self.render_prog)
        GL.glDeleteTextures(1, [self.tex_envmap])


if __name__ == "__main__":
    application.run(EquirectangularApp)
