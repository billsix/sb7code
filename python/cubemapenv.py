"""cubemapenv — dragon reflecting a cubemap skybox.

Port of ``src/cubemapenv/cubemapenv.cpp``. Loads
``mountaincube.ktx`` (a cubemap KTX — 6 faces) and ``dragon.sbm``.
The skybox program draws a fullscreen quad sampled from the cubemap
using the inverse view direction; the render program looks up
the reflection of each fragment's view direction off the dragon's
normal in the same cubemap, giving the dragon a reflective gold/silver
look.

First demo to validate ``ktx.load`` against the ``GL_TEXTURE_CUBE_MAP``
target — the loader picks it because the file's ``faces`` field is 6.

Reads its shaders from ``media/shaders/cubemapenv/``.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, ktx, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "cubemapenv", name)


class CubemapenvApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_prog: int = 0
        self.skybox_prog: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Cubic Environment Map"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        self.tex_envmap: int = ktx.load(
            os.path.join(media, "textures", "envmaps", "mountaincube.ktx")
        )
        if not self.tex_envmap:
            raise RuntimeError("failed to load mountaincube.ktx")

        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glEnable(GL.GL_TEXTURE_CUBE_MAP_SEAMLESS)

        self.dragon: sb6m.Object = sb6m.Object()
        self.dragon.load(os.path.join(media, "objects", "dragon.sbm"))

        self.load_shaders()

        self.skybox_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.skybox_vao)

        GL.glDepthFunc(GL.GL_LEQUAL)

    def load_shaders(self) -> None:
        if self.render_prog:
            GL.glDeleteProgram(self.render_prog)
        if self.skybox_prog:
            GL.glDeleteProgram(self.skybox_prog)
        self.render_prog = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )
        self.uniform_render_mv: int = GL.glGetUniformLocation(
            self.render_prog, "mv_matrix"
        )
        self.uniform_render_proj: int = GL.glGetUniformLocation(
            self.render_prog, "proj_matrix"
        )

        self.skybox_prog = common.compile_program_from_files(
            vs=_shader_path("skybox.vs.glsl"), fs=_shader_path("skybox.fs.glsl")
        )
        self.uniform_skybox_view: int = GL.glGetUniformLocation(
            self.skybox_prog, "view_matrix"
        )

    @override
    def render(self, current_time: float) -> None:
        gray: list[float] = [0.2, 0.2, 0.2, 1.0]
        ones: list[float] = [1.0]
        t: float = float(current_time) * 0.1

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(60.0, aspect, 0.1, 1000.0)
        view: np.ndarray = vmath.lookat(
            (15.0 * math.sin(t), 0.0, 15.0 * math.cos(t)),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        mv: np.ndarray = (
            view
            @ vmath.rotate(t, 1.0, 0.0, 0.0)
            @ vmath.rotate(t * 130.1, 0.0, 1.0, 0.0)
            @ vmath.translate(0.0, -4.0, 0.0)
        )

        GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, ones)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.tex_envmap)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)

        # Skybox pass — depth disabled so the quad fills behind the
        # dragon.
        GL.glUseProgram(self.skybox_prog)
        GL.glBindVertexArray(self.skybox_vao)
        vmath.upload(self.uniform_skybox_view, view)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # Render the dragon over it.
        GL.glUseProgram(self.render_prog)
        vmath.upload(self.uniform_render_mv, mv)
        vmath.upload(self.uniform_render_proj, proj)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.dragon.render()

    @override
    def draw_controls(self) -> None:
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.dragon.free()
        GL.glDeleteProgram(self.render_prog)
        GL.glDeleteProgram(self.skybox_prog)
        GL.glDeleteTextures(1, [self.tex_envmap])


if __name__ == "__main__":
    application.run(CubemapenvApp)
