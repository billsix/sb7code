"""dispmap — heightfield landscape via tessellation + displacement-mapped patches.

Port of ``src/dispmap/dispmap.cpp``. Draws 4096 patches
in a 64×64 grid, each tessellated by the TES into a high-density mesh
that samples ``terragen1.ktx`` (height) and ``terragen_color.ktx``
(color). The camera orbits the heightfield at varying radius/height.

Reads its shaders from ``media/shaders/dispmap/``.

Imgui controls: Wireframe / Fog / Displacement / Pause checkboxes,
Displacement-depth slider, Reload button.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, ktx, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "dispmap", name)


class DispmapApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.program: int = 0
        self.dmap_depth: float = 6.0
        self.enable_displacement: bool = True
        self.wireframe: bool = False
        self.enable_fog: bool = True
        self.paused: bool = False
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Displacement Mapping"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 4)
        GL.glEnable(GL.GL_CULL_FACE)

        media: str = common.require_media()
        # Texture unit 0: displacement / height map
        GL.glActiveTexture(GL.GL_TEXTURE0)
        self.tex_displacement: int = ktx.load(
            os.path.join(media, "textures", "terragen1.ktx")
        )
        # Texture unit 1: color map
        GL.glActiveTexture(GL.GL_TEXTURE1)
        self.tex_color: int = ktx.load(
            os.path.join(media, "textures", "terragen_color.ktx")
        )
        if not self.tex_displacement or not self.tex_color:
            raise RuntimeError("failed to load dispmap KTX textures")
        GL.glActiveTexture(GL.GL_TEXTURE0)

    def load_shaders(self) -> None:
        if self.program:
            GL.glDeleteProgram(self.program)
        self.program = common.compile_program_from_files(
            vs=_shader_path("dispmap.vs.glsl"),
            tcs=_shader_path("dispmap.tcs.glsl"),
            tes=_shader_path("dispmap.tes.glsl"),
            fs=_shader_path("dispmap.fs.glsl"),
        )
        self.uniform_mv: int = GL.glGetUniformLocation(
            self.program, "mv_matrix"
        )
        self.uniform_mvp: int = GL.glGetUniformLocation(
            self.program, "mvp_matrix"
        )
        self.uniform_proj: int = GL.glGetUniformLocation(
            self.program, "proj_matrix"
        )
        self.uniform_dmap_depth: int = GL.glGetUniformLocation(
            self.program, "dmap_depth"
        )
        self.uniform_enable_fog: int = GL.glGetUniformLocation(
            self.program, "enable_fog"
        )

    @override
    def render(self, current_time: float) -> None:
        sky: list[float] = [0.85, 0.95, 1.0, 1.0]
        one: list[float] = [1.0]

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        t = self._total_time * 0.03
        r: float = math.sin(t * 5.37) * 15.0 + 16.0
        h: float = math.cos(t * 4.79) * 2.0 + 3.2

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, sky)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        eye = (math.sin(t) * r, h, math.cos(t) * r)
        mv: np.ndarray = vmath.lookat(eye, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(60.0, aspect, 0.1, 1000.0)

        GL.glUseProgram(self.program)
        vmath.upload(self.uniform_mv, mv)
        vmath.upload(self.uniform_proj, proj)
        vmath.upload(self.uniform_mvp, proj @ mv)
        GL.glUniform1f(
            self.uniform_dmap_depth,
            self.dmap_depth if self.enable_displacement else 0.0,
        )
        GL.glUniform1i(self.uniform_enable_fog, 1 if self.enable_fog else 0)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glPolygonMode(
            GL.GL_FRONT_AND_BACK, GL.GL_LINE if self.wireframe else GL.GL_FILL
        )
        GL.glDrawArraysInstanced(GL.GL_PATCHES, 0, 4, 64 * 64)

    @override
    def draw_controls(self) -> None:
        _, self.enable_displacement = imgui.checkbox(
            "Displacement", self.enable_displacement
        )
        if self.enable_displacement:
            _, self.dmap_depth = imgui.slider_float(
                "Depth", self.dmap_depth, 0.0, 20.0
            )
        _, self.wireframe = imgui.checkbox("Wireframe", self.wireframe)
        _, self.enable_fog = imgui.checkbox("Fog", self.enable_fog)
        _, self.paused = imgui.checkbox("Pause camera", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)
        GL.glDeleteTextures(2, [self.tex_displacement, self.tex_color])


if __name__ == "__main__":
    application.run(DispmapApp)
