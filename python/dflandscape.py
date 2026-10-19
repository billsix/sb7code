"""dflandscape — distance-field "landscape" with mossy/rocky terrain.

Port of ``src/dflandscape/dflandscape.cpp``. A
distance-field map (``psycho-map-df-sm.ktx``) selects between
``mossygrass.ktx`` and ``rocks.ktx`` per fragment, giving a quasi-
hand-drawn appearance that stays sharp at any zoom. The view drifts
slowly via cosine/sine on time.

Reads its shaders from ``media/shaders/dflandscape/``.
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
    return os.path.join(common.require_media(), "shaders", "dflandscape", name)


class DflandscapeApp(application.Application):
    # Assigned via setattr() in startup()'s texture-loading loop, which
    # ty can't see through; declared here so the attributes exist for
    # the type checker (bare annotations create no runtime attribute).
    map_texture: int
    grass_texture: int
    rocks_texture: int

    def __init__(self) -> None:
        super().__init__()
        self.sdf_program: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Distance Field Landscaping"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        for name in ("psycho-map-df-sm.ktx", "mossygrass.ktx", "rocks.ktx"):
            tex: int = ktx.load(os.path.join(media, "textures", name))
            if not tex:
                raise RuntimeError(f"failed to load {name}")
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D,
                GL.GL_TEXTURE_MIN_FILTER,
                GL.GL_LINEAR_MIPMAP_LINEAR,
            )
            attr: str = {
                "psycho-map-df-sm.ktx": "map_texture",
                "mossygrass.ktx": "grass_texture",
                "rocks.ktx": "rocks_texture",
            }[name]
            setattr(self, attr, tex)

        self.vao: int = GL.glGenVertexArrays(1)
        self.load_shaders()

    def load_shaders(self) -> None:
        if self.sdf_program:
            GL.glDeleteProgram(self.sdf_program)
        self.sdf_program = common.compile_program_from_files(
            vs=_shader_path("dflandscape.vs.glsl"),
            fs=_shader_path("dflandscape.fs.glsl"),
        )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )

        GL.glBindVertexArray(self.vao)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        scale: float = (
            math.cos(current_time * 0.2) * math.sin(current_time * 0.15) * 3.0
            + 3.2
        )
        cos_t: float = math.cos(current_time * 0.03) * 0.25
        sin_t: float = math.sin(current_time * 0.02) * 0.25

        # uv_transform — same matrix as the C++ source's m2.
        uv_xform: np.ndarray = np.array(
            [
                cos_t * scale,
                sin_t * scale,
                0.0,
                -sin_t * scale,
                cos_t * scale,
                0.0,
                cos_t,
                sin_t,
                1.0,
            ],
            dtype=np.float32,
        )

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.map_texture)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.grass_texture)
        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.rocks_texture)
        GL.glActiveTexture(GL.GL_TEXTURE0)

        transform: np.ndarray = vmath.translate(0.0, 0.0, -1.0) @ vmath.scale(
            10.0, 10.0, 1.0
        )
        # Same Y-flipped frustum as sdfdemo.
        proj: np.ndarray = vmath.frustum(-aspect, aspect, 1.0, -1.0, 1.0, 100.0)

        GL.glUseProgram(self.sdf_program)
        loc_uv: int = GL.glGetUniformLocation(self.sdf_program, "uv_transform")
        loc_xf: int = GL.glGetUniformLocation(self.sdf_program, "transform")
        if loc_uv < 0:
            loc_uv = 0
        if loc_xf < 0:
            loc_xf = 1
        GL.glUniformMatrix3fv(loc_uv, 1, GL.GL_FALSE, uv_xform)
        vmath.upload(loc_xf, proj @ transform)
        GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, 1)

    @override
    def draw_controls(self) -> None:
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.sdf_program)
        GL.glDeleteTextures(
            3, [self.map_texture, self.grass_texture, self.rocks_texture]
        )
        GL.glDeleteVertexArrays(1, [self.vao])


if __name__ == "__main__":
    application.run(DflandscapeApp)
