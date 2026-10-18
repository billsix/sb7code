"""sdfdemo — signed-distance-field rendering of the GL logo and text.

Port of ``src/sdfdemo/sdfdemo.cpp``. Two modes:

  - LOGO: ``gllogodistsmarray.ktx`` (single-layer SDF of the GL logo)
    is rendered onto a rotating quad. Sharp edges at any zoom level
    because the FS reads the pre-baked distance field instead of
    sampling color.

  - TEXT: ``chars-df-array.ktx`` (a 2D-array texture of SDF glyph
    images) is rendered as 24 instanced quads, each picking a
    different array layer in the shader.

Reads its shaders from ``media/shaders/sdfdemo/``.
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
    return os.path.join(common.require_media(), "shaders", "sdfdemo", name)


MODE_LOGO: int = 0
MODE_TEXT: int = 1


class SdfdemoApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.sdf_program: int = 0
        self.render_mode: int = MODE_LOGO

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Distance Field"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        self.logo_texture: int = ktx.load(
            os.path.join(media, "textures", "gllogodistsmarray.ktx")
        )
        if not self.logo_texture:
            raise RuntimeError("failed to load gllogodistsmarray.ktx")
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.logo_texture)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D_ARRAY,
            GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_LINEAR_MIPMAP_LINEAR,
        )

        self.sdf_texture: int = ktx.load(
            os.path.join(media, "textures", "chars-df-array.ktx")
        )
        if not self.sdf_texture:
            raise RuntimeError("failed to load chars-df-array.ktx")
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.sdf_texture)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D_ARRAY,
            GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_LINEAR_MIPMAP_LINEAR,
        )

        self.vao: int = GL.glGenVertexArrays(1)
        self.load_shaders()

    def load_shaders(self) -> None:
        if self.sdf_program:
            GL.glDeleteProgram(self.sdf_program)
        self.sdf_program = common.compile_program_from_files(
            vs=_shader_path("sdf.vs.glsl"), fs=_shader_path("sdf.fs.glsl")
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
            math.cos(current_time * 0.2) * math.sin(current_time * 0.15) * 6.0
            + 3.001
        )
        cos_t: float = math.cos(current_time) * scale
        sin_t: float = math.sin(current_time) * scale

        identity3: np.ndarray = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

        if self.render_mode == MODE_LOGO:
            num_chars: int = 1
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.logo_texture)
            transform: np.ndarray = (
                vmath.translate(0.0, 0.0, -6.0 + cos_t * 0.25)
                @ vmath.rotate(cos_t * 12.0, 1.0, 0.0, 0.0)
                @ vmath.rotate(sin_t * 15.0, 0.0, 1.0, 0.0)
                @ vmath.scale(6.0, 6.0, 1.0)
            )
        else:
            num_chars = 24
            tz: float = -10.0
            rs: float = 4.0
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.sdf_texture)
            transform = (
                vmath.translate(0.0, 0.0, tz + cos_t)
                @ vmath.rotate(cos_t * 4.0 * rs, 1.0, 0.0, 0.0)
                @ vmath.rotate(sin_t * 3.0 * rs, 0.0, 0.0, 1.0)
                @ vmath.translate(-float(num_chars - 1) + sin_t, 0.0, 0.0)
            )

        # The C++ source uses an unusual ``frustum(-aspect, aspect,
        # 1.0, -1.0, 1.0, 100.0)`` — note bottom > top, which gives
        # a Y-flipped projection. Preserve.
        proj: np.ndarray = vmath.frustum(-aspect, aspect, 1.0, -1.0, 1.0, 100.0)

        GL.glUseProgram(self.sdf_program)
        # The C++ source hardcodes locations 0 and 1 for these
        # uniforms (matches `layout(location = N)` in the shader).
        # Query them anyway to be safe — see the
        # `feedback_query_uniform_locations` lesson.
        loc_uv: int = GL.glGetUniformLocation(self.sdf_program, "uv_transform")
        loc_mvp: int = GL.glGetUniformLocation(self.sdf_program, "transform")
        if loc_uv < 0:
            loc_uv = 0
        if loc_mvp < 0:
            loc_mvp = 1
        GL.glUniformMatrix3fv(loc_uv, 1, GL.GL_FALSE, identity3)
        vmath.upload(loc_mvp, proj @ transform)
        GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, num_chars)

    @override
    def draw_controls(self) -> None:
        if imgui.radio_button(
            "Logo (single SDF layer)", self.render_mode == MODE_LOGO
        ):
            self.render_mode = MODE_LOGO
        if imgui.radio_button(
            "Text (24 SDF glyph layers)", self.render_mode == MODE_TEXT
        ):
            self.render_mode = MODE_TEXT
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.sdf_program)
        GL.glDeleteTextures(2, [self.sdf_texture, self.logo_texture])
        GL.glDeleteVertexArrays(1, [self.vao])


if __name__ == "__main__":
    application.run(SdfdemoApp)
