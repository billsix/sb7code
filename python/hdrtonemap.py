"""hdrtonemap — three tone-mapping operators on the same HDR image.

Port of ``src/hdrtonemap/hdrtonemap.cpp``. Loads the
``treelights_2k.ktx`` HDR image and a 20-entry 1D float LUT used by
the adaptive tone-mapper. Three shader programs share the same vertex
shader; the imgui radio group picks which fragment shader (naive
exposure clamp / global exposure / luminance-adaptive) is active.

Reads its shaders from ``media/shaders/hdrtonemap/``.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, ktx


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "hdrtonemap", name)


# Same per-luminance LUT as the C++ source.
EXPOSURE_LUT: np.ndarray = np.array(
    [
        11.0,
        6.0,
        3.2,
        2.8,
        2.2,
        1.90,
        1.80,
        1.80,
        1.70,
        1.70,
        1.60,
        1.60,
        1.50,
        1.50,
        1.40,
        1.40,
        1.30,
        1.20,
        1.10,
        1.00,
    ],
    dtype=np.float32,
)


MODE_NAIVE: int = 0
MODE_EXPOSURE: int = 1
MODE_ADAPTIVE: int = 2
MODE_NAMES: list[str] = [
    "Naive (clamp)",
    "Global exposure",
    "Adaptive (luminance LUT)",
]


class HdrtonemapApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.exposure: float = 1.0
        self.mode: int = MODE_ADAPTIVE
        self.program_adaptive: int = 0
        self.program_exposure: int = 0
        self.program_naive: int = 0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - HDR Tone Mapping"

    @override
    def startup(self) -> None:
        media: str = common.require_media()
        self.tex_src: int = ktx.load(
            os.path.join(media, "textures", "treelights_2k.ktx")
        )
        if not self.tex_src:
            raise RuntimeError("failed to load treelights_2k.ktx")
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_src)

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.load_shaders()

        # 1D LUT for the adaptive tone mapper.
        self.tex_lut: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_1D, self.tex_lut)
        GL.glTexStorage1D(GL.GL_TEXTURE_1D, 1, GL.GL_R32F, 20)
        GL.glTexSubImage1D(
            GL.GL_TEXTURE_1D, 0, 0, 20, GL.GL_RED, GL.GL_FLOAT, EXPOSURE_LUT
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )

    def load_shaders(self) -> None:
        for prog_attr in (
            "program_naive",
            "program_exposure",
            "program_adaptive",
        ):
            old: int = getattr(self, prog_attr)
            if old:
                GL.glDeleteProgram(old)
        self.program_naive = common.compile_program_from_files(
            vs=_shader_path("tonemap.vs.glsl"),
            fs=_shader_path("tonemap_naive.fs.glsl"),
        )
        self.program_exposure = common.compile_program_from_files(
            vs=_shader_path("tonemap.vs.glsl"),
            fs=_shader_path("tonemap_exposure.fs.glsl"),
        )
        self.program_adaptive = common.compile_program_from_files(
            vs=_shader_path("tonemap.vs.glsl"),
            fs=_shader_path("tonemap_adaptive.fs.glsl"),
        )
        self.exposure_loc: int = GL.glGetUniformLocation(
            self.program_exposure, "exposure"
        )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.25, 0.0, 1.0]
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_1D, self.tex_lut)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_src)

        if self.mode == MODE_NAIVE:
            GL.glUseProgram(self.program_naive)
        elif self.mode == MODE_EXPOSURE:
            GL.glUseProgram(self.program_exposure)
            GL.glUniform1f(self.exposure_loc, self.exposure)
        else:
            GL.glUseProgram(self.program_adaptive)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        for i, name in enumerate(MODE_NAMES):
            if imgui.radio_button(name, self.mode == i):
                self.mode = i
        if self.mode == MODE_EXPOSURE:
            _, self.exposure = imgui.slider_float(
                "Exposure",
                self.exposure,
                0.01,
                32.0,
                flags=imgui.SliderFlags_.logarithmic.value,
            )
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        for prog in (
            self.program_naive,
            self.program_exposure,
            self.program_adaptive,
        ):
            GL.glDeleteProgram(prog)
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteTextures(1, [self.tex_src])
        GL.glDeleteTextures(1, [self.tex_lut])


if __name__ == "__main__":
    application.run(HdrtonemapApp)
