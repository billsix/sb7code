"""deferredshading — classic deferred-shading pipeline.

Port of ``src/deferredshading/deferredshading.cpp``.

  Pass 1 (geometry): render 225 instanced ladybugs into a 2-color
                     G-buffer:
                       COLOR0 (RGBA32UI): packed normal + diffuse + meta
                       COLOR1 (RGBA32F):  world-space position
                       DEPTH:             standard depth buffer

  Pass 2 (lighting): a fullscreen quad samples both G-buffer textures
                     and accumulates contributions from 64 animated
                     point lights (positions/colors in a UBO). The
                     result is the final lit image.

Visualization modes (radio in imgui) replace the lighting shader with
one that outputs the raw G-buffer channels for debugging:

  - off (final lighting)
  - normals (decoded normal as color)
  - world-space coords
  - diffuse albedo
  - meta (specular params, etc.)

Reads its shaders from ``media/shaders/deferredshading/``.
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
    return os.path.join(
        common.require_media(), "shaders", "deferredshading", name
    )


MAX_DISPLAY_WIDTH: int = 2048
MAX_DISPLAY_HEIGHT: int = 2048
NUM_LIGHTS: int = 64
NUM_INSTANCES: int = 15 * 15

VIS_OFF: int = 0
VIS_NORMALS: int = 1
VIS_WS_COORDS: int = 2
VIS_DIFFUSE: int = 3
VIS_META: int = 4
VIS_LABELS: list[str] = [
    "Off (final lighting)",
    "Normals",
    "World-space coords",
    "Diffuse albedo",
    "Meta",
]


class DeferredshadingApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_program: int = 0
        self.render_program_nm: int = 0
        self.light_program: int = 0
        self.vis_program: int = 0
        self.use_nm: bool = True
        self.paused: bool = False
        self.vis_mode: int = VIS_OFF
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Deferred Shading"

    @override
    def startup(self) -> None:
        self.gbuffer: int = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.gbuffer)

        self.gbuffer_tex: np.ndarray = GL.glGenTextures(3)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.gbuffer_tex[0])
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D,
            1,
            GL.GL_RGBA32UI,
            MAX_DISPLAY_WIDTH,
            MAX_DISPLAY_HEIGHT,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.gbuffer_tex[1])
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D,
            1,
            GL.GL_RGBA32F,
            MAX_DISPLAY_WIDTH,
            MAX_DISPLAY_HEIGHT,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.gbuffer_tex[2])
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D,
            1,
            GL.GL_DEPTH_COMPONENT32F,
            MAX_DISPLAY_WIDTH,
            MAX_DISPLAY_HEIGHT,
        )

        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.gbuffer_tex[0], 0
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, self.gbuffer_tex[1], 0
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.gbuffer_tex[2], 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self.fs_quad_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.fs_quad_vao)

        media: str = common.require_media()
        self.ladybug: sb6m.Object = sb6m.Object()
        self.ladybug.load(os.path.join(media, "objects", "ladybug.sbm"))
        self.tex_nm: int = ktx.load(
            os.path.join(media, "textures", "ladybug_nm.ktx")
        )
        self.tex_diffuse: int = ktx.load(
            os.path.join(media, "textures", "ladybug_co.ktx")
        )

        self.load_shaders()

        self.light_ubo: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.light_ubo)
        # Each light_t: vec3 position + 4 pad + vec3 color + 4 pad =
        # 32 bytes. NUM_LIGHTS * 32 = 2048 bytes.
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER, NUM_LIGHTS * 32, None, GL.GL_DYNAMIC_DRAW
        )

        self.render_transform_ubo: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.render_transform_ubo)
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER,
            (2 + NUM_INSTANCES) * 64,
            None,
            GL.GL_DYNAMIC_DRAW,
        )

    def load_shaders(self) -> None:
        for attr in (
            "render_program",
            "render_program_nm",
            "light_program",
            "vis_program",
        ):
            old = getattr(self, attr)
            if old:
                GL.glDeleteProgram(old)
        self.render_program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )
        self.render_program_nm = common.compile_program_from_files(
            vs=_shader_path("render-nm.vs.glsl"),
            fs=_shader_path("render-nm.fs.glsl"),
        )
        self.light_program = common.compile_program_from_files(
            vs=_shader_path("light.vs.glsl"), fs=_shader_path("light.fs.glsl")
        )
        self.vis_program = common.compile_program_from_files(
            vs=_shader_path("light.vs.glsl"),
            fs=_shader_path("render-vis.fs.glsl"),
        )
        self.loc_vis_mode: int = GL.glGetUniformLocation(
            self.vis_program, "vis_mode"
        )

    @override
    def render(self, current_time: float) -> None:
        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        t: float = float(self._total_time)

        # Pass 1: geometry into G-buffer.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.gbuffer)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glDrawBuffers(2, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1])
        zeros_u: np.ndarray = np.zeros(4, dtype=np.uint32)
        ones_f: list[float] = [1.0, 1.0, 1.0, 1.0]
        GL.glClearBufferuiv(GL.GL_COLOR, 0, zeros_u)
        GL.glClearBufferuiv(GL.GL_COLOR, 1, zeros_u)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, ones_f)

        # Build the transform UBO: proj, view, then 225 model
        # matrices for the ladybug grid.
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        d: float = (math.sin(t * 0.131) + 2.0) * 0.15
        eye = (
            d * 120.0 * math.sin(t * 0.11),
            5.5,
            d * 120.0 * math.cos(t * 0.01),
        )
        view: np.ndarray = vmath.lookat(eye, (0.0, -20.0, 0.0), (0.0, 1.0, 0.0))

        ubo: np.ndarray = np.empty((2 + NUM_INSTANCES, 4, 4), dtype=np.float32)
        ubo[0] = proj.T
        ubo[1] = view.T
        for j in range(15):
            for i in range(15):
                m: np.ndarray = vmath.translate(
                    (i - 7.5) * 7.0, 0.0, (j - 7.5) * 11.0
                )
                ubo[2 + j * 15 + i] = m.T
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.render_transform_ubo)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER, 0, ubo.nbytes, np.ascontiguousarray(ubo)
        )

        GL.glUseProgram(
            self.render_program_nm if self.use_nm else self.render_program
        )
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_diffuse)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_nm)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

        self.ladybug.render(NUM_INSTANCES)

        # Pass 2: fullscreen quad lighting / visualization.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glDrawBuffer(GL.GL_BACK)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.gbuffer_tex[0])
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.gbuffer_tex[1])

        if self.vis_mode == VIS_OFF:
            GL.glUseProgram(self.light_program)
        else:
            GL.glUseProgram(self.vis_program)
            GL.glUniform1i(self.loc_vis_mode, self.vis_mode)

        GL.glDisable(GL.GL_DEPTH_TEST)

        # Lights UBO update.
        lights: np.ndarray = np.zeros((NUM_LIGHTS, 8), dtype=np.float32)
        for i in range(NUM_LIGHTS):
            i_f: float = (float(i) - 7.5) * 0.1 + 0.3
            lights[i, 0] = (
                100.0
                * math.sin(t * 1.1 + 5.0 * i_f)
                * math.cos(t * 2.3 + 9.0 * i_f)
            )
            lights[i, 1] = 15.0
            lights[i, 2] = (
                100.0
                * math.sin(t * 1.5 + 6.0 * i_f)
                * math.cos(t * 1.9 + 11.0 * i_f)
            )
            # pad at 3
            lights[i, 4] = math.cos(i_f * 14.0) * 0.5 + 0.8
            lights[i, 5] = math.sin(i_f * 17.0) * 0.5 + 0.8
            lights[i, 6] = (
                math.sin(i_f * 13.0) * math.cos(i_f * 19.0) * 0.5 + 0.8
            )
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.light_ubo)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER, 0, lights.nbytes, np.ascontiguousarray(lights)
        )

        GL.glBindVertexArray(self.fs_quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    @override
    def draw_controls(self) -> None:
        _, self.use_nm = imgui.checkbox("Normal mapping", self.use_nm)
        imgui.text("Visualization:")
        for i, label in enumerate(VIS_LABELS):
            if imgui.radio_button(label, self.vis_mode == i):
                self.vis_mode = i
        _, self.paused = imgui.checkbox("Pause animation", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.ladybug.free()
        GL.glDeleteTextures(3, self.gbuffer_tex)
        GL.glDeleteFramebuffers(1, [self.gbuffer])
        for prog in (
            self.render_program,
            self.render_program_nm,
            self.light_program,
            self.vis_program,
        ):
            GL.glDeleteProgram(prog)
        GL.glDeleteTextures(1, [self.tex_nm])
        GL.glDeleteTextures(1, [self.tex_diffuse])
        GL.glDeleteBuffers(1, [self.light_ubo])
        GL.glDeleteBuffers(1, [self.render_transform_ubo])
        GL.glDeleteVertexArrays(1, [self.fs_quad_vao])


if __name__ == "__main__":
    application.run(DeferredshadingApp)
