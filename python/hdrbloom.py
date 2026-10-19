"""hdrbloom — separable Gaussian-blur bloom on a 32-torus scene.

Port of ``src/hdrbloom/hdrbloom.cpp``. Three-pass HDR
bloom pipeline:

  Pass 1 (scene):  render 32 colored torii into a float FBO with TWO
                   color attachments — full HDR scene at COLOR0,
                   "bright pass" (above ``bloom_thresh_min``) at
                   COLOR1.

  Pass 2 (filter): two 1D Gaussian blurs of the bright-pass image.
                   First filter_fbo[0] (blur in one axis) reads
                   tex_brightpass and writes a transposed result;
                   second filter_fbo[1] reads that transpose and
                   writes the final blurred bloom texture. Net
                   effect: separable 2D Gaussian blur.

  Pass 3 (resolve): fullscreen quad combines the HDR scene with the
                    blurred bloom (additive), applies exposure tone-
                    mapping, outputs to default framebuffer.

The C++ source uses 2048×2048 textures for everything — bigger than
most windows, so the rendered region lives in the bottom-left corner
and the resolve shader's ``gl_FragCoord / textureSize`` math samples
correctly. We keep the same setup.

Reads its shaders from ``media/shaders/hdrbloom/``.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "hdrbloom", name)


MAX_SCENE_WIDTH: int = 2048
MAX_SCENE_HEIGHT: int = 2048
SPHERE_COUNT: int = 32


class HdrbloomApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.exposure: float = 1.0
        self.program_render: int = 0
        self.program_filter: int = 0
        self.program_resolve: int = 0
        self.paused: bool = False
        self.bloom_factor: float = 1.0
        self.show_bloom: bool = True
        self.show_scene: bool = True
        self.bloom_thresh_min: float = 0.8
        self.bloom_thresh_max: float = 1.2
        self.show_prefilter: bool = False
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - HDR Bloom"

    @override
    def startup(self) -> None:
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self.load_shaders()

        # Scene FBO with HDR color (RGBA16F), bright-pass color
        # (RGBA16F), and depth (32F).
        self.render_fbo: int = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.render_fbo)

        self.tex_scene: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_scene)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D,
            1,
            GL.GL_RGBA16F,
            MAX_SCENE_WIDTH,
            MAX_SCENE_HEIGHT,
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.tex_scene, 0
        )

        self.tex_brightpass: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_brightpass)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D,
            1,
            GL.GL_RGBA16F,
            MAX_SCENE_WIDTH,
            MAX_SCENE_HEIGHT,
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, self.tex_brightpass, 0
        )

        self.tex_depth: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_depth)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D,
            1,
            GL.GL_DEPTH_COMPONENT32F,
            MAX_SCENE_WIDTH,
            MAX_SCENE_HEIGHT,
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.tex_depth, 0
        )
        GL.glDrawBuffers(2, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1])

        # Two ping-pong FBOs for the separable blur. Per the C++
        # source, filter_fbo[0]'s texture is *transposed* (height x
        # width) so the second blur reads the rotated layout.
        self.filter_fbo: np.ndarray = GL.glGenFramebuffers(2)
        self.tex_filter: np.ndarray = GL.glGenTextures(2)
        for i in range(2):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.filter_fbo[i])
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_filter[i])
            w: int = MAX_SCENE_WIDTH if i else MAX_SCENE_HEIGHT
            h: int = MAX_SCENE_HEIGHT if i else MAX_SCENE_WIDTH
            GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGBA16F, w, h)
            GL.glFramebufferTexture(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                self.tex_filter[i],
                0,
            )
            GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # Adaptive-tone-map LUT (unused by the resolve shader in the
        # C++ source's default mode, but build it anyway for parity).
        self.tex_lut: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_1D, self.tex_lut)
        GL.glTexStorage1D(GL.GL_TEXTURE_1D, 1, GL.GL_R32F, 20)
        lut: np.ndarray = np.array(
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
        GL.glTexSubImage1D(
            GL.GL_TEXTURE_1D, 0, 0, 20, GL.GL_RED, GL.GL_FLOAT, lut
        )

        # Torus model rendered SPHERE_COUNT times via instancing.
        self.torus: sb6m.Object = sb6m.Object()
        self.torus.load(
            os.path.join(common.require_media(), "objects", "torus.sbm")
        )

        # Transform UBO: proj + view + 32 model matrices.
        self.ubo_transform: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.ubo_transform)
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER,
            (2 + SPHERE_COUNT) * 64,
            None,
            GL.GL_DYNAMIC_DRAW,
        )

        # Material UBO: 32 entries, each std140 layout
        # vec3 + (4 pad bytes via vec3-then-pad) + vec3 + float + vec3
        # + (pad) = 16 + 16 + 16 = 48 bytes. C++ uses 16-byte aligned
        # vec3 + pad → matches.
        self.ubo_material: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.ubo_material)
        materials: np.ndarray = np.zeros((SPHERE_COUNT, 12), dtype=np.float32)
        ambient: float = 0.002
        for i in range(SPHERE_COUNT):
            fi: float = math.pi * i / 8.0
            materials[i, 0] = math.sin(fi) * 0.5 + 0.5
            materials[i, 1] = math.sin(fi + 1.345) * 0.5 + 0.5
            materials[i, 2] = math.sin(fi + 2.567) * 0.5 + 0.5
            # pad at index 3
            materials[i, 4] = 2.8
            materials[i, 5] = 2.8
            materials[i, 6] = 2.9
            materials[i, 7] = 30.0  # specular_power
            materials[i, 8] = ambient * 0.025
            materials[i, 9] = ambient * 0.025
            materials[i, 10] = ambient * 0.025
            ambient *= 1.5
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER, materials.nbytes, materials, GL.GL_STATIC_DRAW
        )

    def load_shaders(self) -> None:
        for attr in ("program_render", "program_filter", "program_resolve"):
            old: int = getattr(self, attr)
            if old:
                GL.glDeleteProgram(old)
        self.program_render = common.compile_program_from_files(
            vs=_shader_path("hdrbloom-scene.vs.glsl"),
            fs=_shader_path("hdrbloom-scene.fs.glsl"),
        )
        self.uniform_scene_thresh_min: int = GL.glGetUniformLocation(
            self.program_render, "bloom_thresh_min"
        )
        self.uniform_scene_thresh_max: int = GL.glGetUniformLocation(
            self.program_render, "bloom_thresh_max"
        )

        self.program_filter = common.compile_program_from_files(
            vs=_shader_path("hdrbloom-filter.vs.glsl"),
            fs=_shader_path("hdrbloom-filter.fs.glsl"),
        )

        self.program_resolve = common.compile_program_from_files(
            vs=_shader_path("hdrbloom-resolve.vs.glsl"),
            fs=_shader_path("hdrbloom-resolve.fs.glsl"),
        )
        self.uniform_resolve_exposure: int = GL.glGetUniformLocation(
            self.program_resolve, "exposure"
        )
        self.uniform_resolve_bloom: int = GL.glGetUniformLocation(
            self.program_resolve, "bloom_factor"
        )
        self.uniform_resolve_scene: int = GL.glGetUniformLocation(
            self.program_resolve, "scene_factor"
        )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 1.0]
        one: list[float] = [1.0]

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        t: float = float(self._total_time)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)

        # Pass 1: scene into render_fbo.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.render_fbo)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_COLOR, 1, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

        GL.glUseProgram(self.program_render)

        # Build the transform UBO: proj, view, then 32 model matrices.
        # std140 mat4 is column-major.
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 1.0, 1000.0)
        view: np.ndarray = vmath.translate(0.0, 0.0, -20.0)
        ubo_data: np.ndarray = np.empty(
            (2 + SPHERE_COUNT, 4, 4), dtype=np.float32
        )
        ubo_data[0] = proj.T
        ubo_data[1] = view.T
        for i in range(SPHERE_COUNT):
            fi: float = math.pi * i / 16.0
            r: float = 0.6 if (i & 2) else 1.5
            m: np.ndarray = vmath.translate(
                math.cos(t + fi) * 5.0 * r,
                math.sin(t + fi * 4.0) * 4.0,
                math.sin(t + fi) * 5.0 * r,
            ) @ vmath.rotate_xyz(
                math.sin(t + fi * 2.13) * 75.0,
                math.cos(t + fi * 1.37) * 92.0,
                0.0,
            )
            ubo_data[2 + i] = m.T

        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.ubo_transform)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER,
            0,
            ubo_data.nbytes,
            np.ascontiguousarray(ubo_data),
        )
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 1, self.ubo_material)

        GL.glUniform1f(self.uniform_scene_thresh_min, self.bloom_thresh_min)
        GL.glUniform1f(self.uniform_scene_thresh_max, self.bloom_thresh_max)

        self.torus.render(SPHERE_COUNT)

        GL.glDisable(GL.GL_DEPTH_TEST)

        # Pass 2: separable Gaussian blur via two ping-pong FBOs.
        # The first one's viewport is (height, width) — transposed —
        # because tex_filter[0] is allocated transposed. Second pass
        # reads that and writes back to native orientation.
        GL.glUseProgram(self.program_filter)
        GL.glBindVertexArray(self.vao)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.filter_fbo[0])
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_brightpass)
        GL.glViewport(0, 0, self.info.window_height, self.info.window_width)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.filter_fbo[1])
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_filter[0])
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # Pass 3: resolve. Reset viewport to window dims for the
        # default framebuffer.
        GL.glUseProgram(self.program_resolve)
        GL.glUniform1f(self.uniform_resolve_exposure, self.exposure)
        if self.show_prefilter:
            GL.glUniform1f(self.uniform_resolve_bloom, 0.0)
            GL.glUniform1f(self.uniform_resolve_scene, 1.0)
        else:
            GL.glUniform1f(
                self.uniform_resolve_bloom,
                self.bloom_factor if self.show_bloom else 0.0,
            )
            GL.glUniform1f(
                self.uniform_resolve_scene, 1.0 if self.show_scene else 0.0
            )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_filter[1])
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(
            GL.GL_TEXTURE_2D,
            self.tex_brightpass if self.show_prefilter else self.tex_scene,
        )
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        _, self.show_scene = imgui.checkbox("Show scene", self.show_scene)
        _, self.show_bloom = imgui.checkbox("Show bloom", self.show_bloom)
        _, self.show_prefilter = imgui.checkbox(
            "View bright-pass (pre-filter)", self.show_prefilter
        )
        _, self.exposure = imgui.slider_float(
            "Exposure",
            self.exposure,
            0.01,
            16.0,
            flags=imgui.SliderFlags_.logarithmic.value,
        )
        _, self.bloom_factor = imgui.slider_float(
            "Bloom factor", self.bloom_factor, 0.0, 4.0
        )
        _, self.bloom_thresh_min = imgui.slider_float(
            "Bloom threshold min", self.bloom_thresh_min, 0.0, 4.0
        )
        _, self.bloom_thresh_max = imgui.slider_float(
            "Bloom threshold max", self.bloom_thresh_max, 0.0, 4.0
        )
        _, self.paused = imgui.checkbox("Pause animation", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.torus.free()
        for prog in (
            self.program_render,
            self.program_filter,
            self.program_resolve,
        ):
            GL.glDeleteProgram(prog)
        GL.glDeleteFramebuffers(1, [self.render_fbo])
        GL.glDeleteFramebuffers(2, self.filter_fbo)
        GL.glDeleteTextures(1, [self.tex_scene])
        GL.glDeleteTextures(1, [self.tex_brightpass])
        GL.glDeleteTextures(1, [self.tex_depth])
        GL.glDeleteTextures(2, self.tex_filter)
        GL.glDeleteTextures(1, [self.tex_lut])
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.ubo_transform])
        GL.glDeleteBuffers(1, [self.ubo_material])


if __name__ == "__main__":
    application.run(HdrbloomApp)
