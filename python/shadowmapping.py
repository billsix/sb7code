"""shadowmapping — depth-buffer shadows on a 4-object scene.

Port of ``src/shadowmapping/shadowmapping.cpp``. Renders
4 SBM models (dragon, sphere, cube, torus) under a single light, with
two passes:

  Pass 1: render scene from the light's POV into a 4096×4096 depth
          texture (the shadow map). FBO has both a depth attachment
          (the shadow map itself) and a color attachment (a
          visualization of the depth values for the RENDER_DEPTH
          mode).

  Pass 2: render the scene from the camera, sampling the depth
          texture as a ``sampler2DShadow`` (compare-mode enabled) to
          decide whether each fragment is lit or shadowed.

The depth texture stays at the C++ source's 4096×4096 — that's
intentional oversampling for shadow quality, independent of the
screen size. We *do* reset the viewport back to the window
dimensions before drawing the depth-debug quad, so that quad fills
the whole window instead of getting clipped to the FBO's 4096×4096
viewport region.

Three view modes via imgui:
  - Full shading        — final lit + shadowed scene
  - Light-only shading  — same scene with shadow term only
  - Depth visualization — the raw shadow map, stretched to fill the
                          window (the underlying texture is square
                          so it'll look horizontally compressed on a
                          16:9 monitor; that's expected)

Reads its shaders from ``media/shaders/shadowmapping/``.
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
    return os.path.join(
        common.require_media(), "shaders", "shadowmapping", name
    )


# Replacement for the C++ source's ``shadowmapping-light-view.{vs,fs}.glsl``.
# The original FS does ``texelFetch(tex, ivec2(gl_FragCoord * 3.0) + ivec2(850, 1050))``
# — a hardcoded 3× zoom into a fixed pixel region of the depth texture
# — so on any screen smaller than ~5760×3240 only the bottom-left
# corner has valid samples and the rest is black. We replace it with
# a normalized-UV sample so the entire shadow map fills the entire
# window regardless of dimensions.
DEBUG_VS_SOURCE: str = """
#version 420 core

out vec2 vUV;

void main(void)
{
    const vec4 verts[] = vec4[](vec4(-1.0, -1.0, 0.5, 1.0),
                                vec4( 1.0, -1.0, 0.5, 1.0),
                                vec4(-1.0,  1.0, 0.5, 1.0),
                                vec4( 1.0,  1.0, 0.5, 1.0));
    const vec2 uvs[] = vec2[](vec2(0.0, 0.0),
                              vec2(1.0, 0.0),
                              vec2(0.0, 1.0),
                              vec2(1.0, 1.0));
    gl_Position = verts[gl_VertexID];
    vUV = uvs[gl_VertexID];
}
"""

DEBUG_FS_SOURCE: str = """
#version 420 core

layout (binding = 0) uniform sampler2D tex_depth;
in vec2 vUV;
layout (location = 0) out vec4 color;

void main(void)
{
    float d = texture(tex_depth, vUV).r;
    // Same contrast remap the C++ FS used (depth values cluster
    // tightly near 1.0 since the light is far from objects).
    d = (d - 0.95) * 15.0;
    color = vec4(d, d, d, 1.0);
}
"""


DEPTH_TEXTURE_SIZE: int = 4096
OBJECT_COUNT: int = 4

RENDER_FULL: int = 0
RENDER_LIGHT: int = 1
RENDER_DEPTH: int = 2

OBJECT_NAMES: list[str] = ["dragon.sbm", "sphere.sbm", "cube.sbm", "torus.sbm"]


# Half-pixel scale-bias to map clip-space [-1,1] into [0,1] for shadow
# sampling. Same matrix as the C++ source's ``scale_bias_matrix``.
SCALE_BIAS_MATRIX: np.ndarray = np.array(
    [
        [0.5, 0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


class ShadowmappingApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.light_program: int = 0
        self.view_program: int = 0
        self.show_light_depth_program: int = 0
        self.mode: int = RENDER_FULL
        self.paused: bool = False
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Shadow Mapping"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.objects = []
        media: str = common.require_media()
        for name in OBJECT_NAMES:
            obj: sb6m.Object = sb6m.Object()
            obj.load(os.path.join(media, "objects", name))
            self.objects.append([obj, vmath.identity()])

        self.depth_fbo: int = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_fbo)

        self.depth_tex: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D,
            11,
            GL.GL_DEPTH_COMPONENT32F,
            DEPTH_TEXTURE_SIZE,
            DEPTH_TEXTURE_SIZE,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_COMPARE_MODE,
            GL.GL_COMPARE_REF_TO_TEXTURE,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_FUNC, GL.GL_LEQUAL
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.depth_tex, 0
        )

        # Color attachment used to visualize the depth values when in
        # RENDER_DEPTH mode.
        self.depth_debug_tex: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_debug_tex)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D,
            1,
            GL.GL_R32F,
            DEPTH_TEXTURE_SIZE,
            DEPTH_TEXTURE_SIZE,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.depth_debug_tex, 0
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glEnable(GL.GL_DEPTH_TEST)

        self.quad_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.quad_vao)

    def load_shaders(self) -> None:
        if self.light_program:
            GL.glDeleteProgram(self.light_program)
        if self.view_program:
            GL.glDeleteProgram(self.view_program)
        if self.show_light_depth_program:
            GL.glDeleteProgram(self.show_light_depth_program)

        self.light_program = common.compile_program_from_files(
            vs=_shader_path("shadowmapping-light.vs.glsl"),
            fs=_shader_path("shadowmapping-light.fs.glsl"),
        )
        self.uniform_light_mvp: int = GL.glGetUniformLocation(
            self.light_program, "mvp"
        )

        self.view_program = common.compile_program_from_files(
            vs=_shader_path("shadowmapping-camera.vs.glsl"),
            fs=_shader_path("shadowmapping-camera.fs.glsl"),
        )
        self.uniform_view_mv: int = GL.glGetUniformLocation(
            self.view_program, "mv_matrix"
        )
        self.uniform_view_proj: int = GL.glGetUniformLocation(
            self.view_program, "proj_matrix"
        )
        self.uniform_view_shadow: int = GL.glGetUniformLocation(
            self.view_program, "shadow_matrix"
        )
        self.uniform_view_full: int = GL.glGetUniformLocation(
            self.view_program, "full_shading"
        )

        # Debug-view shaders are inline here, not loaded from
        # media/shaders/shadowmapping/, so that we can use a
        # screen-size-independent UV sample (see DEBUG_FS_SOURCE
        # above for why the C++ original is broken on small screens).
        self.show_light_depth_program = common.compile_program(
            vs=DEBUG_VS_SOURCE, fs=DEBUG_FS_SOURCE
        )

    @override
    def render(self, current_time: float) -> None:
        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        f: float = float(self._total_time) + 30.0

        # Light + camera matrices.
        self.light_proj: np.ndarray = vmath.frustum(
            -1.0, 1.0, -1.0, 1.0, 1.0, 200.0
        )
        self.light_view: np.ndarray = vmath.lookat(
            (20.0, 20.0, 20.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        )
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        self.camera_proj: np.ndarray = vmath.perspective(
            50.0, aspect, 1.0, 200.0
        )
        self.camera_view: np.ndarray = vmath.lookat(
            (0.0, 0.0, 40.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        )

        # Animate the four model matrices.
        self.objects[0][1] = (
            vmath.rotate(f * 14.5, 0.0, 1.0, 0.0)
            @ vmath.rotate(20.0, 1.0, 0.0, 0.0)
            @ vmath.translate(0.0, -4.0, 0.0)
        )
        self.objects[1][1] = (
            vmath.rotate(f * 3.7, 0.0, 1.0, 0.0)
            @ vmath.translate(
                math.sin(f * 0.37) * 12.0, math.cos(f * 0.37) * 12.0, 0.0
            )
            @ vmath.scale(2.0)
        )
        self.objects[2][1] = (
            vmath.rotate(f * 6.45, 0.0, 1.0, 0.0)
            @ vmath.translate(
                math.sin(f * 0.25) * 10.0, math.cos(f * 0.25) * 10.0, 0.0
            )
            @ vmath.rotate(f * 99.0, 0.0, 0.0, 1.0)
            @ vmath.scale(2.0)
        )
        self.objects[3][1] = (
            vmath.rotate(f * 5.25, 0.0, 1.0, 0.0)
            @ vmath.translate(
                math.sin(f * 0.51) * 14.0, math.cos(f * 0.51) * 14.0, 0.0
            )
            @ vmath.rotate(f * 120.3, 0.707106, 0.0, 0.707106)
            @ vmath.scale(2.0)
        )

        GL.glEnable(GL.GL_DEPTH_TEST)
        self._render_scene(from_light=True)

        if self.mode == RENDER_DEPTH:
            # Reset viewport — _render_scene left it at the FBO's
            # DEPTH_TEXTURE_SIZE — and bind texture unit 0 so the
            # debug shader's ``layout (binding = 0)`` resolves.
            GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glBindVertexArray(self.quad_vao)
            GL.glUseProgram(self.show_light_depth_program)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_debug_tex)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        else:
            self._render_scene(from_light=False)

    @override
    def draw_controls(self) -> None:
        imgui.text("View mode:")
        if imgui.radio_button("Full lit + shadowed", self.mode == RENDER_FULL):
            self.mode = RENDER_FULL
        if imgui.radio_button("Light-only", self.mode == RENDER_LIGHT):
            self.mode = RENDER_LIGHT
        if imgui.radio_button("Shadow-map depth", self.mode == RENDER_DEPTH):
            self.mode = RENDER_DEPTH
        _, self.paused = imgui.checkbox("Pause animation", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    def _render_scene(self, from_light: bool) -> None:
        ones: list[float] = [1.0]
        zero: list[float] = [0.0]
        gray: list[float] = [0.1, 0.1, 0.1, 0.0]

        light_vp: np.ndarray = self.light_proj @ self.light_view
        shadow_sbpv: np.ndarray = SCALE_BIAS_MATRIX @ light_vp

        if from_light:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_fbo)
            GL.glViewport(0, 0, DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE)
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(4.0, 4.0)
            GL.glUseProgram(self.light_program)
            GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
            GL.glClearBufferfv(GL.GL_COLOR, 0, zero)
        else:
            GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
            GL.glClearBufferfv(GL.GL_COLOR, 0, gray)
            GL.glUseProgram(self.view_program)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)
            vmath.upload(self.uniform_view_proj, self.camera_proj)
            GL.glDrawBuffer(GL.GL_BACK)

        GL.glClearBufferfv(GL.GL_DEPTH, 0, ones)

        for obj, model in self.objects:
            if from_light:
                vmath.upload(self.uniform_light_mvp, light_vp @ model)
            else:
                vmath.upload(self.uniform_view_shadow, shadow_sbpv @ model)
                vmath.upload(self.uniform_view_mv, self.camera_view @ model)
                GL.glUniform1i(
                    self.uniform_view_full, 1 if self.mode == RENDER_FULL else 0
                )
            # self.objects holds [Object, model-matrix] pairs in plain
            # lists (the matrix slot is reassigned every frame), so
            # each element types as `Object | ndarray`; slot 0 is
            # always the Object, but a list's element type can't say
            # so positionally.
            obj.render()  # ty: ignore[unresolved-attribute]

        if from_light:
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    @override
    def shutdown(self) -> None:
        for obj, _ in self.objects:
            # Same [Object, matrix] pair layout as in _draw_scene —
            # slot 0 is always the Object.
            obj.free()  # ty: ignore[unresolved-attribute]
        GL.glDeleteProgram(self.light_program)
        GL.glDeleteProgram(self.view_program)
        GL.glDeleteProgram(self.show_light_depth_program)
        GL.glDeleteFramebuffers(1, [self.depth_fbo])
        GL.glDeleteTextures(2, [self.depth_tex, self.depth_debug_tex])
        GL.glDeleteVertexArrays(1, [self.quad_vao])


if __name__ == "__main__":
    application.run(ShadowmappingApp)
