"""ssao — screen-space ambient occlusion on a dragon + cube floor.

Port of ``src/ssao/ssao.cpp``. Two passes:

  Pass 1: render scene (dragon + flat cube as floor) into an FBO with
          two color attachments — RGB16F for shaded color and RGBA32F
          for normal+depth — plus a DEPTH_COMPONENT32F attachment.

  Pass 2: a fullscreen quad samples both color attachments. The SSAO
          shader picks N random sample directions per fragment from a
          UBO of 256 unit vectors, projects each into screen space,
          and accumulates how many samples are occluded — that's the
          ambient occlusion factor. Multiplied with the shaded color
          → the final image.

The C++ source uses a 2048×2048 FBO; we keep that. As long as the
window is ≤ 2048×2048, the resolve shader's
``gl_FragCoord.xy / textureSize`` math correctly samples the
rendered bottom-left region of the FBO regardless of window size.

Reads its shaders from ``media/shaders/ssao/``.
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
    return os.path.join(common.require_media(), "shaders", "ssao", name)


FBO_SIZE: int = 2048


# Same LCG as the rest of the v7 demos.
class _Random:
    def __init__(self, seed: int = 0x13371337) -> None:
        self.seed: int = seed

    def next_float(self) -> float:
        self.seed = (self.seed * 16807) & 0xFFFFFFFF
        tmp = self.seed ^ (self.seed >> 4) ^ ((self.seed << 15) & 0xFFFFFFFF)
        bits = (tmp >> 9) | 0x3F800000
        f = np.frombuffer(
            np.uint32(bits & 0xFFFFFFFF).tobytes(), dtype=np.float32
        )[0]
        return float(f - 1.0)


def _build_sample_points() -> np.ndarray:
    """256 unit-sphere directions + 256 random vec4s. Layout matches
    the SAMPLE_POINTS UBO in ssao.fs.glsl: pos[256] then
    random_vectors[256], each as vec4."""
    rng: _Random = _Random()
    pts: np.ndarray = np.zeros((256, 4), dtype=np.float32)
    for i in range(256):
        while True:
            v: np.ndarray = np.array(
                [
                    rng.next_float() * 2.0 - 1.0,
                    rng.next_float() * 2.0 - 1.0,
                    rng.next_float(),
                    0.0,
                ],
                dtype=np.float32,
            )
            length: float = math.sqrt(
                float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2
            )
            if length <= 1.0 and length > 0.0:
                v[:3] /= length
                pts[i] = v
                break
    rand: np.ndarray = np.zeros((256, 4), dtype=np.float32)
    for i in range(256):
        rand[i, 0] = rng.next_float()
        rand[i, 1] = rng.next_float()
        rand[i, 2] = rng.next_float()
        rand[i, 3] = rng.next_float()
    return np.concatenate([pts, rand]).astype(np.float32)


class SsaoApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_program: int = 0
        self.ssao_program: int = 0
        self.paused: bool = False
        self.ssao_level: float = 1.0
        self.ssao_radius: float = 0.05
        self.show_shading: bool = True
        self.show_ao: bool = True
        self.weight_by_angle: bool = True
        self.randomize_points: bool = True
        self.point_count: int = 10
        self._last_time: float = 0.0
        self._total_time: float = 0.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Screen-Space Ambient Occlusion"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.render_fbo: int = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.render_fbo)

        # Three attachments: shaded color (RGB16F), normal+depth
        # (RGBA32F), depth (DEPTH_COMPONENT32F).
        self.fbo_textures = GL.glGenTextures(3)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_textures[0])
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGB16F, FBO_SIZE, FBO_SIZE)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_textures[1])
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D, 1, GL.GL_RGBA32F, FBO_SIZE, FBO_SIZE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_textures[2])
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D, 1, GL.GL_DEPTH_COMPONENT32F, FBO_SIZE, FBO_SIZE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )

        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.fbo_textures[0], 0
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, self.fbo_textures[1], 0
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.fbo_textures[2], 0
        )
        GL.glDrawBuffers(2, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1])
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self.quad_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.quad_vao)

        media: str = common.require_media()
        self.dragon: sb6m.Object = sb6m.Object()
        self.dragon.load(os.path.join(media, "objects", "dragon.sbm"))
        self.cube: sb6m.Object = sb6m.Object()
        self.cube.load(os.path.join(media, "objects", "cube.sbm"))

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

        # Build and upload the sample-points UBO.
        sample_data: np.ndarray = _build_sample_points()
        self.points_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.points_buffer)
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER,
            sample_data.nbytes,
            sample_data,
            GL.GL_STATIC_DRAW,
        )

    def load_shaders(self) -> None:
        if self.render_program:
            GL.glDeleteProgram(self.render_program)
        if self.ssao_program:
            GL.glDeleteProgram(self.ssao_program)
        self.render_program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )
        self.uniform_render_mv: int = GL.glGetUniformLocation(
            self.render_program, "mv_matrix"
        )
        self.uniform_render_proj: int = GL.glGetUniformLocation(
            self.render_program, "proj_matrix"
        )
        self.uniform_render_shading: int = GL.glGetUniformLocation(
            self.render_program, "shading_level"
        )

        self.ssao_program = common.compile_program_from_files(
            vs=_shader_path("ssao.vs.glsl"), fs=_shader_path("ssao.fs.glsl")
        )
        self.uniform_ssao_radius: int = GL.glGetUniformLocation(
            self.ssao_program, "ssao_radius"
        )
        self.uniform_ssao_level: int = GL.glGetUniformLocation(
            self.ssao_program, "ssao_level"
        )
        self.uniform_ssao_randomize: int = GL.glGetUniformLocation(
            self.ssao_program, "randomize_points"
        )
        self.uniform_ssao_count: int = GL.glGetUniformLocation(
            self.ssao_program, "point_count"
        )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        one: list[float] = [1.0]

        if not self.paused:
            self._total_time += current_time - self._last_time
        self._last_time = current_time
        f: float = float(self._total_time)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)

        # Pass 1: render scene into the FBO.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.render_fbo)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_COLOR, 1, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.points_buffer)
        GL.glUseProgram(self.render_program)

        view: np.ndarray = vmath.lookat(
            (0.0, 3.0, 15.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        )
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)
        vmath.upload(self.uniform_render_proj, proj)

        # Dragon
        mv: np.ndarray = vmath.translate(0.0, -5.0, 0.0) @ vmath.rotate(
            f * 5.0, 0.0, 1.0, 0.0
        )
        vmath.upload(self.uniform_render_mv, view @ mv)
        GL.glUniform1f(
            self.uniform_render_shading,
            0.7
            if self.show_shading and self.show_ao
            else 1.0
            if self.show_shading
            else 0.0,
        )
        self.dragon.render()

        # Floor (a flat cube scaled out)
        mv = (
            vmath.translate(0.0, -4.5, 0.0)
            @ vmath.rotate(f * 5.0, 0.0, 1.0, 0.0)
            @ vmath.scale(4000.0, 0.1, 4000.0)
        )
        vmath.upload(self.uniform_render_mv, view @ mv)
        self.cube.render()

        # Pass 2: SSAO resolve fullscreen quad.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glUseProgram(self.ssao_program)

        # Scale the SSAO sample radius by window width relative to
        # 1000 px (matches the C++ source's heuristic).
        GL.glUniform1f(
            self.uniform_ssao_radius,
            self.ssao_radius * float(self.info.window_width) / 1000.0,
        )
        GL.glUniform1f(
            self.uniform_ssao_level,
            0.3
            if self.show_ao and self.show_shading
            else 1.0
            if self.show_ao
            else 0.0,
        )
        GL.glUniform1i(
            self.uniform_ssao_randomize, 1 if self.randomize_points else 0
        )
        GL.glUniform1ui(self.uniform_ssao_count, self.point_count)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_textures[0])
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_textures[1])

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(self.quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        _, self.show_shading = imgui.checkbox("Show shading", self.show_shading)
        _, self.show_ao = imgui.checkbox("Show ambient occlusion", self.show_ao)
        _, self.randomize_points = imgui.checkbox(
            "Randomize sample points per pixel", self.randomize_points
        )
        _, self.ssao_radius = imgui.slider_float(
            "SSAO radius", self.ssao_radius, 0.0, 0.5
        )
        pc_changed, pc = imgui.slider_int(
            "Sample points", self.point_count, 1, 64
        )
        if pc_changed:
            self.point_count = max(1, pc)
        _, self.paused = imgui.checkbox("Pause rotation", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.dragon.free()
        self.cube.free()
        GL.glDeleteProgram(self.render_program)
        GL.glDeleteProgram(self.ssao_program)
        GL.glDeleteFramebuffers(1, [self.render_fbo])
        GL.glDeleteTextures(3, self.fbo_textures)
        GL.glDeleteBuffers(1, [self.points_buffer])
        GL.glDeleteVertexArrays(1, [self.quad_vao])


if __name__ == "__main__":
    application.run(SsaoApp)
