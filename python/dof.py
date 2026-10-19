"""dof — depth-of-field via summed-area-table compute filter.

Port of ``src/dof/dof.cpp``.

Three-pass pipeline:

  Pass 1 (FBO render): scene goes into ``color_tex`` (RGBA32F) plus
         ``depth_tex`` (DEPTH_COMPONENT32F mip-allocated). Five
         instances of the dragon (the C++ has a copy/paste bug at
         render time that always uses ``objects[0].obj`` instead of
         ``objects[i].obj``; we keep the bug, it's how the upstream
         demo actually runs).

  Pass 2 (compute): ``gensat.cs`` builds a summed-area table — first
         vertically (color_tex → temp_tex), then horizontally
         (temp_tex → color_tex) via two compute dispatches.

  Pass 3 (display): ``display.fs`` samples the SAT for each pixel,
         using the pixel's depth to derive a circle-of-confusion size
         and reading a box-blurred neighborhood out of the SAT in O(1)
         taps. The ``focal_distance`` / ``focal_depth`` uniforms are
         imgui sliders.

Reads its shaders from ``media/shaders/dof/`` and the dragon model
from ``media/objects/dragon.sbm``.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "dof", name)


def _model_path(name: str) -> str:
    return os.path.join(common.require_media(), "objects", name)


FBO_SIZE: int = 2048
OBJECT_COUNT: int = 5
COLORS: np.ndarray = np.array(
    [
        [1.0, 0.7, 0.8, 1.0],
        [0.7, 0.8, 1.0, 1.0],
        [0.3, 0.9, 0.4, 1.0],
        [0.6, 0.4, 0.9, 1.0],
        [0.8, 0.2, 0.1, 1.0],
    ],
    dtype=np.float32,
)


class DofApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.view_program: int = 0
        self.filter_program: int = 0
        self.display_program: int = 0
        self.paused: bool = False
        self.focal_distance: float = 40.0
        self.focal_depth: float = 50.0
        self.total_time: float = 0.0
        self._last_time: float | None = None

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Depth Of Field"

    @override
    def startup(self) -> None:
        self.load_shaders()

        # All five objects load the dragon — see module docstring for
        # the upstream copy/paste bug that caused this. The original
        # cpp loaded sphere/cube/cube/cube but only ever called
        # objects[0].obj.render(), so the user-visible behavior is
        # five dragons.
        self.objects: list[sb6m.Object] = []
        for _ in range(OBJECT_COUNT):
            obj: sb6m.Object = sb6m.Object()
            obj.load(_model_path("dragon.sbm"))
            self.objects.append(obj)

        self.depth_fbo: int = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_fbo)

        self.depth_tex: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D, 11, GL.GL_DEPTH_COMPONENT32F, FBO_SIZE, FBO_SIZE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )

        self.color_tex: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D, 1, GL.GL_RGBA32F, FBO_SIZE, FBO_SIZE
        )

        self.temp_tex: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.temp_tex)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D, 1, GL.GL_RGBA32F, FBO_SIZE, FBO_SIZE
        )

        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.depth_tex, 0
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.color_tex, 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        GL.glEnable(GL.GL_DEPTH_TEST)
        self.quad_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.quad_vao)

    def load_shaders(self) -> None:
        if self.view_program:
            GL.glDeleteProgram(self.view_program)
        if self.display_program:
            GL.glDeleteProgram(self.display_program)
        if self.filter_program:
            GL.glDeleteProgram(self.filter_program)
        self.view_program = common.compile_program_from_files(
            vs=_shader_path("render.vs.glsl"), fs=_shader_path("render.fs.glsl")
        )
        self.display_program = common.compile_program_from_files(
            vs=_shader_path("display.vs.glsl"),
            fs=_shader_path("display.fs.glsl"),
        )
        self.filter_program = common.compile_program_from_files(
            cs=_shader_path("gensat.cs.glsl")
        )
        self.u_view_proj: int = GL.glGetUniformLocation(
            self.view_program, "proj_matrix"
        )
        self.u_view_mv: int = GL.glGetUniformLocation(
            self.view_program, "mv_matrix"
        )
        self.u_view_diffuse: int = GL.glGetUniformLocation(
            self.view_program, "diffuse_albedo"
        )
        self.u_focal_distance: int = GL.glGetUniformLocation(
            self.display_program, "focal_distance"
        )
        self.u_focal_depth: int = GL.glGetUniformLocation(
            self.display_program, "focal_depth"
        )

    @override
    def render(self, current_time: float) -> None:
        if self._last_time is None:
            self._last_time = current_time
        if not self.paused:
            self.total_time += current_time - self._last_time
        self._last_time = current_time
        f: float = float(self.total_time) + 30.0

        view_pos: np.ndarray = np.array([0.0, 0.0, 40.0], dtype=np.float32)
        proj: np.ndarray = vmath.perspective(
            50.0,
            max(self.info.window_width, 1) / max(self.info.window_height, 1),
            2.0,
            300.0,
        )
        view: np.ndarray = vmath.lookat(
            view_pos,
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        # Five instances staircased into the distance.
        offsets: list[tuple[float, float, float]] = [
            (5.0, 0.0, 20.0),
            (-5.0, 0.0, 0.0),
            (-15.0, 0.0, -20.0),
            (-25.0, 0.0, -40.0),
            (-35.0, 0.0, -60.0),
        ]
        models: list[np.ndarray] = []
        for ox, oy, oz in offsets:
            m: np.ndarray = (
                vmath.translate(ox, oy, oz)
                @ vmath.rotate(f * 14.5, 0, 1, 0)
                @ vmath.rotate(20.0, 1, 0, 0)
                @ vmath.translate(0, -4, 0)
            )
            models.append(m)

        # Pass 1: scene → FBO.
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.depth_fbo)
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, [0.1, 0.1, 0.1, 0.0])
        GL.glClearBufferfv(GL.GL_DEPTH, 0, [1.0])
        GL.glUseProgram(self.view_program)
        GL.glUniformMatrix4fv(self.u_view_proj, 1, GL.GL_FALSE, proj.T)
        for i, model in enumerate(models):
            mv: np.ndarray = view @ model
            GL.glUniformMatrix4fv(self.u_view_mv, 1, GL.GL_FALSE, mv.T)
            GL.glUniform3fv(self.u_view_diffuse, 1, COLORS[i, :3])
            self.objects[i].render()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # Pass 2: SAT generation (vertical then horizontal).
        GL.glUseProgram(self.filter_program)
        GL.glBindImageTexture(
            0, self.color_tex, 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_RGBA32F
        )
        GL.glBindImageTexture(
            1, self.temp_tex, 0, GL.GL_FALSE, 0, GL.GL_WRITE_ONLY, GL.GL_RGBA32F
        )
        GL.glDispatchCompute(self.info.window_height, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        GL.glBindImageTexture(
            0, self.temp_tex, 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_RGBA32F
        )
        GL.glBindImageTexture(
            1,
            self.color_tex,
            0,
            GL.GL_FALSE,
            0,
            GL.GL_WRITE_ONLY,
            GL.GL_RGBA32F,
        )
        GL.glDispatchCompute(self.info.window_width, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Pass 3: display.
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.display_program)
        GL.glUniform1f(self.u_focal_distance, self.focal_distance)
        GL.glUniform1f(self.u_focal_depth, self.focal_depth)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glBindVertexArray(self.quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        _, self.focal_distance = imgui.slider_float(
            "Focal distance", self.focal_distance, 1.0, 200.0
        )
        _, self.focal_depth = imgui.slider_float(
            "Focal depth", self.focal_depth, 1.0, 300.0
        )
        _, self.paused = imgui.checkbox("Paused", self.paused)
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        for obj in self.objects:
            obj.free()
        GL.glDeleteProgram(self.view_program)
        GL.glDeleteProgram(self.display_program)
        GL.glDeleteProgram(self.filter_program)
        GL.glDeleteTextures(1, [self.depth_tex])
        GL.glDeleteTextures(1, [self.color_tex])
        GL.glDeleteTextures(1, [self.temp_tex])
        GL.glDeleteFramebuffers(1, [self.depth_fbo])
        GL.glDeleteVertexArrays(1, [self.quad_vao])


if __name__ == "__main__":
    application.run(DofApp)
