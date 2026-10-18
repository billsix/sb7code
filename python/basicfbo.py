"""basicfbo — render a cube to an FBO, then sample it onto a second cube.

Port of ``src/basicfbo/basicfbo.cpp``. Two passes:

  Pass 1: bind a 512x512 FBO with a color texture and a depth texture
          attached. Render a procedurally-shaded cube into it.

  Pass 2: bind the default framebuffer. Render the cube again, this
          time sampling the FBO's color texture as a 2D texture.

The result is a cube whose faces show a smaller cube spinning inside
them — the classic "render to texture" demo. First demo to use FBO,
glTexStorage2D, and texture sampling.
"""

import math
from typing import override

import numpy as np
import OpenGL.GL as GL

from sb7 import application, colors, common, vmath

VS_SOURCE: str = """
#version 410 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec2 texcoord;

out VS_OUT
{
    vec4 color;
    vec2 texcoord;
} vs_out;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

void main(void)
{
    gl_Position = proj_matrix * mv_matrix * position;
    vs_out.color = position * 2.0 + vec4(0.5, 0.5, 0.5, 0.0);
    vs_out.texcoord = texcoord;
}
"""

# Used in pass 1: writes to the FBO. Procedural color, no texture lookup.
FS_SOURCE_PASS1: str = """
#version 410 core

in VS_OUT
{
    vec4 color;
    vec2 texcoord;
} fs_in;

out vec4 color;

void main(void)
{
    color = sin(fs_in.color * vec4(40.0, 20.0, 30.0, 1.0)) * 0.5 + vec4(0.5);
}
"""

# Used in pass 2: samples the FBO color texture, mixed with vertex color.
FS_SOURCE_PASS2: str = """
#version 420 core

uniform sampler2D tex;

out vec4 color;

in VS_OUT
{
    vec4 color;
    vec2 texcoord;
} fs_in;

void main(void)
{
    color = mix(fs_in.color, texture(tex, fs_in.texcoord), 0.7);
}
"""


VERTEX_DATA: np.ndarray = np.array(
    [
        # Position             Tex Coord
        -0.25,
        -0.25,
        0.25,
        0.0,
        1.0,
        -0.25,
        -0.25,
        -0.25,
        0.0,
        0.0,
        0.25,
        -0.25,
        -0.25,
        1.0,
        0.0,
        0.25,
        -0.25,
        -0.25,
        1.0,
        0.0,
        0.25,
        -0.25,
        0.25,
        1.0,
        1.0,
        -0.25,
        -0.25,
        0.25,
        0.0,
        1.0,
        0.25,
        -0.25,
        -0.25,
        0.0,
        0.0,
        0.25,
        0.25,
        -0.25,
        1.0,
        0.0,
        0.25,
        -0.25,
        0.25,
        0.0,
        1.0,
        0.25,
        0.25,
        -0.25,
        1.0,
        0.0,
        0.25,
        0.25,
        0.25,
        1.0,
        1.0,
        0.25,
        -0.25,
        0.25,
        0.0,
        1.0,
        0.25,
        0.25,
        -0.25,
        1.0,
        0.0,
        -0.25,
        0.25,
        -0.25,
        0.0,
        0.0,
        0.25,
        0.25,
        0.25,
        1.0,
        1.0,
        -0.25,
        0.25,
        -0.25,
        0.0,
        0.0,
        -0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        0.25,
        0.25,
        0.25,
        1.0,
        1.0,
        -0.25,
        0.25,
        -0.25,
        1.0,
        0.0,
        -0.25,
        -0.25,
        -0.25,
        0.0,
        0.0,
        -0.25,
        0.25,
        0.25,
        1.0,
        1.0,
        -0.25,
        -0.25,
        -0.25,
        0.0,
        0.0,
        -0.25,
        -0.25,
        0.25,
        0.0,
        1.0,
        -0.25,
        0.25,
        0.25,
        1.0,
        1.0,
        -0.25,
        0.25,
        -0.25,
        0.0,
        1.0,
        0.25,
        0.25,
        -0.25,
        1.0,
        1.0,
        0.25,
        -0.25,
        -0.25,
        1.0,
        0.0,
        0.25,
        -0.25,
        -0.25,
        1.0,
        0.0,
        -0.25,
        -0.25,
        -0.25,
        0.0,
        0.0,
        -0.25,
        0.25,
        -0.25,
        0.0,
        1.0,
        -0.25,
        -0.25,
        0.25,
        0.0,
        0.0,
        0.25,
        -0.25,
        0.25,
        1.0,
        0.0,
        0.25,
        0.25,
        0.25,
        1.0,
        1.0,
        0.25,
        0.25,
        0.25,
        1.0,
        1.0,
        -0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        -0.25,
        -0.25,
        0.25,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

STRIDE: int = 5 * 4  # 5 floats per vertex


class BasicfboApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Basic Framebuffer Object"

    @override
    def startup(self) -> None:
        self.program1: int = common.compile_program(
            vs=VS_SOURCE, fs=FS_SOURCE_PASS1
        )
        self.program2: int = common.compile_program(
            vs=VS_SOURCE, fs=FS_SOURCE_PASS2
        )

        self.mv_location: int = GL.glGetUniformLocation(
            self.program1, "mv_matrix"
        )
        self.proj_location: int = GL.glGetUniformLocation(
            self.program1, "proj_matrix"
        )
        self.mv_location2: int = GL.glGetUniformLocation(
            self.program2, "mv_matrix"
        )
        self.proj_location2: int = GL.glGetUniformLocation(
            self.program2, "proj_matrix"
        )

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.position_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.position_buffer)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            VERTEX_DATA.nbytes,
            VERTEX_DATA,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, STRIDE, None)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, STRIDE, GL.GLvoidp(3 * 4)
        )
        GL.glEnableVertexAttribArray(1)

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

        # FBO with color and depth attachments — both 512x512, 9 mip levels.
        self.fbo: int = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        self.color_texture: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_texture)
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 9, GL.GL_RGBA8, 512, 512)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )

        self.depth_texture: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_texture)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D, 9, GL.GL_DEPTH_COMPONENT32F, 512, 512
        )

        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.color_texture, 0
        )
        GL.glFramebufferTexture(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.depth_texture, 0
        )
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])

    @override
    def render(self, current_time: float) -> None:
        blue: list[float] = [0.0, 0.0, 0.3, 1.0]
        one: list[float] = [1.0]

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)

        f = current_time * 0.3
        mv: np.ndarray = (
            vmath.translate(0.0, 0.0, -4.0)
            @ vmath.translate(
                math.sin(2.1 * f) * 0.5,
                math.cos(1.7 * f) * 0.5,
                math.sin(1.3 * f) * math.cos(1.5 * f) * 2.0,
            )
            @ vmath.rotate(current_time * 45.0, 0.0, 1.0, 0.0)
            @ vmath.rotate(current_time * 81.0, 1.0, 0.0, 0.0)
        )

        # Pass 1: into the FBO.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, 512, 512)
        GL.glClearBufferfv(GL.GL_COLOR, 0, colors.Color.Green)
        GL.glClearBufferfi(GL.GL_DEPTH_STENCIL, 0, 1.0, 0)

        GL.glUseProgram(self.program1)
        vmath.upload(self.proj_location, proj)
        vmath.upload(self.mv_location, mv)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)

        # Pass 2: to the default framebuffer, sampling the FBO color
        # texture.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, blue)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_texture)
        GL.glUseProgram(self.program2)
        vmath.upload(self.proj_location2, proj)
        vmath.upload(self.mv_location2, mv)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program1)
        GL.glDeleteProgram(self.program2)
        GL.glDeleteBuffers(1, [self.position_buffer])
        GL.glDeleteFramebuffers(1, [self.fbo])
        GL.glDeleteTextures(1, [self.color_texture])
        GL.glDeleteTextures(1, [self.depth_texture])


if __name__ == "__main__":
    application.run(BasicfboApp)
