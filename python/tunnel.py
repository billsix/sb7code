"""tunnel — endless 4-walled tunnel, three KTX textures.

Port of ``src/tunnel/tunnel.cpp``. A long planar quad is
drawn 4 times — once per wall, ceiling, floor, and the opposite wall —
each with a different KTX texture (brick/ceiling/floor) and offset
``tex_coord.x`` along time, giving the illusion of moving through a
brick corridor.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

from sb7 import application, common, ktx, vmath

VS_SOURCE: str = """
#version 420 core

out VS_OUT
{
    vec2 tc;
} vs_out;

uniform mat4 mvp;
uniform float offset;

void main(void)
{
    const vec2[4] position = vec2[4](vec2(-0.5, -0.5),
                                     vec2( 0.5, -0.5),
                                     vec2(-0.5,  0.5),
                                     vec2( 0.5,  0.5));
    vs_out.tc = (position[gl_VertexID].xy + vec2(offset, 0.5)) * vec2(30.0, 1.0);
    gl_Position = mvp * vec4(position[gl_VertexID], 0.0, 1.0);
}
"""

FS_SOURCE: str = """
#version 420 core

layout (location = 0) out vec4 color;

in VS_OUT
{
    vec2 tc;
} fs_in;

layout (binding = 0) uniform sampler2D tex;

void main(void)
{
    color = texture(tex, fs_in.tc);
}
"""


class TunnelApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Tunnel"

    @override
    def startup(self) -> None:
        self.render_prog: int = common.compile_program(
            vs=VS_SOURCE, fs=FS_SOURCE
        )
        self.mvp_loc: int = GL.glGetUniformLocation(self.render_prog, "mvp")
        self.offset_loc: int = GL.glGetUniformLocation(
            self.render_prog, "offset"
        )

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        media: str = common.require_media()
        self.tex_wall: int = ktx.load(
            os.path.join(media, "textures", "brick.ktx")
        )
        self.tex_ceiling: int = ktx.load(
            os.path.join(media, "textures", "ceiling.ktx")
        )
        self.tex_floor: int = ktx.load(
            os.path.join(media, "textures", "floor.ktx")
        )
        for tex in (self.tex_wall, self.tex_ceiling, self.tex_floor):
            if not tex:
                raise RuntimeError("failed to load tunnel KTX textures")
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D,
                GL.GL_TEXTURE_MIN_FILTER,
                GL.GL_LINEAR_MIPMAP_LINEAR,
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
            )

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        t: float = float(current_time)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        GL.glUseProgram(self.render_prog)

        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(60.0, aspect, 0.1, 100.0)

        GL.glUniform1f(self.offset_loc, t * 0.003)

        # Walls in order: right wall, floor, left wall, ceiling.
        textures: list[int] = [
            self.tex_wall,
            self.tex_floor,
            self.tex_wall,
            self.tex_ceiling,
        ]
        for i in range(4):
            mv: np.ndarray = (
                vmath.rotate(90.0 * i, 0.0, 0.0, 1.0)
                @ vmath.translate(-0.5, 0.0, -10.0)
                @ vmath.rotate(90.0, 0.0, 1.0, 0.0)
                @ vmath.scale(30.0, 1.0, 1.0)
            )
            mvp: np.ndarray = proj @ mv
            vmath.upload(self.mvp_loc, mvp)
            GL.glBindTexture(GL.GL_TEXTURE_2D, textures[i])
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.render_prog)
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteTextures(
            3, [self.tex_wall, self.tex_ceiling, self.tex_floor]
        )


if __name__ == "__main__":
    application.run(TunnelApp)
