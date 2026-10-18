"""movingtri — singletri's triangle, animated via vertex attribute.

Port of ``src/movingtri/movingtri.cpp``. Same vertex
positions baked into the vertex shader, but the shader adds a ``vec4
offset`` attribute and ``glVertexAttrib4fv`` is called every frame from
the CPU side to push a sine/cosine offset. Demonstrates the difference
between vertex *attributes* set per draw (no buffer needed) and per-
vertex attributes pulled from a VBO.
"""

import math
from typing import override

import OpenGL.GL as GL

from sb7 import application, common

VS_SOURCE: str = """
#version 410 core

layout (location = 0) in vec4 offset;

void main(void)
{
    const vec4 vertices[] = vec4[](vec4( 0.25, -0.25, 0.5, 1.0),
                                   vec4(-0.25, -0.25, 0.5, 1.0),
                                   vec4( 0.25,  0.25, 0.5, 1.0));

    gl_Position = vertices[gl_VertexID] + offset;
}
"""

FS_SOURCE: str = """
#version 410 core

out vec4 color;

void main(void)
{
    color = vec4(0.0, 0.8, 1.0, 1.0);
}
"""


class MovingtriApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Moving Triangle"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

    @override
    def render(self, current_time: float) -> None:
        green: list[float] = [0.0, 0.25, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, green)

        GL.glUseProgram(self.program)

        attrib: list[float] = [
            math.sin(current_time) * 0.5,
            math.cos(current_time) * 0.6,
            0.0,
            0.0,
        ]
        GL.glVertexAttrib4fv(0, attrib)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(MovingtriApp)
