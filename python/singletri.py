"""singletri — first triangle, no VBO.

Port of ``src/singletri/singletri.cpp``.

The C++ original has the vertex positions hard-coded in the vertex
shader as a ``vec4[]`` indexed by ``gl_VertexID`` — so there's no VBO,
no glVertexAttribPointer, no client-side vertex data. Just bind a VAO,
``glDrawArrays(GL_TRIANGLES, 0, 3)``, and the vertex shader manufactures
the positions from the vertex index. This is the smallest possible
"draw something" example in v7.
"""

from typing import override

import OpenGL.GL as GL

from sb7 import application, common

VS_SOURCE: str = """
#version 420 core

void main(void)
{
    const vec4 vertices[] = vec4[](vec4( 0.25, -0.25, 0.5, 1.0),
                                   vec4(-0.25, -0.25, 0.5, 1.0),
                                   vec4( 0.25,  0.25, 0.5, 1.0));

    gl_Position = vertices[gl_VertexID];
}
"""

FS_SOURCE: str = """
#version 420 core

out vec4 color;

void main(void)
{
    color = vec4(0.0, 0.8, 1.0, 1.0);
}
"""


class SingletriApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Single Triangle"

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
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(SingletriApp)
