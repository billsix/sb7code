"""singlepoint — one big GL_POINT at the origin.

Port of ``src/singlepoint/singlepoint.cpp``. Demonstrates
``glPointSize`` and the GL_POINTS primitive on top of the same shader-
manufactures-position-from-gl_VertexID idea as singletri.
"""

from typing import override

import OpenGL.GL as GL

from sb7 import application, common

VS_SOURCE: str = """
#version 420 core

void main(void)
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
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


class SinglepointApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Single Point"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

    @override
    def render(self, current_time: float) -> None:
        red: list[float] = [1.0, 0.0, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, red)
        GL.glUseProgram(self.program)
        GL.glPointSize(40.0)
        GL.glDrawArrays(GL.GL_POINTS, 0, 1)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(SinglepointApp)
