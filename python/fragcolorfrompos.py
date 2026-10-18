"""fragcolorfrompos — fragment color from gl_FragCoord, or from
interpolated per-vertex color.

Port of ``src/fragcolorfrompos/fragcolorfrompos.cpp``. The
C++ source uses ``#define INTERPOLATE_COLOR`` to switch between two
shader pairs at compile time (default: interpolated). We expose the
same toggle as the module-level ``INTERPOLATE_COLOR`` flag so a student
flipping it sees the difference.
"""

from typing import override

import OpenGL.GL as GL

from sb7 import application, common

INTERPOLATE_COLOR: bool = True


VS_INTERPOLATE: str = """
#version 420 core

out vec4 vs_color;

void main(void)
{
    const vec4 vertices[] = vec4[](vec4( 0.25, -0.25, 0.5, 1.0),
                                   vec4(-0.25, -0.25, 0.5, 1.0),
                                   vec4( 0.25,  0.25, 0.5, 1.0));
    const vec4 colors[] = vec4[](vec4(1.0, 0.0, 0.0, 1.0),
                                 vec4(0.0, 1.0, 0.0, 1.0),
                                 vec4(0.0, 0.0, 1.0, 1.0));
    gl_Position = vertices[gl_VertexID];
    vs_color = colors[gl_VertexID];
}
"""

FS_INTERPOLATE: str = """
#version 420 core

in vec4 vs_color;
out vec4 color;

void main(void)
{
    color = vs_color;
}
"""

VS_FROMPOS: str = """
#version 420 core

void main(void)
{
    const vec4 vertices[] = vec4[](vec4( 0.25, -0.25, 0.5, 1.0),
                                   vec4(-0.25, -0.25, 0.5, 1.0),
                                   vec4( 0.25,  0.25, 0.5, 1.0));
    gl_Position = vertices[gl_VertexID];
}
"""

FS_FROMPOS: str = """
#version 420 core

out vec4 color;

void main(void)
{
    color = vec4(sin(gl_FragCoord.x * 0.25) * 0.5 + 0.5,
                 cos(gl_FragCoord.y * 0.25) * 0.5 + 0.5,
                 sin(gl_FragCoord.x * 0.15) * cos(gl_FragCoord.y * 0.1),
                 1.0);
}
"""


class FragcolorfromposApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Frag Color from Position"

    @override
    def startup(self) -> None:
        if INTERPOLATE_COLOR:
            self.program: int = common.compile_program(
                vs=VS_INTERPOLATE, fs=FS_INTERPOLATE
            )
        else:
            self.program = common.compile_program(vs=VS_FROMPOS, fs=FS_FROMPOS)
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
    application.run(FragcolorfromposApp)
