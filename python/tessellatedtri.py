"""tessellatedtri — singletri's triangle, tessellated via TCS/TES.

Port of ``src/tessellatedtri/tessellatedtri.cpp``. Adds
a tessellation control shader that sets all four tess levels to 5,
and a tessellation evaluation shader that interpolates the triangle's
patch coordinates. Drawn as ``GL_PATCHES`` (not ``GL_TRIANGLES``) and
the polygon mode is ``GL_LINE`` so the tessellated triangles are
visible as a wireframe.
"""

from typing import override

import OpenGL.GL as GL

from sb7 import application, common

VS_SOURCE: str = """
#version 410 core

void main(void)
{
    const vec4 vertices[] = vec4[](vec4( 0.25, -0.25, 0.5, 1.0),
                                   vec4(-0.25, -0.25, 0.5, 1.0),
                                   vec4( 0.25,  0.25, 0.5, 1.0));

    gl_Position = vertices[gl_VertexID];
}
"""

TCS_SOURCE: str = """
#version 410 core

layout (vertices = 3) out;

void main(void)
{
    if (gl_InvocationID == 0)
    {
        gl_TessLevelInner[0] = 5.0;
        gl_TessLevelOuter[0] = 5.0;
        gl_TessLevelOuter[1] = 5.0;
        gl_TessLevelOuter[2] = 5.0;
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
"""

TES_SOURCE: str = """
#version 410 core

layout (triangles, equal_spacing, cw) in;

void main(void)
{
    gl_Position = (gl_TessCoord.x * gl_in[0].gl_Position) +
                  (gl_TessCoord.y * gl_in[1].gl_Position) +
                  (gl_TessCoord.z * gl_in[2].gl_Position);
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


class TessellatedtriApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Tessellated Triangle"

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(
            vs=VS_SOURCE, tcs=TCS_SOURCE, tes=TES_SOURCE, fs=FS_SOURCE
        )
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

    @override
    def render(self, current_time: float) -> None:
        green: list[float] = [0.0, 0.25, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, green)
        GL.glUseProgram(self.program)
        GL.glDrawArrays(GL.GL_PATCHES, 0, 3)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(TessellatedtriApp)
