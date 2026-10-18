"""tessellatedgstri — tessellated triangle, then a geometry shader
turns each tessellated triangle into 3 points.

Port of ``src/tessellatedgstri/tessellatedgstri.cpp``. A
single triangle is tessellated to level 5, then routed through a GS
that emits each vertex as a ``GL_POINTS`` primitive instead of
triangles. Result: a 5×5 lattice of points.

Demonstrates pipeline stage chaining: VS → TCS → TES → GS → FS, where
the GS changes the output primitive type from triangles to points.
"""

from typing import override

import glfw  # noqa: F401
import OpenGL.GL as GL
from imgui_bundle import imgui  # noqa: F401

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

GS_SOURCE: str = """
#version 410 core

layout (triangles) in;
layout (points, max_vertices = 3) out;

void main(void)
{
    for (int i = 0; i < gl_in.length(); i++)
    {
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
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


class TessellatedgstriApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = (
            "OpenGL SuperBible - Tessellation and Geometry Shaders"
        )

    @override
    def startup(self) -> None:
        self.program: int = common.compile_program(
            vs=VS_SOURCE,
            tcs=TCS_SOURCE,
            tes=TES_SOURCE,
            gs=GS_SOURCE,
            fs=FS_SOURCE,
        )
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

    @override
    def render(self, current_time: float) -> None:
        green: list[float] = [0.0, 0.25, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, green)

        GL.glUseProgram(self.program)
        GL.glPointSize(5.0)
        GL.glDrawArrays(GL.GL_PATCHES, 0, 3)

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(TessellatedgstriApp)
