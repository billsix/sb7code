"""tessmodes — cycle through the 4 tessellation domain modes.

Port of ``src/tessmodes/tessmodes.cpp``. Builds 4 shader
programs that differ only in their TES ``layout`` declaration:

  - QUADS — bilinear interpolation across a 4-vertex patch
  - TRIANGLES — barycentric interpolation across a 3-vertex patch
  - QUADS AS POINTS — same patch as QUADS but with ``point_mode``
  - ISOLINES — concentric spiral generated from gl_TessCoord

Press M to cycle which program is active. Wireframe rendered for
QUADS/TRIANGLES; ISOLINES draws lines naturally; AS_POINTS shows the
domain points.

The C++ side uses ``sb7::text_overlay`` and ``M`` for cycling. We
expose a radio-button group instead.
"""

from typing import override

import glfw  # noqa: F401
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common

VS_SOURCE: str = """
#version 420 core

void main(void)
{
    const vec4 vertices[] = vec4[](vec4( 0.4, -0.4, 0.5, 1.0),
                                   vec4(-0.4, -0.4, 0.5, 1.0),
                                   vec4( 0.4,  0.4, 0.5, 1.0),
                                   vec4(-0.4,  0.4, 0.5, 1.0));
    gl_Position = vertices[gl_VertexID];
}
"""

TCS_TRIANGLES: str = """
#version 420 core
layout (vertices = 3) out;
void main(void)
{
    if (gl_InvocationID == 0)
    {
        gl_TessLevelInner[0] = 5.0;
        gl_TessLevelOuter[0] = 8.0;
        gl_TessLevelOuter[1] = 8.0;
        gl_TessLevelOuter[2] = 8.0;
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
"""

TES_TRIANGLES: str = """
#version 420 core
layout (triangles) in;
void main(void)
{
    gl_Position = (gl_TessCoord.x * gl_in[0].gl_Position) +
                  (gl_TessCoord.y * gl_in[1].gl_Position) +
                  (gl_TessCoord.z * gl_in[2].gl_Position);
}
"""

TES_TRIANGLES_AS_POINTS: str = """
#version 420 core
layout (triangles, point_mode) in;
void main(void)
{
    gl_Position = (gl_TessCoord.x * gl_in[0].gl_Position) +
                  (gl_TessCoord.y * gl_in[1].gl_Position) +
                  (gl_TessCoord.z * gl_in[2].gl_Position);
}
"""

TCS_QUADS: str = """
#version 420 core
layout (vertices = 4) out;
void main(void)
{
    if (gl_InvocationID == 0)
    {
        gl_TessLevelInner[0] = 9.0;
        gl_TessLevelInner[1] = 7.0;
        gl_TessLevelOuter[0] = 3.0;
        gl_TessLevelOuter[1] = 5.0;
        gl_TessLevelOuter[2] = 3.0;
        gl_TessLevelOuter[3] = 5.0;
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
"""

TES_QUADS: str = """
#version 420 core
layout (quads) in;
void main(void)
{
    vec4 p1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
    vec4 p2 = mix(gl_in[2].gl_Position, gl_in[3].gl_Position, gl_TessCoord.x);
    gl_Position = mix(p1, p2, gl_TessCoord.y);
}
"""

TCS_ISOLINES: str = """
#version 420 core
layout (vertices = 4) out;
void main(void)
{
    if (gl_InvocationID == 0)
    {
        gl_TessLevelOuter[0] = 5.0;
        gl_TessLevelOuter[1] = 5.0;
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
"""

TES_ISOLINES: str = """
#version 420 core
layout (isolines) in;
void main(void)
{
    float r = (gl_TessCoord.y + gl_TessCoord.x / gl_TessLevelOuter[0]);
    float t = gl_TessCoord.x * 2.0 * 3.14159;
    gl_Position = vec4(sin(t) * r, cos(t) * r, 0.5, 1.0);
}
"""

FS_SOURCE: str = """
#version 420 core
out vec4 color;
void main(void)
{
    color = vec4(1.0);
}
"""


# (TCS, TES, label) — order matches the C++ source's program[] indexing.
PROGRAMS_SPEC = [
    (TCS_QUADS, TES_QUADS, "QUADS"),
    (TCS_TRIANGLES, TES_TRIANGLES, "TRIANGLES"),
    (TCS_TRIANGLES, TES_TRIANGLES_AS_POINTS, "QUADS AS POINTS"),
    (TCS_ISOLINES, TES_ISOLINES, "ISOLINES"),
]


class TessmodesApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.program_index: int = 0
        self.programs: list[int] = []

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Tessellation Modes"

    @override
    def startup(self) -> None:
        self.programs = [
            common.compile_program(vs=VS_SOURCE, tcs=tcs, tes=tes, fs=FS_SOURCE)
            for tcs, tes, _ in PROGRAMS_SPEC
        ]

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 4)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        GL.glUseProgram(self.programs[self.program_index])
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glDrawArrays(GL.GL_PATCHES, 0, 4)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

    @override
    def draw_controls(self) -> None:
        for i, (_, _, label) in enumerate(PROGRAMS_SPEC):
            if imgui.radio_button(label, self.program_index == i):
                self.program_index = i

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        for p in self.programs:
            GL.glDeleteProgram(p)


if __name__ == "__main__":
    application.run(TessmodesApp)
