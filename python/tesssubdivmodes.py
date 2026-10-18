"""tesssubdivmodes — compare the three tessellation subdivision modes.

Port of ``src/tesssubdivmodes/tesssubdivmodes.cpp``. Three
shader programs differ only in the TES's ``layout`` declaration:
``equal_spacing``, ``fractional_even_spacing``, ``fractional_odd_spacing``.
A 4-vertex patch is tessellated with level 5.3 in each mode and drawn
in wireframe. The imgui panel has radio buttons for the three modes
so you can see the visual differences in how fractional subdivisions
are placed at the patch boundary.
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
    const vec4 vertices[] = vec4[](vec4( 0.8, -0.8, 0.5, 1.0),
                                   vec4(-0.8, -0.8, 0.5, 1.0),
                                   vec4( 0.8,  0.8, 0.5, 1.0),
                                   vec4(-0.8,  0.8, 0.5, 1.0));

    gl_Position = vertices[gl_VertexID];
}
"""

TCS_SOURCE: str = """
#version 420 core

layout (vertices = 3) out;

uniform float tess_level = 2.7;

void main(void)
{
    if (gl_InvocationID == 0)
    {
        gl_TessLevelInner[0] = tess_level;
        gl_TessLevelOuter[0] = tess_level;
        gl_TessLevelOuter[1] = tess_level;
        gl_TessLevelOuter[2] = tess_level;
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
"""


def _tes_source(spacing: str) -> str:
    return f"""
#version 420 core

layout (triangles{spacing}) in;

void main(void)
{{
    gl_Position = (gl_TessCoord.x * gl_in[0].gl_Position) +
                  (gl_TessCoord.y * gl_in[1].gl_Position) +
                  (gl_TessCoord.z * gl_in[2].gl_Position);
}}
"""


TES_EQUAL = _tes_source("")
TES_FRACT_EVEN = _tes_source(", fractional_even_spacing")
TES_FRACT_ODD = _tes_source(", fractional_odd_spacing")

FS_SOURCE: str = """
#version 420 core

out vec4 color;

void main(void)
{
    color = vec4(1.0);
}
"""


class TesssubdivmodesApp(application.Application):
    MODE_NAMES = ("equal", "fractional_even", "fractional_odd")

    def __init__(self) -> None:
        super().__init__()
        self.program_index: int = 0
        self.programs: list[int] = [0, 0, 0]

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Subdivision Modes"

    @override
    def startup(self) -> None:
        self.tess_level_locs: list[int] = []
        for i, tes_src in enumerate([TES_EQUAL, TES_FRACT_EVEN, TES_FRACT_ODD]):
            self.programs[i] = common.compile_program(
                vs=VS_SOURCE, tcs=TCS_SOURCE, tes=tes_src, fs=FS_SOURCE
            )
            self.tess_level_locs.append(
                GL.glGetUniformLocation(self.programs[i], "tess_level")
            )

        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 4)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

        prog: int = self.programs[self.program_index]
        GL.glUseProgram(prog)
        GL.glUniform1f(self.tess_level_locs[self.program_index], 5.3)
        GL.glDrawArrays(GL.GL_PATCHES, 0, 4)

    @override
    def draw_controls(self) -> None:
        for i, name in enumerate(self.MODE_NAMES):
            if imgui.radio_button(name, self.program_index == i):
                self.program_index = i

    @override
    def shutdown(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        for p in self.programs:
            GL.glDeleteProgram(p)


if __name__ == "__main__":
    application.run(TesssubdivmodesApp)
