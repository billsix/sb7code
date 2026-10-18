"""shapedpoints — discard inside the fragment shader to make point shapes.

Port of ``src/shapedpoints/shapedpoints.cpp``. Four 200-
pixel points, each with a different shape carved out via ``discard``
in the fragment shader using ``gl_PointCoord``: a circle, a five-armed
star, an annulus, and an X. Shows that point sprites are just quads
that you can paint freely.
"""

from typing import override

import OpenGL.GL as GL

from sb7 import application, common

VS_SOURCE: str = """
#version 410 core

flat out int shape;

void main(void)
{
    const vec4[4] position = vec4[4](vec4(-0.4, -0.4, 0.5, 1.0),
                                     vec4( 0.4, -0.4, 0.5, 1.0),
                                     vec4(-0.4,  0.4, 0.5, 1.0),
                                     vec4( 0.4,  0.4, 0.5, 1.0));
    gl_Position = position[gl_VertexID];
    shape = gl_VertexID;
}
"""

FS_SOURCE: str = """
#version 410 core

layout (location = 0) out vec4 color;

flat in int shape;

void main(void)
{
    color = vec4(1.0);
    vec2 p = gl_PointCoord * 2.0 - vec2(1.0);
    if (shape == 0)
    {
        if (dot(p, p) > 1.0) discard;
    }
    else if (shape == 1)
    {
        if (dot(p, p) > sin(atan(p.y, p.x) * 5.0)) discard;
    }
    else if (shape == 2)
    {
        if (abs(0.8 - dot(p, p)) > 0.2) discard;
    }
    else if (shape == 3)
    {
        if (abs(p.x) < abs(p.y)) discard;
    }
}
"""


class ShapedpointsApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Shaped Points"

    @override
    def startup(self) -> None:
        self.render_prog: int = common.compile_program(
            vs=VS_SOURCE, fs=FS_SOURCE
        )
        self.render_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.render_vao)

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 0.0]
        one: list[float] = [1.0]
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)
        GL.glClearBufferfv(GL.GL_DEPTH, 0, one)

        GL.glUseProgram(self.render_prog)
        GL.glPointSize(200.0)
        GL.glBindVertexArray(self.render_vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, 4)


if __name__ == "__main__":
    application.run(ShapedpointsApp)
