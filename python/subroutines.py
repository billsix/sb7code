"""subroutines — switch fragment-shader behavior at draw time without
relinking the program.

Port of ``src/subroutines/subroutines.cpp``. The
fragment shader declares two subroutine functions (``myFunction1`` and
``myFunction2``) plus a subroutine uniform (``mySubroutineUniform``).
At draw time we pick which subroutine is active via
``glUniformSubroutinesuiv`` — alternating each second, so the output
toggles between the two implementations.

GLSL subroutines are like a typed function pointer in the shader:
declared once, bound to one of N named subroutines per draw. Faster
than recompiling/relinking when you have a small enumeration of
visual variants.

Reads its shaders from ``media/shaders/subroutines/``.

Imgui controls: Reload-shaders button. Subroutine choice cycles
automatically once per second based on render time, matching the C++
demo's behavior.
"""

import os
from typing import override

import glfw  # noqa: F401
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "subroutines", name)


class SubroutinesApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.render_program: int = 0
        self.active: int = 0  # set by render each frame, read by draw_controls

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Shader Subroutines"

    @override
    def startup(self) -> None:
        self.load_shaders()
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

    def load_shaders(self) -> None:
        if self.render_program:
            GL.glDeleteProgram(self.render_program)
        self.render_program = common.compile_program_from_files(
            vs=_shader_path("subroutines.vs.glsl"),
            fs=_shader_path("subroutines.fs.glsl"),
        )
        self.subroutines: list[int] = [
            GL.glGetSubroutineIndex(
                self.render_program, GL.GL_FRAGMENT_SHADER, "myFunction1"
            ),
            GL.glGetSubroutineIndex(
                self.render_program, GL.GL_FRAGMENT_SHADER, "myFunction2"
            ),
        ]
        self.subroutine1_loc: int = GL.glGetSubroutineUniformLocation(
            self.render_program, GL.GL_FRAGMENT_SHADER, "mySubroutineUniform"
        )

    @override
    def render(self, current_time: float) -> None:
        i: int = int(current_time)
        active: int = i & 1
        self.active = active  # read by draw_controls

        GL.glUseProgram(self.render_program)
        GL.glUniformSubroutinesuiv(
            GL.GL_FRAGMENT_SHADER, 1, [self.subroutines[active]]
        )
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        imgui.text(f"Active subroutine: myFunction{self.active + 1}")
        imgui.text("(toggles automatically each second)")
        if imgui.button("Reload shaders"):
            self.load_shaders()


if __name__ == "__main__":
    application.run(SubroutinesApp)
