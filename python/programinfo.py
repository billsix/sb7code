"""programinfo — query active program outputs and dump them to overlay.

Port of ``src/programinfo/programinfo.cpp``. Builds a
fragment shader with several outputs of different types (vec4, ivec2,
float) plus interface blocks, then walks ``GL_PROGRAM_OUTPUT`` via
``glGetProgramInterfaceiv`` / ``glGetProgramResource*`` and prints
each output's name, type, location, and array size. The C++ side prints
through ``sb7::text_overlay``; we use an imgui panel since that's the
mvp convention.

Doesn't draw any geometry — it's a pure introspection demo.
"""

from typing import override

import numpy as np
import OpenGL.GL as GL
from OpenGL.constant import Constant

from sb7 import application
from imgui_bundle import imgui  # noqa: E402

VS_SOURCE: str = """
#version 420 core

in vec4 position;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

out VS_OUT
{
    vec2 tc;
} vs_out;

void main(void)
{
    gl_Position = proj_matrix * mv_matrix * position;
    vs_out.tc = position.xy;
}
"""

FS_SOURCE: str = """
#version 420 core

out vec4 color;
layout (location = 2) out ivec2 data;
out float extra;

in BLOCK0
{
    vec2 tc;
    vec4 color;
    flat int foo;
} fs_in0;

in BLOCK1
{
    vec3 normal[4];
    flat ivec3 layers;
    double bar;
} fs_in1;

void main(void)
{
    float val = abs(fs_in0.tc.x + fs_in0.tc.y) * 20.0f;
    color = vec4(fract(val) >= 0.5 ? 1.0 : 0.25) + fs_in1.normal[3].xyzy;
    data = ivec2(1, 2);
    extra = 9.0;
}
"""


# Mirror of type_to_name_table from the C++ source.
_TYPE_NAMES: dict[Constant, str] = {
    GL.GL_FLOAT: "float",
    GL.GL_FLOAT_VEC2: "vec2",
    GL.GL_FLOAT_VEC3: "vec3",
    GL.GL_FLOAT_VEC4: "vec4",
    GL.GL_DOUBLE: "double",
    GL.GL_DOUBLE_VEC2: "dvec2",
    GL.GL_DOUBLE_VEC3: "dvec3",
    GL.GL_DOUBLE_VEC4: "dvec4",
    GL.GL_INT: "int",
    GL.GL_INT_VEC2: "ivec2",
    GL.GL_INT_VEC3: "ivec3",
    GL.GL_INT_VEC4: "ivec4",
    GL.GL_UNSIGNED_INT: "uint",
    GL.GL_UNSIGNED_INT_VEC2: "uvec2",
    GL.GL_UNSIGNED_INT_VEC3: "uvec3",
    GL.GL_UNSIGNED_INT_VEC4: "uvec4",
    GL.GL_BOOL: "bool",
    GL.GL_BOOL_VEC2: "bvec2",
    GL.GL_BOOL_VEC3: "bvec3",
    GL.GL_BOOL_VEC4: "bvec4",
}


class PrograminfoApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Program Information"
        self.lines: list[str] = []

    @override
    def startup(self) -> None:
        # Build a SEPARABLE program with just a fragment shader — we
        # don't need to render anything, only introspect outputs.
        self.program: int = GL.glCreateProgram()
        GL.glProgramParameteri(
            self.program, GL.GL_PROGRAM_SEPARABLE, GL.GL_TRUE
        )

        fs: int = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, FS_SOURCE)
        GL.glCompileShader(fs)
        GL.glAttachShader(self.program, fs)
        GL.glLinkProgram(self.program)

        log: bytes | str = GL.glGetProgramInfoLog(self.program)
        log_str: str = (
            log.decode(errors="replace") if isinstance(log, bytes) else str(log)
        )
        self.lines.append("Program linked")
        if log_str.strip():
            self.lines.append(log_str.strip())

        n_outputs: int = GL.glGetProgramInterfaceiv(
            self.program, GL.GL_PROGRAM_OUTPUT, GL.GL_ACTIVE_RESOURCES
        )

        props: list[Constant] = [GL.GL_TYPE, GL.GL_LOCATION, GL.GL_ARRAY_SIZE]
        for i in range(n_outputs):
            name_bytes: bytes | str = GL.glGetProgramResourceName(
                self.program, GL.GL_PROGRAM_OUTPUT, i, 64
            )
            name: str = (
                name_bytes.decode(errors="replace")
                if isinstance(name_bytes, bytes)
                else str(name_bytes)
            )
            params: np.ndarray = GL.glGetProgramResourceiv(
                self.program, GL.GL_PROGRAM_OUTPUT, i, 3, props, 3
            )
            type_name: str = _TYPE_NAMES.get(
                params[0], f"<type 0x{params[0]:04X}>"
            )
            loc: int = params[1]
            arr_size: int = params[2]
            if arr_size != 0:
                self.lines.append(
                    f"Index {i}: {type_name} {name}[{arr_size}] @ location {loc}."
                )
            else:
                self.lines.append(
                    f"Index {i}: {type_name} {name} @ location {loc}."
                )

        # Also dump to stdout so students can read without the window.
        for line in self.lines:
            print(line)  # noqa: T201

    @override
    def render(self, current_time: float) -> None:
        black: list[float] = [0.0, 0.0, 0.0, 1.0]
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, black)

    @override
    def draw_controls(self) -> None:
        for line in self.lines:
            imgui.text(line)

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.program)


if __name__ == "__main__":
    application.run(PrograminfoApp)
