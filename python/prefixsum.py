"""prefixsum — 1D parallel prefix sum on the GPU via compute shader.

Port of ``src/prefixsum/prefixsum.cpp``.

What it does: given an array of 2048 random floats, compute the
**cumulative sum** at each position (output[i] = input[0] + input[1]
+ ... + input[i]) using one GPU compute shader workgroup of 1024
threads with shared-memory tree-reduction. The whole demo is about
the algorithm — the upstream version has no graphics output at all,
just a row of text in the sb7 overlay showing the first 16 results.

To make the result visible, this port plots the input array and the
GPU's output array as two imgui line graphs, plus shows the first 16
values numerically and the max-abs-error vs a CPU ``np.cumsum``
reference.

The compute shader is **inlined** rather than read from disk because
the upstream version sizes its SSBO arrays as ``[gl_WorkGroupSize.x]``
(= 1024) but indexes up to 2047. NVIDIA tolerates that; Mesa zeroes
the out-of-bounds writes. The inline version uses unsized SSBO arrays
which is well-defined on both.
"""

from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common

NUM_ELEMENTS: int = 2048


def _make_input(seed: int = 0x13371337) -> np.ndarray:
    rng: np.random.Generator = np.random.default_rng(seed)
    return rng.random(NUM_ELEMENTS, dtype=np.float32)


# The upstream shader declares the SSBO arrays as sized
# `float input_data[gl_WorkGroupSize.x]` (= 1024) but each thread
# accesses `id*2` and `id*2 + 1`, going up to index 2047 — past the
# declared bound. NVIDIA's driver lets that slide; Mesa zeroes the
# out-of-bounds writes, so the GPU output is all zero on Bill's
# Fedora box. We replace it with the canonical unsized-array form.
PREFIXSUM_CS_SOURCE: str = """\
#version 430 core

layout (local_size_x = 1024) in;

layout (binding = 0) coherent readonly buffer block1
{
    float input_data[];
};

layout (binding = 1) coherent writeonly buffer block2
{
    float output_data[];
};

shared float shared_data[gl_WorkGroupSize.x * 2];

void main(void)
{
    uint id = gl_LocalInvocationID.x;
    uint rd_id;
    uint wr_id;
    uint mask;

    const uint steps = uint(log2(gl_WorkGroupSize.x)) + 1;
    uint step = 0;

    shared_data[id * 2]     = input_data[id * 2];
    shared_data[id * 2 + 1] = input_data[id * 2 + 1];

    barrier();

    for (step = 0; step < steps; step++)
    {
        mask  = (1 << step) - 1;
        rd_id = ((id >> step) << (step + 1)) + mask;
        wr_id = rd_id + 1 + (id & mask);

        shared_data[wr_id] += shared_data[rd_id];

        barrier();
    }

    output_data[id * 2]     = shared_data[id * 2];
    output_data[id * 2 + 1] = shared_data[id * 2 + 1];
}
"""


class PrefixsumApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.prog: int = 0
        self.input_data: np.ndarray = _make_input()
        self.cpu_output: np.ndarray = np.cumsum(self.input_data).astype(
            np.float32
        )
        # Pre-allocated readback buffer reused every frame.
        self.gpu_output: np.ndarray = np.zeros(NUM_ELEMENTS, dtype=np.float32)
        self._diagnostics: str = ""

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - 1D Prefix Sum"

    @override
    def startup(self) -> None:
        self.vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.data_buffer = GL.glGenBuffers(2)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.data_buffer[0])
        GL.glBufferData(
            GL.GL_SHADER_STORAGE_BUFFER,
            NUM_ELEMENTS * 4,
            None,
            GL.GL_DYNAMIC_DRAW,
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.data_buffer[1])
        GL.glBufferData(
            GL.GL_SHADER_STORAGE_BUFFER,
            NUM_ELEMENTS * 4,
            None,
            GL.GL_DYNAMIC_COPY,
        )

        self.load_shaders()

    def load_shaders(self) -> None:
        if self.prog:
            GL.glDeleteProgram(self.prog)
        self.prog = common.compile_program(cs=PREFIXSUM_CS_SOURCE)
        GL.glShaderStorageBlockBinding(self.prog, 0, 0)
        GL.glShaderStorageBlockBinding(self.prog, 1, 1)

    @override
    def render(self, current_time: float) -> None:
        # Upload the input into SSBO 0.
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.data_buffer[0])
        GL.glBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER,
            0,
            self.input_data.nbytes,
            np.ascontiguousarray(self.input_data),
        )

        # Bind both SSBOs to their shader binding points.
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self.data_buffer[0])
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, self.data_buffer[1])

        # Dispatch the compute shader and barrier so the readback
        # below sees the writes.
        GL.glUseProgram(self.prog)
        GL.glDispatchCompute(1, 1, 1)
        # PyOpenGL constants are int subclasses at runtime, but the
        # stubs' `Constant` declares no `__or__`, so OR-ing two masks
        # trips ty.
        GL.glMemoryBarrier(
            GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT  # ty: ignore[unsupported-operator]
        )

        # Read the result with glGetBufferSubData. PyOpenGL's
        # glGetBufferSubData accepts either a preallocated array or
        # `None`; with `None` it returns a numpy array of the right
        # dtype/shape. We use the return value so we don't depend on
        # in-place fill semantics.
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.data_buffer[1])
        readback: np.ndarray = GL.glGetBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER, 0, NUM_ELEMENTS * 4
        )
        gpu: np.ndarray = np.frombuffer(readback, dtype=np.float32).copy()
        if gpu.size != NUM_ELEMENTS:
            # Some PyOpenGL versions return a bytes-like; coerce.
            gpu = np.frombuffer(bytes(readback), dtype=np.float32).copy()
        self.gpu_output = gpu

        err: int = GL.glGetError()
        if err != GL.GL_NO_ERROR:
            self._diagnostics = "GL error after dispatch/readback: 0x%x" % err
        elif float(gpu[-1]) == 0.0:
            self._diagnostics = (
                "Readback all zero — shader may not have "
                "run; try Reload Shaders"
            )
        else:
            self._diagnostics = ""

        GL.glClearBufferfv(GL.GL_COLOR, 0, [0.0, 0.0, 0.0, 1.0])

    @override
    def draw_controls(self) -> None:
        gpu: np.ndarray = self.gpu_output  # render's readback of this frame
        imgui.text_wrapped(
            "Compute shader takes %d random floats in [0, 1) and returns "
            "the cumulative sum at every index. The max output equals the "
            "total sum of the input — about NUM_ELEMENTS/2 ≈ %.0f on "
            "average." % (NUM_ELEMENTS, NUM_ELEMENTS / 2.0)
        )
        imgui.separator()

        graph_size: tuple[int, int] = (680, 140)
        imgui.text("Input  (per-element values, range [0, 1))")
        imgui.plot_lines(
            "##input",
            np.ascontiguousarray(self.input_data, dtype=np.float32),
            scale_min=0.0,
            scale_max=1.0,
            graph_size=graph_size,
        )
        # Auto-scale the output graph to whichever is bigger: the GPU
        # max or the CPU max. A flat-zero output would otherwise sit
        # exactly on the bottom border and be invisible — we pick a
        # nonzero floor so the line is always drawn.
        gpu_max: float = float(np.max(gpu)) if gpu.size else 0.0
        cpu_max: float = float(self.cpu_output[-1])
        scale_max: float = max(gpu_max, cpu_max, 1e-3)
        imgui.text(
            "Output (GPU cumulative sum — should be monotonically increasing)"
        )
        imgui.plot_lines(
            "##gpu_output",
            np.ascontiguousarray(gpu, dtype=np.float32),
            scale_min=0.0,
            scale_max=scale_max,
            graph_size=graph_size,
        )
        if self._diagnostics:
            imgui.text_colored((1.0, 0.4, 0.4, 1.0), self._diagnostics)

        imgui.separator()
        imgui.text(
            "First 16 GPU outputs: " + " ".join("%5.2f" % v for v in gpu[:16])
        )
        imgui.text(
            "Final value (GPU): %.4f   (CPU reference: %.4f)"
            % (float(gpu[-1]), float(self.cpu_output[-1]))
        )
        diff: float = float(np.max(np.abs(gpu - self.cpu_output)))
        imgui.text("Max abs error vs CPU reference: %.6f" % diff)
        if imgui.button("New random input"):
            self.input_data = _make_input(
                seed=int(np.random.default_rng().integers(1, 1 << 31))
            )
            self.cpu_output = np.cumsum(self.input_data).astype(np.float32)
        imgui.same_line()
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.prog)
        GL.glDeleteBuffers(2, self.data_buffer)
        GL.glDeleteVertexArrays(1, [self.vao])


if __name__ == "__main__":
    application.run(PrefixsumApp)
