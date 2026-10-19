"""prefixsum2d — 2D summed-area table via two compute-shader passes.

Port of ``src/prefixsum2d/prefixsum2d.cpp``.

The point: a **summed-area table** (SAT) lets you compute the average
of any axis-aligned rectangular region of an image in **four texture
fetches** — one at each corner. SATs make variable-radius box blurs
constant-time, and that's what this demo shows.

Pipeline:

  1. Load a grayscale 2048×2048 KTX (``salad-gray.ktx``).
  2. Compute pass 1 (NUM_ELEMENTS=2048 workgroups, one per row):
     row-wise prefix sums of the source → image[1].
  3. Compute pass 2 (2048 workgroups, one per column): column-wise
     prefix sums of image[1] → image[2]. Image[2] is now the SAT.
  4. Fragment shader: for each output pixel, pick a blur radius that
     grows from left to right, sample the four SAT corners,
     reconstruct ``a − b − c + d`` (= sum within the box), divide by
     the box area to get the average. Result: the original image
     blurred more strongly toward the right edge.

The upstream display shader has hardcoded screen-space constants
``1800`` / ``100`` baked into its sampling math. Those fit only the
original demo's window. We override the display shader inline so the
SAT readback is normalized to whatever resolution the window happens
to be.

Reads the compute shader from ``media/shaders/prefixsum2d/`` and the
texture from ``media/textures/salad-gray.ktx``.
"""

import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, ktx


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "prefixsum2d", name)


def _texture_path(name: str) -> str:
    return os.path.join(common.require_media(), "textures", name)


NUM_ELEMENTS: int = 2048


# Replacement for showimage.fs.glsl. The upstream version hardcodes
# (1800, 100) screen-space offsets that crop badly on any other
# window size. This version uses normalized UVs and an explicit blur
# radius slider.
SHOW_IMAGE_FS: str = """\
#version 430 core

layout (binding = 0) uniform sampler2D input_image;
uniform float blur_radius_pixels = 32.0;

layout (location = 0) out vec4 color;

void main(void)
{
    vec2 sz = vec2(textureSize(input_image, 0));
    // Map fragment to image UV; flip Y so the salad is upright.
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(input_image, 0));
    uv.y = 1.0 - uv.y;

    // Blur radius scales linearly across the screen, 0 at the left.
    // Range is [0, blur_radius_pixels] in source-image pixels.
    float u = uv.x;  // already 0..1 across image width
    float m = 1.0 + u * blur_radius_pixels;
    vec2 mvec = vec2(m) / sz;

    vec2 P0 = uv + vec2(-mvec.x, -mvec.y);
    vec2 P1 = uv + vec2(-mvec.x,  mvec.y);
    vec2 P2 = uv + vec2( mvec.x, -mvec.y);
    vec2 P3 = uv + vec2( mvec.x,  mvec.y);

    float a = textureLod(input_image, P0, 0).r;
    float b = textureLod(input_image, P1, 0).r;
    float c = textureLod(input_image, P2, 0).r;
    float d = textureLod(input_image, P3, 0).r;

    float f = a - b - c + d;
    float area = (2.0 * m) * (2.0 * m);
    color = vec4(vec3(f / area), 1.0);
}
"""


class Prefixsum2dApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.prefix_sum_prog: int = 0
        self.show_image_prog: int = 0
        self.blur_radius_pixels: float = 32.0

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - 2D Prefix Sum"

    @override
    def startup(self) -> None:
        self.images: list[int] = [0, 0, 0]
        # Source: grayscale KTX texture.
        self.images[0] = ktx.load(_texture_path("salad-gray.ktx"))
        # Two scratch images that the compute pass writes into.
        scratch: np.ndarray = GL.glGenTextures(2)
        self.images[1] = int(scratch[0])
        self.images[2] = int(scratch[1])
        for i in (1, 2):
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.images[i])
            GL.glTexStorage2D(
                GL.GL_TEXTURE_2D, 1, GL.GL_R32F, NUM_ELEMENTS, NUM_ELEMENTS
            )

        self.dummy_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.dummy_vao)
        self.load_shaders()

    def load_shaders(self) -> None:
        if self.prefix_sum_prog:
            GL.glDeleteProgram(self.prefix_sum_prog)
        if self.show_image_prog:
            GL.glDeleteProgram(self.show_image_prog)
        self.prefix_sum_prog = common.compile_program_from_files(
            cs=_shader_path("prefixsum2d.cs.glsl")
        )
        with open(_shader_path("showimage.vs.glsl")) as fh:
            show_vs: str = fh.read()
        self.show_image_prog = common.compile_program(
            vs=show_vs, fs=SHOW_IMAGE_FS
        )
        self.u_blur_radius: int = GL.glGetUniformLocation(
            self.show_image_prog, "blur_radius_pixels"
        )

    @override
    def render(self, current_time: float) -> None:
        # Pass 1: row prefix sums (image[0] → image[1]).
        GL.glUseProgram(self.prefix_sum_prog)
        GL.glBindImageTexture(
            0, self.images[0], 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_R32F
        )
        GL.glBindImageTexture(
            1, self.images[1], 0, GL.GL_FALSE, 0, GL.GL_WRITE_ONLY, GL.GL_R32F
        )
        GL.glDispatchCompute(NUM_ELEMENTS, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Pass 2: column prefix sums (image[1] → image[2]). Same
        # shader, but the second dispatch transposes the work — the
        # shader interprets gl_GlobalInvocationID.x as the column
        # index for the second pass via internal swizzling.
        GL.glBindImageTexture(
            0, self.images[1], 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_R32F
        )
        GL.glBindImageTexture(
            1, self.images[2], 0, GL.GL_FALSE, 0, GL.GL_WRITE_ONLY, GL.GL_R32F
        )
        GL.glDispatchCompute(NUM_ELEMENTS, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Display the summed-area table on a fullscreen quad.
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.images[2])
        GL.glUseProgram(self.show_image_prog)
        if self.u_blur_radius >= 0:
            GL.glUniform1f(self.u_blur_radius, self.blur_radius_pixels)
        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        GL.glBindVertexArray(self.dummy_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        imgui.text_wrapped(
            "Summed-area table (SAT) of a grayscale image, computed via "
            "two compute-shader passes (rows then columns). The fragment "
            "shader uses the SAT to do a variable-radius box blur — wider "
            "blur on the right side, sharp on the left."
        )
        _, self.blur_radius_pixels = imgui.slider_float(
            "Max blur radius", self.blur_radius_pixels, 0.0, 128.0
        )
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        GL.glDeleteProgram(self.prefix_sum_prog)
        GL.glDeleteProgram(self.show_image_prog)
        GL.glDeleteTextures(3, self.images)
        GL.glDeleteVertexArrays(1, [self.dummy_vao])


if __name__ == "__main__":
    application.run(Prefixsum2dApp)
