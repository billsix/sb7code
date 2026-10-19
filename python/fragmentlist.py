"""fragmentlist — per-pixel A-buffer linked lists (visualized as thickness).

Port of ``src/fragmentlist/fragmentlist.cpp``. Demonstrates
the per-pixel linked-list / "A-buffer" primitive that's the building
block for order-independent transparency, deep G-buffers, etc. This
particular demo uses it to compute and visualize the dragon's
*thickness* per pixel — but the SAME linked-list machinery underlies
real OIT.

Three passes:

  Pass 1 (clear):   wipe a window-sized R32UI image (the head-pointer
                    texture) to 0xFFFFFFFF — "no fragments here yet".
  Pass 2 (append):  render the dragon. For every fragment (including
                    back-facing ones — depth-test off), the FS atomically
                    claims a slot in a SHADER_STORAGE_BUFFER and prepends
                    that slot's index to the per-pixel linked list rooted
                    in the head-pointer image. Each record stores
                    ``(color, depth, facing, next)``.
  Pass 3 (resolve): a fullscreen quad walks each pixel's list and
                    accumulates ``-front_depth + back_depth`` across all
                    visited entries — producing the *thickness* of the
                    dragon along that view ray. Brighter pixels = more
                    dragon between you and the back surface. Thin parts
                    (legs, wing edges) come out dim; the body comes out
                    bright. **It is *not* a blurry render** — it's a
                    grayscale thickness map, which is precisely what the
                    C++ resolve shader computes.

Reads its shaders from ``media/shaders/fragmentlist/``.

The C++ source hardcodes a 1024×1024 head-pointer image (calibrated
for a 1024² window). Fragments outside that range silently get
``imageAtomicExchange`` no-ops on append, and read 0 on resolve →
white background on a wider window. We size the image to the actual
framebuffer instead, recreating it whenever the window resizes.

The SSBO holding fragment records is also bumped from the C++
source's 16 MB to 64 MB to give 1080p-class screens enough
depth-complexity headroom.

Reads its shaders from ``media/shaders/fragmentlist/``.
"""

import math
import os
from typing import override

import glfw  # noqa: F401
import numpy as np
import OpenGL.GL as GL
from imgui_bundle import imgui

from sb7 import application, common, sb6m, vmath


def _shader_path(name: str) -> str:
    return os.path.join(common.require_media(), "shaders", "fragmentlist", name)


# SSBO size: each record is ~16 bytes (color + depth + next). For
# 1920×1080 with depth complexity ~2, 64 MB ≈ 4M records gives ~2
# records/pixel which is enough for the dragon-only scene. Bump
# higher if you see flickering at the deepest parts.
FRAGMENT_BUFFER_BYTES: int = 64 * 1024 * 1024


class FragmentlistApp(application.Application):
    def __init__(self) -> None:
        super().__init__()
        self.clear_program: int = 0
        self.append_program: int = 0
        self.resolve_program: int = 0
        self.head_pointer_image: int = 0
        self._head_tex_size: tuple[int, int] = (0, 0)

    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Fragment List"

    @override
    def startup(self) -> None:
        self.load_shaders()

        self.dragon: sb6m.Object = sb6m.Object()
        self.dragon.load(
            os.path.join(common.require_media(), "objects", "dragon.sbm")
        )

        # SSBO: storage for the linked-list records.
        self.fragment_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.fragment_buffer)
        GL.glBufferData(
            GL.GL_SHADER_STORAGE_BUFFER,
            FRAGMENT_BUFFER_BYTES,
            None,
            GL.GL_DYNAMIC_COPY,
        )

        # Atomic counter to claim slot indices in the SSBO.
        self.atomic_counter_buffer: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ATOMIC_COUNTER_BUFFER, self.atomic_counter_buffer)
        GL.glBufferData(
            GL.GL_ATOMIC_COUNTER_BUFFER, 4, None, GL.GL_DYNAMIC_COPY
        )

        # Head-pointer image is sized lazily in render() to match
        # the framebuffer; see _ensure_head_image.

        self.dummy_vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.dummy_vao)

    def _ensure_head_image(self, w: int, h: int) -> None:
        """(Re)create the R32UI head-pointer image whenever the
        window size changes. The C++ source hardcodes 1024² and
        breaks on any other window — for the Python port we size to
        match. ``glTexStorage2D`` is immutable, so the only way to
        resize is to delete + recreate."""
        if self._head_tex_size == (w, h) and self.head_pointer_image:
            return
        if self.head_pointer_image:
            GL.glDeleteTextures(1, [self.head_pointer_image])
        self.head_pointer_image = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.head_pointer_image)
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D, 1, GL.GL_R32UI, max(1, w), max(1, h)
        )
        self._head_tex_size = (w, h)

    def load_shaders(self) -> None:
        for attr in ("clear_program", "append_program", "resolve_program"):
            old: int = getattr(self, attr)
            if old:
                GL.glDeleteProgram(old)
        self.clear_program = common.compile_program_from_files(
            vs=_shader_path("clear.vs.glsl"), fs=_shader_path("clear.fs.glsl")
        )
        self.append_program = common.compile_program_from_files(
            vs=_shader_path("append.vs.glsl"), fs=_shader_path("append.fs.glsl")
        )
        self.uniform_mvp: int = GL.glGetUniformLocation(
            self.append_program, "mvp"
        )
        self.resolve_program = common.compile_program_from_files(
            vs=_shader_path("resolve.vs.glsl"),
            fs=_shader_path("resolve.fs.glsl"),
        )

    @override
    def render(self, current_time: float) -> None:
        f: float = float(current_time)

        # Resize the head-pointer image if needed BEFORE setting the
        # viewport — the texture is bound by the clear/append/resolve
        # passes via glBindImageTexture below.
        self._ensure_head_image(self.info.window_width, self.info.window_height)

        GL.glViewport(0, 0, self.info.window_width, self.info.window_height)
        # Make sure the previous frame's writes are visible before we
        # start clearing.
        # PyOpenGL constants are int subclasses at runtime, but the
        # stubs' `Constant` declares no `__or__`, so OR-ing two masks
        # trips ty.
        GL.glMemoryBarrier(
            GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT  # ty: ignore[unsupported-operator]
            | GL.GL_ATOMIC_COUNTER_BARRIER_BIT
            | GL.GL_SHADER_STORAGE_BARRIER_BIT
        )

        # The clear pass also needs the head image bound so its FS
        # can imageStore 0xFFFFFFFF into every texel. The C++ source
        # only binds it for append; binding it before the clear pass
        # is what actually makes the clear effective.
        GL.glBindImageTexture(
            0,
            self.head_pointer_image,
            0,
            GL.GL_FALSE,
            0,
            GL.GL_READ_WRITE,
            GL.GL_R32UI,
        )

        # Pass 1: clear the head-pointer image to "no fragments".
        GL.glUseProgram(self.clear_program)
        GL.glBindVertexArray(self.dummy_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # Pass 2: append fragments to per-pixel lists.
        GL.glUseProgram(self.append_program)

        model: np.ndarray = vmath.scale(7.0)
        eye: tuple[float, float, float] = (
            math.cos(f * 0.35) * 120.0,
            math.cos(f * 0.4) * 30.0,
            math.sin(f * 0.35) * 120.0,
        )
        view: np.ndarray = vmath.lookat(eye, (0.0, 30.0, 0.0), (0.0, 1.0, 0.0))
        aspect: float = float(self.info.window_width) / float(
            max(1, self.info.window_height)
        )
        proj: np.ndarray = vmath.perspective(50.0, aspect, 0.1, 1000.0)

        vmath.upload(self.uniform_mvp, proj @ view @ model)

        # Reset the atomic counter to 0.
        zero: np.ndarray = np.zeros(1, dtype=np.uint32)
        GL.glBindBufferBase(
            GL.GL_ATOMIC_COUNTER_BUFFER, 0, self.atomic_counter_buffer
        )
        GL.glBufferSubData(GL.GL_ATOMIC_COUNTER_BUFFER, 0, 4, zero)

        # Bind the SSBO and the head-pointer image for the FS to use.
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER, 0, self.fragment_buffer
        )
        GL.glBindImageTexture(
            0,
            self.head_pointer_image,
            0,
            GL.GL_FALSE,
            0,
            GL.GL_READ_WRITE,
            GL.GL_R32UI,
        )

        # PyOpenGL constants are int subclasses at runtime, but the
        # stubs' `Constant` declares no `__or__`, so OR-ing two masks
        # trips ty.
        GL.glMemoryBarrier(
            GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT  # ty: ignore[unsupported-operator]
            | GL.GL_ATOMIC_COUNTER_BARRIER_BIT
            | GL.GL_SHADER_STORAGE_BARRIER_BIT
        )

        self.dragon.render()

        # Pass 3: resolve — walk each pixel's list, sort, blend.
        # PyOpenGL constants are int subclasses at runtime, but the
        # stubs' `Constant` declares no `__or__`, so OR-ing two masks
        # trips ty.
        GL.glMemoryBarrier(
            GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT  # ty: ignore[unsupported-operator]
            | GL.GL_ATOMIC_COUNTER_BARRIER_BIT
            | GL.GL_SHADER_STORAGE_BARRIER_BIT
        )
        GL.glUseProgram(self.resolve_program)
        GL.glBindVertexArray(self.dummy_vao)
        # PyOpenGL constants are int subclasses at runtime, but the
        # stubs' `Constant` declares no `__or__`, so OR-ing two masks
        # trips ty.
        GL.glMemoryBarrier(
            GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT  # ty: ignore[unsupported-operator]
            | GL.GL_ATOMIC_COUNTER_BARRIER_BIT
            | GL.GL_SHADER_STORAGE_BARRIER_BIT
        )
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    @override
    def draw_controls(self) -> None:
        imgui.text("Per-pixel linked lists (A-buffer).")
        imgui.text("Resolve pass sums (back - front) depth")
        imgui.text("across all surface hits → thickness map.")
        if imgui.button("Reload shaders"):
            self.load_shaders()

    @override
    def shutdown(self) -> None:
        self.dragon.free()
        for prog in (
            self.clear_program,
            self.append_program,
            self.resolve_program,
        ):
            GL.glDeleteProgram(prog)
        GL.glDeleteBuffers(1, [self.fragment_buffer])
        GL.glDeleteBuffers(1, [self.atomic_counter_buffer])
        GL.glDeleteTextures(1, [self.head_pointer_image])
        GL.glDeleteVertexArrays(1, [self.dummy_vao])


if __name__ == "__main__":
    application.run(FragmentlistApp)
