"""Shared helpers for the SuperBible v7 ports.

This module:

  1. Re-exports the window/menubar/camera helpers from the vendored
     ``window.py`` (see the banner in that file) — so demos can write
     ``import common`` and get ``WindowState``, ``draw_menubar``,
     ``Camera``, etc.

  2. Adds shader-compile helpers v7 demos need: ``compile_shader`` for
     a single stage, ``compile_program`` for a vs/fs pair (or any
     combination of stages), and convenience wrappers that read GLSL
     source from a file. These mirror the role of ``shader.h`` /
     ``sb7shader.cpp`` in the C++ framework.

  3. Adds a ``MEDIA_DIR`` constant — the path under this directory where
     ``superbible7-media.zip`` is unpacked. Demos that need KTX
     textures or SBM models read from there.
"""

from __future__ import annotations

import ctypes
import os
import re
from typing import Iterable, Optional, Sequence

import OpenGL.GL as GL
from OpenGL.constant import Constant

from . import window

#: A value OpenGL accepts where it wants an enum — PyOpenGL constants
#: resolve to ``OpenGL.constant.Constant``, not ``int``, so a plain
#: ``int`` annotation rejects them (house pattern, same as
#: ``src/modelviewprojection/wxapp.py``).
GLenum = int | Constant

_HERE = os.path.dirname(os.path.abspath(__file__))

# Re-export the vendored window/menubar/camera helpers (from window.py).
Camera = window.Camera
SceneObject = window.SceneObject
WindowState = window.WindowState
apply_camera = window.apply_camera
bind_camera_inputs = window.bind_camera_inputs
draw_camera_controls = window.draw_camera_controls
draw_menubar = window.draw_menubar
init_imgui = window.init_imgui
resolve_default_window_size = window.resolve_default_window_size
toggle_fullscreen = window.toggle_fullscreen
update_camera = window.update_camera


# Path where v7 media (KTX textures, SBM models, GLSL shaders) lives.
# This repo commits its media under ``<repo>/bin/media`` -- the same place
# the C++ samples read from (cwd-relative ``media/...`` from ``bin/``), so
# both ports share one copy. (That the media is committed here at all is a
# deliberate departure from sb7code's external-download default -- see
# tasks/pythonize-and-containerize-port.md.)
#
# Resolved at import time in this order:
#   1. ``$SB7_MEDIA`` env var, if set.
#   2. ``<repo>/bin/media`` -- one directory up from this ``python/`` port.
#      Works in the container and on a host checkout with no env var.
#   3. a copy dropped into the port directory, then a sibling checkout named
#      ``sb7code``/``superbiblev7``.
#
# If none exist, ``MEDIA_DIR`` is None and ``require_media()`` raises a clear
# error. Demos that load no media (inline-GLSL basics) work without it.


def _find_media_dir() -> str | None:
    env = os.environ.get("SB7_MEDIA")
    if env and os.path.isdir(env):
        return os.path.abspath(env)

    # Normal case: this port lives at ``<repo>/python/`` and the media is at
    # ``<repo>/bin/media`` -- one level up. Both ports share that one copy.
    in_repo: str = os.path.join(os.path.dirname(_HERE), "bin", "media")
    if os.path.isdir(in_repo):
        return in_repo

    # Fallback: a copy dropped into the port directory.
    in_port: str = os.path.join(_HERE, "bin", "media")
    if os.path.isdir(in_port):
        return in_port

    # Walk up looking for a sibling/cousin checkout of the C repo.
    current = _HERE
    for _ in range(8):
        parent = os.path.dirname(current)
        if parent == current:
            break
        for name in ("sb7code", "superbiblev7"):
            candidate: str = os.path.join(parent, name, "bin", "media")
            if os.path.isdir(candidate):
                return candidate
        current = parent

    for fallback in ("/sb7code/bin/media", "/superbiblev7/bin/media"):
        if os.path.isdir(fallback):
            return fallback

    return None


MEDIA_DIR = _find_media_dir()


def require_media() -> str:
    """Return ``MEDIA_DIR`` or raise a clear FileNotFoundError. Call
    this at the top of any demo that loads from ``MEDIA_DIR`` so the
    error names exactly what's missing and how to fix it."""
    if MEDIA_DIR is None:
        raise FileNotFoundError(
            "SuperBible v7 media directory not found.\n"
            "  Tried $SB7_MEDIA, sibling-walk for superbiblev7/bin/media, "
            "and /superbiblev7/bin/media.\n"
            "  Set SB7_MEDIA to the absolute path of the unpacked "
            "superbible7-media.zip "
            "(e.g. export SB7_MEDIA=/path/to/superbiblev7/bin/media)."
        )
    return MEDIA_DIR


# ---------------------------------------------------------------------------
# Shader compilation
# ---------------------------------------------------------------------------


class ShaderError(RuntimeError):
    """Raised when GLSL compilation or program linking fails. The
    message is the driver's info log so students see exactly what the
    GLSL compiler said."""


def _ensure_flat_on_integer_varyings(source: str) -> str:
    """The GLSL spec requires that integer interface-block members
    (``int``, ``uint``, ``ivec*``, ``uvec*``) carry the ``flat``
    interpolation qualifier — they cannot be smooth-interpolated.
    Some C++ v7 shaders (e.g. ``indirectmaterial``'s VS_OUT block)
    omit ``flat`` and rely on driver leniency. Mesa rejects with
    ``error: if a fragment input is (or contains) an integer, then
    it must be qualified with 'flat'``.

    Inject ``flat`` only on declarations that don't already carry an
    interpolation qualifier (so we never emit ``flat flat int``)."""

    def _patch_block(m: re.Match[str]) -> str:
        prefix = m.group(1)
        body = m.group(2)
        body = re.sub(
            r"^([ \t]*)(?!(?:flat|noperspective|smooth|centroid|patch)\b)"
            r"(int|uint|ivec[234]|uvec[234])(\s+\w+\s*;)",
            r"\1flat \2\3",
            body,
            flags=re.MULTILINE,
        )
        return prefix + "{" + body + "}"

    pattern = re.compile(
        r"((?:^|\n)\s*(?:in|out)\s+\w+\s*)\{([^{}]*)\}",
        flags=re.DOTALL,
    )
    return pattern.sub(_patch_block, source)


def _ensure_binding_qualifier_version(source: str) -> str:
    """Some v7 shader files use ``layout (binding = N) uniform
    sampler2D ...`` but declare ``#version 410 core`` — that binding
    syntax wasn't added to GLSL until 4.20. Permissive drivers
    accept it under 4.10 anyway; Mesa is strict and rejects it with
    ``unrecognized layout identifier 'binding'``.

    If we see both a ``binding`` qualifier and a low ``#version``,
    bump the version. 4.20 is backward-compatible with 4.10."""
    if "binding" not in source:
        return source
    # Match `#version 4XX core` where XX < 20 and bump to 420.
    m = re.search(r"^\s*#version\s+(\d+)\s+core", source, re.MULTILINE)
    if not m:
        return source
    ver: int = int(m.group(1))
    if ver < 420 and ver >= 100:
        return re.sub(
            r"^(\s*#version\s+)\d+(\s+core)",
            r"\g<1>420\g<2>",
            source,
            count=1,
            flags=re.MULTILINE,
        )
    return source


def compile_shader(source: str, shader_type: GLenum) -> int:
    """Compile a single GLSL shader stage. ``shader_type`` is one of
    ``GL.GL_VERTEX_SHADER`` / ``GL_FRAGMENT_SHADER`` / ``GL_GEOMETRY_SHADER``
    / ``GL_TESS_CONTROL_SHADER`` / ``GL_TESS_EVALUATION_SHADER`` /
    ``GL_COMPUTE_SHADER``. Returns the shader name on success, raises
    ``ShaderError`` with the info log on failure.

    Equivalent of ``sb7::shader::from_string`` in the C++ framework.
    """
    source = _ensure_binding_qualifier_version(source)
    source = _ensure_flat_on_integer_varyings(source)
    shader: int = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not status:
        log = GL.glGetShaderInfoLog(shader).decode(errors="replace")
        GL.glDeleteShader(shader)
        raise ShaderError(f"shader compile failed: {log}")
    return shader


def compile_shader_from_file(path: str, shader_type: GLenum) -> int:
    # errors="replace": upstream shaders are ASCII except for the odd Latin-1
    # byte in a copyright comment (rgtc/rgtccompress.cs.glsl has a 0xA9 (c));
    # the C++ handed the compiler raw bytes, so decoding must not raise.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return compile_shader(f.read(), shader_type)


def link_program(
    shaders: Iterable[int],
    *,
    delete_shaders: bool = True,
    tf_varyings: Optional[Sequence[str]] = None,
    tf_buffer_mode: int | None = None,
) -> int:
    """Link a program from already-compiled shader objects. Detaches
    and (by default) deletes the shaders after linking, since v7's
    convention is that the program owns the binaries. Raises
    ``ShaderError`` on link failure.

    Optional ``tf_varyings`` enables transform feedback: pass a list
    of varying names (as Python ``str`` — we'll encode and box them
    into a ``ctypes.c_char_p`` array, which is what PyOpenGL's
    ``glTransformFeedbackVaryings`` actually expects). The default
    buffer mode is ``GL_SEPARATE_ATTRIBS`` if ``tf_buffer_mode`` is
    not specified."""
    program: int = GL.glCreateProgram()
    shaders = list(shaders)
    for s in shaders:
        GL.glAttachShader(program, s)
    if tf_varyings is not None:
        # PyOpenGL's argtype is ``LP_LP_c_char`` (pointer-to-pointer-
        # to-char). A bare ``(c_char_p * N)`` array fails the strict
        # type check even though it has the right memory layout. Build
        # a backing list of ``c_char`` buffers, an array of pointers
        # into them, and cast to LP_LP_c_char.
        encoded = [
            v.encode("ascii") if isinstance(v, str) else v for v in tf_varyings
        ]
        # Keep the buffers alive until link finishes — assign back so
        # they don't get GC'd.
        _buffers = [ctypes.create_string_buffer(s) for s in encoded]
        lp_char = ctypes.POINTER(ctypes.c_char)
        ptrs = (lp_char * len(_buffers))(
            *[ctypes.cast(b, lp_char) for b in _buffers]
        )
        names_ptr = ctypes.cast(ptrs, ctypes.POINTER(lp_char))
        mode = (
            tf_buffer_mode
            if tf_buffer_mode is not None
            else GL.GL_SEPARATE_ATTRIBS
        )
        GL.glTransformFeedbackVaryings(
            program, len(tf_varyings), names_ptr, mode
        )
    GL.glLinkProgram(program)
    status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
    if not status:
        log = GL.glGetProgramInfoLog(program).decode(errors="replace")
        GL.glDeleteProgram(program)
        raise ShaderError(f"program link failed: {log}")
    for s in shaders:
        GL.glDetachShader(program, s)
        if delete_shaders:
            GL.glDeleteShader(s)
    return program


def compile_program(
    *,
    vs: Optional[str] = None,
    fs: Optional[str] = None,
    gs: Optional[str] = None,
    tcs: Optional[str] = None,
    tes: Optional[str] = None,
    cs: Optional[str] = None,
    tf_varyings: Optional[Sequence[str]] = None,
    tf_buffer_mode: int | None = None,
) -> int:
    """Compile a program from inline GLSL strings. Pass any subset of
    stages by keyword. Most demos use just ``vs`` + ``fs``::

        program = common.compile_program(vs=VS_SOURCE, fs=FS_SOURCE)

    A compute demo passes ``cs=...`` alone. Tessellation demos pass
    ``vs`` + ``tcs`` + ``tes`` + ``fs``. The function picks the right
    GL shader-type enums and links them.

    Pass ``tf_varyings`` (a list of varying names) to enable transform
    feedback before linking. ``tf_buffer_mode`` defaults to
    ``GL_SEPARATE_ATTRIBS``; pass ``GL.GL_INTERLEAVED_ATTRIBS`` to
    pack into one buffer.
    """
    pairs = [
        (vs, GL.GL_VERTEX_SHADER),
        (fs, GL.GL_FRAGMENT_SHADER),
        (gs, GL.GL_GEOMETRY_SHADER),
        (tcs, GL.GL_TESS_CONTROL_SHADER),
        (tes, GL.GL_TESS_EVALUATION_SHADER),
        (cs, GL.GL_COMPUTE_SHADER),
    ]
    shaders = [
        compile_shader(src, kind) for src, kind in pairs if src is not None
    ]
    if not shaders:
        raise ShaderError("compile_program called with no stages")
    return link_program(
        shaders, tf_varyings=tf_varyings, tf_buffer_mode=tf_buffer_mode
    )


def compile_program_from_files(
    *,
    vs: Optional[str] = None,
    fs: Optional[str] = None,
    gs: Optional[str] = None,
    tcs: Optional[str] = None,
    tes: Optional[str] = None,
    cs: Optional[str] = None,
    tf_varyings: Optional[Sequence[str]] = None,
    tf_buffer_mode: int | None = None,
) -> int:
    """Same as ``compile_program`` but each argument is a path to a
    GLSL file. Convention: paths end in ``.vs.glsl`` / ``.fs.glsl`` /
    ``.gs.glsl`` / ``.tcs.glsl`` / ``.tes.glsl`` / ``.cs.glsl`` — the
    same suffixes as this repo's ``shaders/`` tree.
    """

    def _read(p: str | None) -> str | None:
        if p is None:
            return None
        # See compile_shader_from_file for why errors="replace".
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    return compile_program(
        vs=_read(vs),
        fs=_read(fs),
        gs=_read(gs),
        tcs=_read(tcs),
        tes=_read(tes),
        cs=_read(cs),
        tf_varyings=tf_varyings,
        tf_buffer_mode=tf_buffer_mode,
    )
