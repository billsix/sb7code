"""KTX 1.0 texture loader.

Port of ``src/sb7/sb7ktx.cpp``. Reads a KTX 1.0 file and
uploads the texture to GL via ``glTexStorage2D`` + ``glTexSubImage2D``
(or 1D/3D/CubeMap variants), returning the texture name.

KTX 1.0 file layout (a fixed 64-byte header then image data):

  bytes  0..11   identifier — 0xAB 'K' 'T' 'X' ' ' '1' '1' 0xBB '\\r\\n\\x1a\\n'
  bytes 12..15   endianness — 0x04030201 native, 0x01020304 swap
  bytes 16..63   13 × uint32: gltype, gltypesize, glformat,
                 glinternalformat, glbaseinternalformat, pixelwidth,
                 pixelheight, pixeldepth, arrayelements, faces,
                 miplevels, keypairbytes
  bytes 64..(64+keypairbytes-1)   key/value metadata (skipped here)
  rest:          image data, mip level 0 first, then 1, ...

Supported targets: GL_TEXTURE_1D, GL_TEXTURE_2D, GL_TEXTURE_3D,
GL_TEXTURE_1D_ARRAY, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_CUBE_MAP,
GL_TEXTURE_CUBE_MAP_ARRAY. Both compressed (gltype==GL_NONE) and
uncompressed formats are handled. Target is auto-detected from
pixelheight / pixeldepth / arrayelements / faces, same as the C++
loader.

Usage::

    tex = ktx.load("/path/to/foo.ktx")
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)

If the file can't be opened or the identifier doesn't match, returns
0 (matching the C++ semantic — caller must check).
"""

from __future__ import annotations

import dataclasses
import io
import struct
from typing import TypeAlias

import OpenGL.GL as GL
from OpenGL.constant import Constant

#: A value OpenGL accepts where it wants an enum — PyOpenGL constants
#: resolve to ``OpenGL.constant.Constant``, not ``int``, so a plain
#: ``int`` annotation rejects them (house pattern, same as
#: ``src/modelviewprojection/wxapp.py``).
GLenum: TypeAlias = int | Constant

_IDENTIFIER: bytes = bytes(
    [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A]
)

# KTX header is 12 (identifier) + 13 × 4 = 64 bytes.
_HEADER_FMT_NATIVE: str = "<13I"  # little-endian after we know endianness
_HEADER_FMT_SWAP: str = ">13I"


@dataclasses.dataclass(slots=True)
class Header:
    """Plain container mirroring the C++ ``sb7::ktx::file::header``
    struct. Each field is the corresponding GL enum / size. Field order
    matches the on-disk struct — ``Header(*struct.unpack(...))``."""

    endianness: int
    gltype: int
    gltypesize: int
    glformat: int
    glinternalformat: int
    glbaseinternalformat: int
    pixelwidth: int
    pixelheight: int
    pixeldepth: int
    arrayelements: int
    faces: int
    miplevels: int
    keypairbytes: int


def _parse_header(buf: bytes) -> Header:
    """Parse 64 bytes of KTX header. Auto-detects byte order from the
    ``endianness`` field — files with 0x04030201 are little-endian-on-disk,
    files with 0x01020304 came from a big-endian writer and need swap."""
    if len(buf) < 64:
        raise ValueError("KTX header truncated")
    if buf[:12] != _IDENTIFIER:
        raise ValueError("not a KTX 1.0 file (bad identifier)")
    # The endianness field is the first uint32 *after* the identifier.
    # Try native (LE) first; if that gives 0x04030201, we're correct.
    fields_le: tuple[int, ...] = struct.unpack(_HEADER_FMT_NATIVE, buf[12:64])
    if fields_le[0] == 0x04030201:
        return Header(*fields_le)
    fields_be: tuple[int, ...] = struct.unpack(_HEADER_FMT_SWAP, buf[12:64])
    if fields_be[0] == 0x04030201:
        return Header(*fields_be)
    raise ValueError(f"unrecognized endianness 0x{fields_le[0]:08X}")


def _sized_internal_format(internal: int, gltype: int) -> GLenum:
    """``glTexStorage*`` requires a *sized* internal format. Some KTX
    files store the unsized base format (GL_RGB / GL_RGBA / GL_RED /
    GL_BGR / etc.) in ``glinternalformat`` — that's legal for
    ``glTexImage*`` but not ``glTexStorage*``. Upgrade to the sized
    equivalent based on the GL type. Anything we don't recognize is
    passed through unchanged so already-sized formats work."""
    # Map (base-format, gltype) → sized-format. We cover the common
    # 8/16/32-bit unsigned-byte / half-float / float combos. BGR and
    # BGRA are storage-equivalent to RGB/RGBA — only the upload swizzle
    # differs.
    base_to_sized: dict[tuple[GLenum, GLenum], GLenum] = {
        (GL.GL_RED, GL.GL_UNSIGNED_BYTE): GL.GL_R8,
        (GL.GL_RG, GL.GL_UNSIGNED_BYTE): GL.GL_RG8,
        (GL.GL_RGB, GL.GL_UNSIGNED_BYTE): GL.GL_RGB8,
        (GL.GL_RGBA, GL.GL_UNSIGNED_BYTE): GL.GL_RGBA8,
        (GL.GL_BGR, GL.GL_UNSIGNED_BYTE): GL.GL_RGB8,
        (GL.GL_BGRA, GL.GL_UNSIGNED_BYTE): GL.GL_RGBA8,
        (GL.GL_RED, GL.GL_HALF_FLOAT): GL.GL_R16F,
        (GL.GL_RG, GL.GL_HALF_FLOAT): GL.GL_RG16F,
        (GL.GL_RGB, GL.GL_HALF_FLOAT): GL.GL_RGB16F,
        (GL.GL_RGBA, GL.GL_HALF_FLOAT): GL.GL_RGBA16F,
        (GL.GL_RED, GL.GL_FLOAT): GL.GL_R32F,
        (GL.GL_RG, GL.GL_FLOAT): GL.GL_RG32F,
        (GL.GL_RGB, GL.GL_FLOAT): GL.GL_RGB32F,
        (GL.GL_RGBA, GL.GL_FLOAT): GL.GL_RGBA32F,
        (GL.GL_RED, GL.GL_UNSIGNED_SHORT): GL.GL_R16,
        (GL.GL_RG, GL.GL_UNSIGNED_SHORT): GL.GL_RG16,
        (GL.GL_RGB, GL.GL_UNSIGNED_SHORT): GL.GL_RGB16,
        (GL.GL_RGBA, GL.GL_UNSIGNED_SHORT): GL.GL_RGBA16,
    }
    return base_to_sized.get((internal, gltype), internal)


def _channels_for_format(fmt: int) -> int:
    """How many color channels the given GL base internal format has.
    Used for stride calculation. Unknown formats fall back to 0,
    which yields a zero stride — fine for compressed paths since they
    skip the stride loop entirely."""
    if fmt == GL.GL_RED:
        return 1
    if fmt == GL.GL_RG:
        return 2
    if fmt in (GL.GL_RGB, GL.GL_BGR):
        return 3
    if fmt in (GL.GL_RGBA, GL.GL_BGRA):
        return 4
    return 0


def _calculate_stride(h: Header, width: int, pad: int = 4) -> int:
    """Bytes per row, padded up to ``pad`` (default 4). Mirrors the
    C++ ``calculate_stride`` helper."""
    channels: int = _channels_for_format(h.glbaseinternalformat)
    stride: int = h.gltypesize * channels * width
    stride = (stride + (pad - 1)) & ~(pad - 1)
    return stride


def _calculate_face_size(h: Header) -> int:
    return _calculate_stride(h, h.pixelwidth) * h.pixelheight


def _detect_target(h: Header) -> GLenum:
    """Replicate sb7's ``// Guess target`` block: pick a GL texture
    target based on which dimensional fields are non-zero."""
    if h.pixelheight == 0:
        if h.arrayelements == 0:
            return GL.GL_TEXTURE_1D
        return GL.GL_TEXTURE_1D_ARRAY
    if h.pixeldepth == 0:
        if h.arrayelements == 0:
            if h.faces == 0:
                return GL.GL_TEXTURE_2D
            return GL.GL_TEXTURE_CUBE_MAP
        if h.faces == 0:
            return GL.GL_TEXTURE_2D_ARRAY
        return GL.GL_TEXTURE_CUBE_MAP_ARRAY
    return GL.GL_TEXTURE_3D


def load(filename: str, tex: int = 0) -> int:
    """Load a KTX 1.0 file. Returns the GL texture name on success,
    or 0 on failure. If ``tex`` is non-zero, that texture is reused;
    otherwise a new one is created with ``glGenTextures``.

    Generates mipmaps via ``glGenerateMipmap`` when the file contains
    only a single level (matches the C++ behavior)."""
    try:
        f: io.BufferedReader
        with open(filename, "rb") as f:
            data: bytes = f.read()
    except OSError:
        return 0

    try:
        h: Header = _parse_header(data[:64])
    except ValueError:
        return 0

    width: int
    height: int
    depth: int
    width, height, depth = h.pixelwidth, h.pixelheight, h.pixeldepth
    if width == 0 or (height == 0 and depth != 0):
        return 0

    target: GLenum = _detect_target(h)

    if tex == 0:
        tex: int = GL.glGenTextures(1)
    GL.glBindTexture(target, tex)

    image_start: int = 64 + h.keypairbytes
    image_data: bytes = data[image_start:]

    miplevels: int = h.miplevels if h.miplevels != 0 else 1

    # glTexStorage* needs a sized internal format; some KTX files
    # store an unsized base format (GL_RGB, GL_RGBA, …). Upgrade.
    sized: GLenum = _sized_internal_format(h.glinternalformat, h.gltype)

    if target == GL.GL_TEXTURE_1D:
        GL.glTexStorage1D(GL.GL_TEXTURE_1D, miplevels, sized, width)
        GL.glTexSubImage1D(
            GL.GL_TEXTURE_1D, 0, 0, width, h.glformat, h.gltype, image_data
        )
    elif target == GL.GL_TEXTURE_2D:
        if h.gltype == GL.GL_NONE:
            # Compressed format. The C++ source has a hard-coded
            # ``420 * 380 / 2`` byte count that's only correct for the
            # one demo it was tested with; use the actual remaining
            # data length so this works for any compressed KTX.
            GL.glCompressedTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                h.glinternalformat,
                width,
                height,
                0,
                len(image_data),
                image_data,
            )
        else:
            GL.glTexStorage2D(GL.GL_TEXTURE_2D, miplevels, sized, width, height)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            offset: int = 0
            mw: int
            mh: int
            mw, mh = width, height
            level: int
            for level in range(miplevels):
                stride: int = _calculate_stride(h, mw, 1)
                level_size: int = stride * mh
                GL.glTexSubImage2D(
                    GL.GL_TEXTURE_2D,
                    level,
                    0,
                    0,
                    mw,
                    mh,
                    h.glformat,
                    h.gltype,
                    image_data[offset : offset + level_size],
                )
                offset += level_size
                mw = max(1, mw >> 1)
                mh = max(1, mh >> 1)
    elif target == GL.GL_TEXTURE_3D:
        GL.glTexStorage3D(
            GL.GL_TEXTURE_3D, miplevels, sized, width, height, depth
        )
        GL.glTexSubImage3D(
            GL.GL_TEXTURE_3D,
            0,
            0,
            0,
            0,
            width,
            height,
            depth,
            h.glformat,
            h.gltype,
            image_data,
        )
    elif target == GL.GL_TEXTURE_1D_ARRAY:
        GL.glTexStorage2D(
            GL.GL_TEXTURE_1D_ARRAY, miplevels, sized, width, h.arrayelements
        )
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_1D_ARRAY,
            0,
            0,
            0,
            width,
            h.arrayelements,
            h.glformat,
            h.gltype,
            image_data,
        )
    elif target == GL.GL_TEXTURE_2D_ARRAY:
        GL.glTexStorage3D(
            GL.GL_TEXTURE_2D_ARRAY,
            miplevels,
            sized,
            width,
            height,
            h.arrayelements,
        )
        GL.glTexSubImage3D(
            GL.GL_TEXTURE_2D_ARRAY,
            0,
            0,
            0,
            0,
            width,
            height,
            h.arrayelements,
            h.glformat,
            h.gltype,
            image_data,
        )
    elif target == GL.GL_TEXTURE_CUBE_MAP:
        GL.glTexStorage2D(
            GL.GL_TEXTURE_CUBE_MAP, miplevels, sized, width, height
        )
        face_size: int = _calculate_face_size(h)
        i: int
        for i in range(h.faces):
            face_data: bytes = image_data[face_size * i : face_size * (i + 1)]
            GL.glTexSubImage2D(
                # PyOpenGL constants are int subclasses at runtime, but
                # the stubs' `Constant` declares no `__add__`.
                GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,  # ty: ignore[unsupported-operator]
                0,
                0,
                0,
                width,
                height,
                h.glformat,
                h.gltype,
                face_data,
            )
    elif target == GL.GL_TEXTURE_CUBE_MAP_ARRAY:
        GL.glTexStorage3D(
            GL.GL_TEXTURE_CUBE_MAP_ARRAY,
            miplevels,
            sized,
            width,
            height,
            h.arrayelements,
        )
        GL.glTexSubImage3D(
            GL.GL_TEXTURE_CUBE_MAP_ARRAY,
            0,
            0,
            0,
            0,
            width,
            height,
            h.faces * h.arrayelements,
            h.glformat,
            h.gltype,
            image_data,
        )
    else:
        return 0

    if miplevels == 1:
        GL.glGenerateMipmap(target)

    return tex
