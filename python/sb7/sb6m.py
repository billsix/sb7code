"""SBM (SB6M) model loader.

Port of ``src/sb7/sb7object.cpp`` and the chunk format
declared in ``include/sb6mfile.h``.

A ``.sbm`` file consists of:

  - 16-byte header: magic ``b'SB6M'``, header_size, num_chunks, flags
  - ``num_chunks`` chunk *headers* (each starts with a 4-byte FourCC name
    and a 4-byte size, followed by chunk-specific fields)
  - the raw vertex / index data, referenced by ``data_offset`` fields
    inside the chunk headers (offsets are relative to file start)

Recognized chunk types:

  - ``b'VRTX'`` (vertex data) — packed buffer + total vertex count
  - ``b'ATRB'`` (vertex attribute decls) — 84 bytes per attribute:
    a 64-byte name, then 5 uints (size, type, stride, flags, offset)
  - ``b'INDX'`` (index data) — index_type / count / offset
  - ``b'OLST'`` (sub-object list) — array of (first, count) pairs for
    indexed draws
  - ``b'DATA'`` — alternative single-chunk encoding (none of the
    shipped models use it but the C++ loader handles it)
  - ``b'CMNT'`` — comment, ignored

Usage::

    obj = sb6m.Object()
    obj.load("/.../cube.sbm")
    obj.render()                     # one full draw
    obj.render(instance_count=24)    # 24 instances
    obj.render_sub_object(2)         # draw sub-object index 2

After ``load`` the ``Object``'s VAO is left bound to nothing — like
the C++ loader, we restore ``glBindVertexArray(0)`` and
``glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)``.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from typing import List, Optional, TypeAlias

import OpenGL.GL as GL
from OpenGL.constant import Constant

#: A value OpenGL accepts where it wants an enum — PyOpenGL constants
#: resolve to ``OpenGL.constant.Constant``, not ``int``, so a plain
#: ``int`` annotation rejects them (house pattern, same as
#: ``src/modelviewprojection/wxapp.py``).
GLenum: TypeAlias = int | Constant

_MAGIC: bytes = b"SB6M"
_HEADER_FMT: str = "<4sIII"  # magic, size, num_chunks, flags
_HEADER_SIZE: int = 16
_CHUNK_HDR: str = "<4sI"  # name, size

_FLAG_NORMALIZED: int = 0x00000001
_FLAG_INTEGER: int = 0x00000002


@dataclass
class _AttribDecl:
    name: str
    size: int
    type: int
    stride: int
    flags: int
    data_offset: int


@dataclass
class _SubObject:
    first: int
    count: int


@dataclass
class _Chunks:
    """Chunks the loader cared about, captured by FourCC."""

    # (data_size, data_offset, total_v)
    vertex_data: Optional[tuple[int, int, int]] = None
    # (index_type, count, offset)
    index_data: Optional[tuple[int, int, int]] = None
    attrib_chunk: Optional[List[_AttribDecl]] = None
    sub_objects: Optional[List[_SubObject]] = None
    # (encoding, offset, length)
    data_chunk: Optional[tuple[int, int, int]] = None


def _parse_chunks(data: bytes) -> tuple[_Chunks, int]:
    if len(data) < _HEADER_SIZE:
        raise ValueError("SBM file too short for header")
    magic: bytes
    hsize: int
    n_chunks: int
    _flags: int
    magic, hsize, n_chunks, _flags = struct.unpack(
        _HEADER_FMT, data[:_HEADER_SIZE]
    )
    if magic != _MAGIC:
        raise ValueError(f"not an SBM file (got {magic!r})")

    chunks: _Chunks = _Chunks()
    ptr: int = hsize
    for _ in range(n_chunks):
        cname: bytes
        csize: int
        cname, csize = struct.unpack(_CHUNK_HDR, data[ptr : ptr + 8])
        if cname == b"ATRB":
            attrib_count: int
            (attrib_count,) = struct.unpack("<I", data[ptr + 8 : ptr + 12])
            attrs: List[_AttribDecl] = []
            off: int = ptr + 12
            for _ in range(attrib_count):
                name: str = (
                    data[off : off + 64]
                    .rstrip(b"\0")
                    .decode("ascii", errors="replace")
                )
                size: int
                type_: int
                stride: int
                flags: int
                doff: int
                size, type_, stride, flags, doff = struct.unpack(
                    "<5I", data[off + 64 : off + 84]
                )
                attrs.append(
                    _AttribDecl(name, size, type_, stride, flags, doff)
                )
                off += 84
            chunks.attrib_chunk = attrs
        elif cname == b"VRTX":
            data_size: int
            total_v: int
            data_size, doff, total_v = struct.unpack(
                "<3I", data[ptr + 8 : ptr + 20]
            )
            chunks.vertex_data = (data_size, doff, total_v)
        elif cname == b"INDX":
            idx_type: int
            idx_count: int
            idx_off: int
            idx_type, idx_count, idx_off = struct.unpack(
                "<3I", data[ptr + 8 : ptr + 20]
            )
            chunks.index_data = (idx_type, idx_count, idx_off)
        elif cname == b"OLST":
            count: int
            (count,) = struct.unpack("<I", data[ptr + 8 : ptr + 12])
            subs: List[_SubObject] = []
            off = ptr + 12
            for _ in range(count):
                first: int
                cnt: int
                first, cnt = struct.unpack("<2I", data[off : off + 8])
                subs.append(_SubObject(first, cnt))
                off += 8
            chunks.sub_objects = subs
        elif cname == b"DATA":
            encoding: int
            dlen: int
            encoding, doff, dlen = struct.unpack(
                "<3I", data[ptr + 8 : ptr + 20]
            )
            chunks.data_chunk = (encoding, doff, dlen)
        # CMNT and any unknown chunks are silently skipped.
        ptr += csize
    return chunks, n_chunks


def _index_size(index_type: GLenum) -> int:
    if index_type == GL.GL_UNSIGNED_SHORT:
        return 2
    if index_type == GL.GL_UNSIGNED_INT:
        return 4
    return 1  # GL_UNSIGNED_BYTE / GL_NONE


@dataclass
class Object:
    """Loaded SBM model. ``load(path)`` populates it; ``render()`` /
    ``render_sub_object(i)`` issues the draw call(s)."""

    data_buffer: int = 0
    vao: int = 0
    index_type: GLenum = GL.GL_NONE
    index_offset: int = 0
    sub_objects: List[_SubObject] = field(default_factory=list)

    def free(self) -> None:
        if self.vao:
            GL.glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
        if self.data_buffer:
            GL.glDeleteBuffers(1, [self.data_buffer])
            self.data_buffer = 0

    def load(self, filename: str) -> None:
        f: io.BufferedReader
        with open(filename, "rb") as f:
            data: bytes = f.read()
        self._load_from_bytes(data)

    def _load_from_bytes(self, data: bytes) -> None:
        self.free()
        chunks: _Chunks
        chunks, _ = _parse_chunks(data)

        if chunks.attrib_chunk is None:
            raise ValueError("SBM file missing ATRB chunk")

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.data_buffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.data_buffer)

        if chunks.data_chunk is not None:
            # Single packed DATA chunk. Use it directly.
            _enc: int
            doff: int
            dlen: int
            _enc, doff, dlen = chunks.data_chunk
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                dlen,
                data[doff : doff + dlen],
                GL.GL_STATIC_DRAW,
            )
        else:
            # Vertex data + (optional) index data. The C++ loader
            # packs both into the same buffer. Vertex data goes at
            # offset 0; index data follows immediately after.
            v_size: int = 0
            if chunks.vertex_data is not None:
                v_size = chunks.vertex_data[0]
            i_size: int = 0
            if chunks.index_data is not None:
                idx_type: int
                idx_count: int
                _idx_off: int
                idx_type, idx_count, _idx_off = chunks.index_data
                i_size = idx_count * _index_size(idx_type)

            total: int = v_size + i_size
            GL.glBufferData(GL.GL_ARRAY_BUFFER, total, None, GL.GL_STATIC_DRAW)

            if chunks.vertex_data is not None:
                v_data_size: int
                v_doff: int
                v_data_size, v_doff, _ = chunks.vertex_data
                GL.glBufferSubData(
                    GL.GL_ARRAY_BUFFER,
                    0,
                    v_data_size,
                    data[v_doff : v_doff + v_data_size],
                )
            if chunks.index_data is not None:
                idx_off: int
                idx_type, idx_count, idx_off = chunks.index_data
                GL.glBufferSubData(
                    GL.GL_ARRAY_BUFFER,
                    v_size,
                    i_size,
                    data[idx_off : idx_off + i_size],
                )

        # Wire up vertex attributes per ATRB chunk.
        i: int
        ad: _AttribDecl
        for i, ad in enumerate(chunks.attrib_chunk):
            normalized: GLenum = (
                GL.GL_TRUE if (ad.flags & _FLAG_NORMALIZED) else GL.GL_FALSE
            )
            if ad.flags & _FLAG_INTEGER:
                GL.glVertexAttribIPointer(
                    i, ad.size, ad.type, ad.stride, GL.GLvoidp(ad.data_offset)
                )
            else:
                GL.glVertexAttribPointer(
                    i,
                    ad.size,
                    ad.type,
                    normalized,
                    ad.stride,
                    GL.GLvoidp(ad.data_offset),
                )
            GL.glEnableVertexAttribArray(i)

        if chunks.index_data is not None:
            idx_type, idx_count, _idx_off = chunks.index_data
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.data_buffer)
            self.index_type = idx_type
            # Element-buffer offset is right after the vertex data.
            self.index_offset = (
                chunks.vertex_data[0] if chunks.vertex_data else 0
            )
        else:
            self.index_type = GL.GL_NONE
            self.index_offset = 0

        if chunks.sub_objects is not None:
            self.sub_objects = chunks.sub_objects[:256]
        else:
            # No sub-object list: synthesize a single one covering
            # all the geometry.
            if chunks.index_data is not None:
                count: int = chunks.index_data[1]
            else:
                count = chunks.vertex_data[2] if chunks.vertex_data else 0
            self.sub_objects = [_SubObject(0, count)]

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def render(self, instance_count: int = 1, base_instance: int = 0) -> None:
        self.render_sub_object(0, instance_count, base_instance)

    def render_sub_object(
        self, index: int, instance_count: int = 1, base_instance: int = 0
    ) -> None:
        if not self.sub_objects:
            return
        sub: _SubObject = self.sub_objects[index]
        GL.glBindVertexArray(self.vao)

        if self.index_type != GL.GL_NONE:
            # Element-buffer offset = sub-object's first element ×
            # bytes-per-index, plus this object's own index_offset
            # within the buffer.
            byte_offset: int = self.index_offset + sub.first * _index_size(
                self.index_type
            )
            GL.glDrawElementsInstancedBaseInstance(
                GL.GL_TRIANGLES,
                sub.count,
                self.index_type,
                GL.GLvoidp(byte_offset),
                instance_count,
                base_instance,
            )
        else:
            GL.glDrawArraysInstancedBaseInstance(
                GL.GL_TRIANGLES,
                sub.first,
                sub.count,
                instance_count,
                base_instance,
            )

    def get_sub_object_count(self) -> int:
        return len(self.sub_objects)

    def get_sub_object_info(self, index: int) -> tuple[int, int]:
        if 0 <= index < len(self.sub_objects):
            s: _SubObject = self.sub_objects[index]
            return s.first, s.count
        return 0, 0
