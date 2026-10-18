"""Subset of ``vmath.h`` (SuperBible 7e) ported to numpy.

Covers the operations the Tier 1 / Tier 2 demos use: ``mat4`` with
``perspective`` / ``lookat`` / ``translate`` / ``rotate`` / ``scale``,
and basic ``vec3`` / ``vec4`` aliases. More gets added as demos need it.

Convention: matrices are stored as ``numpy.float32`` arrays of shape
``(4, 4)`` indexed by ``mat[row, col]`` — the natural mathematical
layout. Multiplication is ``A @ B`` (numpy ``matmul``). When you upload
to a GLSL ``uniform mat4`` you must pass ``transpose=GL_TRUE`` because
GLSL expects column-major. ``upload(loc, mat)`` does that for you::

    proj = vmath.perspective(50.0, aspect, 0.1, 1000.0)
    vmath.upload(self.proj_location, proj)

The C++ ``vmath::perspective`` takes degrees for FOV and returns a GL
clip-space (right-handed, -Z forward) matrix. Same here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import OpenGL.GL as GL


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float32)


def vec4(
    x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0
) -> np.ndarray:
    return np.array([x, y, z, w], dtype=np.float32)


def identity() -> np.ndarray:
    return np.identity(4, dtype=np.float32)


def translate(
    x: float | Sequence[float] | np.ndarray,
    y: float | None = None,
    z: float | None = None,
) -> np.ndarray:
    """Translation matrix. Accepts either three scalars or a single
    iterable (``vec3``)."""
    if y is None and z is None:
        # Dispatch is on y/z being None, which ty can't tie back to x
        # being the sequence branch of the union (that would take
        # @overload stubs, which the no-restructuring rule rules out).
        x, y, z = float(x[0]), float(x[1]), float(x[2])  # ty: ignore[not-subscriptable]
    m = identity()
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(
    x: float | Sequence[float] | np.ndarray,
    y: float | None = None,
    z: float | None = None,
) -> np.ndarray:
    if y is None and z is None:
        if hasattr(x, "__len__"):
            # hasattr(__len__) narrows to a len-only protocol, which
            # still lacks __getitem__ in ty's eyes; every runtime type
            # with __len__ that reaches here (list/tuple/ndarray) also
            # subscripts.
            x, y, z = float(x[0]), float(x[1]), float(x[2])  # ty: ignore[not-subscriptable]
        else:
            x = y = z = float(x)
    m = identity()
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z
    return m


def rotate(angle_degrees: float, ax: float, ay: float, az: float) -> np.ndarray:
    """Rotation by ``angle_degrees`` around the unit vector ``(ax, ay,
    az)``. Same signature as ``vmath::rotate(angle, x, y, z)``. The
    axis is normalized internally so you don't have to."""
    n: float = math.sqrt(ax * ax + ay * ay + az * az)
    if n == 0.0:
        return identity()
    ax, ay, az = ax / n, ay / n, az / n
    a: float = math.radians(angle_degrees)
    c: float = math.cos(a)
    s = math.sin(a)
    omc = 1.0 - c
    m = identity()
    m[0, 0] = c + ax * ax * omc
    m[0, 1] = ax * ay * omc - az * s
    m[0, 2] = ax * az * omc + ay * s
    m[1, 0] = ay * ax * omc + az * s
    m[1, 1] = c + ay * ay * omc
    m[1, 2] = ay * az * omc - ax * s
    m[2, 0] = az * ax * omc - ay * s
    m[2, 1] = az * ay * omc + ax * s
    m[2, 2] = c + az * az * omc
    return m


def rotate_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Z-Y-X Euler convention used by the second overload of
    ``vmath::rotate(x, y, z)`` in the book. Angles in degrees."""
    return rotate(rz, 0, 0, 1) @ rotate(ry, 0, 1, 0) @ rotate(rx, 1, 0, 0)


def perspective(
    fovy_degrees: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """Right-handed perspective projection matching ``vmath::perspective``.
    FOV is in degrees (vertical), ``aspect = width / height``, depth
    range is GL's [-1, 1] post-perspective-divide."""
    f = 1.0 / math.tan(math.radians(fovy_degrees) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def frustum(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = (2.0 * near) / (right - left)
    m[0, 2] = (right + left) / (right - left)
    m[1, 1] = (2.0 * near) / (top - bottom)
    m[1, 2] = (top + bottom) / (top - bottom)
    m[2, 2] = -(far + near) / (far - near)
    m[2, 3] = -(2.0 * far * near) / (far - near)
    m[3, 2] = -1.0
    return m


def ortho(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    m = identity()
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def lookat(
    eye: Sequence[float] | np.ndarray,
    center: Sequence[float] | np.ndarray,
    up: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Right-handed view matrix matching ``vmath::lookat``. Each arg is
    a 3-element iterable."""
    eye: np.ndarray = np.asarray(eye, dtype=np.float32)
    center: np.ndarray = np.asarray(center, dtype=np.float32)
    up: np.ndarray = np.asarray(up, dtype=np.float32)

    f = center - eye
    f /= np.linalg.norm(f)
    upn = up / np.linalg.norm(up)
    s = np.cross(f, upn)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)

    m = identity()
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


# ---------------------------------------------------------------------------
# GL upload
# ---------------------------------------------------------------------------


def upload(location: int, mat: np.ndarray) -> None:
    """Upload a row-major numpy mat4 to a GLSL ``mat4`` uniform.

    Calls ``glUniformMatrix4fv(location, 1, GL_TRUE, mat)`` — the
    ``GL_TRUE`` transposes our row-major storage into the column-major
    layout GLSL expects.
    """
    GL.glUniformMatrix4fv(
        location, 1, GL.GL_TRUE, np.ascontiguousarray(mat, dtype=np.float32)
    )
