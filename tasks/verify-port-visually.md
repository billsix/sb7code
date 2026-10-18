# Display-check the SuperBible 7e Python port against the C++

**Status:** blocked
**Priority:** 4
**Difficulty:** 3
**Blocked on:** a real display — the maintainer running each demo and the C++
original side by side.
**Recheck:** `make run DEMO=spinnycube` on a display shows a lit, spinning cube;
this task closes when the rows below are confirmed (or a regression is filed in
`tasks/reference/python-port/fidelity-audit.md`).

## BLUF

All 77 Python demos import clean (`make check`) and were brought toward parity
with the C++ originals, but correctness needs eyes on a display — an import
smoke test cannot judge a picture. This is the maintainer's pass: run each demo
next to its C++ original (`make run DEMO=<name>` vs `make run-c DEMO=<name>`)
and confirm it draws the same thing. "Done" = every demo confirmed good, or a
regression recorded in `tasks/reference/python-port/fidelity-audit.md` with a
follow-up.

## How to run

    make run   DEMO=<name>    # the Python port (Wayland)
    make run-c DEMO=<name>    # the C++ original (X11)

Both open at the C++'s window size; the imgui menubar (File→Quit,
View→Fullscreen/F11, Controls) is the one visible addition on the Python side.

## Focus areas (higher risk first)

These are the demos most likely to differ, from the port's own notes
(`tasks/reference/python-port/fidelity-audit.md` and `mesa-and-pyopengl-traps.md`):

- **Compute / image-store:** `prefixsum`, `prefixsum2d`, `dof`, `compressrgtc`,
  `fragmentlist`, `cullindirect` — memory-barrier ordering and
  framebuffer-sized images; `prefixsum`'s empty-output diagnosis was parked.
- **HDR chain:** `hdrbloom`, `hdrexposure`, `hdrtonemap` — exposure/tonemap
  math and the uniform-location-0 trap.
- **Deferred / lighting:** `deferredshading` (a known "looks off" item),
  `ssao`, `shadowmapping`.
- **Geometry / tessellation / viewport:** `gsquads`, `gstessellate`,
  `tessmodes`, `tesssubdivmodes`, `multiviewport` (viewport-array collapse).
- **KTX / textures:** `ktxview`, `bumpmapping`, `cubemapenv`, `equirectangular`.
- **HiDPI:** every demo — confirm the framebuffer-size path is right on a
  HiDPI display (the port's framebuffer-size callback).

## Notes

- The C examples use gl3w (GLX) → X11; the Python port uses PyOpenGL (EGL) →
  Wayland. `make run-c` unsets `WAYLAND_DISPLAY` for the C side.
- Headless back-buffer captures are triage, not proof (animation phase differs
  between two captures).
