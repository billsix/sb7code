# The Python port — reference set map (L0)

**Reference set** — durable knowledge about the Python port under
[`python/`](../../../python/) of this repository: the C++→Python translation rules
it settled on, the Mesa + PyOpenGL + imgui/GLFW traps met along the way, how to
verify it without a display, and the per-demo fidelity picture against the C++
originals that sit beside it in [`src/`](../../../src/). Read this first; each row
links down to the doc with the detail. Updated in place, never archived.

This knowledge was harvested from the port's development history — the port was
first written inside the `modelviewprojection` course repository
(github.com/billsix/modelviewprojection) and extracted here so this repository is
the single home for the C++ SuperBible 7e samples and their port. The docs are
rewritten to read cold against *this* repo alone: everything a fresh contributor
needs is in these docs or the files they point at.

## What the port is

A Python port of **77** of the C++ OpenGL SuperBible 7th-edition examples in this
repository (the upstream is github.com/openglsuperbible/sb7code), using PyOpenGL +
glfw + Dear ImGui (imgui-bundle) and rendering the same scenes. It lives in
`python/` as a **local editable pip package** — `python/sb7/` holds the shared
modules and the demos are flat scripts beside it (`python blinnphong.py`, or
`make run DEMO=blinnphong`), installed with `pip install -e python/` (local and
editable only, never a PyPI wheel; maintainer's call). Each `python/<name>.py`
maps 1:1 to a `src/<name>/<name>.cpp` here (85 C++ examples; the port covers 77
of them).

- Shared infrastructure is the installed `sb7` package (`python/sb7/`):
  `application.py` (the `sb7::application` framework — glfw + imgui_bundle
  menubar, lifecycle hooks, `run`), `common.py` (shader compile/link, media
  resolution), `vmath.py` (the `vmath.h` math), `ktx.py` (KTX textures),
  `sb6m.py` (`.sbm` meshes), `colors.py` (`sb7color.h`), and `window.py` (a
  frozen vendored copy of the menubar/camera/imgui helpers — `frameworks.md` §6).
- **The C++ examples' media is committed in this repo** at
  [`bin/media/`](../../../bin/media/) (objects, `.ktx`/`.sbm` assets, and the GLSL
  shaders under `bin/media/shaders/`). The port references it, never duplicates it
  (`frameworks.md` §7). Committing the pack is a **deliberate departure** from
  sb7code's external-download design — see the note in `frameworks.md` §7 and
  [`tasks/archive/2026/10/19/pythonize-and-containerize-port.md`](../../archive/2026/10/19/pythonize-and-containerize-port.md).
- The C++ originals are right here: each `python/<name>.py` has its
  counterpart in `src/<name>/<name>.cpp` (+ the `sb7` framework in `lib/` and
  `include/`), which is what the fidelity picture compares against.

## The docs in this set

| Doc | One breath |
|---|---|
| `c-to-python-rules.md` | The settled C++→Python rules: the type table, imgui-widgets-not-keyboard (menubar Controls menu, F11, Esc), "set" a uniform vs "upload" a buffer, inline small fixtures, **shaders are the C++ build's** (§7), and **keyword every argument a callee accepts, and which PyOpenGL functions cannot** (§8). |
| `frameworks.md` | What the shared modules provide: `application.Application` (lifecycle + `run`), `common` (shader + media helpers), `vmath`/`ktx`/`sb6m`/`colors`, and the vendored `window` (why it's a frozen copy and where the source of truth is). |
| `mesa-and-pyopengl-traps.md` | The modern-GL failure modes on Mesa + PyOpenGL and the imgui/GLFW integration traps: the `contextdata` patch, `import glfw` first, GlfwRenderer not chaining callbacks, `glMemoryBarrier`, the `#version` bump and `flat`-on-integer injection, KTX unsized formats, image targets sized to the framebuffer, uniform location 0, transform-feedback varyings, the viewport-array collapse, and the loop's `glUseProgram(0)`. |
| `headless-verification.md` | What `make check` (the import smoke test) proves and what it cannot, the back-buffer capture method for a real render check, and the X11/Wayland backend agreement a headless run needs. Correctness needs a display. |
| `fidelity-audit.md` | The verification status against the C++ originals and the per-demo latent-risk list (the compute/image-store demos, the HDR chain, `deferredshading`, `ktxview`, `prefixsum`). Modest by design: the on-display pass is still pending. |

## Verification status

A headless sweep from the port's development history rendered all 77 demos (one,
`compressrgtc`, needed two fixes first — `mesa-and-pyopengl-traps.md` §9 and the
raw-signature `glCompressedTexImage2D`). `make check` imports every demo module
without a display and so catches import-time and asset-resolution failures; it
cannot confirm the picture matches — that needs a display, which is the
maintainer's to run (`fidelity-audit.md`).

## Related (elsewhere in this repo)

- [`python/README.md`](../../../python/README.md) — the per-demo run notes.
- [`tasks/archive/2026/10/19/pythonize-and-containerize-port.md`](../../archive/2026/10/19/pythonize-and-containerize-port.md)
  — the extraction task: the new layout, the editable-package and committed-media
  decisions, and the vendoring of `window`.
