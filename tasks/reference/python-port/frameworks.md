# The port framework — what the shared modules provide (L2)

Up: `README.md`. The port keeps each demo readable top-to-bottom and pushes the machinery a
C++ demo got for free from `sb7::application` (and the `lib/` support code) into a few
modules in the `sb7` package (`python/sb7/`). This doc is the map of those modules: what each
provides and how they fit together. Facts are from the source under `python/`.

## 1. `application.py` — the `sb7::application` port

`Application` is the base class with the C++ virtuals as overridable hooks: `init()` (set
`self.info`, an `AppInfo` dataclass carrying title, window size, GL version, MSAA samples and
an `AppFlags`), `startup()`, `render(current_time)` (seconds, like the C++ `render(double)`),
`shutdown()`, and `on_resize` / `on_key` / `on_mouse_button` / `on_mouse_move` /
`on_mouse_wheel`. `run(cls)` is the `DECLARE_MAIN` entry point: it does the GLFW init / window /
context dance, then calls the demo's `init` / `startup` / loop / `shutdown`. GLFW key codes are
re-exported as `Application.KEY_*` so a demo never imports glfw to read them.

Decisions baked into the base class and `run`, several learned the hard way:

- **Opens at 800×600** (`AppInfo.window_width/height`) matching the C++ default, but the base
  `init()` first calls `window.resolve_default_window_size()` (a 1920×1080-capped size from
  the primary monitor); a demo that needs the C++'s exact 800×600 sets it back in its `init()`.
- **GL context is 4.3 core, forward-compatible** (`AppInfo.major/minor_version`) — compute,
  tessellation and geometry shaders need 4.3; a demo overrides it in `init()` (e.g. a
  `#version 420` demo).
- **A default VAO is left bound for the process lifetime** — never `glBindVertexArray(0)` as a
  teardown (macOS Core Profile prohibits VAO 0; `mesa-and-pyopengl-traps.md` §1).
- **`glUseProgram(0)` runs every frame between `render()` and imgui's draw** — v7 demos rebind
  their program each frame, and "Reload shaders" buttons delete the bound program, so resetting
  to 0 before imgui snapshots GL state stops imgui restoring a freed program id
  (`mesa-and-pyopengl-traps.md` §10).
- **The `contextdata` monkey-patch** runs before the first `import OpenGL.GL`
  (`mesa-and-pyopengl-traps.md` §1), and the three `glGetString` prints after
  `make_context_current` and before `init_imgui` are **load-bearing**, not decoration (they
  force PyOpenGL's context detection — same section).
- **`on_resize` receives framebuffer PIXELS** via `set_framebuffer_size_callback`, not the
  window-size callback (HiDPI — `mesa-and-pyopengl-traps.md` §2). A `gl_FragCoord`-sized FBO or
  image target must use these (`fragmentlist`).
- **Quit and Fullscreen are polled, not callback-driven.** The loop checks
  `glfw.get_key(KEY_ESCAPE)` each frame and `draw_menubar` polls F11, so neither depends on the
  GLFW key callback — which matters because imgui's `GlfwRenderer` replaces that callback
  without chaining (`mesa-and-pyopengl-traps.md` §2).

`run`'s loop draws the menubar (`window.draw_menubar`, with the demo's `draw_controls` as
the Controls body), calls the demo's `render(glfw.get_time())`, resets the program to 0, then
imgui's draw, then swaps.

## 2. `common.py` — shaders and media

- **Media resolution** (§7): `MEDIA_DIR`, `require_media()`, and `_find_media_dir`.
- **Shader pipeline:** `compile_shader`, `compile_program(vs=…, fs=…, gs=…, tcs=…, tes=…,
  cs=…)` from inline strings, `compile_program_from_files(...)` from paths, and
  `link_program(shaders, tf_varyings=…, tf_buffer_mode=…)` — the only correct way to call
  `glTransformFeedbackVaryings` from PyOpenGL (`mesa-and-pyopengl-traps.md` §7). A `ShaderError`
  carries the driver info log. Two mechanical source rewrites run on every shader at load:
  `_ensure_binding_qualifier_version` (bumps a `layout(binding)` shader's `#version` to 420) and
  `_ensure_flat_on_integer_varyings` (injects `flat` on integer interface-block members) —
  `mesa-and-pyopengl-traps.md` §8. Shader files are read `encoding="utf-8", errors="replace"`
  (traps §9).
- The `GLenum = int | Constant` alias (`c-to-python-rules.md` §2) is defined here, and also in
  `ktx.py`/`sb6m.py` so each stands alone.
- `common` also **re-exports the vendored v4 helpers** (`WindowState`, `draw_menubar`,
  `Camera`, `init_imgui`, `resolve_default_window_size`, `toggle_fullscreen`, …) so a demo can
  `from sb7 import common` and reach them; `application` loads the same vendored module directly (§6).

## 3. `vmath.py` — `vmath::` as numpy

`vec3`/`vec4`, `identity`/`translate`/`scale`/`rotate`/`rotate_xyz`,
`perspective`/`frustum`/`ortho`/`lookat`, and `upload(loc, mat)` which passes
`transpose=GL_TRUE` to convert row-major numpy to GLSL's column-major layout. Degrees for
`rotate`/`perspective`, right-handed projections, matching `vmath.h`. `translate`/`scale`
accept either three scalars or one `vec3` (the dispatch is on `y`/`z` being `None`).

## 4. `ktx.py` — the KTX 1.0 loader

`Header` (a `slots=True` dataclass mirroring the C++ `sb7::ktx::file::header`) + `load(filename)`:
reads a KTX 1.0 file, auto-detects the target (1D/2D/3D/array/cubemap) from the header, and
uploads through `glTexStorage*` + `glTexSubImage*` for both compressed and uncompressed formats
— a port of `src/sb7/sb7ktx.cpp`. It **upgrades unsized internal formats to sized ones** because
`glTexStorage*` rejects unsized (`mesa-and-pyopengl-traps.md` §6). Returns 0 on a bad
identifier, like the C++ (the caller checks).

## 5. `sb6m.py` / `colors.py` — meshes and colors

`sb6m.Object.load(filename)` reads the `.sbm` (SB6M) chunk format — `VRTX`/`ATRB`/`INDX`/`OLST`/
`DATA`/`CMNT` chunks — uploads the vertex blob, sets the attribute pointers, and draws via
`render()` / `render(instance_count=…)` / `render_sub_object(i)`, restoring
`glBindVertexArray(0)` afterward like the C++ `src/sb7/sb7object.cpp`. `colors.py` holds the
`sb7color.h` named colors inlined as `vec4`s (`c-to-python-rules.md` §6).

## 6. `window.py` — the vendored menubar/camera helpers (a frozen copy)

`window.py` is a **frozen vendored copy** of `ports/openglsuperbiblev4/_common.py` from the
`modelviewprojection` repository (github.com/billsix/modelviewprojection), where the port was
first written. Those helpers are mvp-wide (a shared menubar, imgui setup, window sizing,
fullscreen, a walk-around camera), not v7-specific, so they were copied in rather than
reimplemented when the port was extracted here. **modelviewprojection is the source of truth;
re-sync by copying that file over this one, and do not reshape it to this repo's style** (the
banner at the top of the file says so). Because it is an independent copy it will drift from
mvp's over time — that is inherent to this repo standing alone.

Both `common.py` and `application.py` reach it as a sibling module of the installed `sb7`
package (`from sb7 import window`), so no demo touches `sys.path` to find it. It stays a
**verbatim** drop-in of mvp's `_common.py` — kept as a plain copied module, never reshaped to
this repo's import style — so a re-sync is a file copy, not an edit (the banner, above).

What the two modules take from it: `WindowState`, `draw_menubar(window, state, controls=…)`,
`draw_view_menu`, `poll_fullscreen_key`/`toggle_fullscreen` (F11), `init_imgui`,
`resolve_default_window_size`, and `Camera`/`SceneObject`/`apply_camera`/`update_camera`/
`bind_camera_inputs`/`draw_camera_controls`. The camera helpers are re-exported for the demos
that use the walk-around camera; `MVP_CONTROLS_WINDOW` (the headless hook of
`headless-verification.md`) is read in `draw_menubar` here.

## 7. Media resolution — the one rule

Nothing hardcodes an absolute path. `common._find_media_dir()` resolves in order:

1. `$SB7_MEDIA`, if set and a directory (override).
2. **`<repo>/bin/media`** — one directory up from `python/`, the C sources' own media. This is
   the normal case and needs no env var, in the container or on a host checkout.
3. `<port>/bin/media` — a copy dropped into the port directory (not normally present).
4. An 8-level ancestor walk for a sibling `sb7code`/`superbiblev7` checkout, then
   `/sb7code/bin/media` and `/superbiblev7/bin/media`.

`require_media()` raises a `FileNotFoundError` naming what it tried when nothing matches; demos
with inline GLSL and no assets run without any media at all.

**The media is committed in this repo**, a deliberate maintainer decision (William Emerison Six
<billsix@gmail.com>, 2026-10-18) that **departs from sb7code's design** — upstream treats the
SuperBible 7e media as an external, separately-licensed download from openglsuperbible.com (the
`bin/` gitignore, the CMake download step, and the `bin/media/!OBTAIN_THE_MEDIA_PACK…` placeholder
are its mechanism). Committing it makes the repo self-contained; the recorded caveat (the pack
is not this project's to redistribute, so it may be a licensing concern) and how to reverse the
choice are in [`tasks/pythonize-and-containerize-port.md`](../../pythonize-and-containerize-port.md).
