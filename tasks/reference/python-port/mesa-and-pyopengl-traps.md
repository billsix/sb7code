# Mesa + PyOpenGL and imgui/GLFW traps (Python port) (L2)

Up: `README.md`. The failure modes met porting shader-era, compute-era C++ (OpenGL SuperBible
7e) to PyOpenGL + imgui-bundle on a **Fedora / Mesa radeonsi / Wayland** box, plus the
imgui/GLFW integration traps. Each cost a debugging round; several are invisible on NVIDIA or
AMD proprietary drivers, which is why the C++ originals never needed them. Check here before
writing or debugging GL/imgui/GLFW code in the port. Code paths are under `python/`.

## 1. PyOpenGL "no valid context" → patch `contextdata`

`imgui_bundle`'s `GlfwRenderer.__init__` dies in its first `glVertexAttribPointer(…,
c_void_p(offset))` because PyOpenGL's pointer-retention bookkeeping asks
`OpenGL.platform.GetCurrentContext()` and Mesa returns 0 even after GLFW made the context
current. The old fix (`OpenGL.STORE_POINTERS = False`) is **silently ignored** by current
PyOpenGL unless `ERROR_ON_COPY` is also on, which would break every list-argument
`glUniform*fv` call. The fix, applied in `application.py` *before the first `import OpenGL.GL`*, makes
the per-context bucket constant:

```python
import OpenGL.contextdata as _contextdata
_contextdata.getContext = lambda context=None: context if context is not None else 1
import OpenGL.GL as GL   # AFTER the patch
```

Retention degenerates to one global table, harmless because the port keeps its buffers alive
on instance attributes. Two companions: issue **one GL call after `make_context_current` and
before `init_imgui`** (the three `glGetString` vendor/version/renderer prints in `application.run` are
load-bearing, not decoration), and never call `glBindVertexArray(0)` as a teardown — a default
VAO stays bound for the process lifetime (macOS Core Profile prohibits VAO 0).

## 2. `import glfw` before `import imgui_bundle`, and GlfwRenderer does NOT chain callbacks

**Import order.** The first GLFW loaded into the process wins. imgui-bundle ships its own
`libglfw.so.3`, an **X11-only** build; if its native module loads first, pyglfw binds that
copy, the window gets a GLX context through XWayland, and PyOpenGL — which picks its EGL
backend purely from `WAYLAND_DISPLAY` — sees no current context. Its extension check then
fails, so every GL 4.2+ function `OpenGL.GL` exposes as an ARB alias (`glBindImageTexture`,
`glMemoryBarrier`, `glTexStorage2D`, `glDispatchCompute`) raises `NullFunctionError` while
`glGetString` still says "4.6". `application` imports glfw before imgui, so a demo that only imports
`application`/`common` is covered; a demo that imports `imgui_bundle` directly carries
`import glfw  # noqa: F401` above it. The container also exports `PYGLFW_LIBRARY` to the system
dual-backend build, which masks the problem — so a green container run is not evidence the
import order is right; keep the rule.

**Callbacks.** `GlfwRenderer(window)` installs its own GLFW key/char/cursor callbacks in its
constructor and does **not** chain to any previously-installed ones. `application.run` wires its own
callbacks in `_wire_callbacks()` *before* `init_imgui`, so imgui's renderer overwrites them.
The port side-steps this rather than depending on the clobbered callbacks: **Quit is polled**
(`glfw.get_key(KEY_ESCAPE)` each frame in the loop) and **Fullscreen is polled**
(`window.poll_fullscreen_key` in `draw_menubar`), and every demo control is an imgui widget
— which is why no demo overrides `on_key`. If a future demo does need `on_key`, move
`_wire_callbacks()` to *after* `init_imgui` and forward each event to
`imgui_impl.keyboard_callback`/`char_callback` first, or imgui will swallow it.

And `application` wires the **framebuffer-size** callback, not the window-size callback: the
window-size callback delivers screen coordinates, which under HiDPI / fractional scaling are
smaller than the framebuffer, so `glViewport` fed with them draws the scene into the
bottom-left `1/scale` of the window with black bars. `application` seeds the viewport once from
`get_framebuffer_size()` after startup and updates it in `on_resize`.

## 3. A bare `imgui.begin` floating window saves a too-narrow size

A bare `imgui.begin(title)` floating window is a trap — imgui auto-fits it once, sizing each
slider to ~65 % of a near-zero first-frame width, and the saved `imgui.ini` (written to the
cwd) then pins the narrow size. The port avoids this by putting every demo's widgets in the
menubar's Controls menu (`c-to-python-rules.md` §3). The one place a plain window is drawn —
`draw_menubar`'s `MVP_CONTROLS_WINDOW` verification window — opens with
`WindowFlags_.always_auto_resize`.

## 4. Mesa needs the `glMemoryBarrier` the proprietary drivers insert for you

Two programs sharing an image-load-store target (or SSBO, or atomic counter) across
draws/dispatches show **artifacts that look like z-fighting** — a partially-flushed image.
Insert the barrier between them:

| After … then … | Barrier bit |
|---|---|
| `imageStore` in one program, read in another | `GL_SHADER_IMAGE_ACCESS_BARRIER_BIT` |
| SSBO write, SSBO read | `GL_SHADER_STORAGE_BARRIER_BIT` |
| atomic counter write, read | `GL_ATOMIC_COUNTER_BARRIER_BIT` |
| compute `imageStore`, then `texture()` sampling | `GL_TEXTURE_FETCH_BARRIER_BIT` |
| compute-written attribute/draw-indirect buffer, then draw | `GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT` / `GL_COMMAND_BARRIER_BIT` |
| compute → CPU readback (`glGetBufferSubData`) | `GL_BUFFER_UPDATE_BARRIER_BIT`, and prefer `glFinish()` |

`dof`, `prefixsum`, `prefixsum2d`, `compressrgtc`, `fragmentlist` and `cullindirect` all issue
`glMemoryBarrier` for the pattern they hit; a demo that writes a GPU buffer in one stage and
reads it in another and shows shimmer on Mesa is missing one.

## 5. `glUniform*(0, …)` — location 0 is not reliably yours

Some C++ demos (`hdrexposure`, `ktxview`, `tesssubdivmodes`) write to location 0 for a uniform
with no `layout(location=0)`. NVIDIA-style drivers hand the first scalar uniform location 0;
Mesa does not and rejects with `INVALID_OPERATION`. **Always `glGetUniformLocation` after
linking**, even where the C++ hard-codes 0. Uniform *blocks* with `layout(binding=N)` are fine.

## 6. KTX with an *unsized* internal format cannot go through `glTexStorage*`

Some KTX files store `GL_RGB`/`GL_RGBA`/`GL_BGR`/`GL_BGRA` (unsized) in `glinternalformat`.
Legal for `glTexImage*`, `INVALID_ENUM` for `glTexStorage*`. `ktx` upgrades the unsized format
to its sized equivalent (`GL_RGB8`/`GL_RGBA8`/…) from `(internalformat, gltype)` before
allocating storage.

## 7. `glTransformFeedbackVaryings` from PyOpenGL

PyOpenGL type-checks the third argument as `LP_LP_c_char`; no list of `str`/`bytes` satisfies
it directly. `common.link_program(tf_varyings=…, tf_buffer_mode=…)` builds the
`create_string_buffer`s and the `POINTER(c_char)` array and keeps them referenced until
`glLinkProgram` returns (default mode `GL_SEPARATE_ATTRIBS`). Request transform feedback through
that keyword (`springmass`); never call the raw function.

## 8. Two GLSL rewrites applied to shaders at load (the shader text stays verbatim)

The shaders are the C++ build's (`c-to-python-rules.md` §7); these rewrites happen in the
*loader*, not the files on disk:

- **`layout(binding = N)` on samplers needs `#version 420`.** A shader that says `410 core` and
  uses it compiles on permissive drivers and fails on Mesa ("unrecognized layout identifier
  'binding'"). `common._ensure_binding_qualifier_version` bumps the `#version` line (4.20 is
  backward-compatible with 4.10).
- **Integer interface-block members need `flat`.** An `int material_id;` inside an `out VS_OUT
  {…}` without an interpolation qualifier compiles on NVIDIA and fails on Mesa.
  `common._ensure_flat_on_integer_varyings` injects `flat` (with a negative lookahead so it
  never emits `flat flat`); `indirectmaterial`'s VS_OUT block is the case that needs it.

## 9. Upstream shader files are not all UTF-8

An upstream shader can carry a Latin-1 byte (e.g. `©`, 0xA9) in a copyright comment; the C++
handed the compiler raw bytes, Python's default decode raises `UnicodeDecodeError`.
`shaders/rgtc/rgtccompress.cs.glsl` is the one that bit — it made `compressrgtc` the only crash
of this class in the headless sweep. `common` reads shaders with `encoding="utf-8",
errors="replace"`. When grepping for such bytes use **`LC_ALL=C grep -P`** — under a UTF-8
locale `grep -P "[\x80-\xff]"` skips invalid bytes and reports nothing.

## 10. The loop resets the bound program to 0 before imgui

v7 demos rebind their program every `render()`, and several have "Reload shaders" buttons that
delete the bound program, so `application.run` **does** `glUseProgram(0)` after `render()` and before
imgui snapshots GL state — otherwise imgui would save a freed program id, draw its UI, and crash
restoring it in `glUseProgram(stale_id)`. (This is the opposite of a fixed-pipeline tree where a
program is bound once and never rebound; there, resetting to 0 breaks the next
`glUniform*` — not the case here.)

## 11. Image-store / SSBO targets indexed by `gl_FragCoord` must be framebuffer-sized

The C++ `fragmentlist` hard-codes a 1024² head-pointer image. Fragments beyond 1024 in either
axis silently drop their `imageStore`/`imageAtomic*`; the resolve pass reads the image's initial
state there — white or garbage on the right and top of a larger window. `glTexStorage2D` is
immutable, so `fragmentlist._ensure_head_image(w, h)` allocates lazily in `render()` and deletes
and recreates on a size change, using the framebuffer pixels from `on_resize`.

## 12. imgui's state restore collapses the viewport array

imgui_bundle's GL backend restores state with `glViewport`, which sets **every** index of the
viewport array. A demo using `glViewportIndexedf` (`multiviewport`) must re-issue its viewports
each frame, or from frame 2 everything draws in the first viewport's region.

## Where each rule is enforced in code

| Rule | Code |
|---|---|
| §1 contextdata patch, default VAO, one-GL-call-before-imgui | `application.py` |
| §2 glfw-before-imgui, framebuffer-size callback, polled Quit/F11 | `application.py`, each demo's `import glfw  # noqa: F401`, `window.poll_fullscreen_key` |
| §6 KTX sized-format upgrade | `ktx.py` |
| §7 transform-feedback varyings | `common.link_program(tf_varyings=…)` |
| §8 version bump / `flat` injection | `common._ensure_binding_qualifier_version` / `_ensure_flat_on_integer_varyings` |
| §9 shader decode | `common.compile_shader_from_file` / `compile_program_from_files` |
| §10 reset program to 0 | `application.run` loop |
| §11 framebuffer-sized image target | `fragmentlist._ensure_head_image` |
