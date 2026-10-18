# C++ → Python: the settled rules for the port (L2)

Up: `README.md`. The rules the Python port under `python/` settled on, translating this
repository's C++ OpenGL SuperBible 7e examples (`src/<name>/<name>.cpp` + the `sb7`
framework in `lib/`/`include/`). **Behaviour-faithful throughout:** nothing here may change
GL call order, RNG, or what is drawn. Code paths are under `python/`.

## 1. Structure follows the C++, names follow Python

Each demo keeps its C++ original's *shape* — a class per demo overriding the
`sb7::application` lifecycle hooks — so a reader can hold `src/<name>/<name>.cpp` and
`python/<name>.py` side by side. The hooks are renamed to Python house style, which is
safe because the *Python* base class (`application.Application`) defines them, so the name is ours:

| C++ (`sb7.h`) | Python | Note |
|---|---|---|
| `init` / `startup` / `render(double)` / `shutdown` | `init` / `startup` / `render(current_time)` / `shutdown` | overrides carry `@override` |
| `onKey` / `onResize` / `onMouseButton` / `onMouseMove` / `onMouseWheel` | `on_key` / `on_resize` / `on_mouse_button` / `on_mouse_move` / `on_mouse_wheel` | |
| the demo class `my_application` (lowercase) | `MyApplication` (PascalCase) | the one renamed-for-house-style class per demo |

The framework docstrings and each demo's docstring keep the C++↔Python mapping (every port
cites its upstream `.cpp`), so the side-by-side reading still works. `@override` is not
decoration: a hook whose name drifts from the base becomes a type error instead of
silently-dead dispatch, and `make check` imports every demo so the mismatch is caught.

## 2. The type table (C++ / vmath → Python)

| C / library type | Python | Note |
|---|---|---|
| `GLuint`, `GLint`, `GLsizei` | `int` | names, locations, counts |
| `GLenum` | `GLenum = int \| Constant` | the alias defined in `common`/`ktx`/`sb6m` — PyOpenGL constants are `IntConstant`, so a bare `int` annotation on a GL enum needs a suppression; the alias avoids them |
| `GLfloat` / `GLboolean` / `GLchar*` | `float` / `bool` / `str` | |
| `GLfloat[16]`/`[9]`/`[3]`/`[4]`, `vmath::vec*`/`mat4` | `np.ndarray` | 4×4 **row-major**, transposed at upload (`vmath.upload` passes `GL_TRUE`) |
| `KTX_header`, the SB6M chunk headers | `slots=True` dataclasses (`ktx.Header`, `sb6m`) | |
| a GLFW window handle | `Any` | **the one sanctioned `Any`** — glfw's stubs type it as an opaque pointer |

Two footguns: `glGen*(1)` returns an `int` but `glGen*(n>1)` returns an array — annotate
only the `(1)` form; and treat any other `Any` as "I didn't check" — leave the name bare
instead. **Typing finds real bugs by counting:** count the `glUniform*` calls per program
against the C++'s `glGetUniformLocation`s — a mismatch is either a missing upload or a uniform
the C++ fetches but never sets.

## 3. Every adjustable parameter is an imgui widget; keys are accelerators at most

A C++ `case 'M': mode = (mode + 1) % N;` becomes an `imgui.radio_button` per mode or a combo;
`case 'P': paused = !paused;` an `imgui.checkbox`; `case '+': iterations += 1;` an
`imgui.slider_int`; `case 'R': load_shaders();` an `imgui.button`. **Show the widget — do not
write "press M to toggle" in the panel.** 49 of the 77 demos define widgets.

**The widgets live in the menubar's Controls menu, never a floating window** (the
maintainer's call, William Emerison Six <billsix@gmail.com>, 2026-10-14: a floating panel
overlaps the drawing, and a bare `imgui.begin` saves a too-narrow size that `imgui.ini` then
pins — `mesa-and-pyopengl-traps.md` §3). The menubar is `File → Quit (Esc)`, `View →
Fullscreen (F11, polled from the keyboard)`, `Controls → the demo's widgets`. A demo overrides
`draw_controls(self)` on its `application.Application`; the framework adds a Controls menu only when a
demo defines one (`_controls_hook`). A menu body runs only while the menu is open, so apply a
change on the spot or store it on `self` — never rely on it running every frame. Quit and
Fullscreen are **polled** every frame (`glfw.get_key(KEY_ESCAPE)` in the loop, `F11` in
`window.poll_fullscreen_key`), so they work without a per-demo key callback — which is why
no demo overrides `on_key` (`frameworks.md` §1, and the callback-clobber caveat in
`mesa-and-pyopengl-traps.md` §2). Headless gates set `MVP_CONTROLS_WINDOW=1`, which also draws
the Controls body in a plain window every frame so every widget executes
(`headless-verification.md`).

## 4. What a framework base class may and may not define

Do **not** pre-define, as a `@property`, an attribute demos assign to. A base class that turns
such an attribute into a read-only property breaks every demo that writes it in a lifecycle
hook, exactly as the C++ does. Either leave the attribute to the demo or give it a plain
default in `__init__` (as `application.Application.__init__` does for `window`/`imgui_impl`/
`win_state`).

## 5. "Set" a uniform; "upload" a buffer or a texture

`glUniform*` specifies program state (it moves no bytes to GPU memory); `glBufferData`/
`glBufferSubData`/`glTexImage*`/`glTexStorage*` do. So a demo's uniform helper is named
`set_uniforms`, and "upload" is reserved for VBO/UBO creation, `glBufferSubData`, and texture
loads. (`vmath.upload(loc, mat)` is the one exception kept for parity with `vmath`'s own name
— it is a thin `glUniformMatrix4fv` wrapper, not a buffer upload.)

## 6. Inline small static fixtures; never parse a C++ header at runtime

A static table a C++ demo keeps in a header belongs inline as Python data, not regexed out of
the C++ source tree at load — a runtime parse fails on a machine that has the media but not the
include tree. `colors.py` is the worked case: it is a mechanical translation of `sb7color.h` +
`src/sb7/sb7color.cpp` into module-level `vec4`s, not a parse of those files. Same for anything
you would otherwise read out of the C++ source at runtime.

## 7. The shaders are the C++ build's; parity work is Python-side

**Every shader a port runs is the shader its C++ binary runs.** This repo holds it in two
forms:

- **The 37 demos whose C++ loads shader files** load the same files through
  `common.compile_program_from_files`, reading straight from this repo's
  `bin/media/shaders/<demo>/` — byte-identical to what the C++ compiles.
- **The 40 demos whose C++ embeds GLSL as string literals** hold the decoded literal text in a
  module-level Python string, byte-equal to the C++ — re-verify against `src/<demo>/<demo>.cpp`
  after any edit.

Never hand-reduce, "simplify", restyle or re-derive a shader; if the port needs different
behaviour, that is a Python-side change (uniforms, texture formats/units, GL state, camera).
Decided by the maintainer (William Emerison Six <billsix@gmail.com>, 2026-10-13) after a
hand-written PBR shader hid two Python-side bugs (a 16-bit-float HDR overflow and unfiltered
importance sampling) behind shading differences (learned across the mvp port trees). The loader
makes two *mechanical* rewrites at load time that do not touch the shader's logic — the
`#version` bump and the `flat` injection of `mesa-and-pyopengl-traps.md` §8 — so the files on
disk stay verbatim.

## 8. Keyword arguments at every call site that can take them

Every call whose callee accepts keyword arguments passes them by keyword, under one mechanical
rule (maintainer's call, William Emerison Six <billsix@gmail.com>, 2026-10-14):

- **Project APIs and keyword-capable library calls keyword every argument:** the port's own
  functions and dataclasses, PyOpenGL's Python-level convenience functions
  (`compile_program_from_files(vs=…, fs=…)`, `glBufferData(target=…, size=…, …)` where the
  wrapper allows it), imgui-bundle (`imgui.slider_float(label=…, v=…, v_min=…, v_max=…)`) and
  glfw.
- **Library callees keep required leading parameters positional** and keyword only those with a
  default. **Single-argument and `*`/`**`-splat calls are left alone.**

**What cannot take keywords, and stays positional — do not "finish the job" by hand:**
PyOpenGL's raw ctypes functions (`glUseProgram`, `glBindTexture`, `glUniform*f/i/fv`,
`glEnable`, `glViewport`, `glClearBufferfv`, `glDrawArrays`, …) and its array-converting
wrappers (`glGen*`, `glBufferSubData`, `glUniformMatrix4fv`, `glGetIntegerv`, …) raise
`TypeError` on a keyword argument. numpy ufuncs and `math.*` take no keywords either. This is a
convention, not a gate (ruff has no rule for it).

## 9. Gates for any edit to the port

`ruff` clean and format-idempotent at line-length 80 (`select = E4/E7/E9/F/T201` in
`python/pyproject.toml`); `ty` clean (the per-site `# ty: ignore[...]` reasons are the GL
`Constant`-vs-`int` cases, the `contextdata` patch, and the importlib boilerplate); and
`make check` must still import all 77 demos (which executes `@override` and the import-time
media resolution). For anything that changes what is drawn, a back-buffer render before/after
(`headless-verification.md`). Correctness ultimately needs the maintainer's display.
