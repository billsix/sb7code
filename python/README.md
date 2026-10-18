# OpenGL SuperBible 7e — Python port

Python translations of the example programs in this repository's C++ sources
(`../src/`), from **OpenGL SuperBible, 7th Edition** (Sellers, Wright, Haemel).
77 of the ~85 samples are ported; they use PyOpenGL + GLFW + imgui_bundle and
render the same pictures as the C++ originals.

This port is a **locally pip-installable package** — `python/` holds the
editable-installable package `sb7` (the shared modules) plus the demo scripts,
and the container runs `pip install -e python/`. The install is **local and
editable only** (never a remote/PyPI wheel). Its media (objects, shaders,
textures) is read from the C sources' own `../bin/media/`, never duplicated here.

## Run

From the repo root, in the container (builds both C and Python — see the
top-level `README.md`):

```bash
make run DEMO=blinnphong      # a Python demo (Wayland)
make run-c DEMO=blinnphong    # the C++ original (X11)
make check                    # import every demo, no display
make list                     # list the demos
```

Or directly, in or out of the container, from this `python/` directory (after
`pip install -e .`; no env var needed — the media directory is found by walking
up to `../bin/media`):

```bash
python singletri.py
python blinnphong.py
```

Most compute / tessellation / geometry demos need OpenGL 4.3+. To point at a
media directory elsewhere, set `SB7_MEDIA=/path/to/bin/media` (an override;
normally unnecessary). Demos with inline GLSL and no assets (`simpleclear`,
`singletri`, `spinnycube`, …) run even without a media directory.

## Layout

One `<name>.py` per demo at the `python/` root, mirroring `../src/`; its GLSL is
either inline in the `.py` or under `../bin/media/shaders/<name>/` (the same
files the C++ loads). The shared infrastructure is the installed `sb7` package
(`python/sb7/`):

| File | Purpose |
|---|---|
| `sb7/application.py` | `Application` base class (Python equivalent of `sb7::application`) + the `run(cls)` entry helper; glfw windowing + imgui menubar |
| `sb7/common.py` | Shader compile/link helpers, media resolution (`MEDIA_DIR`/`require_media`), and re-exports of the vendored window/menubar/camera helpers |
| `sb7/window.py` | Window/menubar/camera + imgui helpers, vendored from github.com/billsix/modelviewprojection (see the banner in the file) |
| `sb7/vmath.py` | numpy-backed subset of `vmath.h` (vec/mat, perspective/lookat/translate/rotate/scale) |
| `sb7/ktx.py` | KTX texture loader (mirrors `sb7ktx`) |
| `sb7/sb6m.py` | `.sbm` model loader (mirrors `sb7object`) |
| `sb7/colors.py` | named color constants from `sb7color.h` |

Demos import the package at the top of the file, like any installed module:

```python
from sb7 import common, application, window
```

## Conventions

- **GLFW + imgui_bundle** for windowing + UI (no GLUT). `application.Application`
  mirrors `sb7::application`: `init` / `startup` / `render(current_time)` /
  `shutdown` / `on_key` / `on_resize` hooks, `run(cls)` drives the loop. ESC
  closes; File→Quit and View→Fullscreen (F11, polled) via the menubar; a demo's
  own controls go in a Controls menu (`draw_controls`).
- Demo classes are PascalCase here where the C++ has `class wrapmodes_app` —
  one case convention, house style over source fidelity (William Emerison Six
  <billsix@gmail.com>, 2026-07-30); each docstring cites its upstream `.cpp`,
  and hook overrides carry `@override`.
- **Behaviour-faithful:** shaders, winding, sampler state and matrix math
  follow the C++ exactly. See `../tasks/reference/python-port/`.

## Fidelity and verification

All 77 demos import clean (`make check`) and were brought to parity with the
C++; correctness still needs **eyes on a display** (an import smoke test cannot
judge a picture). The porting rules, the frameworks, the Mesa/PyOpenGL traps,
and the per-demo fidelity notes are in `../tasks/reference/python-port/` — start
at its `README.md`.
