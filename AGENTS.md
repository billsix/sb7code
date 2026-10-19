# AGENTS — sb7code

OpenGL SuperBible 7th Edition example source (public upstream
github.com/openglsuperbible/sb7code; this checkout's remote is a private
mirror): C++ examples (`src/`, built by CMake against the `sb7` static lib in
`src/sb7/` + `include/`, using glfw + **gl3w** + OpenGL 4.x) plus a
behaviour-faithful **Python port** of 77 of them (`python/`). The media —
objects, shaders, textures — lives once under `bin/media/` and is used by both
halves.

## Repo shape

- `src/<name>/<name>.cpp` — one C++ example per directory (~85; the CMake
  `EXAMPLES` list is authoritative). CMake links each to `sb7`.
- `include/`, `src/sb7/` — the sb7 framework (gl3w, KTX/object/shader loaders).
- `bin/media/` — the assets, loaded by the C++ examples via the **cwd-relative**
  path `"media/..."` (so an example runs with cwd = `bin/`). See the media note
  below.
- `python/` — the Python port (below).
- `tasks/`, `tasks/reference/` — task docs and durable knowledge.

## Build and run in a container

The repo-root `Makefile` + `Dockerfile` build **both** halves in one image:

```
make image                   # build the image (bakes + compiles the C++ examples)
make shell                   # repo-root shell, both X11 + Wayland; helpers: run <dir> / run-c <dir>
make run    DEMO=blinnphong  # Python port of a demo (Wayland)
make run-c  DEMO=blinnphong  # C++ original (X11)
make build-c                 # (re)build the C++ examples
make check                   # import every Python demo module, no display
make list                    # list the demos
```

Nested podman (inside a runClaudeInContainer/runCrushInContainer sandbox) works
as-is: `PODMAN_RUN_FLAGS` auto-adds `--cgroups=disabled` when `NESTED_PODMAN=1`
is exported. Never pass `NESTED_PODMAN=1` on a downstream command.

### Two display backends, on purpose

**The C++ examples need X11**: gl3w resolves GL entry points through GLX, which
does not exist on an EGL/Wayland context, so a C demo under Wayland opens a
window and cannot draw. `make run-c` / the `run-c` helper launch with
`WAYLAND_DISPLAY` unset (→ X11, or XWayland on a Wayland host). **The Python
port runs on Wayland**: PyOpenGL handles EGL, so `make run` / `run` stay on
Wayland (and also work on X11).

### The C build tree is outside the mount

`make shell` bind-mounts the repo over `/sb7code`. The C build tree is
`/opt/sb7-build` (NOT under the mount), so the host's own build cache cannot
collide with it. CMake writes binaries into `/opt/sb7-build/bin`; `run-c`
launches them with cwd = `bin/` so the `"media/..."` paths resolve against the
committed `bin/media/`. CMake is invoked with
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` so modern CMake accepts the project's
`cmake_minimum_required(VERSION 2.6)`.

## The Python port

- **A local editable pip package** (maintainer's decision). `python/` holds the
  installable package `sb7` (the shared modules) plus the flat demo scripts; the
  container runs `pip install -e python/`. Demos run as `python <name>.py` (or
  `make run DEMO=<name>`) and import the package at the top of the file;
  `sb7.application`'s `run()` is the entry point. The install is local/editable
  only (never a remote/PyPI wheel).
- **Media is referenced, never duplicated.** `python/sb7/common.py`'s
  `_find_media_dir()` walks up from `python/` to this repo's `bin/media`, so the
  port needs no env var in or out of the container. `$SB7_MEDIA` overrides.
- **`python/sb7/window.py` is vendored** from
  github.com/billsix/modelviewprojection
  (`ports/openglsuperbiblev4/_common.py`) — the window/menubar/camera + imgui
  helpers the port leans on. Frozen copy (mvp is the source of truth; re-sync by
  copying that file over it). Both `sb7/common.py` and `sb7/application.py` load
  it as a sibling.
- **Packaged, plus lint/type config.** `python/pyproject.toml` carries
  `[project]`/`[build-system]` (the editable install) alongside `[tool.ruff]`.
  `make check` is the headless gate (import every demo); render correctness
  needs a display.

## Media note (committed here, deliberately)

`bin/media/` is committed in this repo, which **departs from sb7code's design**
(upstream gitignores `bin/` and downloads the pack from openglsuperbible.com at
build time). This is a deliberate maintainer decision for a self-contained repo,
with a recorded licensing caveat — see `bin/media/README.md` and
`tasks/archive/2026/10/19/pythonize-and-containerize-port.md`. The CMakeLists was adjusted to skip
its media download when the committed media is present.

## Cold-start knowledge for the port

Read before touching the port: `tasks/reference/python-port/` (start at its
`README.md`) — the C++→Python porting rules, the frameworks (`application`,
`common`, `ktx`, `sb6m`, the vendored `window`), the Mesa + PyOpenGL traps,
headless verification, and the fidelity notes vs the C++ originals.
