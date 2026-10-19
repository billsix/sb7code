OpenGL SuperBible 7th Edition Source Code
=========================================

This is the repository for the example source code for the 7th edition of the
OpenGL SuperBible. Please read HOWTOBUILD.txt. It tells you how to build and
run the examples.

This fork also carries a **Python port** of the samples (`python/`) and a
container that builds and runs **both** halves. See "Container" and "Python
port" below.

Container (C + Python)
----------------------

One image builds the C++ examples and carries the Python port; `make shell`
drops you at the repo root with both display backends. From the repo root:

```
make image                  # build the image (compiles the C++ examples)
make run-c DEMO=blinnphong   # run a C++ example on your display (X11)
make run   DEMO=blinnphong   # run the Python port of it (Wayland)
make check                   # import every Python demo, no display
make type-check              # ty + annotate-every-binding checker (no display)
make format                  # ruff check --fix + ruff format the port
make list                    # list the demo directories
make shell                   # interactive shell; then: run-c <dir> / run <dir>
```

> The C++ examples use gl3w, which resolves GL through GLX, so they run on
> X11/XWayland (`run-c` unsets `WAYLAND_DISPLAY`); the Python port uses PyOpenGL
> (EGL-capable) and runs on Wayland. The C build tree lives at `/opt/sb7-build`,
> outside the mount. Details: `AGENTS.md`.

Python port
-----------

`python/` holds a behaviour-faithful Python translation of 77 of the samples
(PyOpenGL + GLFW + imgui_bundle). It is a flat set of scripts run by path
(`python blinnphong/blinnphong.py`), not a pip package, and it reads the media
from this repo's own `bin/media/` rather than carrying a copy. See
`python/README.md` to run it, and `tasks/reference/python-port/` for the port's
cold-start knowledge.

> **Media note:** `bin/media/` is committed in this repo (a deliberate decision
> that departs from sb7code's external-download default) — see
> `bin/media/README.md`.