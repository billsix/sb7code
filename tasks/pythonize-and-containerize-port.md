# Bring the Python port into this repo + containerize C and Python together

## BLUF

The Python port of these SuperBible 7e samples currently lives **inside the
`modelviewprojection` repo** (github.com/billsix/modelviewprojection) at
`ports/openglsuperbiblev7/`. Move it here so this repo is the single home for
both the C++ samples and their Python port, add a top-level container that
builds **both**, and delete the port from mvp. "Done" means: `python/` holds
the port, runnable in and out of the container; the C++ examples and the Python
demos both build/run from one `make` interface; the cold-start knowledge the
port depended on is harvested here; and mvp no longer mentions v7.

This mirrors the McNopper extraction (github.com/billsix/OpenGL) and the
Red Book 9e extraction (github.com/openglredbook... the local openglredbook9
repo). The one real departure is the **media**, below.

## Context (cold-start)

- **This repo** (sb7code): the OpenGL SuperBible 7th Edition example source
  (public upstream github.com/openglsuperbible/sb7code; this checkout's remote
  is a private mirror). CMake builds ~85 C++ examples (`src/<name>/<name>.cpp`)
  against the `sb7` static lib (`lib/`, `include/`) into `bin/` (RUN_DIR =
  `bin/`). Examples load assets by the **cwd-relative** path `"media/..."`, so
  they run from `bin/`.
- **The port** (now `python/`, originally mvp `ports/openglsuperbiblev7/`): 77
  demos (a subset of the 85; every one maps 1:1 to an `src/<name>/` here), one
  flat `python/<name>.py` each, importing the installed package
  (`from sb7 import common, application, ...`) + `application.run()`. The `sb7`
  package holds `common.py`, `application.py` (the `sb7::application` framework,
  glfw + imgui_bundle menubar), `ktx.py` (KTX textures), `sb6m.py` (`.sbm`
  meshes), `colors.py`, `vmath.py`, and `window.py`. Run model is a **local
  editable pip package** (`pip install -e python/`; maintainer's call).

### Same as the earlier extractions

1. **The port is not self-contained.** `common.py` AND `application.py` load
   `ports/openglsuperbiblev4/_common.py` from mvp at import (for `WindowState`
   / `draw_menubar` / `Camera` / `init_imgui` / `toggle_fullscreen`). That v4
   module imports only stdlib + glfw + OpenGL + imgui_bundle, so it is
   vendorable. **Fix:** copy it here as `python/sb7/window.py` and repoint both
   loaders. v4 stays in mvp untouched; this repo gets a frozen copy.
2. **Escape/menubar need no fix** — `application.py` already chains the imgui
   callbacks and draws the menubar (File→Quit / View→Fullscreen / Controls).

### The one real departure — MEDIA (decision below)

Unlike the Red Book repo (which shipped its `bin/media`), **sb7code does NOT
commit its media**: `.gitignore` excludes all of `bin/`, and CMake downloads
`superbible7-media.zip` from openglsuperbible.com at build time (the tracked
`bin/media/!OBTAIN_THE_MEDIA_PACK...txt` placeholder documents this). The mvp
port, by contrast, committed the 257 MB pack (objects/shaders/textures;
verified byte-identical to this repo's local `bin/media`).

## Plan (phases; each a staging boundary)

1. **This task doc.**
2. **Port in + self-contained.** Copy mvp `ports/openglsuperbiblev7/` →
   `python/` (minus `bin/media`); vendor v4 `_common.py` as `python/sb7/window.py`
   and repoint `common.py`/`application.py`; point the media resolver at
   `../bin/media`; add `python/requirements.txt` (PyOpenGL, glfw, imgui-bundle,
   numpy), a `pyproject.toml` (`[project]`/`[build-system]` for the editable
   install, plus ruff) and `.gitignore`; strip leaked sandbox paths
   (`/superbiblev7`, `/mvp`) from committed code.
3. **Media — commit it here (see Decisions).** Track `bin/media`
   (objects/shaders/textures), overriding the `bin/` gitignore, labelled as a
   deliberate decision. The port references `../bin/media`; the mvp copy is
   deleted.
4. **Container at top level.** `Dockerfile`/`Makefile`/`entrypoint/` building
   the C (CMake, build tree OUTSIDE the mount) + the Python deps; both X11 and
   Wayland. Editable install (`pip install -e python/`).
5. **Cold-start docs + tooling.** Harvest mvp's (now v7-focused)
   `tasks/reference/ports/*` into `tasks/reference/python-port/`.
6. **Remove from mvp.** `git rm` the port; move the v7-centric shared docs and
   the `moreports-*` verification tasks here or retire them; re-point
   `keywordify` (→ v4), `headless_sweep`, `format.sh`, `pyproject.toml`,
   CLAUDE.md. (After this, mvp's `ports/` holds only v4.)
7. **Gates + sweep + stage.**

## Decisions

- **A local editable pip package** (`pip install -e python/`): the `sb7`
  package holds the shared modules, the demos are flat scripts that import it.
- **Commit the media into this repo** (William Emerison Six
  <billsix@gmail.com>, 2026-10-18 — the maintainer's explicit, deliberate
  decision). This **departs from sb7code's design**, which treats the media as
  an external, separately-licensed download from openglsuperbible.com (hence
  the `bin/` gitignore + the CMake download + the placeholder). **Caveat,
  recorded on purpose:** the SuperBible 7e media pack is not this project's to
  redistribute, so committing it may be a licensing concern; the maintainer
  chose to commit it anyway (it was already committed in mvp) for a
  self-contained repo, and this note labels that choice rather than making it
  silently. To reverse: `git rm -r --cached bin/media`, restore the `bin/`
  gitignore, and rely on the CMake download again.
- **v4 `_common.py` vendored** as `python/sb7/window.py` (frozen copy; mvp is the
  source of truth).
- **McNopper (github.com/billsix/OpenGL) and the Red Book 9e repo are left
  as-is.**

## Status

**Priority:** 3  **Difficulty:** 6

Done and staged (2026-10-18, William Emerison Six <billsix@gmail.com>'s
session):

- Phases 2–5 ✓ — port at `python/` (77 demos, a local editable `sb7` package +
  flat demo scripts); v4 helpers vendored (see restructure note); media committed +
  labelled (`bin/media/README.md`, `.gitignore`, the CMakeLists download
  guard); container (`Dockerfile`/`Makefile`/`entrypoint/`) builds C + Python,
  both X11 (gl3w) + Wayland, build tree `/opt/sb7-build`,
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`; cold-start docs in
  `tasks/reference/python-port/`; README + AGENTS; `verify-port-visually.md`.
- Verified: `make image` → 82 C binaries + `libsb7.a`; `make check` →
  **77 / 77 demos import**.
- Phase 6 ✓ — the port + the (now-subsumed) morePorts doc/task footprint
  removed from mvp; `keywordify`/`headless_sweep`/`format.sh`/`pyproject`/
  CLAUDE.md re-pointed off v7. After this, mvp's `ports/` holds only v4.
- Package restructure ✓ — the shared modules moved into the `python/sb7/`
  package and dropped their leading `_` (`_sb7`→`application`, `_common`→
  `common`, the vendored helpers → `window.py`, and `ktx`/`sb6m`/`colors`/
  `vmath`); `python/` is a local editable pip package (`pip install -e
  python/`), and the demos are flat `python/<name>.py` that import it
  (`from sb7 import ...`) with no `sys.path` bootstrap.

Nothing is committed (the maintainer commits); this repo's additions are
staged.
