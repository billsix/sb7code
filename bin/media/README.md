# Media committed here (a deliberate decision)

This directory holds the OpenGL SuperBible 7th-edition media pack (objects,
shaders, textures) used by both the C++ examples and the Python port.

Upstream sb7code does **not** commit this media: its `.gitignore` excludes
`bin/`, and CMake downloads `superbible7-media.zip` from openglsuperbible.com at
build time (this directory previously held only a placeholder saying so).

It is committed here by a deliberate decision of the maintainer (William
Emerison Six <billsix@gmail.com>, 2026-10-18) so the repository is
self-contained for both halves — the Python port references this one copy
rather than duplicating it, and the C build no longer needs a network download.

**Caveat, recorded on purpose:** the SuperBible media pack is not this
project's to redistribute, so committing it may be a licensing concern. It is
tracked here anyway as a conscious choice, labelled rather than done silently.
To reverse: `git rm -r --cached bin/media`, restore the `bin/` line in the
top-level `.gitignore`, and rely on the CMake download again. Full rationale:
`tasks/pythonize-and-containerize-port.md`.
