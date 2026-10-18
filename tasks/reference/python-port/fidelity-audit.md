# Python port — fidelity against the C++ originals (L2)

Up: `README.md`. What is known about how faithfully each Python demo in `python/` reproduces its
C++ original — the `src/<name>/<name>.cpp` (+ the `sb7` framework in `lib/`/`include/` and the
shader files under `bin/media/shaders/`) that sits beside it in this repository. This doc is
deliberately **modest**: a full per-demo pixel audit against the C++ has not been run in this
repo, so it records the verification status, the one pending step, and the per-demo areas most
likely to need a close look on a display. Anchors are file + symbol; line numbers rot.

## Verification status

- **Headless render sweep (from the port's development history):** all 77 demos put a plausible
  picture up under Xvfb. One demo, `compressrgtc`, crashed first and needed two fixes — a
  Latin-1 byte in its upstream compute shader (`mesa-and-pyopengl-traps.md` §9) and the
  C-signature `glCompressedTexImage2D` call (the raw PyOpenGL entry point drops the C++
  `imageSize` parameter). Both are fixed; it renders.
- **`make check`** imports all 77 demo modules against the referenced `bin/media/` without a
  display, so import-time and asset-resolution failures are caught on every run
  (`headless-verification.md`).
- **Media is complete in this repo.** The committed `bin/media/` holds the full upstream pack
  (33 `.ktx` files among the objects/shaders/textures), so the asset-loading demos resolve
  everything they ask for — including the envmap and distance-field demos. (A demo with inline
  GLSL and no assets needs no media at all.)

**What is NOT yet done:** an on-display, side-by-side correctness pass against the C++ build.
`RENDERED` means *worth a human look*, not *matches the C++* — a headless sweep cannot judge
colour, animation phase, or geometry correctness. That pass is the remaining open item and is
the maintainer's to run (it needs the display and GPU; `headless-verification.md` §4 is the
C++-vs-Python method). No task doc tracks it in this repo yet — when one is opened, record
per-demo verdicts there, and promote any durable finding back into this doc.

## Per-demo risk areas (inferred from the port, not yet pixel-verified)

These are where a cold reader should look first on the display, because the port either hits a
Mesa trap the C++ never needed or reworks something the C++ got from a driver quirk. Each is a
*risk to check*, not a known defect.

- **Compute / image-store / SSBO demos** — `csflocking`, `prefixsum`, `prefixsum2d`,
  `fragmentlist`, `dof`, `cullindirect`, `compressrgtc`. These depend on the `glMemoryBarrier`
  Mesa needs but proprietary drivers insert for free (`mesa-and-pyopengl-traps.md` §4) and, for
  `fragmentlist`, on sizing the head-pointer image to the framebuffer (§11). **`prefixsum` (1-D)
  carries a known-parked diagnosis from the port's history** — its output read empty on Mesa
  across several rounds of barrier/readback work; look here first and expect it may still be
  wrong.
- **The HDR chain** — `hdrbloom`, `hdrexposure`, `hdrtonemap`. A hand-written HDR shader once hid
  a 16-bit-float overflow (the reason the "shaders are the C++ build's" rule exists,
  `c-to-python-rules.md` §7); these now use the C++ shaders, but the float-format and exposure
  paths are the sensitive part. `hdrexposure` also writes uniform location 0
  (`mesa-and-pyopengl-traps.md` §5).
- **`deferredshading`** — swept "looks off" in the port's history; the G-buffer fill and the
  lighting resolve pass are worth a careful comparison.
- **`ktxview`** — writes uniform location 0 (`mesa-and-pyopengl-traps.md` §5) and exercises
  `ktx`'s format coverage, including the unsized-internal-format upgrade (§6).
- **`multiviewport`** — the only `glViewportIndexedf` user; imgui's state restore collapses the
  viewport array, so it re-issues its viewports each frame (`mesa-and-pyopengl-traps.md` §12).
  Check that all viewports draw from frame 2 onward.
- **Tessellation / geometry demos** — faithful only if the C++ shaders compile verbatim on Mesa;
  the `#version` bump and `flat`-on-integer injection (`mesa-and-pyopengl-traps.md` §8) are what
  make that work. A black or degenerate surface points at a shader the loader did not rewrite
  correctly.

## Deliberate deviations to keep (do not "fix" the port toward the C++)

- Core-profile **4.3 forward-compatible** context and the default VAO
  (`mesa-and-pyopengl-traps.md` §1).
- The loader's **`#version` bump and `flat` injection** (§8) and the **shader decode with
  `errors="replace"`** (§9) — mechanical, logic-preserving.
- The **`glMemoryBarrier`s** (§4) and **`fragmentlist`'s framebuffer-sized image** (§11) that
  Mesa needs and the C++ omits.
- **Escape closes the window**, and **Quit/Fullscreen are polled** rather than wired through the
  (imgui-clobbered) GLFW key callback (`c-to-python-rules.md` §3, `mesa-and-pyopengl-traps.md`
  §2).
- Every C++ keyboard control is an **imgui widget** in the Controls menu; the key, where one
  exists, is at most an accelerator (`c-to-python-rules.md` §3).

## How to use this doc

On the display pass, go demo by demo against `make run-c DEMO=<name>` vs `make run DEMO=<name>`,
start from the risk list above, and record each verdict (faithful / drift / gap / broken) with
a file+symbol anchor. The shaders are the C++ build's (`c-to-python-rules.md` §7), so a
picture difference is a Python-side bug (uniforms, texture formats/units, GL state, camera) or
a Mesa trap, not a shader to rewrite — re-verify a suspected inline shader by diffing it against
`src/<demo>/<demo>.cpp`, and capture a frame next to the C++ build
(`headless-verification.md` §4).
