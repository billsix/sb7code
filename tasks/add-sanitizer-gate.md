# Add an ASan + UBSan(trap) build gate

**Status:** proposed — needs go-ahead
**Created:** 2026-06-16

## Goal

Mirror the sanitizer gate the sibling **spimulator** project just gained: build
this code under **UBSan trap mode** (`-fsanitize=undefined
-fsanitize-trap=undefined` — any undefined behavior aborts with SIGILL, the
pass/fail gate) and under **AddressSanitizer** (`-fsanitize=address` — buffer
overflows, use-after-free, etc.), exercise something under each, and **fail the
build on any UB or memory error**. For C++, UBSan additionally checks `vptr`
(bad virtual dispatch / wrong-type downcast) and array bounds — relevant here
because the sb7 framework uses a virtual `sb7::application` base subclassed by
every demo.

Primer on what ASan/UBSan are, why trap mode is the reliable gate (diagnostic
UBSan *under-reports*), and the integer-UB patterns to expect:
`/billopt/spimulator/tasks/archive/2026/06/16/ubsan-sweep.md`. How spimulator
wired it in (a `RUN_SANITIZERS` arg in the Dockerfile that does a second
instrumented build + test run, gated by the Makefile): search `RUN_SANITIZERS`
in `/billopt/spimulator/Dockerfile` and `/billopt/spimulator/Makefile`.

## What this project is (relevant constraints)

- **OpenGL SuperBible, 7th ed.** example source (`superbible7`). **CMake**
  (`CMakeLists.txt`, `cmake_minimum_required(2.6)` — very old; uses
  `LINK_DIRECTORIES`, `${OPENGL_LIBRARIES}`). C++ (`-std=c++0x` on Linux). No
  Meson, **no Dockerfile/Makefile container template, no `CLAUDE.md`** — a bare
  upstream tree.
- Layout: a static **`sb7`** framework lib (`src/sb7/`: `sb7.cpp`, `sb7ktx`,
  `sb7object`, `sb7shader`, `sb7color`, `sb7textoverlay`, `gl3w.c`) plus ~80
  example executables, one per `src/<name>/<name>.cpp`, each linking `sb7`.
- **glfw** comes from **pkg-config** (`pkg_check_modules(GLFW REQUIRED glfw3)`)
  and links a pile of X libs; `extern/glfw-3.0.4` is also vendored. glfw is
  third-party and **must not be the gate's target**.
- **No test suite** (no `enable_testing`/`add_test`/ctest).
- Examples are **interactive GLFW windows** requiring **OpenGL 4.x**; render
  loop until closed. Demos load **media** (KTX textures etc.) from a
  `superbible7-media.zip` the CMake **downloads at configure time** — a smoke
  run needs that media present or it will fail at asset load.

## How to add the flags (CMake)

CMake here is old (2.6); `target_link_options` needs CMake ≥3.13, so either bump
the minimum or use `target_compile_options` + `LINK_FLAGS` properties. Scope to
**`sb7` + the example targets only**, never glfw:

```cmake
option(ENABLE_SANITIZERS "Build sample code under ASan/UBSan" OFF)
# (with a modern enough CMake)
add_library(sb7_sanitizers INTERFACE)
if(ENABLE_SANITIZERS)
  target_compile_options(sb7_sanitizers INTERFACE
    -fsanitize=address,undefined -fsanitize-trap=undefined
    -fno-omit-frame-pointer -g)
  target_link_options(sb7_sanitizers INTERFACE
    -fsanitize=address,undefined -fsanitize-trap=undefined)
endif()
# target_link_libraries(sb7 PUBLIC sb7_sanitizers)
# in the example foreach: target_link_libraries(${EXAMPLE} sb7_sanitizers)
```

Do **not** add the flags to the global `CMAKE_CXX_FLAGS`/`CMAKE_C_FLAGS` (the
file already appends OpenMP and `-D_LINUX -std=c++0x` there) — that would also
instrument the vendored glfw if it ever builds in-tree.

**Default LSan off** for these leaky interactive demos. Add the standard
ASan-guarded weak hook to one framework TU (e.g. `src/sb7/sb7.cpp`):

```cpp
#if defined(__SANITIZE_ADDRESS__) || \
    (defined(__has_feature) && __has_feature(address_sanitizer))
extern "C" const char *__asan_default_options(void) { return "detect_leaks=0"; }
#endif
```

(spimulator does the same in `spim.c`.) The gate is for corruption, not the
demos' intentional exit-time leaks. Note `gl3w.c` resolves GL function pointers
at runtime — UBSan `function`-pointer-type checks can be noisy across that
boundary; narrow sub-checks or suppress as needed.

## Caveats (be honest about these)

1. **Interactive GL demos needing a display, and OpenGL 4.x.** Each example
   blocks in a GLFW loop until closed and needs a 4.x context a headless
   software stack may not give. A realistic gate asserts
   **builds-clean-under-ASan/UBSan** plus, where a context can be created, a
   **headless render-N-frames-then-exit** smoke run (Xvfb/EGL).
2. **Media dependency.** Demos load assets from the downloaded
   `superbible7-media.zip`; a smoke run must have that media unpacked or it
   fails at load (unrelated to any sanitizer finding). The zip download itself
   is a configure-time network dependency to account for.
3. **Largely third-party/upstream code.** The `sb7` framework and every example
   are imported book sample code; glfw is fully third-party. Findings are mostly
   *upstream's*. Scope the gate to first-party targets; keep
   `ASAN_OPTIONS`/`UBSAN_OPTIONS` suppressions for noise from linked
   vendored/driver code.
4. **No test suite.** A smoke harness (headless few-frames-then-exit for a
   representative subset, with media staged) is a **prerequisite** — it's what
   makes the *dynamic* sanitizers worthwhile, and it's the bulk of the work.
5. **No container template / old CMake.** Unlike spimulator there's no `make
   image`/CI to hang the second instrumented build on, and the CMake minimum may
   need bumping for `target_link_options`. You'd add scaffold or document a
   manual `cmake -DENABLE_SANITIZERS=ON` + Xvfb recipe.

## Priority / feasibility verdict

**Lower priority than apue/spimulator.** Those are non-interactive, test-backed,
mostly first-party — sanitizing the existing suite is high-signal and nearly
free. Here there's no test suite, no container gate, ~80 display-bound demos that
need a 4.x context *and* downloaded media, and the code is imported book samples
whose findings we'd mostly punt upstream. Flag wiring is small (modulo the old
CMake-minimum bump); the **headless-context + media-staged smoke harness is the
real cost**, with thin payoff. Treat as a *nice-to-have*. If pursued, do the
cheap, high-certainty half first — add `ENABLE_SANITIZERS`, confirm `sb7` + the
examples **compile and link clean** under both sanitizers — and defer any
dynamic run-frames gate until a working headless harness (with media) exists.
