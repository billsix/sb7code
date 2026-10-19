"""Python port of the ``sb7`` framework from OpenGL SuperBible 7e.

The C++ side (``include/sb7.h``) defines an
``sb7::application`` base class with virtual ``init()`` / ``startup()`` /
``render(currentTime)`` / ``shutdown()`` and key/mouse callbacks; each
demo subclasses it and ``DECLARE_MAIN`` instantiates and runs it.

This module is the same shape in Python:

  - ``AppInfo``          — dataclass mirroring the C++ ``APPINFO`` struct
                           (title, window size, GL version, samples, flags)
  - ``Application``      — base class with overridable lifecycle methods
  - ``run(cls)``         — entry point; the equivalent of ``DECLARE_MAIN(cls)``

Internally it uses **glfw** for windowing and **imgui_bundle** for the
menubar (File→Quit, View→Fullscreen) — that's the mvp convention. The
KEY/MOUSE events are forwarded as integer-args methods (no GLFW types
leak past the framework) so subclasses don't import glfw just to read
key codes; the constants live as ``Application.KEY_*`` for convenience.

A typical demo reads as::

    class my_demo(application.Application):
        def init(self):
            super().init()
            self.info.title = "OpenGL SuperBible - My Demo"

        def startup(self):
            self.program = common.compile_program_from_files(
                here("my_demo.vs.glsl"), here("my_demo.fs.glsl"))
            self.vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(self.vao)

        def render(self, current_time):
            GL.glClearBufferfv(GL.GL_COLOR, 0, [0, 0.25, 0, 1])
            GL.glUseProgram(self.program)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

        def shutdown(self):
            GL.glDeleteProgram(self.program)
            GL.glDeleteVertexArrays(1, [self.vao])

    if __name__ == "__main__":
        application.run(my_demo)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Work around the Mesa/Wayland PyOpenGL+imgui_bundle crash. PyOpenGL
# keeps a contextdata table to extend the lifetime of buffers passed
# to ``glVertexAttribPointer`` etc.; its lookup calls
# ``getContext`` → ``OpenGL.platform.GetCurrentContext()`` which can
# return 0/None on Mesa/Wayland even when GLFW has made the context
# current. The lookup then raises "Attempt to retrieve context when
# no valid context" inside imgui_bundle's renderer init.
#
# Earlier versions of this file set ``OpenGL.STORE_POINTERS = False``
# to skip the retention bookkeeping entirely. A newer PyOpenGL
# release ignores that flag (with a warning) unless
# ``ERROR_ON_COPY = True`` is also set — but ERROR_ON_COPY would
# then fail every ``glUniform3fv(loc, 1, [a,b,c])`` call. The robust
# fix is to monkey-patch ``contextdata.getContext`` to return a
# constant non-None bucket id, making per-context bookkeeping
# degenerate to one global table per process. Buffers are also kept
# alive via instance attributes so this loses nothing.
# Must run BEFORE importing OpenGL.GL.
import OpenGL.contextdata as _contextdata  # noqa: E402


def _stable_get_context(context: int | None = None) -> int:
    return context if context is not None else 1


# Deliberate monkeypatch of a PyOpenGL internal (see comment above);
# call-compatible with the untyped original, but ty compares the
# function objects' declared types and rejects the swap.
_contextdata.getContext = _stable_get_context  # ty: ignore[invalid-assignment]

import glfw  # noqa: E402
import OpenGL.GL as GL  # noqa: E402
from imgui_bundle import imgui  # noqa: E402

from . import window  # noqa: E402  (after the GL-setup patch above)

# The vendored window/menubar/camera helpers are imported at the top
# (`from . import window`), after the GL-setup patch above.


@dataclass
class AppFlags:
    """Mirrors the bit-flag union ``APPINFO.flags`` in ``sb7.h``.

    GLFW exposes most of these as window hints; we plumb only the ones
    demos actually toggle (debug, vsync, samples, fullscreen, cursor).
    ``stereo`` and ``robust`` are kept as fields for parity with the C++
    side but are no-ops for now — no demo we've planned uses them.
    """

    fullscreen: bool = False
    vsync: bool = True
    cursor: bool = True
    stereo: bool = False
    debug: bool = False
    robust: bool = False


@dataclass
class AppInfo:
    """Mirrors the ``APPINFO`` struct in ``sb7.h`` (title, size, GL
    version, MSAA samples, flags). Default GL is 4.3 core, matching v7's
    default. Demos override this in ``init()`` if they want a different
    version (e.g. simpletri uses ``#version 420 core``)."""

    title: str = "OpenGL SuperBible Example"
    window_width: int = 800
    window_height: int = 600
    major_version: int = 4
    minor_version: int = 3
    samples: int = 0
    flags: AppFlags = field(default_factory=AppFlags)


class Application:
    """Base class for SuperBible 7e Python demos.

    Override the lifecycle hooks (``init``, ``startup``, ``render``,
    ``shutdown``) and the input callbacks (``on_key``, ``on_resize``,
    ``on_mouse_button``, ``on_mouse_move``, ``on_mouse_wheel``) the same
    way you'd override the virtuals on ``sb7::application``.

    The instance attributes available inside those methods:

      - ``self.info``      — the ``AppInfo`` (mutable in ``init()``)
      - ``self.window``    — the GLFW window handle (after ``startup()``)
      - ``self.imgui_impl`` — the imgui_bundle GlfwRenderer
      - ``self.win_state`` — ``common.WindowState`` for the menubar
    """

    # Re-export GLFW key codes as class attributes so demos that
    # override ``on_key`` don't have to import glfw themselves. Add more
    # aliases as ports need them.
    KEY_ESCAPE = glfw.KEY_ESCAPE
    KEY_SPACE = glfw.KEY_SPACE
    KEY_ENTER = glfw.KEY_ENTER
    KEY_LEFT = glfw.KEY_LEFT
    KEY_RIGHT = glfw.KEY_RIGHT
    KEY_UP = glfw.KEY_UP
    KEY_DOWN = glfw.KEY_DOWN
    PRESS = glfw.PRESS
    RELEASE = glfw.RELEASE
    REPEAT = glfw.REPEAT

    def __init__(self) -> None:
        self.info = AppInfo()
        # Any is the sanctioned type for glfw window handles: the handle
        # is None until run() creates the window, and the glfw stubs'
        # `_Pointer[_GLFWwindow]` union would otherwise poison every
        # glfw call made outside run()'s narrowing scope.
        self.window: Any = None
        self.imgui_impl = None
        self.win_state = window.WindowState()

    # -- Lifecycle hooks (override these) -----------------------------

    def init(self) -> None:
        """Called once before window creation. Set ``self.info`` here.

        Default: pick a sensible default window size from the primary
        monitor (1920x1080 capped, mvp convention) so demos that don't
        override ``init()`` still get a properly-sized window."""
        w: int
        h: int
        w, h = window.resolve_default_window_size()
        self.info.window_width = w
        self.info.window_height = h

    def startup(self) -> None:
        """Called once after the GL context is current. Load shaders,
        create VAOs/VBOs, allocate textures here."""

    def render(self, current_time: float) -> None:
        """Called every frame. ``current_time`` is ``glfwGetTime()`` in
        seconds — same semantics as the C++ ``render(double)``."""

    def shutdown(self) -> None:
        """Called once after the main loop exits. Free GL resources."""

    # -- Controls menu (rules §3: every C++ control is a widget) --------

    def draw_controls(self) -> None:
        """The body of the menubar's Controls menu: the demo's widgets
        (checkboxes, sliders, radio buttons, a "Reload shaders" button).
        Override it; ``run`` adds a Controls menu only when a demo does.
        imgui runs a menu body only while the menu is open, so apply a
        widget's change on the spot or store it where ``render`` reads
        it, never rely on this running every frame. ``MVP_CONTROLS_WINDOW``
        (the gates' hook) also draws it in a plain window every frame."""

    def _controls_hook(self) -> Optional[Callable[[], None]]:
        if type(self).draw_controls is Application.draw_controls:
            return None
        return self.draw_controls

    # -- Input callbacks (override these) -----------------------------

    def on_resize(self, width: int, height: int) -> None:
        """``width``/``height`` are FRAMEBUFFER PIXELS, not screen units.
        On HiDPI/fractional-scaling displays the two differ; the viewport
        (and any gl_FragCoord-sized FBO/image target) must use pixels or
        the scene renders into the bottom-left fraction of the window
        with black bars top and right (seen 2026-07-30; see
        tasks/v7-framework-hidpi-viewport.md)."""
        self.info.window_width = width
        self.info.window_height = height
        GL.glViewport(0, 0, max(1, width), max(1, height))

    def on_key(self, key: int, action: int) -> None:
        """``key`` is a GLFW key code (use ``Application.KEY_*``).
        ``action`` is ``PRESS`` / ``RELEASE`` / ``REPEAT``."""

    def on_mouse_button(self, button: int, action: int) -> None:
        pass

    def on_mouse_move(self, x: int, y: int) -> None:
        pass

    def on_mouse_wheel(self, pos: int) -> None:
        pass

    # -- Helpers a demo can call --------------------------------------

    def set_window_title(self, title: str) -> None:
        if self.window is not None:
            glfw.set_window_title(self.window, title)

    def set_vsync(self, enable: bool) -> None:
        self.info.flags.vsync = enable
        glfw.swap_interval(1 if enable else 0)

    # -- Main loop (don't override unless you know why) ---------------

    def run(self) -> None:
        if not glfw.init():
            print("Failed to initialize GLFW", file=sys.stderr)  # noqa: T201
            return
        try:
            self.init()

            glfw.window_hint(
                glfw.CONTEXT_VERSION_MAJOR, self.info.major_version
            )
            glfw.window_hint(
                glfw.CONTEXT_VERSION_MINOR, self.info.minor_version
            )
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            # The glfw stubs demand `int` hint values; GL_TRUE/GL_FALSE
            # are PyOpenGL `Constant`s — int subclasses at runtime that
            # the stubs don't credit as such (next three suppressions).
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)  # ty: ignore[invalid-argument-type]
            glfw.window_hint(glfw.SAMPLES, self.info.samples)
            glfw.window_hint(
                glfw.STEREO,
                GL.GL_TRUE if self.info.flags.stereo else GL.GL_FALSE,  # ty: ignore[invalid-argument-type]
            )
            if self.info.flags.debug:
                glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, GL.GL_TRUE)  # ty: ignore[invalid-argument-type]

            monitor: Any = (
                glfw.get_primary_monitor()
                if self.info.flags.fullscreen
                else None
            )
            self.window = glfw.create_window(
                self.info.window_width,
                self.info.window_height,
                self.info.title,
                monitor,
                None,
            )
            if not self.window:
                print("Failed to open window", file=sys.stderr)  # noqa: T201
                return

            glfw.make_context_current(self.window)
            glfw.swap_interval(1 if self.info.flags.vsync else 0)
            if not self.info.flags.cursor:
                glfw.set_input_mode(
                    self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN
                )

            self._wire_callbacks()

            # Issue a PyOpenGL call BEFORE init_imgui. Some platforms
            # (Linux with PyOpenGL 3.x in particular) need PyOpenGL to
            # see at least one GL call after glfw.make_context_current
            # before its context-detection works. Without this,
            # imgui_bundle's GlfwRenderer init crashes inside
            # ``glVertexAttribPointer`` with "Attempt to retrieve
            # context when no valid context".
            print(f"VENDOR:   {GL.glGetString(GL.GL_VENDOR).decode()}")  # noqa: T201
            print(f"VERSION:  {GL.glGetString(GL.GL_VERSION).decode()}")  # noqa: T201
            print(f"RENDERER: {GL.glGetString(GL.GL_RENDERER).decode()}")  # noqa: T201

            self.imgui_impl = window.init_imgui(self.window)

            self.startup()

            # Seed the viewport from the real framebuffer size. GL's
            # default viewport matches the framebuffer at context
            # creation, but under Wayland fractional scaling the
            # compositor can rescale immediately after — and demos'
            # on_resize overrides need their aspect/proj set at least
            # once before the first frame.
            self.on_resize(*glfw.get_framebuffer_size(self.window))

            while not glfw.window_should_close(self.window):
                self.imgui_impl.process_inputs()
                imgui.new_frame()
                window.draw_menubar(
                    self.window, self.win_state, controls=self._controls_hook()
                )

                self.render(glfw.get_time())

                # Reset the bound program before imgui's GL backend
                # snapshots GL state. Without this, a "Reload shaders"
                # button that deletes the currently-bound program
                # leaves the current-program id pointing at a freed
                # name; imgui saves it, draws its own UI, then crashes
                # in glUseProgram(stale_id) trying to restore. Binding
                # 0 makes the save/restore a no-op.
                GL.glUseProgram(0)

                imgui.render()
                self.imgui_impl.render(imgui.get_draw_data())
                glfw.swap_buffers(self.window)
                glfw.poll_events()

                if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    glfw.set_window_should_close(self.window, True)

            self.shutdown()
        finally:
            if self.imgui_impl is not None:
                self.imgui_impl.shutdown()
            if self.window is not None:
                glfw.destroy_window(self.window)
            glfw.terminate()

    # -- Internal -----------------------------------------------------

    def _wire_callbacks(self) -> None:
        def _resize(_w: Any, w: int, h: int) -> None:
            self.on_resize(w, h)

        def _key(_w: Any, key: int, _sc: int, action: int, _mods: int) -> None:
            self.on_key(key, action)

        def _mb(_w: Any, button: int, action: int, _mods: int) -> None:
            self.on_mouse_button(button, action)

        def _move(_w: Any, x: float, y: float) -> None:
            self.on_mouse_move(int(x), int(y))

        def _wheel(_w: Any, _xo: float, yo: float) -> None:
            self.on_mouse_wheel(int(yo))

        # Framebuffer-size callback, NOT window-size: the window-size
        # callback delivers screen coordinates, which under HiDPI are
        # smaller than the framebuffer — glViewport fed with them shows
        # the scene in the bottom-left fraction of the window. The other
        # port framework (_vapp.py) already polls get_framebuffer_size for
        # the same reason.
        glfw.set_framebuffer_size_callback(self.window, _resize)
        glfw.set_key_callback(self.window, _key)
        glfw.set_mouse_button_callback(self.window, _mb)
        glfw.set_cursor_pos_callback(self.window, _move)
        glfw.set_scroll_callback(self.window, _wheel)


def run(cls: type[Application]) -> None:
    """Entry point. Instantiate ``cls``, call ``run()``, dispose.

    Equivalent of the C++ ``DECLARE_MAIN(cls)`` macro::

        if __name__ == "__main__":
            application.run(my_demo)
    """
    cls().run()
