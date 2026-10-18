"""simpleclear — clear the back buffer to red.

Port of ``src/simpleclear/simpleclear.cpp``. The smallest
possible v7 demo: no geometry, no shaders, just glClearBufferfv.
"""

from typing import override

import OpenGL.GL as GL

from sb7 import application


class SimpleclearApp(application.Application):
    def init(self) -> None:
        super().init()
        self.info.title = "OpenGL SuperBible - Simple Clear"

    @override
    def render(self, current_time: float) -> None:
        red: list[float] = [1.0, 0.0, 0.0, 1.0]
        GL.glClearBufferfv(GL.GL_COLOR, 0, red)


if __name__ == "__main__":
    application.run(SimpleclearApp)
