FROM registry.fedoraproject.org/fedora:44

COPY entrypoint/*.sh /usr/local/bin/
COPY entrypoint/entrypoint.sh /
COPY python/requirements.txt /requirements.txt

RUN  --mount=type=cache,target=/var/cache/libdnf5 \
     --mount=type=cache,target=/var/lib/dnf \
     echo "keepcache=True" >> /etc/dnf/dnf.conf && \
     dnf upgrade -y && \
     dnf install -y \
                   gcc \
                   gcc-c++ \
                   make \
                   cmake \
                   ninja-build \
                   git \
                   pkgconf-pkg-config \
                   glfw \
                   glfw-devel \
                   python3-glfw \
                   python3-numpy \
                   python3-pyopengl \
                   python3-pytest \
                   python3-virtualenv \
                   ruff \
                   uv \
                   tmux \
                   which \
                   libatomic \
                   libgomp \
                   mesa-dri-drivers \
                   mesa-libGL-devel \
                   mesa-libEGL-devel \
                   mesa-libGLU-devel \
                   libX11-devel \
                   libXrandr-devel \
                   libXinerama-devel \
                   libXi-devel \
                   libXcursor-devel \
                   libXxf86vm-devel \
                   libwayland-client \
                   libwayland-cursor \
                   libwayland-egl \
                   libxkbcommon \
                   libdecor ; \
    python3 -m venv --system-site-packages /venv/ && \
    export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
    source /venv/bin/activate && \
    uv pip install setuptools ty && \
    uv pip install --python "$(which python)" -r /requirements.txt && \
    rm /requirements.txt

# Bake the C examples from the source so a built image is validated at build
# time and a pulled image is runnable standalone. `make shell` bind-mounts the
# live repo over /sb7code, which SHADOWS this baked copy; the entrypoint then
# re-builds the C (incremental) against the live mount. The Python port's shared
# `sb7` package is editable-installed (-e) so the demo scripts import it
# (`from sb7 import ...`) with no sys.path hacks; demos still run by path
# (python <name>.py) and read media from the sibling bin/media/.
#
# The build TREE lives at /opt/sb7-build, OUTSIDE the /sb7code mount, so the
# host's own build cache cannot collide with it. The examples load assets by the
# cwd-relative path "media/...", so they are launched with cwd set to the repo's
# bin/ (where the committed bin/media/ lives). CMAKE_POLICY_VERSION_MINIMUM=3.5
# lets modern CMake accept sb7code's `cmake_minimum_required(VERSION 2.6)`.
COPY . /sb7code
RUN cmake -S /sb7code -B /opt/sb7-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
    cmake --build /opt/sb7-build && \
    export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
    source /venv/bin/activate && \
    cd /sb7code/python && \
    uv pip install --no-deps --no-index --no-build-isolation -e . \
        --python "$(which python)"

ENTRYPOINT ["/entrypoint.sh"]
