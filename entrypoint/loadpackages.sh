#!/bin/env bash
# Prepare the mounted repo: build the C examples (incremental) and
# editable-install the Python port's `sb7` package so demos import it. Called by
# shell.sh on `make shell`. No `set -e`, so a C build failure still lets the
# shell open (the errors print; you debug inside).

export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate

# Build the C examples. The build tree /opt/sb7-build is OUTSIDE the mount: the
# baked image already configured + built it, so this is a fast incremental
# relink against the live source. CMake writes binaries into /opt/sb7-build/bin.
# CMAKE_POLICY_VERSION_MINIMUM=3.5 lets modern CMake accept sb7code's
# `cmake_minimum_required(VERSION 2.6)`.
cmake -S /sb7code -B /opt/sb7-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null
cmake --build /opt/sb7-build

# Editable-install the shared `sb7` package (deps pre-installed in the image).
( cd /sb7code/python && \
  uv pip install --no-deps --no-index --no-build-isolation -e . \
      --python "$(which python)" )
