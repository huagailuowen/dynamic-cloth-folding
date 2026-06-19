#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export PATH="$REPO_ROOT/.conda/bin:$PATH"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/x86_64-linux-gnu:$REPO_ROOT/.conda/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="/usr/include/eigen3:${CPLUS_INCLUDE_PATH:-}"

# The EGL-backed mujoco-py extension segfaults on this host/driver stack.
# CPU/OSMesa rendering is slower, but it is reliable for reproduction and demo export.
export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"
