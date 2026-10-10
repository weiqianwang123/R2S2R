#!/bin/bash
# Create the conda env "newton" (r2s2r.paths.ENV_NEWTON) with Newton (newton-physics, on
# NVIDIA Warp), where cloths settle and move beside MuJoCo (scripts/tools/cloth_job.py,
# r2s2r.sim.cloth): its VBD solver, on the GPU when there is one. Its own env: Newton follows Warp's releases,
# while Isaac Lab, in .venv, keeps the Warp it was installed with.
#
# Usage: bash scripts/setup/install_newton.sh
set -euo pipefail

ENV=newton
NEWTON_VERSION=1.6.1
MAMBA="$(command -v mamba || command -v conda || echo "${HOME}/miniforge3/bin/mamba")"
if ! "${MAMBA}" env list | awk '{print $1}' | grep -qx "${ENV}"; then
    "${MAMBA}" create -y -n "${ENV}" python=3.11
fi
"${MAMBA}" run -n "${ENV}" python -m pip install --quiet "newton==${NEWTON_VERSION}"
"${MAMBA}" run -n "${ENV}" python -c \
    "import newton, warp; print('newton', newton.__version__, 'warp', warp.__version__)"
