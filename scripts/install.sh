#!/bin/bash
# Install r2s2r with Isaac Sim 5.1 and a pinned Isaac Lab 2.3.x into ./.venv.
#
# The core package (capture loading, reconstruction adapters, scene specs) needs
# none of this; only the Isaac Lab scene builder does. SimFoundry keeps its own
# conda envs (see scripts/link_simfoundry_resources.sh).
#
# Usage: bash scripts/install.sh            # R2S2R_ISAACLAB_REF overrides the tag
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAACLAB_REF="${R2S2R_ISAACLAB_REF:-v2.3.2}"
ISAACLAB_ROOT="${ROOT}/_isaaclab/IsaacLab"
PYTHON="${ROOT}/.venv/bin/python"
NVIDIA_INDEX="https://pypi.nvidia.com"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

cd "${ROOT}"
[ -x "${PYTHON}" ] || uv venv --python 3.11 .venv

echo "[INFO] Installing r2s2r"
uv pip install --python "${PYTHON}" -e ".[develop]"

echo "[INFO] Installing torch 2.7.0 (cu128) and Isaac Sim 5.1.0"
uv pip install --python "${PYTHON}" "torch==2.7.0" "torchvision==0.22.0" \
    --index-url "${TORCH_INDEX}"
uv pip install --python "${PYTHON}" "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url "${NVIDIA_INDEX}"

if [ ! -d "${ISAACLAB_ROOT}/.git" ]; then
    echo "[INFO] Cloning Isaac Lab ${ISAACLAB_REF}"
    mkdir -p "$(dirname "${ISAACLAB_ROOT}")"
    git clone --depth 1 --branch "${ISAACLAB_REF}" \
        https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_ROOT}"
fi

echo "[INFO] Installing Isaac Lab packages (editable)"
for pkg in isaaclab isaaclab_assets; do
    uv pip install --python "${PYTHON}" -e "${ISAACLAB_ROOT}/source/${pkg}" \
        --extra-index-url "${NVIDIA_INDEX}" --index-strategy unsafe-best-match
done

echo "[INFO] Done. Run Isaac with OMNI_KIT_ACCEPT_EULA=YES."
