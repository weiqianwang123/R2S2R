#!/bin/bash
# Fetch Robotiq's own 2F-85 for Isaac Sim (robotiq/isaacsim_assets, CC-BY-4.0) into
# ~/.cache/r2s2r (R2S2R_CACHE), which r2s2r.robots.droid_franka mounts on the Franka in
# Isaac Lab: grippers/Robotiq_2F_85 at a pinned commit, a sparse clone with its meshes
# from Git LFS (needs git-lfs).
#
# Usage: bash scripts/setup/fetch_robotiq_isaac.sh
set -euo pipefail

CACHE="${R2S2R_CACHE:-$HOME/.cache/r2s2r}"
DIR="${CACHE}/robotiq_isaacsim_assets"
COMMIT=02caed3c9a5f60528adbf9a68ef9b97c7dc0c9b1  # 2026-10-01
mkdir -p "${CACHE}"
if [ ! -d "${DIR}/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --sparse \
        https://github.com/robotiq/isaacsim_assets.git "${DIR}"
fi
git -C "${DIR}" sparse-checkout set grippers/Robotiq_2F_85
GIT_LFS_SKIP_SMUDGE=1 git -C "${DIR}" -c advice.detachedHead=false checkout --quiet \
    "${COMMIT}"
git -C "${DIR}" lfs pull --include="grippers/Robotiq_2F_85/**"
echo "Robotiq's 2F-85 for Isaac in ${DIR}/grippers/Robotiq_2F_85"
