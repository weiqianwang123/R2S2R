#!/bin/bash
# Fetch what the MuJoCo "real" world needs into ~/.cache/r2s2r (R2S2R_CACHE):
# the Franka Panda and DROID's Robotiq 2F-85 from mujoco_menagerie, and a few Google
# Scanned Objects (kevinzakka/mujoco_scanned_objects). Sparse, shallow clones.
#
# Usage: bash scripts/setup/fetch_mujoco_assets.sh [MODEL_NAME...]
set -euo pipefail

CACHE="${R2S2R_CACHE:-$HOME/.cache/r2s2r}"
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    # The objects of r2s2r.real.mujoco.world.default_objects().
    MODELS=(Crayola_Crayons_24_count Cole_Hardware_Mug_Classic_Blue Android_Figure_Orange)
fi
mkdir -p "${CACHE}"

sparse_clone() {  # url dir paths...
    local url="$1" dir="$2"
    shift 2
    if [ ! -d "${dir}/.git" ]; then
        git clone --depth 1 --filter=blob:none --sparse "${url}" "${dir}"
    fi
    git -C "${dir}" sparse-checkout add "$@"
}

sparse_clone https://github.com/google-deepmind/mujoco_menagerie.git \
    "${CACHE}/mujoco_menagerie" franka_emika_panda robotiq_2f85 robotiq_2f85
paths=()
for m in "${MODELS[@]}"; do paths+=("models/${m}"); done
sparse_clone https://github.com/kevinzakka/mujoco_scanned_objects.git \
    "${CACHE}/gso" "${paths[@]}"
echo "MuJoCo assets in ${CACHE}"
