#!/bin/bash
# The whole MuJoCo-as-real loop: capture -> SimFoundry -> refine -> pick in Isaac
# Lab -> deploy the same program in MuJoCo, plus the ground-truth (oracle) baseline.
#
# Usage: bash scripts/mujoco_loop.sh [OUT_DIR] [TARGET]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${ROOT}/outputs/mujoco_pick}"
TARGET="${2:-crayon}"
CAMERA="${CAMERA:-ext2}"  # the view SimFoundry reconstructs from
PY="${ROOT}/.venv/bin/python"
R2S2R="${ROOT}/.venv/bin/r2s2r"
export OMNI_KIT_ACCEPT_EULA=YES

"${R2S2R}" mujoco-capture --out "${OUT}/capture"
# The arm stands on the table and runs off the frame edge; SimFoundry's frame
# selection would count it as a clipped object.
"${R2S2R}" reconstruct "${OUT}/capture" --workdir "${OUT}/simfoundry" \
    --camera-role "${CAMERA}" --override s3_ground.frame_selection.max_clipped_frac=0.9
SCENE="$(ls -d "${OUT}"/simfoundry/*/scene)"
"${R2S2R}" refine "${SCENE}" --capture "${OUT}/capture" --capture-depth \
    --out "${OUT}/scene_refined"

# Oracle: ground-truth objects, to separate policy failures from reconstruction ones.
"${R2S2R}" mujoco-deploy --oracle --capture "${OUT}/capture" --target "${TARGET}" \
    --out "${OUT}/deploy_oracle"
# Isaac ignores SIGTERM and can hang on exit; bound it.
timeout -s KILL 900 "${PY}" "${ROOT}/scripts/run_policy_isaac.py" \
    "${OUT}/scene_refined" --target "${TARGET}" --out "${OUT}/isaac" --headless
"${R2S2R}" mujoco-deploy "${OUT}/scene_refined" --capture "${OUT}/capture" \
    --target "${TARGET}" --out "${OUT}/deploy"
