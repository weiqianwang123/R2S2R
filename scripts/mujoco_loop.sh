#!/bin/bash
# The whole MuJoCo-as-real loop: capture -> SimFoundry -> refine -> calibrate joints
# against the video -> pick in Isaac Lab -> deploy the same program in MuJoCo, plus the
# ground-truth (oracle) baseline.
#
# Usage: CAMERAS="ext2 wrist" WORLD=cabinet bash scripts/mujoco_loop.sh [OUT_DIR]
#   CAMERAS: the cameras to record and reconstruct from (default: ext1 ext2 wrist).
#   WORLD:   the MuJoCo world preset, pick (default) or cabinet (articulated).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${ROOT}/outputs/mujoco_pick}"
read -r -a CAMERAS <<< "${CAMERAS:-ext1 ext2 wrist}"
PY="${ROOT}/.venv/bin/python"
R2S2R="${ROOT}/.venv/bin/r2s2r"
export OMNI_KIT_ACCEPT_EULA=YES

"${R2S2R}" mujoco-capture --out "${OUT}/capture" --cameras "${CAMERAS[@]}" \
    --world "${WORLD:-pick}"
"${R2S2R}" reconstruct "${OUT}/capture" --workdir "${OUT}/simfoundry"
SCENE="$(ls -d "${OUT}"/simfoundry/*/scene)"
"${R2S2R}" refine "${SCENE}" --capture "${OUT}/capture" --capture-depth \
    --out "${OUT}/scene_refined"
# Joints fitted to the demonstration (the cabinet world); a Codex agent remodels any
# that fail the check. A no-op without articulated objects.
"${R2S2R}" calibrate-joints "${OUT}/scene_refined" --capture "${OUT}/capture" \
    --out "${OUT}/scene_fitted"
"${R2S2R}" mujoco-eval "${OUT}/scene_fitted" --capture "${OUT}/capture"
# Which reconstructed object the task is about: the ground truth stands in for the
# language grounding an agent would do. The policy only sees the reconstruction.
TARGET="$("${R2S2R}" mujoco-eval "${OUT}/scene_fitted" --capture "${OUT}/capture" \
    --match-target 2>/dev/null | tail -1)"

# Oracle: ground-truth objects, to separate policy failures from reconstruction ones.
"${R2S2R}" mujoco-deploy --oracle --capture "${OUT}/capture" --out "${OUT}/deploy_oracle"
VIDEO=""
if [[ " ${CAMERAS[*]} " == *" ext1 "* ]]; then VIDEO=ext1; fi
# Isaac ignores SIGTERM and can hang on exit; bound it.
timeout -s KILL 900 "${PY}" "${ROOT}/scripts/isaaclab/run_pick.py" \
    "${OUT}/scene_fitted" --target "${TARGET}" --out "${OUT}/isaac" --headless \
    --video-camera "${VIDEO}"
"${R2S2R}" mujoco-deploy "${OUT}/scene_fitted" --capture "${OUT}/capture" \
    --target "${TARGET}" --out "${OUT}/deploy"
