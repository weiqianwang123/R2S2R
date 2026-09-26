#!/bin/bash
# Demonstration mode, MuJoCo as the real world: the robot's calibrated cameras record
# the scene with its joint states -> SimFoundry -> refine -> pick in Isaac Lab ->
# deploy the same program in MuJoCo, plus the ground-truth (oracle) baseline.
#
# Usage: CAMERAS="wrist" bash scripts/mujoco_demo.sh [OUT_DIR]
#   CAMERAS: the cameras to record and reconstruct from (default: ext1 wrist).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${ROOT}/outputs/mujoco_demo}"
read -r -a CAMERAS <<< "${CAMERAS:-ext1 wrist}"
PY="${ROOT}/.venv/bin/python"
R2S2R="${ROOT}/.venv/bin/r2s2r"
export OMNI_KIT_ACCEPT_EULA=YES

"${R2S2R}" mujoco-capture --out "${OUT}/capture" --cameras "${CAMERAS[@]}"
"${R2S2R}" reconstruct "${OUT}/capture" --workdir "${OUT}/simfoundry"
SCENE="$(ls -d "${OUT}"/simfoundry/*/scene)"
"${R2S2R}" refine "${SCENE}" --capture "${OUT}/capture" --capture-depth \
    --out "${OUT}/scene_refined"
"${R2S2R}" mujoco-eval "${OUT}/scene_refined" --capture "${OUT}/capture"
# Which reconstructed object the task is about: the ground truth stands in for the
# language grounding an agent would do. The policy only sees the reconstruction.
TARGET="$("${R2S2R}" mujoco-eval "${OUT}/scene_refined" --capture "${OUT}/capture" \
    --match-target 2>/dev/null | tail -1)"

# Oracle: ground-truth objects, to separate policy failures from reconstruction ones.
"${R2S2R}" mujoco-deploy --oracle --capture "${OUT}/capture" --out "${OUT}/deploy_oracle"
VIDEO=""
if [[ " ${CAMERAS[*]} " == *" ext1 "* ]]; then VIDEO=ext1; fi
# Isaac ignores SIGTERM and can hang on exit; bound it.
timeout -s KILL 900 "${PY}" "${ROOT}/scripts/isaaclab/run_pick.py" \
    "${OUT}/scene_refined" --target "${TARGET}" --out "${OUT}/isaac" --headless \
    --video-camera "${VIDEO}"
"${R2S2R}" mujoco-deploy "${OUT}/scene_refined" --capture "${OUT}/capture" \
    --target "${TARGET}" --out "${OUT}/deploy"
