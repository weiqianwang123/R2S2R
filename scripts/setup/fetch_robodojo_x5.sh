#!/bin/bash
# Fetch RoboDojo's ARX X5 arm (its X5A URDF and meshes: Assets/Robots/x5 of the
# RoboDojo-Benchmark/RoboDojo dataset on Hugging Face, at a pinned commit) into
# ~/.cache/r2s2r/robodojo_x5 (R2S2R_CACHE), which r2s2r.robots.dual_x5 puts two of side
# by side. File by file: the dataset is large, and Git LFS scans all of it.
#
# Usage: bash scripts/setup/fetch_robodojo_x5.sh
set -euo pipefail

CACHE="${R2S2R_CACHE:-$HOME/.cache/r2s2r}"
DIR="${CACHE}/robodojo_x5"
COMMIT=35efbc7dedfdbeeb6e95fb749bd885d73d483e41  # 2026-09-28
URL="https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/${COMMIT}"
FILES=(X5A.urdf meshes/camera.glb meshes/camera_base.glb meshes/base_link.STL)
for i in 1 2 3 4 5 6 7 8; do FILES+=("meshes/link${i}.STL"); done
mkdir -p "${DIR}/meshes"
for f in "${FILES[@]}"; do
    if [ ! -s "${DIR}/${f}" ]; then
        curl -sSfL -o "${DIR}/${f}.part" "${URL}/Assets/Robots/x5/${f}"
        mv "${DIR}/${f}.part" "${DIR}/${f}"
    fi
done
echo "RoboDojo's ARX X5 in ${DIR}"
