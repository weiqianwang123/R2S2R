#!/bin/bash
# Fetch ABC-Sim's bimanual i2rt YAM into ~/.cache/r2s2r/i2rt_yam (R2S2R_CACHE), which
# r2s2r.robots.dual_yam re-roots to the arms' base frame: the robot's MJCF
# (yam_bimanual_empty.xml, from our fork of amazon-far/abc at a pinned commit) and
# i2rt's mesh package (MIT; its sha256 checked), laid out as
#   i2rt_yam/{yam_bimanual_empty.xml, assets/*.stl, LICENSE}.
#
# Usage: bash scripts/setup/fetch_abc_yam.sh
set -euo pipefail

CACHE="${R2S2R_CACHE:-$HOME/.cache/r2s2r}"
DIR="${CACHE}/i2rt_yam"
COMMIT=d0832d12651d1b260a652861a14648dc5f3660c7  # ABC-Sim, 2026-10-15
XML_URL="https://raw.githubusercontent.com/weiqianwang123/abc/${COMMIT}/abc_sim/models/yam_bimanual_empty.xml"
MESHES_URL="https://abc-data.timehorizons.org/sim_assets/i2rt_yam.tar.gz"
MESHES_SHA256=ed5144c0b4ae55da75a3701b5bad9ca3599977cc8b73c59996e5e1633749aad7
mkdir -p "${DIR}"
if [ ! -s "${DIR}/yam_bimanual_empty.xml" ]; then
    curl -sSfL -o "${DIR}/yam_bimanual_empty.xml.part" "${XML_URL}"
    mv "${DIR}/yam_bimanual_empty.xml.part" "${DIR}/yam_bimanual_empty.xml"
fi
if [ ! -s "${DIR}/LICENSE" ] || [ ! -s "${DIR}/assets/model2.stl" ]; then
    curl -sSfL -o "${DIR}/i2rt_yam.tar.gz" "${MESHES_URL}"
    echo "${MESHES_SHA256}  ${DIR}/i2rt_yam.tar.gz" | sha256sum --check --quiet
    # The archive holds one folder, i2rt_yam/, with assets/ and LICENSE in it.
    tar -xzf "${DIR}/i2rt_yam.tar.gz" -C "${DIR}" --strip-components=1
    rm "${DIR}/i2rt_yam.tar.gz"
fi
echo "ABC-Sim's i2rt YAM in ${DIR}"
