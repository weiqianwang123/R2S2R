#!/bin/bash
# Point the SimFoundry submodule at the heavy, git-ignored resources of an
# existing SimFoundry install (model repos under deps/, checkpoints/, assets/).
# The conda envs built by SimFoundry's installers are reused as-is; r2s2r runs
# the submodule's code in them via PYTHONPATH.
#
# Usage: bash scripts/link_simfoundry_resources.sh [SIMFOUNDRY_INSTALL_DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-${SIMFOUNDRY_HOME:-$HOME/SimFoundry}}"
DST="${ROOT}/third_party/SimFoundry"

if [ ! -d "${DST}/simfoundry" ]; then
    echo "[ERROR] submodule missing; run: git submodule update --init" >&2
    exit 1
fi

for name in deps checkpoints assets; do
    if [ ! -e "${SRC}/${name}" ]; then
        echo "[WARN] ${SRC}/${name} does not exist; skipped"
        continue
    fi
    if [ -e "${DST}/${name}" ] && [ ! -L "${DST}/${name}" ]; then
        echo "[WARN] ${DST}/${name} is a real directory; left untouched"
        continue
    fi
    ln -sfn "$(cd "${SRC}/${name}" && pwd)" "${DST}/${name}"
    echo "[INFO] ${DST}/${name} -> ${SRC}/${name}"
done
