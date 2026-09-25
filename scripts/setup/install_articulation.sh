#!/bin/bash
# Install what SimFoundry's stage 9 (articulated objects) needs, without sudo:
# git-lfs and Blender 4.2.3 into ~/.local, then articulate-anything-sf (into the
# SimFoundry deps/, with the fork's r2s2r patch) with its Hunyuan3D-Part
# environment, libigl and the P3-SAM weights. This is SimFoundry's
# scripts/installation/install_articulate.sh minus the apt/sudo steps and, unless
# ARTICULATION_PARTFIELD=1, the optional PartField environment.
#
# Note: `git lfs install` adds the LFS filter to ~/.gitconfig.
#
# Usage: bash scripts/setup/install_articulation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SIMFOUNDRY="${ROOT}/third_party/SimFoundry"
LOCAL="${HOME}/.local"
BLENDER_VERSION="${BLENDER_VERSION:-4.2.3}"
GIT_LFS_VERSION="${GIT_LFS_VERSION:-v3.8.0}"
ARTICULATE_ANYTHING_REPO="${ARTICULATE_ANYTHING_REPO:-https://github.com/nadunRanawaka1/articulate-anything-sf.git}"
mkdir -p "${LOCAL}/bin" "${LOCAL}/opt"
export PATH="${LOCAL}/bin:${PATH}"

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "=== git-lfs ${GIT_LFS_VERSION} -> ${LOCAL}/bin ==="
    tmp="$(mktemp -d)"
    curl -sL "https://github.com/git-lfs/git-lfs/releases/download/${GIT_LFS_VERSION}/git-lfs-linux-amd64-${GIT_LFS_VERSION}.tar.gz" \
        | tar -xz -C "${tmp}"
    install -m 755 "${tmp}"/git-lfs-*/git-lfs "${LOCAL}/bin/git-lfs"
    rm -rf "${tmp}"
fi
git lfs install

if ! blender --version 2>/dev/null | grep -q "Blender ${BLENDER_VERSION}"; then
    echo "=== Blender ${BLENDER_VERSION} -> ${LOCAL}/opt ==="
    series="${BLENDER_VERSION%.*}"
    dest="${LOCAL}/opt/blender-${BLENDER_VERSION}"
    rm -rf "${dest}" && mkdir -p "${dest}"
    curl -sL "https://download.blender.org/release/Blender${series}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
        | tar -xJ -C "${dest}" --strip-components=1
    ln -sf "${dest}/blender" "${LOCAL}/bin/blender"
fi
blender --version | head -1

eval "$(mamba shell hook --shell bash)"
cd "${SIMFOUNDRY}"
mkdir -p deps
cd deps
if [ ! -d articulate-anything ]; then
    git clone -b main "${ARTICULATE_ANYTHING_REPO}" articulate-anything
fi
cd articulate-anything
# Codex for its VLM calls (SIMFOUNDRY_VLM_BACKEND=codex) and a GLB node-name fix.
patch="${SIMFOUNDRY}/patches/articulate-anything-r2s2r.patch"
if git apply --reverse --check "${patch}" 2>/dev/null; then
    echo "r2s2r patch already applied"
else
    git apply "${patch}"
fi
if ! mamba env list | grep -q "^articulate-anything-hunyuan "; then
    bash installation_hunyuan.sh
fi
if [ "${ARTICULATION_PARTFIELD:-0}" = 1 ] && ! mamba env list | grep -q "^articulate-anything-partfield "; then
    bash installation_partfield.sh
fi

echo "=== libigl in articulate-anything-hunyuan ==="
mamba install -n articulate-anything-hunyuan -c conda-forge igl -y

weights="deps/Hunyuan3D-Part/P3-SAM/weights"
mkdir -p "${weights}"
if [ ! -f "${weights}/p3sam.safetensors" ]; then
    echo "=== P3-SAM weights ==="
    curl -L -o "${weights}/p3sam.safetensors" \
        https://huggingface.co/tencent/Hunyuan3D-Part/resolve/main/p3sam/p3sam.safetensors
fi
echo "articulation dependencies installed"
