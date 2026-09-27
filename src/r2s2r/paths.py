"""Where r2s2r finds what it does not ship: its own checkout (scripts), the SimFoundry
submodule, the asset cache, physcoder's assets, the conda environments the model jobs run
in, and the executables it drives.

The one home of these; every other module asks here.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIMFOUNDRY_DIR = REPO_ROOT / "third_party" / "SimFoundry"
# Downloaded models (MuJoCo Menagerie robots, GSO objects).
CACHE_DIR = Path(os.environ.get("R2S2R_CACHE", Path.home() / ".cache" / "r2s2r"))
# The root of physcoder's assets (holding ``mujoco/`` and ``isaaclab/``), used by path
# and never copied: they are git-ignored there and not ours to redistribute. physcoder's
# own override, PHYSCODER_ASSETS_DIR, names the ``isaaclab`` directory in it.
PHYSCODER_ASSETS = (
    Path(os.environ["PHYSCODER_ASSETS_DIR"]).expanduser().parent
    if os.environ.get("PHYSCODER_ASSETS_DIR")
    else Path(os.environ.get("PHYSCODER_ROOT", Path.home() / "physcoder")).expanduser()
    / "assets"
)

# Conda environments: SimFoundry's (its stages, SAM3, CoACD, FoundationStereo) and
# Hunyuan3D-2.1's.
ENV_SIMFOUNDRY = "simfoundry"
ENV_MESH = "hunyuan"

# The ChatGPT desktop app's bundled Codex CLI is kept current; a separately installed
# `codex` can be too old for the newest models.
DESKTOP_APP_CODEX = Path("/usr/lib/chatgpt/resources/codex")


def mamba_exe() -> str:
    """The mamba (or conda) executable; ``mamba`` when none is found."""
    for name in ("mamba", "conda"):
        found = shutil.which(name)
        if found:
            return found
    miniforge = Path.home() / "miniforge3" / "bin" / "mamba"
    return str(miniforge) if miniforge.exists() else "mamba"


def codex_bin() -> str:
    """``$SIMFOUNDRY_CODEX_BIN``, else the desktop app's Codex CLI, else ``codex`` on the
    PATH."""
    explicit = os.environ.get("SIMFOUNDRY_CODEX_BIN")
    if explicit:
        return explicit
    if os.access(DESKTOP_APP_CODEX, os.X_OK):
        return str(DESKTOP_APP_CODEX)
    return shutil.which("codex") or "codex"
