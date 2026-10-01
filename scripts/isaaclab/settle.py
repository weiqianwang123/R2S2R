"""Let a scene's objects come to rest in Isaac Lab, the capture's robot held as it was
at the start of the static period, and save the settled scene.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/settle.py SCENE_DIR CAPTURE_DIR \
        --out OUT_DIR --headless [--seconds 2]

Writes ``OUT_DIR/scene.json`` (object poses where they came to rest; the report under
``provenance.settle``), ``OUT_DIR/settle.json`` (how far each object moved) and
``OUT_DIR/objects/<name>/`` (each object as one USD file with physcoder's
``metadata.yaml``, which the scene refers to; a cloth's settled surface and URDF too).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from r2s2r.sim.isaac import RENDERING_MODE, SETTLE_SECONDS, settle_summary

parser = argparse.ArgumentParser()
parser.add_argument("scene_dir", type=Path)
parser.add_argument("capture_dir", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=SETTLE_SECONDS)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(rendering_mode=RENDERING_MODE)
args = parser.parse_args()
app = AppLauncher(args).app

# pylint: disable=wrong-import-position
from r2s2r.sim.isaaclab.replay import SettleConfig, settle  # noqa: E402
from r2s2r.structs import Capture, SceneSpec  # noqa: E402


def main() -> None:
    """Settle, then save."""
    spec = SceneSpec.load(args.scene_dir)
    capture = Capture.load(args.capture_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    settled, report = settle(
        spec,
        capture,
        args.out,
        SettleConfig(seconds=args.seconds, device=args.device),
    )
    settled.save(args.out)
    (args.out / "settle.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    print(f"settled: {settle_summary(report)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
    # Everything is written; Omniverse can take minutes to shut down, or hang.
    sys.stdout.flush()
    os._exit(0)  # pylint: disable=protected-access
