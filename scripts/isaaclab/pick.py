"""Run the pick program on a scene in Isaac Lab and score it.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/pick.py SCENE_DIR \
        --target crayon_box --out OUT_DIR --headless [--video-camera ext1]

Writes ``OUT_DIR/result.json`` (success: the target rose more than 5 cm),
``OUT_DIR/commands.json`` (every joint and gripper command, the same stream a
deployment receives) and, with a video camera, ``OUT_DIR/isaac_<role>.mp4`` from that
calibrated camera. ``r2s2r pick`` runs this beside the MuJoCo pick
(:func:`r2s2r.sim.isaac.pick`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("scene_dir", type=Path)
parser.add_argument("--target", required=True, help="object name or unique substring")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--video-camera", default="ext1", help="'' for no video")
parser.add_argument("--physics-dt", type=float, default=0.005)
parser.add_argument("--settle", type=float, default=0.5, help="seconds before acting")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = bool(args.video_camera)
app = AppLauncher(args).app

# pylint: disable=wrong-import-position
from r2s2r.sim.isaaclab.pick import run_pick  # noqa: E402
from r2s2r.structs import SceneSpec  # noqa: E402
from r2s2r.testbed.policy import summarize  # noqa: E402

if __name__ == "__main__":
    result = run_pick(
        SceneSpec.load(args.scene_dir),
        args.target,
        args.out,
        video_camera=args.video_camera,
        physics_dt=args.physics_dt,
        settle=args.settle,
        device=args.device,
    )
    print(f"{summarize(result)} -> {args.out}", flush=True)
    # Everything is written; SimulationApp.close() can hang after headless rendering.
    sys.stdout.flush()
    os._exit(0)  # pylint: disable=protected-access
