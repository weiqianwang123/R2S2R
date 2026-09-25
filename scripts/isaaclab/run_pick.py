"""Run the pick program policy on a SceneSpec in Isaac Lab and score it.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/run_pick.py SCENE_DIR \
        --target crayon --out outputs/mujoco_pick/isaac --headless

Writes ``result.json`` (success = the target rose more than 5 cm), ``commands.json``
(every joint/gripper command, the same stream a deployment receives) and, with
``--video-camera``, ``isaac_<role>.mp4`` from that calibrated camera.
"""

from __future__ import annotations

import argparse
import os
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
from r2s2r.policy.scoring import summarize  # noqa: E402
from r2s2r.sim.isaaclab.rollout import run_pick  # noqa: E402
from r2s2r.structs import SceneSpec  # noqa: E402

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
    # SimulationApp.close() can hang after headless camera rendering.
    os._exit(0)  # pylint: disable=protected-access
