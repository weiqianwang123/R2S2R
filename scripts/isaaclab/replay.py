"""Replay a capture's robot in a reconstructed scene, in Isaac Lab, and compare every
recorded frame with its render.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/replay.py SCENE_DIR CAPTURE_DIR \
        --out OUT_DIR --headless [--mode geometry|physics] [--cameras ext2 wrist]

Writes ``OUT_DIR/frames/`` (the renders), ``OUT_DIR/replay.json`` (where they are, the
objects' poses at every rendered step, how closely the arm tracked) and
``OUT_DIR/compare/`` (real | sim | blend panels with the objects' outlines, depth
residuals where the capture has depth, a contact sheet per camera, ``compare.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("scene_dir", type=Path)
parser.add_argument("capture_dir", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--cameras", nargs="+", help="roles or serials (default: all)")
parser.add_argument("--every", type=int, default=1, help="render every n-th frame step")
parser.add_argument(
    "--mode",
    choices=["geometry", "physics"],
    default="geometry",
    help="geometry: static period, robot set to the recorded state, objects held; "
    "physics: the whole recording, arm tracking it, objects simulated",
)
parser.add_argument("--physics-dt", type=float, default=0.005)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

# pylint: disable=wrong-import-position
from r2s2r.sim.compare import compare_replay  # noqa: E402
from r2s2r.sim.isaaclab.replay import ReplayConfig, replay  # noqa: E402
from r2s2r.structs import Capture, SceneSpec  # noqa: E402


def main() -> None:
    """Replay, then compare."""
    spec = SceneSpec.load(args.scene_dir)
    capture = Capture.load(args.capture_dir)
    config = ReplayConfig(
        mode=args.mode,
        physics_dt=args.physics_dt,
        cameras=tuple(args.cameras) if args.cameras else None,
        every=args.every,
        device=args.device,
    )
    log = replay(spec, capture, args.out, config)
    (args.out / "replay.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    summary = compare_replay(log, spec, capture, args.out / "compare", args.out)
    moved = {
        name: max(d for _, d in rows) for name, rows in summary["motion_m"].items()
    }
    print(
        f"replayed {len(log['frames'])} frames; arm tracking within "
        f"{summary['arm_error_rad_max']:.3f} rad; objects moved (m): "
        + ", ".join(f"{n} {d:.3f}" for n, d in moved.items())
        + f" -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    # Everything is written; Omniverse can take minutes to shut down, or hang.
    sys.stdout.flush()
    os._exit(0)  # pylint: disable=protected-access
