"""Let a scene's objects come to rest in Isaac Lab, the capture's robot held as it was
at the start of the static period, and save the settled scene.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/settle.py SCENE_DIR CAPTURE_DIR \
        --out OUT_DIR --headless [--seconds 2]

Writes ``OUT_DIR/scene.json`` (object poses where they came to rest; the report under
``provenance.settle``) and ``OUT_DIR/settle.json`` (how far each object moved).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from r2s2r.sim.isaac import SETTLE_SECONDS

parser = argparse.ArgumentParser()
parser.add_argument("scene_dir", type=Path)
parser.add_argument("capture_dir", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=SETTLE_SECONDS)
parser.add_argument("--physics-dt", type=float, default=0.005)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

# pylint: disable=wrong-import-position
from r2s2r.sim.isaaclab.replay import SettleConfig, settle  # noqa: E402
from r2s2r.structs import Capture, SceneSpec  # noqa: E402


def main() -> None:
    """Settle, then save."""
    spec = SceneSpec.load(args.scene_dir)
    capture = Capture.load(args.capture_dir)
    settled, report = settle(
        spec,
        capture,
        SettleConfig(
            seconds=args.seconds, physics_dt=args.physics_dt, device=args.device
        ),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    settled.save(args.out)
    (args.out / "settle.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    print(
        "settled: "
        + ", ".join(
            f"{n} moved {r['moved_m']:.3f} m, turned {r['turned_deg']:.1f} deg"
            for n, r in report["objects"].items()
        )
        + f" -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    # Everything is written; Omniverse can take minutes to shut down, or hang.
    sys.stdout.flush()
    os._exit(0)  # pylint: disable=protected-access
