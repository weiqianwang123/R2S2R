"""Render a SceneSpec in Isaac Lab from every real camera and overlay the real frames.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_overlay.py \
        outputs/iris/preview outputs/iris/capture --out outputs/iris/overlay

For each static camera it writes ``<role>_sim.png`` (the render),
``<role>_real.png`` (the capture frame at the scene's reference step, or the
nearest one) and ``<role>_overlay.png`` (a 50/50 blend).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("scene_dir", type=Path)
parser.add_argument("capture_dir", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--settle-steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

# pylint: disable=wrong-import-position
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from r2s2r.sim.isaaclab.scene import (  # noqa: E402
    build_scene_cfg,
    centered_render_size,
)
from r2s2r.structs import Capture, SceneSpec  # noqa: E402


def main() -> None:
    """Render and blend."""
    spec = SceneSpec.load(args.scene_dir)
    capture = Capture.load(args.capture_dir)
    sim = SimulationContext(SimulationCfg(dt=1 / 60, device=args.device))
    scene = InteractiveScene(build_scene_cfg(spec))
    sim.reset()

    robot = scene["robot"]
    for _ in range(args.settle_steps):
        # Hold the recorded robot configuration and the reconstructed object
        # poses exactly; this is a kinematic check, not a physics rollout.
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos, robot.data.default_joint_vel
        )
        for obj in scene.rigid_objects.values():
            obj.write_root_state_to_sim(obj.data.default_root_state)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    args.out.mkdir(parents=True, exist_ok=True)
    for cam in spec.cameras.values():
        key = f"camera_{cam.role}"
        if key not in scene.keys():
            continue
        _, _, x0, y0 = centered_render_size(cam)
        window = (slice(y0, y0 + cam.height), slice(x0, x0 + cam.width))
        rgb = scene[key].data.output["rgb"][0, ..., :3].cpu().numpy()[window]
        depth = scene[key].data.output["distance_to_image_plane"][0, ..., 0]
        rendered = (depth.cpu().numpy()[window] < 50.0)[..., None]  # bg is inf
        sim_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        frames = capture.frames_of(cam.serial)
        real_frame = min(frames, key=lambda f: abs(f.step - spec.reference_step))
        real = cv2.imread(str(capture.root / real_frame.left_image))
        cv2.imwrite(str(args.out / f"{cam.role}_sim.png"), sim_bgr)
        cv2.imwrite(str(args.out / f"{cam.role}_real.png"), real)
        blend = np.where(rendered, cv2.addWeighted(real, 0.4, sim_bgr, 0.6, 0.0), real)
        cv2.imwrite(str(args.out / f"{cam.role}_overlay.png"), blend)
        print(f"{cam.role}: step {real_frame.step} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang after headless camera rendering; everything
    # is written by now, so exit hard.
    os._exit(0)  # pylint: disable=protected-access
