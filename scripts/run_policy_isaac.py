"""Run the pick program policy on a SceneSpec in Isaac Lab and score it.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/run_policy_isaac.py SCENE_DIR \
        --target crayon --out outputs/mujoco_pick/isaac --headless

Writes ``result.json`` (success = the target rose more than 5 cm), ``commands.json``
(every joint/gripper command, the same stream a deployment receives) and, with
``--video-camera``, ``isaac_<role>.mp4`` from that calibrated camera.
"""

from __future__ import annotations

import argparse
import json
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
import numpy as np  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from r2s2r.policy.pick import find_object, pick_up  # noqa: E402
from r2s2r.sim.isaaclab.robot import IsaacLabRobot  # noqa: E402
from r2s2r.sim.isaaclab.scene import (  # noqa: E402
    build_scene_cfg,
    centered_render_size,
)
from r2s2r.structs import SceneSpec  # noqa: E402
from r2s2r.video import VideoRecorder  # noqa: E402

LIFT_SUCCESS_M = 0.05
VIDEO_EVERY = 2  # control steps per video frame (25 fps)


def main() -> None:
    """Settle, run the policy, score."""
    spec = SceneSpec.load(args.scene_dir)
    target = find_object(spec, args.target).name
    roles = [args.video_camera] if args.video_camera else []
    sim = SimulationContext(SimulationCfg(dt=args.physics_dt, device=args.device))
    scene = InteractiveScene(
        build_scene_cfg(spec, with_cameras=bool(roles), camera_roles=roles)
    )
    sim.reset()
    names = {f"object_{i}": obj.name for i, obj in enumerate(spec.objects)}

    def positions() -> dict[str, np.ndarray]:
        return {
            names[key]: obj.data.root_pos_w[0].cpu().numpy().copy()
            for key, obj in scene.rigid_objects.items()
        }

    # sim.reset() leaves the USD's joint state; start from the scene's instead,
    # then let the reconstructed objects come to rest under physics.
    robot_art = scene["robot"]
    robot_art.write_joint_state_to_sim(
        robot_art.data.default_joint_pos, robot_art.data.default_joint_vel
    )
    robot_art.set_joint_position_target(robot_art.data.default_joint_pos)
    for _ in range(int(args.settle / sim.get_physics_dt())):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
    before = positions()

    video = None
    window = None
    if roles:
        cam = next(c for c in spec.cameras.values() if c.role == roles[0])
        _, _, x0, y0 = centered_render_size(cam)
        window = (slice(y0, y0 + cam.height), slice(x0, x0 + cam.width))
        video = VideoRecorder(
            args.out / f"isaac_{roles[0]}.mp4",
            1 / (IsaacLabRobot.control_dt * VIDEO_EVERY),
        )

    def record(robot: IsaacLabRobot) -> None:
        if video is not None and robot.steps % VIDEO_EVERY == 0:
            rgb = scene[f"camera_{roles[0]}"].data.output["rgb"][0, ..., :3]
            video.add(rgb.cpu().numpy().astype(np.uint8)[window])

    robot = IsaacLabRobot(
        sim, scene, on_step=record, render_every=VIDEO_EVERY if video else 0
    )
    policy = pick_up(robot, spec, target)
    after = positions()
    lift = {n: float(after[n][2] - before[n][2]) for n in before}
    result = {
        "deployment": "isaaclab",
        "scene": spec.name,
        "scene_backend": spec.provenance.get("backend"),
        "policy": policy,
        "object_lift_m": lift,
        "object_shift_m": {
            n: float(np.linalg.norm(after[n] - before[n])) for n in before
        },
        "success": bool(lift[target] > LIFT_SUCCESS_M),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.out / "commands.json").write_text(
        json.dumps(robot.log.as_dict()), encoding="utf-8"
    )
    if video is not None:
        video.close()
    print(
        f"{'SUCCESS' if result['success'] else 'FAILURE'}: picked {target!r}; lifted "
        + ", ".join(f"{n} {dz * 100:+.1f} cm" for n, dz in lift.items())
        + f" -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang after headless camera rendering.
    os._exit(0)  # pylint: disable=protected-access
