"""Run a program policy on a SceneSpec in Isaac Lab and score it.

The Isaac Lab counterpart of :func:`r2s2r.real.mujoco.deploy.run_pick`. Import after
``isaaclab.app.AppLauncher`` has started (``scripts/isaaclab/run_pick.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

from r2s2r.policy.pick import find_object, pick_up
from r2s2r.policy.scoring import save_rollout, score_lift
from r2s2r.sim.isaaclab.robot import IsaacLabRobot
from r2s2r.sim.isaaclab.scene import build_scene_cfg, centered_render_size
from r2s2r.structs import SceneSpec
from r2s2r.video import VideoRecorder


def run_pick(
    spec: SceneSpec,
    target: str,
    out_dir: str | Path,
    video_camera: str | None = "ext1",
    video_every: int = 2,
    physics_dt: float = 0.005,
    settle: float = 0.5,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Let the scene settle, run :func:`pick_up`, score by how far ``target`` rose."""
    out_dir = Path(out_dir)
    target = find_object(spec, target).name
    roles = [video_camera] if video_camera else []
    sim = SimulationContext(SimulationCfg(dt=physics_dt, device=device))
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
    arm = scene["robot"]
    arm.write_joint_state_to_sim(arm.data.default_joint_pos, arm.data.default_joint_vel)
    arm.set_joint_position_target(arm.data.default_joint_pos)
    for _ in range(int(settle / sim.get_physics_dt())):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())

    video = None
    window: tuple[slice, slice] | None = None
    if roles:
        cam = next(c for c in spec.cameras.values() if c.role == roles[0])
        _, _, x0, y0 = centered_render_size(cam)
        window = (slice(y0, y0 + cam.height), slice(x0, x0 + cam.width))
        fps = 1 / (IsaacLabRobot.control_dt * video_every)
        video = VideoRecorder(out_dir / f"isaac_{roles[0]}.mp4", fps)

    def record(robot: IsaacLabRobot) -> None:
        if video is not None and robot.steps % video_every == 0:
            rgb = scene[f"camera_{roles[0]}"].data.output["rgb"][0, ..., :3]
            video.add(rgb.cpu().numpy().astype(np.uint8)[window])

    before = positions()
    robot = IsaacLabRobot(
        sim, scene, on_step=record, render_every=video_every if video else 0
    )
    policy = pick_up(robot, spec, target)
    result = {
        "deployment": "isaaclab",
        "scene": spec.name,
        "scene_backend": spec.provenance.get("backend"),
        "policy": policy,
        **score_lift(before, positions(), target),
    }
    save_rollout(out_dir, result, robot.log)
    if video is not None:
        video.close()
    return result
