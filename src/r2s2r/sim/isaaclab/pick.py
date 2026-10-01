"""The pick test in Isaac Lab: the pick program (:mod:`r2s2r.testbed.policy`) on a
scene, for the scene's robot, scored by how far the target rose.

The Isaac Lab counterpart of :func:`r2s2r.testbed.pick.run_pick`; the same program
sends the same commands through :class:`IsaacLabRobot`. Import after
``isaaclab.app.AppLauncher`` has started (``scripts/isaaclab/pick.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from r2s2r.robots import get_robot
from r2s2r.sim.isaaclab.scene import Session, camera_key, crop_window, with_object_usds
from r2s2r.structs import SceneSpec
from r2s2r.testbed.policy import (
    CONTROL_DT,
    VIDEO_EVERY,
    RobotInterface,
    VideoRecorder,
    find_object,
    pick_up,
    save_rollout,
    score_lift,
    summarize,
)

REST_SECONDS = 0.5  # physics before the program starts: the objects come to rest


class IsaacLabRobot(RobotInterface):
    """The robot of a :class:`~r2s2r.sim.isaaclab.scene.Session`, its arm and gripper
    on joint position targets."""

    def __init__(
        self,
        session: Session,
        on_step: Callable[[IsaacLabRobot], None] | None = None,
        render_every: int = 0,  # render on every n-th control step (0: never)
    ) -> None:
        super().__init__(session.robot_spec)
        self.session = session
        self.on_step = on_step
        self.render_every = render_every
        driver, _ = session.articulation.find_joints([self.robot.gripper.isaac_driver])
        self.driver = session.grip_ids.index(driver[0])
        self.decimation = max(1, int(round(CONTROL_DT / session.dt)))
        self.steps = 0

    def joint_positions(self) -> NDArray[np.float64]:
        s = self.session
        return s.articulation.data.joint_pos[0, s.arm_ids].cpu().numpy().astype(float)

    def gripper_level(self) -> float:
        s, k = self.session, self.driver
        q = s.articulation.data.joint_pos[0, s.grip_ids[k]].item()
        lo, hi = s.grip_open[k].item(), s.grip_closed[k].item()
        return float(np.clip((q - lo) / (hi - lo), 0.0, 1.0))

    def _hold(self, q: NDArray[np.float64], level: float) -> None:
        s = self.session
        target = s.articulation.data.joint_pos_target.clone()
        target[0, s.arm_ids] = target.new_tensor(q)
        target[0, s.grip_ids] = s.gripper_targets(level)
        s.articulation.set_joint_position_target(target)
        render = self.render_every > 0 and (self.steps + 1) % self.render_every == 0
        s.step_physics(self.decimation, render)
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self)


def run_pick(
    spec: SceneSpec,
    target: str,
    out_dir: str | Path,
    video_camera: str | None = "ext1",
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Let the scene settle, run :func:`~r2s2r.testbed.policy.pick_up`, score by how
    far ``target`` rose; ``result.json``, ``commands.json`` and the video (from the
    static camera ``video_camera``) go to ``out_dir``, the verdict to stdout."""
    out_dir = Path(out_dir)
    target = find_object(spec, target).name
    video_cams = [
        (c, c.T_base_cam)
        for c in spec.cameras.values()
        if c.role == video_camera and c.T_base_cam is not None
    ]
    session = Session(
        with_object_usds(spec, out_dir),
        get_robot(spec.embodiment),
        False,
        device,
        video_cams,
    )

    def positions() -> dict[str, NDArray[np.float64]]:
        out = {name: T[:3, 3] for name, T in session.object_poses().items()}
        out.update({n: p.mean(axis=0) for n, p in session.cloth_points().items()})
        return out

    # sim.reset() leaves the USD's joint state; start from the scene's instead,
    # then let the objects come to rest under physics.
    arm = session.articulation
    arm.write_joint_state_to_sim(arm.data.default_joint_pos, arm.data.default_joint_vel)
    arm.set_joint_position_target(arm.data.default_joint_pos)
    session.step_physics(int(REST_SECONDS / session.dt))

    video = None
    window: tuple[slice, slice] | None = None
    if video_cams:
        cam = video_cams[0][0]
        window = crop_window(cam)
        video = VideoRecorder(out_dir / f"isaac_{cam.role}.mp4")

    def record(robot: IsaacLabRobot) -> None:
        if video is not None and robot.steps % VIDEO_EVERY == 0:
            output = session.scene[camera_key(video_cams[0][0])].data.output
            video.add(output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)[window])

    before = positions()
    robot = IsaacLabRobot(
        session, on_step=record, render_every=VIDEO_EVERY if video else 0
    )
    policy = pick_up(robot, spec, target)
    result = {
        "deployment": "isaaclab",
        "scene": spec.name,
        "scene_method": spec.provenance.get("method"),
        "policy": policy,
        **score_lift(before, positions(), target),
    }
    save_rollout(out_dir, result, robot.log)
    if video is not None:
        video.close()
    print(f"{summarize(result)} -> {out_dir}", flush=True)
    return result
