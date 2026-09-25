"""Record a :class:`~r2s2r.structs.Capture` in the MuJoCo world.

The capture holds exactly what a real rig would record (RGB, metric depth, intrinsics,
extrinsics, joint states). Ground truth goes into ``capture.metadata`` for scoring only;
the pipeline never reads it.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from r2s2r.io.rgbd import write_depth
from r2s2r.real.mujoco.world import MujocoRobot, MujocoWorld, MujocoWorldConfig
from r2s2r.robots.franka import FRANKA_HAND_MAX_WIDTH
from r2s2r.structs import DEPTH_PNG_SCALE, Capture, FrameRecord

MAX_DEPTH = 10.0  # metres; farther pixels (the sky) are stored as invalid (0)


def _capture_motion(robot: MujocoRobot) -> None:
    """Sweep the hand over the table so the wrist camera sees the scene."""
    T0 = robot.tcp_pose()
    for dx, dy, dz in [(0.12, 0.18, -0.1), (0.12, -0.18, -0.1), (0.0, 0.0, 0.0)]:
        T = T0.copy()
        T[:3, 3] += [dx, dy, dz]
        robot.move_tcp(T, speed=0.12, settle=0.2)


def record_capture(
    out_dir: str | Path,
    cfg: MujocoWorldConfig | None = None,
    name: str = "mujoco_pick",
    instruction: str = "pick up the crayon box",
    every: int = 5,
) -> Capture:
    """Run the capture motion and save RGB-D frames every ``every`` control steps."""
    out_dir = Path(out_dir)
    world = MujocoWorld(cfg)
    world.reset()
    cameras = {n: world.camera_spec(n) for n in world.camera_names()}
    frames: list[FrameRecord] = []

    def grab(robot: MujocoRobot) -> None:
        if robot.steps % every:
            return
        q, width = world.arm_q(), world.finger_width()
        for cam_name, spec in cameras.items():
            rgb, depth = world.render(cam_name)
            rel = Path("frames") / spec.serial
            (out_dir / rel).mkdir(parents=True, exist_ok=True)
            rgb_rel = rel / f"{robot.steps:04d}_rgb.png"
            depth_rel = rel / f"{robot.steps:04d}_depth.png"
            cv2.imwrite(str(out_dir / rgb_rel), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            depth[depth > MAX_DEPTH] = 0.0  # like a real sensor: 0 = no return
            write_depth(out_dir / depth_rel, depth)
            frames.append(
                FrameRecord(
                    step=robot.steps,
                    camera=spec.serial,
                    left_image=str(rgb_rel),
                    right_image=None,
                    depth_image=str(depth_rel),
                    T_base_cam=world.camera_pose(cam_name),
                    joint_positions=q,
                    gripper_position=1.0 - width / FRANKA_HAND_MAX_WIDTH,
                )
            )

    robot = MujocoRobot(world, on_step=grab)
    grab(robot)
    _capture_motion(robot)
    capture = Capture(
        name=name,
        source="mujoco",
        embodiment="franka_panda",
        instruction=instruction,
        cameras={c.serial: c for c in cameras.values()},
        frames=frames,
        static_steps=(0, robot.steps + 1),
        root=out_dir,
        metadata={
            "world_config": world.cfg.as_dict(),
            "depth_png_scale": DEPTH_PNG_SCALE,
            "ground_truth_T_base_obj": {
                o.name: world.object_pose(o.name).tolist() for o in world.cfg.objects
            },
        },
    )
    capture.save()
    world.close()
    return capture
