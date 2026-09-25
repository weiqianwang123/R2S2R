"""Record a :class:`~r2s2r.structs.Capture` in the MuJoCo world.

The capture holds exactly what a real rig would record (RGB, metric depth, intrinsics,
extrinsics, joint states). Ground truth goes into ``capture.metadata`` for scoring only;
the pipeline never reads it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from r2s2r.io.rgbd import write_depth
from r2s2r.real.mujoco.world import (
    MujocoRobot,
    MujocoWorld,
    MujocoWorldConfig,
    look_at,
)
from r2s2r.robots.franka import FRANKA_HAND_MAX_WIDTH, Q_READY
from r2s2r.structs import DEPTH_PNG_SCALE, Capture, FrameRecord
from r2s2r.transforms import invert

MAX_DEPTH = 10.0  # metres; farther pixels (the sky) are stored as invalid (0)


def _capture_motion(
    robot: MujocoRobot,
    world: MujocoWorld,
    elevation_deg: float = 55.0,
    azimuths_deg: tuple[float, ...] = (-35.0, 0.0, 35.0),
) -> None:
    """Point the wrist camera at the world's scan target from a few directions, then
    return to the ready pose.

    The exterior cameras see the arm move (it is masked out).
    """
    if "wrist" not in world.camera_names():
        return
    T_tcp_cam = invert(world.tcp_pose()) @ world.camera_pose("wrist")
    centre = np.asarray(world.cfg.scan_target)
    distance = world.cfg.scan_distance
    for az_deg in azimuths_deg:
        az, el = np.deg2rad(az_deg), np.deg2rad(elevation_deg)
        offset = [-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)]
        T_base_cam = look_at(centre + distance * np.asarray(offset), centre)
        robot.move_tcp(T_base_cam @ invert(T_tcp_cam), speed=0.15, settle=0.3)
    robot.move_joints(Q_READY)


def record_capture(
    out_dir: str | Path,
    cfg: MujocoWorldConfig | None = None,
    name: str = "mujoco_pick",
    instruction: str = "pick up the crayon box",
    every: int = 5,
    cameras: list[str] | None = None,
) -> Capture:
    """Run the capture motion and save RGB-D frames of ``cameras`` (default: all) every
    ``every`` control steps."""
    out_dir = Path(out_dir)
    world = MujocoWorld(cfg)
    world.reset()
    names = cameras or world.camera_names()
    unknown = set(names) - set(world.camera_names())
    if unknown:
        raise KeyError(
            f"unknown cameras {sorted(unknown)}; have {world.camera_names()}"
        )
    cameras_ = {n: world.camera_spec(n) for n in names}
    frames: list[FrameRecord] = []

    def grab(robot: MujocoRobot) -> None:
        if robot.steps % every:
            return
        q, width = world.arm_q(), world.finger_width()
        for cam_name, spec in cameras_.items():
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
    _capture_motion(robot, world)
    capture = Capture(
        name=name,
        source="mujoco",
        embodiment="franka_panda",
        instruction=instruction,
        cameras={c.serial: c for c in cameras_.values()},
        frames=frames,
        static_steps=(0, robot.steps + 1),
        root=out_dir,
        metadata={
            "world_config": world.cfg.as_dict(),
            "depth_png_scale": DEPTH_PNG_SCALE,
            "ground_truth_T_base_obj": {
                name: world.object_pose(name).tolist()
                for name in [o.name for o in world.cfg.objects]
                + [c.name for c in world.cfg.cabinets]
            },
        },
    )
    capture.save()
    world.close()
    return capture
