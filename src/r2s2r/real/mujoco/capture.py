"""Record a :class:`~r2s2r.structs.Capture` in the MuJoCo world.

The capture holds exactly what a real rig would record (RGB, metric depth, intrinsics,
extrinsics, joint states). Ground truth goes into ``capture.metadata`` for scoring only;
the pipeline never reads it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.io.rgbd import write_depth
from r2s2r.real.mujoco.world import (
    MujocoRobot,
    MujocoWorld,
    MujocoWorldConfig,
    look_at,
)
from r2s2r.robots.franka import FRANKA_HAND_MAX_WIDTH, Q_READY
from r2s2r.structs import DEPTH_PNG_SCALE, Capture, FrameRecord
from r2s2r.transforms import invert, make_transform

# The depth sensor's range in metres; outside it there is no return (0). The near
# limit is the ZED Mini's, on DROID's wrist; beyond the far one is the sky.
MIN_DEPTH, MAX_DEPTH = 0.1, 10.0


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


def _demo_interaction(robot: MujocoRobot, world: MujocoWorld) -> None:
    """Open every drawer by ``demo_pull`` and close it again, gripping the handle from
    the front, and pause in between with the arm out of the way.

    MuJoCo is the real world here, so the demonstration may use its ground truth, as a
    teleoperator would use their eyes.
    """
    if world.cfg.demo_pull > 0:
        for cab in world.cfg.cabinets:
            _open_and_close(robot, world, cab.name, world.cfg.demo_pull)


def _open_and_close(
    robot: MujocoRobot, world: MujocoWorld, cabinet: str, pull: float
) -> None:
    """Pull ``cabinet``'s drawer ``pull`` metres out by its handle, then push it
    back."""
    drawer = world.data.body(f"{cabinet}_drawer")
    axis = drawer.xmat.reshape(3, 3) @ world.model.joint(f"{cabinet}_slide").axis
    axis = axis / np.linalg.norm(axis)  # the opening direction
    z = -axis  # the hand approaches against it
    y = np.array([0.0, 0.0, 1.0])  # fingers close vertically around the bar
    grasp = make_transform(np.column_stack([np.cross(y, z), y, z]), np.zeros(3))

    def at(offset: float) -> NDArray[np.float64]:
        T = grasp.copy()
        T[:3, 3] = world.data.geom(f"{cabinet}_handle").xpos + offset * axis
        return T

    for direction in (1.0, -1.0):  # open, then close
        robot.set_gripper(False, 0.5)
        # Turning the hand to face the handle is a large reorientation: go to a
        # well-conditioned configuration in joint space, then approach.
        robot.move_joints(robot.kin.solve(at(0.08), robot.q_command).q)
        robot.move_tcp(at(0.0), speed=0.05)
        robot.set_gripper(True, 1.0)
        T = at(0.0)
        T[:3, 3] += direction * pull * axis
        robot.move_tcp(T, speed=0.05)
        robot.set_gripper(False, 0.8)
        robot.move_tcp(at(0.08), speed=0.1)
        robot.move_joints(Q_READY)
        robot.wait(1.5)


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
    joints: dict[str, dict[str, float]] = {}  # ground truth, for scoring only

    def grab(robot: MujocoRobot) -> None:
        if robot.steps % every:
            return
        joints[str(robot.steps)] = {
            f"{c.name}_slide": world.joint_position(f"{c.name}_slide")
            for c in world.cfg.cabinets
        }
        q, width = world.arm_q(), world.finger_width()
        for cam_name, spec in cameras_.items():
            rgb, depth = world.render(cam_name)
            rel = Path("frames") / spec.serial
            (out_dir / rel).mkdir(parents=True, exist_ok=True)
            rgb_rel = rel / f"{robot.steps:04d}_rgb.png"
            depth_rel = rel / f"{robot.steps:04d}_depth.png"
            cv2.imwrite(str(out_dir / rgb_rel), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            depth[(depth < MIN_DEPTH) | (depth > MAX_DEPTH)] = 0.0
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
    static_end = robot.steps + 1
    _demo_interaction(robot, world)
    capture = Capture(
        name=name,
        source="mujoco",
        embodiment="franka_panda",
        instruction=instruction,
        cameras={c.serial: c for c in cameras_.values()},
        frames=frames,
        static_steps=(0, static_end),
        root=out_dir,
        metadata={
            "world_config": world.cfg.as_dict(),
            "depth_png_scale": DEPTH_PNG_SCALE,
            "ground_truth_joints": joints,
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
