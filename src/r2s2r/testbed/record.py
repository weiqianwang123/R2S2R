"""Record a :class:`~r2s2r.structs.Capture` in a MuJoCo world, as a real rig would.

The world's free objects first come to rest under physics (the arm held at home), and
the ground truth is read then (:meth:`~r2s2r.testbed.worlds.MujocoWorld.ground_truth`,
into ``capture.metadata``, which runs never show a method). The recording itself is
kinematic, and all of it is the static period: the wrist camera is pointed at the
objects from a ring of views, from above, and from wide views off to the sides where
the support fills much of the image (the plane a method fits must be the support, not
the top of a large object). Each view is solved by the robot model's IK from the
previous solution, joint paths are interpolated between views, and a view is dropped
when its path would bring the arm within a few centimetres of the scene.

Every step gets a trajectory row; every ``every``-th step, and each view, gets RGB and
metric depth from every camera, the depth zeroed outside the sensor's range.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.structs import Capture, FrameRecord, RobotTrajectory, write_depth
from r2s2r.testbed.worlds import MujocoWorld
from r2s2r.transforms import look_at, make_transform

# The depth sensor's range in metres; outside it there is no return (0). The near
# limit is where physcoder's (and the real RealSense's) depth starts.
MIN_DEPTH, MAX_DEPTH = 0.07, 10.0
DEPTH_NOTE = (
    f"rendered by MuJoCo (planar, exact; none outside [{MIN_DEPTH}, {MAX_DEPTH}] m)"
)
CONTROL_DT = 0.02  # seconds per recorded step
JOINT_SPEED = 0.6  # rad/s of the joint that moves most
CLEARANCE = 0.03  # m the arm keeps from the scene
IK_POS_TOL, IK_ROT_TOL = 0.01, 0.05  # m, rad: a view reached closely enough
ROLLS = (0.0, 180.0, 90.0, -90.0)  # degrees about the optical axis to try
# The ring of views on the robot's side of the objects (azimuths from the direction
# toward the robot) and the one from above; wide views look at the support beside the
# objects from further out.
RING_AZIMUTHS = (-45.0, -25.0, 0.0, 25.0, 45.0)
RING_ELEVATION, TOP_ELEVATION, WIDE_ELEVATION = 50.0, 80.0, 60.0
WIDE_OFFSET = 0.12  # m beyond the objects' reach, sideways
WIDE_DISTANCE = 1.3  # times the ring's distance
RING_DISTANCE = (0.35, 0.6)  # m, the ring's distance bounds
RING_MARGIN = 1.6  # the image's width over the objects'


@dataclass
class View:
    """A wrist-camera viewpoint: ``eye`` looking at ``target``."""

    name: str
    eye: NDArray[np.float64]
    target: NDArray[np.float64]


def scan_views(world: MujocoWorld, truth: dict[str, Any], camera: str) -> list[View]:
    """Views of the ground-truth objects for ``camera``, in the order to visit them:
    wide on one side, the ring (from above in its middle), wide on the other side."""
    lo = np.min([o["center"] - o["size"] / 2 for o in truth["objects"].values()], 0)
    hi = np.max([o["center"] + o["size"] / 2 for o in truth["objects"].values()], 0)
    centre = (lo + hi) / 2
    reach = float(np.max(hi[:2] - lo[:2])) / 2
    toward_robot = -centre[:2] / np.linalg.norm(centre[:2])
    side = np.array([-toward_robot[1], toward_robot[0]])
    # Near enough that the objects span most of the image's width.
    cam = world.camera_spec(camera)
    half_width = np.arctan(cam.width / 2 / cam.K[0, 0])
    distance = float(np.clip(RING_MARGIN * reach / np.tan(half_width), *RING_DISTANCE))

    def eye(target: NDArray, azimuth: float, elevation: float, d: float) -> NDArray:
        az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
        horizontal = np.cos(az) * toward_robot + np.sin(az) * side
        return target + d * np.r_[np.cos(el) * horizontal, np.sin(el)]

    ring = [
        View(f"ring{az:+.0f}", eye(centre, az, RING_ELEVATION, distance), centre)
        for az in RING_AZIMUTHS
    ]
    ring.insert(
        len(ring) // 2 + 1,
        View("top", eye(centre, 0.0, TOP_ELEVATION, distance), centre),
    )
    support = truth["support"]["height"]
    wide = []
    for sign, name in ((-1.0, "wide_left"), (1.0, "wide_right")):
        target = np.r_[centre[:2] + sign * (reach + WIDE_OFFSET) * side, support]
        wide.append(
            View(
                name, eye(target, 0.0, WIDE_ELEVATION, WIDE_DISTANCE * distance), target
            )
        )
    return [wide[0], *ring, wide[1]]


class _Path:
    """The recorded joint path: a row per step; frames on multiples of ``every``."""

    def __init__(self, world: MujocoWorld, every: int) -> None:
        self.world, self.every = world, every
        self.q = [np.asarray(world.spec.home_q, float)]

    def segment(self, q_goal: NDArray) -> list[NDArray] | None:
        """Joint-interpolated steps from the path's end to ``q_goal`` (a multiple of
        ``every`` of them), or None if one comes within :data:`CLEARANCE` of the
        scene."""
        q0 = self.q[-1]
        n = int(np.ceil(np.max(np.abs(q_goal - q0)) / (JOINT_SPEED * CONTROL_DT)))
        n = max(self.every, int(np.ceil(n / self.every)) * self.every)
        steps = [q0 + (q_goal - q0) * (i / n) for i in range(1, n + 1)]
        for q in steps:
            self.world.robot.set(q, 0.0)
            if self.world.clearance(CLEARANCE) < CLEARANCE:
                return None
        return steps


def _reach(
    world: MujocoWorld, camera: str, T_base_cam: NDArray, q_seed: NDArray
) -> NDArray | None:
    """Arm joints that put ``camera`` at ``T_base_cam`` turned by one of
    :data:`ROLLS` about its optical axis (a wrist camera's images turn with the
    hand; the level one first), or None when none is in reach."""
    for roll in np.deg2rad(ROLLS):
        c, s = np.cos(roll), np.sin(roll)
        turned = T_base_cam @ make_transform([[c, -s, 0], [s, c, 0], [0, 0, 1]], 0)
        result = world.robot.ik(turned, q_seed, camera, "camera")
        if result.pos_error < IK_POS_TOL and result.rot_error < IK_ROT_TOL:
            return result.q
    return None


def record_capture(world: MujocoWorld, out_dir: str | Path, every: int = 5) -> Capture:
    """Settle ``world``, read its ground truth, scan it with the wrist camera and save
    the capture in ``out_dir`` (its name is the directory's)."""
    out_dir = Path(out_dir)
    world.reset()
    truth = world.ground_truth()
    moving = [c for c in world.cameras if not world.camera_spec(c).is_static]
    if not moving:
        raise ValueError(f"world {world.name} has no camera on the robot to scan with")
    path = _Path(world, every)
    visits: list[dict[str, Any]] = []
    for view in scan_views(world, truth, moving[0]):
        row = {"view": view.name, "eye": view.eye, "target": view.target}
        q = _reach(world, moving[0], look_at(view.eye, view.target), path.q[-1])
        if q is None:
            visits.append({**row, "skipped": "out of reach"})
            continue
        steps = path.segment(q)
        if steps is None:
            visits.append({**row, "skipped": f"within {CLEARANCE} m of the scene"})
            continue
        path.q += steps
        visits.append({**row, "step": len(path.q) - 1})
    home = path.segment(np.asarray(world.spec.home_q, float))
    if home is None:
        raise RuntimeError(f"no clear way back home in world {world.name}")
    path.q += home
    return _save(world, out_dir, path, truth, visits)


def _save(
    world: MujocoWorld,
    out_dir: Path,
    path: _Path,
    truth: dict[str, Any],
    visits: list[dict[str, Any]],
) -> Capture:
    """Render the frames along ``path`` and write the capture."""
    cameras = {role: world.camera_spec(role) for role in world.cameras}
    support = world.model.body(truth["support"]["body"]).id
    at_view = {v["step"]: v for v in visits if "step" in v}
    frames, levels = [], []
    for step, q in enumerate(path.q):
        world.robot.set(q, 0.0)
        level = world.gripper_level()
        levels.append(level)
        if step % path.every:
            continue
        for role, cam in cameras.items():
            out = world.render(role)
            rel = Path("frames") / cam.serial
            (out_dir / rel).mkdir(parents=True, exist_ok=True)
            rgb_rel, depth_rel = (
                rel / f"{step:04d}_rgb.png",
                rel / f"{step:04d}_depth.png",
            )
            cv2.imwrite(
                str(out_dir / rgb_rel), cv2.cvtColor(out["rgb"], cv2.COLOR_RGB2BGR)
            )
            depth = out["depth"]
            depth[(depth < MIN_DEPTH) | (depth > MAX_DEPTH)] = 0.0
            write_depth(out_dir / depth_rel, depth)
            frames.append(
                FrameRecord(
                    step=step,
                    camera=cam.serial,
                    left_image=str(rgb_rel),
                    right_image=None,
                    depth_image=str(depth_rel),
                    T_base_cam=world.camera_pose(role),
                    joint_positions=q,
                    gripper_position=level,
                )
            )
            if step in at_view and not cam.is_static:
                at_view[step]["support_fraction"] = float(
                    np.mean(out["body"] == support)
                )
    steps = np.arange(len(path.q))
    capture = Capture(
        name=out_dir.name,
        source="mujoco",
        embodiment=world.spec.name,
        instruction=world.instruction,
        cameras={c.serial: c for c in cameras.values()},
        frames=frames,
        static_steps=(0, len(path.q)),
        root=out_dir,
        metadata={
            "depth": DEPTH_NOTE,
            "world": {"name": world.name, "params": world.params},
            "ground_truth": truth,
            "scan": visits,
        },
        trajectory=RobotTrajectory(
            steps=steps.astype(np.int64),
            times=(steps * CONTROL_DT).astype(np.float64),
            joint_positions=np.array(path.q),
            gripper_position=np.array(levels),
        ),
    )
    capture.save()
    return capture
