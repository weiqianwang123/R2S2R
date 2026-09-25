"""A scripted top-down pick: the smallest program policy that exercises the loop.

The policy only knows the reconstructed :class:`~r2s2r.structs.SceneSpec` (object meshes
and poses in the robot base frame) and the robot interface, which is all a code-writing
agent would get, so the same program runs unchanged in Isaac Lab and on the deployment
target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.policy.robot import RobotInterface
from r2s2r.refine import urdf_visual_points
from r2s2r.robots.franka import FRANKA_HAND_MAX_WIDTH, Q_READY
from r2s2r.structs import ObjectSpec, SceneSpec


@dataclass
class GraspPlan:
    """A top-down grasp on one object."""

    object_name: str
    T_base_tcp: NDArray[np.float64]
    width: float  # object extent along the closing axis
    top_z: float
    bottom_z: float

    def as_dict(self) -> dict:
        """JSON-friendly form."""
        d = asdict(self)
        d["T_base_tcp"] = self.T_base_tcp.tolist()
        return d


def find_object(scene: SceneSpec, query: str) -> ObjectSpec:
    """Exact name, else the unique object whose name or category contains ``query``."""
    for obj in scene.objects:
        if obj.name == query:
            return obj
    q = query.lower()
    hits = [o for o in scene.objects if q in o.name.lower() or q in o.category.lower()]
    if len(hits) != 1:
        names = ", ".join(o.name for o in scene.objects)
        raise KeyError(f"{query!r} matches {len(hits)} objects; scene has: {names}")
    return hits[0]


def object_points(obj: ObjectSpec, n: int = 3000) -> NDArray[np.float64]:
    """Visual surface samples in the robot base frame."""
    pts = urdf_visual_points(obj.asset_path, n)
    return pts @ obj.T_base_obj[:3, :3].T + obj.T_base_obj[:3, 3]


def plan_top_down_grasp(
    scene: SceneSpec,
    target: str,
    reference_x_axis: NDArray | None = None,
    finger_depth: float = 0.03,
    clearance: float = 0.012,
    width_margin: float = 0.01,
) -> GraspPlan:
    """Close across the object's narrowest horizontal side, ``finger_depth`` below its
    top but at least ``clearance`` above its bottom (the fingertips reach ~9 mm below
    the TCP).

    ``reference_x_axis`` picks which of the two symmetric hand yaws to use (the one
    needing the least wrist rotation).
    """
    obj = find_object(scene, target)
    pts = object_points(obj)
    rect = cv2.minAreaRect(pts[:, :2].astype(np.float32))
    (cx, cy), (w, h), angle = rect
    theta = np.deg2rad(angle)
    side_w = np.array([np.cos(theta), np.sin(theta)])
    side_h = np.array([-np.sin(theta), np.cos(theta)])
    closing, width = (side_w, w) if w <= h else (side_h, h)
    if width > FRANKA_HAND_MAX_WIDTH - width_margin:
        raise ValueError(f"{obj.name} is {width:.3f} m wide: too wide for the hand")

    y_axis = np.array([closing[0], closing[1], 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(y_axis, z_axis)
    if reference_x_axis is not None and x_axis @ reference_x_axis < 0:
        x_axis, y_axis = -x_axis, -y_axis
    top_z, bottom_z = float(pts[:, 2].max()), float(pts[:, 2].min())
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = [cx, cy, max(top_z - finger_depth, bottom_z + clearance)]
    return GraspPlan(obj.name, T, float(width), top_z, bottom_z)


def _raised(T: NDArray, dz: float) -> NDArray[np.float64]:
    out = T.copy()
    out[2, 3] += dz
    return out


def pick_up(
    robot: RobotInterface,
    scene: SceneSpec,
    target: str,
    approach: float = 0.12,
    lift: float = 0.15,
) -> dict:
    """Go to the ready pose, open, approach from above, descend, close, lift, and
    report.

    Starting from a fixed pose makes the command stream independent of where the robot
    happened to be.
    """
    robot.set_gripper(False, 0.5)
    robot.move_joints(Q_READY)
    plan = plan_top_down_grasp(scene, target, reference_x_axis=robot.tcp_pose()[:3, 0])
    grasp = plan.T_base_tcp
    robot.move_tcp(_raised(grasp, approach))
    grasp_err = robot.move_tcp(grasp, speed=0.08)
    robot.set_gripper(True, 1.0)
    robot.move_tcp(_raised(grasp, lift), speed=0.1)
    robot.wait(1.0)
    width = robot.gripper_width()
    return {
        "target": plan.object_name,
        "grasp": plan.as_dict(),
        "grasp_ik_error_m": grasp_err,
        "gripper_width_after_lift": width,
        # Fingers that closed fully hold nothing.
        "holding": bool(0.004 < width < plan.width + 0.01),
    }
