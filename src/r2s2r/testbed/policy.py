"""The robot interface program policies are written against, motion primitives, a
top-down pick program, and how a pick is scored and saved.

A simulator (MuJoCo standing in for the real world, Isaac Lab) implements
:class:`RobotInterface` for one :class:`~r2s2r.robots.spec.RobotSpec`: hold
arm joint targets and a gripper opening (0 open, 1 closed) for one control period, and
report the measured ones. Everything above that (inverse kinematics on the robot's own
model, Cartesian interpolation, gripper timing) is here, so a policy sends the same
commands wherever it runs.

The pick program only knows the reconstructed :class:`~r2s2r.structs.SceneSpec` and the
robot, which is all a code-writing agent would get. It grasps from above, across the
object's narrowest side, with the gripper's closing axis, finger reach and empty
closure read off the robot's model.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from r2s2r.assets import object_points
from r2s2r.mjrender import geom_mesh, mujoco
from r2s2r.robots.model import RobotModel, in_subtree
from r2s2r.robots.spec import RobotSpec
from r2s2r.structs import ObjectSpec, SceneSpec
from r2s2r.transforms import invert, make_transform, transform_points

LIFT_SUCCESS_M = 0.05  # the target must rise this much to count as picked
# A gripper holds something when it stopped this much short of where its fingers meet.
HOLDING_MARGIN = 0.05
CONTROL_DT = 0.02  # seconds per command (and per step of a MuJoCo recording)
VIDEO_EVERY = 2  # control steps per video frame
VIDEO_FPS = 1 / (CONTROL_DT * VIDEO_EVERY)
SHEET_FRAMES, SHEET_WIDTH = 6, 1920  # a video's contact sheet


@dataclass
class CommandLog:
    """Every command sent, for comparing the simulators."""

    t: list[float] = field(default_factory=list)
    q: list[list[float]] = field(default_factory=list)
    gripper: list[float] = field(default_factory=list)  # 0 open, 1 closed

    def as_dict(self) -> dict[str, list]:
        """JSON-friendly form."""
        return {"t": self.t, "q": self.q, "gripper": self.gripper}


class RobotInterface(ABC):
    """A one-armed robot with a gripper, driven by joint position targets."""

    def __init__(self, robot: RobotSpec) -> None:
        self.robot = robot
        self.arm = robot.arm  # raises for more arms than one
        self.model = RobotModel(robot)  # the robot alone, for kinematics
        self.log = CommandLog()
        self._q_cmd: NDArray[np.float64] | None = None
        self._level = 0.0
        self._t = 0.0

    # -------------------------------------------------- implemented per target
    @abstractmethod
    def joint_positions(self) -> NDArray[np.float64]:
        """Measured arm joints (dof,)."""

    @abstractmethod
    def gripper_level(self) -> float:
        """Measured gripper opening, 0 open to 1 closed."""

    @abstractmethod
    def _hold(self, q: NDArray[np.float64], level: float) -> None:
        """Track these targets for one :data:`CONTROL_DT`."""

    # ---------------------------------------------------------------- commands
    @property
    def q_command(self) -> NDArray[np.float64]:
        """Last commanded arm joints (the measured ones before any command)."""
        if self._q_cmd is None:
            self._q_cmd = self.joint_positions().copy()
        return self._q_cmd

    def step(self, q: NDArray) -> None:
        """Send one command (the arm at ``q``, the gripper as set) and advance one
        control period."""
        self._q_cmd = np.asarray(q, dtype=float).copy()
        self._hold(self._q_cmd, self._level)
        self._t += CONTROL_DT
        self.log.t.append(round(self._t, 6))
        self.log.q.append([float(v) for v in self._q_cmd])
        self.log.gripper.append(self._level)

    def tcp_pose(self) -> NDArray[np.float64]:
        """Commanded ``T_base_tcp``."""
        self.model.set(self.q_command)
        return self.model.tcp_pose()

    def wait(self, seconds: float) -> None:
        """Hold the current command."""
        for _ in range(max(1, int(round(seconds / CONTROL_DT)))):
            self.step(self.q_command)

    def set_gripper(self, level: float, seconds: float = 0.8) -> None:
        """Open (0) or close (1), holding the arm still while the fingers move."""
        self._level = float(level)
        self.wait(seconds)

    def move_joints(
        self, q_goal: NDArray, speed: float = 0.6, settle: float = 0.3
    ) -> None:
        """Joint-space move (smoothstep timing), ``speed`` in rad/s for the joint that
        moves most."""
        q_start = self.q_command
        q_goal = np.asarray(q_goal, dtype=float)
        duration = max(float(np.max(np.abs(q_goal - q_start))) / speed, CONTROL_DT)
        n = int(np.ceil(duration / CONTROL_DT))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            self.step((1 - s) * q_start + s * q_goal)
        self.wait(settle)

    def move_tcp(
        self,
        T_goal: NDArray,
        speed: float = 0.15,
        angular_speed: float = 1.0,
        settle: float = 0.3,
    ) -> float:
        """Straight-line Cartesian move (smoothstep timing, slerped rotation).

        Returns the TCP position error of the last IK solve.
        """
        T_start = self.tcp_pose()
        dist = float(np.linalg.norm(T_goal[:3, 3] - T_start[:3, 3]))
        angle = float(
            np.linalg.norm(
                Rotation.from_matrix(T_goal[:3, :3] @ T_start[:3, :3].T).as_rotvec()
            )
        )
        duration = max(dist / speed, angle / angular_speed, CONTROL_DT)
        n = int(np.ceil(duration / CONTROL_DT))
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(np.stack([T_start[:3, :3], T_goal[:3, :3]])),
        )
        q = self.q_command
        pos_error = 0.0
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            T = np.eye(4)
            T[:3, :3] = slerp([s]).as_matrix()[0]
            T[:3, 3] = (1 - s) * T_start[:3, 3] + s * T_goal[:3, 3]
            sol = self.model.ik(T, q)
            q, pos_error = sol.q, sol.pos_error
            self.step(q)
        self.wait(settle)
        return pos_error


# ------------------------------------------------------------------- grasping
@dataclass
class GripperGeometry:
    """What a grasp plan needs of a gripper, in the TCP frame, and where it stops
    closed on nothing."""

    closing_axis: NDArray[np.float64]  # the fingers close along it (unit, in x-y)
    tip_depth: float  # how far the fingers reach beyond the TCP along its z (m)
    empty_level: float  # the opening (0 open, 1 closed) at which the fingers meet


def gripper_geometry(model: RobotModel) -> GripperGeometry:
    """Read off the model: the fingers' colliding geoms, open and closed, in the TCP
    frame. They close along the main direction the geoms move in the TCP's x-y plane;
    the tips are their farthest point along its z at either opening; the two sides
    (the geoms moving either way along the closing axis) meet at the empty level."""
    m, d = model.model, model.data
    roots = [m.jnt_bodyid[m.joint(j).id] for j in model.grippers[0].joints]
    geoms = [
        g
        for g in range(m.ngeom)
        if (m.geom_contype[g] or m.geom_conaffinity[g])
        and any(in_subtree(m, int(m.geom_bodyid[g]), int(r)) for r in roots)
    ]
    q = model.data.qpos[model.arm_qadr].copy()
    centres, tip = [], 0.0
    for level in (0.0, 1.0):
        model.set(q, level)
        T_tcp_base = invert(model.tcp_pose())
        centres.append(transform_points(T_tcp_base, d.geom_xpos[geoms]))
        for g in geoms:
            mesh = geom_mesh(m, g)
            if mesh is None:
                continue
            b = m.geom_bodyid[g]
            T_base_body = make_transform(d.xmat[b].reshape(3, 3), d.xpos[b])
            z = transform_points(T_tcp_base @ T_base_body, mesh.vertices)[:, 2]
            tip = max(tip, float(z.max()))
    _, _, vt = np.linalg.svd((centres[1] - centres[0])[:, :2])
    closing_axis = np.r_[vt[0], 0.0]
    moves = (centres[1] - centres[0]) @ closing_axis
    sides = [[g for g, v in zip(geoms, moves) if v * sign > 0] for sign in (1, -1)]

    def meet(level: float) -> bool:
        model.set(q, level)
        # A small distmax: with a large one mj_geomDistance can say 0 for far pairs.
        return any(
            mujoco.mj_geomDistance(m, d, a, b, 1e-3, None) <= 0.0
            for a in sides[0]
            for b in sides[1]
        )

    apart, met = 0.0, 1.0
    if not meet(met):
        apart = met
    while met - apart > 1e-4:
        mid = (apart + met) / 2
        apart, met = (apart, mid) if meet(mid) else (mid, met)
    model.set(q, 0.0)
    return GripperGeometry(closing_axis, tip, apart)


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


def plan_top_down_grasp(
    scene: SceneSpec,
    target: str,
    gripper: GripperGeometry,
    max_opening: float,
    R_reference: NDArray | None = None,
    finger_depth: float = 0.03,
    clearance: float = 0.003,
    width_margin: float = 0.01,
) -> GraspPlan:
    """Close across the object's narrowest horizontal side, ``finger_depth`` below its
    top but with the fingertips ``clearance`` above its bottom.

    ``R_reference`` (a TCP rotation) picks which of the two symmetric yaws to use: the
    one nearer to it.
    """
    obj = find_object(scene, target)
    pts = object_points(obj)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts[:, :2].astype(np.float32))
    theta = np.deg2rad(angle)
    side_w = np.array([np.cos(theta), np.sin(theta), 0.0])
    side_h = np.array([-np.sin(theta), np.cos(theta), 0.0])
    closing, width = (side_w, w) if w <= h else (side_h, h)
    if width > max_opening - width_margin:
        raise ValueError(f"{obj.name} is {width:.3f} m wide: too wide for the gripper")
    down = np.array([0.0, 0.0, -1.0])
    tcp_axes = np.column_stack(
        [gripper.closing_axis, np.cross([0, 0, 1.0], gripper.closing_axis), [0, 0, 1]]
    )
    options = [
        np.column_stack([c, np.cross(down, c), down]) @ tcp_axes.T
        for c in (closing, -closing)
    ]
    if R_reference is not None:
        options.sort(key=lambda R: -float(np.trace(R_reference.T @ R)))
    top_z, bottom_z = float(pts[:, 2].max()), float(pts[:, 2].min())
    T = np.eye(4)
    T[:3, :3] = options[0]
    T[:3, 3] = [
        cx,
        cy,
        max(top_z - finger_depth, bottom_z + gripper.tip_depth + clearance),
    ]
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
) -> dict[str, Any]:
    """From the robot's home pose, open, approach from above, descend, close, lift,
    and report.

    Starting from a fixed pose makes the commands independent of where the robot
    happened to be.
    """
    robot.set_gripper(0.0, 0.5)
    robot.move_joints(np.asarray(robot.robot.home_q))
    geometry = gripper_geometry(robot.model)
    plan = plan_top_down_grasp(
        scene,
        target,
        geometry,
        robot.arm.max_opening,
        robot.tcp_pose()[:3, :3],
    )
    grasp = plan.T_base_tcp
    robot.move_tcp(_raised(grasp, approach))
    grasp_err = robot.move_tcp(grasp, speed=0.08)
    robot.set_gripper(1.0, 1.0)
    robot.move_tcp(_raised(grasp, lift), speed=0.1)
    robot.wait(1.0)
    level = robot.gripper_level()
    return {
        "target": plan.object_name,
        "grasp": plan.as_dict(),
        "grasp_ik_error_m": grasp_err,
        "gripper_level_after_lift": level,
        "holding": bool(level < geometry.empty_level - HOLDING_MARGIN),
    }


# -------------------------------------------------------------------- scoring
def score_lift(
    before: dict[str, NDArray], after: dict[str, NDArray], target: str
) -> dict[str, Any]:
    """Per-object height change and displacement, and whether ``target`` rose."""
    lift = {n: float(after[n][2] - before[n][2]) for n in before}
    return {
        "object_lift_m": lift,
        "object_shift_m": {
            n: float(np.linalg.norm(after[n] - before[n])) for n in before
        },
        "success": bool(lift[target] > LIFT_SUCCESS_M),
    }


def summarize(result: dict[str, Any]) -> str:
    """One line: verdict, what the policy picked, how far every object rose."""
    verdict = "SUCCESS" if result["success"] else "FAILURE"
    lift = ", ".join(
        f"{n} {dz * 100:+.1f} cm" for n, dz in result["object_lift_m"].items()
    )
    return f"{verdict}: picked {result['policy']['target']!r}; lifted {lift}"


def save_rollout(out_dir: Path, result: dict[str, Any], log: CommandLog) -> None:
    """``result.json`` and ``commands.json`` (every command sent)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "commands.json").write_text(json.dumps(log.as_dict()), encoding="utf-8")


class VideoRecorder:
    """Collect RGB frames; write an mp4 and a contact sheet of evenly spaced frames."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.frames: list[NDArray[np.uint8]] = []

    def add(self, rgb: NDArray[np.uint8]) -> None:
        """Append one RGB frame."""
        self.frames.append(np.ascontiguousarray(rgb[..., :3]))

    def close(self) -> None:
        """Write the video and ``<stem>_sheet.jpg``."""
        if not self.frames:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        h, w = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter.fourcc(*"mp4v"), VIDEO_FPS, (w, h)
        )
        for frame in self.frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        idx = np.linspace(0, len(self.frames) - 1, SHEET_FRAMES).round().astype(int)
        cols = 3
        rows = int(np.ceil(len(idx) / cols))
        tile_w = SHEET_WIDTH // cols
        tile_h = int(round(h * tile_w / w))
        sheet = np.zeros((rows * tile_h, cols * tile_w, 3), np.uint8)
        for k, i in enumerate(idx):
            tile = cv2.resize(
                self.frames[i], (tile_w, tile_h), interpolation=cv2.INTER_AREA
            )
            cv2.putText(
                tile,
                f"t={i / VIDEO_FPS:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            r, c = divmod(k, cols)
            sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
        cv2.imwrite(
            str(self.path.with_name(f"{self.path.stem}_sheet.jpg")),
            cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        )
