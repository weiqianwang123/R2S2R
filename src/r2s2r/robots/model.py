"""A robot's MuJoCo model, posed by its arm joints and gripper opening.

:class:`RobotModel` compiles a :class:`~r2s2r.robots.spec.RobotSpec`'s MJCF and sets the
arm joints and the gripper (0 open, 1 closed; the other gripper joints follow the
driver as the spec says), gives the pose of any body, site or camera, and solves
inverse kinematics for one of them by damped least squares. Every simulator takes joint
targets, so kinematics lives here, once, from the same model the masker and the viewer
draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.mjrender import CV_TO_MJ, mujoco
from r2s2r.robots.spec import RobotSpec
from r2s2r.transforms import make_transform

FRAME_KINDS = ("body", "site", "camera")


@dataclass
class IKResult:
    """An arm solution and how far its frame stays from the target."""

    q: NDArray[np.float64]
    pos_error: float  # metres
    rot_error: float  # radians
    success: bool


class GripperPoser:
    """Every gripper joint's position for an opening (0 open, 1 closed).

    ``equality`` followers come from the MJCF's joint equalities on the driver
    (joint1 = polynomial of joint2, about their reference positions). ``simulate``
    solves every joint in the gripper (the subtree of the driver's parent body, so a
    world's other joints are left alone) by simulating the gripper's actuator at a few
    openings (gravity off, arm held), once, and interpolates.
    """

    LEVELS = np.linspace(0.0, 1.0, 9)

    def __init__(self, model: Any, robot: RobotSpec) -> None:
        self.gripper = robot.gripper
        driver = model.joint(self.gripper.driver).id
        if self.gripper.followers == "equality":
            self.joints = [self.gripper.driver]
            self._poly: list[tuple[float, float, NDArray[np.float64]]] = []
            for i in range(model.neq):
                if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                    continue
                if model.eq_obj2id[i] != driver:
                    continue
                follower = int(model.eq_obj1id[i])
                self.joints.append(model.joint(follower).name)
                self._poly.append(
                    (
                        float(model.qpos0[model.jnt_qposadr[follower]]),
                        float(model.qpos0[model.jnt_qposadr[driver]]),
                        np.asarray(model.eq_data[i][:5], float),
                    )
                )
            return
        base = int(model.body_parentid[model.jnt_bodyid[driver]])
        self.joints = [
            model.joint(j).name
            for j in range(model.njnt)
            if in_subtree(model, int(model.jnt_bodyid[j]), base)
        ]
        self._table = np.stack([self._solve(model, robot, lv) for lv in self.LEVELS])

    def positions(self, level: float) -> NDArray[np.float64]:
        """Positions of :attr:`joints` at ``level``."""
        level = float(np.clip(level, 0.0, 1.0))
        if self.gripper.followers == "simulate":
            return np.array(
                [np.interp(level, self.LEVELS, col) for col in self._table.T]
            )
        x = self.gripper.driver_at(level)
        values = [x]
        for y0, x0, c in self._poly:
            dx = x - x0
            values.append(
                y0 + c[0] + c[1] * dx + c[2] * dx**2 + c[3] * dx**3 + c[4] * dx**4
            )
        return np.array(values)

    def _solve(self, model: Any, robot: RobotSpec, level: float) -> NDArray[np.float64]:
        data = mujoco.MjData(model)
        actuator = model.actuator(self.gripper.actuator).id
        data.ctrl[actuator] = self.gripper.ctrl_at(level)
        arm_q = [model.joint(j).qposadr[0] for j in robot.arm_joints]
        arm_v = [model.joint(j).dofadr[0] for j in robot.arm_joints]
        gravity = model.opt.gravity.copy()
        model.opt.gravity[:] = 0.0
        try:
            for _ in range(1500):
                data.qpos[arm_q] = 0.0
                data.qvel[arm_v] = 0.0
                mujoco.mj_step(model, data)
        finally:
            model.opt.gravity[:] = gravity
        driver = data.qpos[model.joint(self.gripper.driver).qposadr[0]]
        bounds = sorted((self.gripper.open, self.gripper.closed))
        if not bounds[0] - 0.05 <= driver <= bounds[1] + 0.05:
            raise RuntimeError(
                f"{robot.name}: the gripper driver settled at {driver:.3f}, outside "
                f"{bounds} at opening {level}"
            )
        return np.array([data.qpos[model.joint(j).qposadr[0]] for j in self.joints])


def in_subtree(model: Any, body: int, root: int) -> bool:
    """Whether ``body`` is ``root`` or hangs below it."""
    while body != root:
        if body == 0:
            return False
        body = int(model.body_parentid[body])
    return True


class RobotModel:
    """A compiled robot, posed with :meth:`set`.

    ``mjspec`` is the robot's ``MjSpec`` when the caller edits it first (say, adds a
    camera); by default the spec's own.
    """

    def __init__(self, robot: RobotSpec, mjspec: Any | None = None) -> None:
        self.robot = robot
        self.model = (robot.mjcf() if mjspec is None else mjspec).compile()
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(j) for j in robot.arm_joints]
        self.arm_qadr = np.array([j.qposadr[0] for j in joints])
        self.arm_dofadr = np.array([j.dofadr[0] for j in joints])
        limited = np.array([bool(self.model.jnt_limited[j.id]) for j in joints])
        ranges = np.array([self.model.jnt_range[j.id] for j in joints])
        self.q_min = np.where(limited, ranges[:, 0], -np.inf)
        self.q_max = np.where(limited, ranges[:, 1], np.inf)
        self.gripper = GripperPoser(self.model, robot)
        self.gripper_qadr = np.array(
            [self.model.joint(j).qposadr[0] for j in self.gripper.joints]
        )
        self.T_body_tcp = make_transform(np.eye(3), [0.0, 0.0, robot.tcp_offset])
        self.set(np.asarray(robot.home_q), 0.0)

    def set(self, q: NDArray, level: float | None = None) -> None:
        """Arm joints ``q`` and, when given, gripper opening ``level``; updates the
        kinematics."""
        self.data.qpos[self.arm_qadr] = q
        if level is not None:
            self.data.qpos[self.gripper_qadr] = self.gripper.positions(level)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def pose(self, name: str, kind: str = "body") -> NDArray[np.float64]:
        """``T_base_frame`` of a body, site or camera (cameras in the OpenCV convention:
        x right, y down, z forward)."""
        return self._frame(name, kind)[0]

    def tcp_pose(self) -> NDArray[np.float64]:
        """``T_base_tcp``."""
        return self._frame(None, "body")[0]

    def ik(
        self,
        T_target: NDArray,
        q_seed: NDArray,
        frame: str | None = None,
        kind: str = "body",
        max_iters: int = 200,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        damping: float = 0.05,
        max_step: float = 0.2,
        nullspace_gain: float = 0.05,
    ) -> IKResult:
        """Arm joints that put ``frame`` (a body, site or camera by ``kind``; the TCP
        when None) at ``T_target``.

        Damped least squares from ``q_seed``, within the joint limits, pulled toward
        the seed in the nullspace so that consecutive targets give continuous joint
        paths. The gripper stays as set; the model is left at the solution.
        """
        q_seed = np.asarray(q_seed, dtype=float)
        q = np.clip(q_seed, self.q_min, self.q_max)
        for _ in range(max_iters):
            self.set(q)
            T, body = self._frame(frame, kind)
            err = np.concatenate(
                [
                    T_target[:3, 3] - T[:3, 3],
                    Rotation.from_matrix(T_target[:3, :3] @ T[:3, :3].T).as_rotvec(),
                ]
            )
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                break
            J = self._jacobian(T[:3, 3], body)
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            # The damped inverse leaks into task space; project with the exact one.
            null = np.eye(len(q)) - np.linalg.pinv(J) @ J
            dq += nullspace_gain * null @ (q_seed - q)
            scale = np.max(np.abs(dq)) / max_step
            if scale > 1.0:
                dq /= scale
            q = np.clip(q + dq, self.q_min, self.q_max)
        self.set(q)
        T, _ = self._frame(frame, kind)
        pos_err = float(np.linalg.norm(T_target[:3, 3] - T[:3, 3]))
        rot_err = float(
            np.linalg.norm(
                Rotation.from_matrix(T_target[:3, :3] @ T[:3, :3].T).as_rotvec()
            )
        )
        return IKResult(q, pos_err, rot_err, pos_err < pos_tol and rot_err < rot_tol)

    def _frame(self, name: str | None, kind: str) -> tuple[NDArray[np.float64], int]:
        """The frame's pose and the body it moves with."""
        m, d = self.model, self.data
        if name is None:
            b = m.body(self.robot.tcp_body).id
            T = make_transform(d.xmat[b].reshape(3, 3), d.xpos[b]) @ self.T_body_tcp
            return T, b
        if kind == "body":
            b = m.body(name).id
            return make_transform(d.xmat[b].reshape(3, 3), d.xpos[b]), b
        if kind == "site":
            s = m.site(name).id
            T = make_transform(d.site_xmat[s].reshape(3, 3), d.site_xpos[s])
            return T, int(m.site_bodyid[s])
        if kind == "camera":
            c = m.camera(name).id
            T = make_transform(d.cam_xmat[c].reshape(3, 3) @ CV_TO_MJ, d.cam_xpos[c])
            return T, int(m.cam_bodyid[c])
        raise ValueError(f"unknown frame kind {kind!r} {FRAME_KINDS}")

    def _jacobian(self, point: NDArray, body: int) -> NDArray[np.float64]:
        """6 x dof Jacobian (linear rows first) of ``point`` moving with ``body``."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(
            self.model, self.data, jacp, jacr, np.ascontiguousarray(point), body
        )
        return np.vstack([jacp[:, self.arm_dofadr], jacr[:, self.arm_dofadr]])
