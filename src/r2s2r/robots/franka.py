"""Franka Panda kinematics in plain numpy.

Every deployment target (Isaac Lab, MuJoCo, the real arm) takes joint position targets,
so inverse kinematics lives here instead of in each simulator: a program policy run in
simulation and on the robot then sends the same commands, and only the plant differs.
The model is Franka's published modified-DH chain; the TCP sits between the fingertip
pads of the Franka Hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

# (a_{i-1}, d_i, alpha_{i-1}) per joint, Craig's convention, from Franka's docs.
PANDA_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
FLANGE_OFFSET = 0.107  # link7 -> flange along z
HAND_YAW = -np.pi / 4  # the Franka Hand is mounted rotated about the flange z axis
FRANKA_HAND_TCP = 0.1034  # hand frame -> fingertip pad centre along z
FRANKA_HAND_MAX_WIDTH = 0.08

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# Hand above the table, pointing down, out of the external cameras' way.
Q_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _dh(a: float, d: float, alpha: float, theta: float) -> NDArray[np.float64]:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rotation_error(R_target: NDArray, R: NDArray) -> NDArray[np.float64]:
    """Axis-angle vector (base frame) rotating ``R`` onto ``R_target``."""
    return Rotation.from_matrix(R_target @ R.T).as_rotvec()


@dataclass
class IKResult:
    """Joint solution and its remaining TCP error."""

    q: NDArray[np.float64]
    pos_error: float
    rot_error: float
    converged: bool


@dataclass
class PandaKinematics:
    """Forward/inverse kinematics of a Panda arm with a fixed tool offset."""

    tcp_offset: float = FRANKA_HAND_TCP
    q_min: NDArray[np.float64] = field(default_factory=Q_MIN.copy)
    q_max: NDArray[np.float64] = field(default_factory=Q_MAX.copy)

    def link_frames(self, q: NDArray) -> list[NDArray[np.float64]]:
        """``T_base_link_i`` for i = 1..7 followed by the TCP frame."""
        frames = []
        T = np.eye(4)
        for (a, d, alpha), theta in zip(PANDA_DH, q):
            T = T @ _dh(a, d, alpha, float(theta))
            frames.append(T)
        T_link7_tcp = np.eye(4)
        T_link7_tcp[:3, :3] = Rotation.from_euler("z", HAND_YAW).as_matrix()
        T_link7_tcp[2, 3] = FLANGE_OFFSET + self.tcp_offset
        frames.append(T @ T_link7_tcp)
        return frames

    def fk(self, q: NDArray) -> NDArray[np.float64]:
        """``T_base_tcp``."""
        return self.link_frames(q)[-1]

    def jacobian(self, q: NDArray) -> NDArray[np.float64]:
        """6x7 geometric Jacobian of the TCP (linear rows first), base frame."""
        frames = self.link_frames(q)
        p_tcp = frames[-1][:3, 3]
        J = np.zeros((6, 7))
        for i in range(7):
            z, p = frames[i][:3, 2], frames[i][:3, 3]
            J[:3, i] = np.cross(z, p_tcp - p)
            J[3:, i] = z
        return J

    def ik(
        self,
        T_target: NDArray,
        q_seed: NDArray,
        max_iters: int = 200,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        damping: float = 0.05,
        max_step: float = 0.2,
        nullspace_gain: float = 0.05,
    ) -> IKResult:
        """Damped least squares from ``q_seed``, pulled toward the seed in the nullspace
        so consecutive waypoints give continuous joint paths."""
        q_seed = np.asarray(q_seed, dtype=float)
        q = q_seed.copy()
        for _ in range(max_iters):
            T = self.fk(q)
            err = np.concatenate(
                [
                    T_target[:3, 3] - T[:3, 3],
                    rotation_error(T_target[:3, :3], T[:3, :3]),
                ]
            )
            pos_err, rot_err = float(np.linalg.norm(err[:3])), float(
                np.linalg.norm(err[3:])
            )
            if pos_err < pos_tol and rot_err < rot_tol:
                return IKResult(q, pos_err, rot_err, True)
            J = self.jacobian(q)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
            dq = J_pinv @ err
            # The damped inverse leaks into task space; project with the exact one.
            null = np.eye(7) - np.linalg.pinv(J) @ J
            dq += nullspace_gain * null @ (q_seed - q)
            scale = np.max(np.abs(dq)) / max_step
            if scale > 1.0:
                dq /= scale
            q = np.clip(q + dq, self.q_min, self.q_max)
        T = self.fk(q)
        pos_err = float(np.linalg.norm(T_target[:3, 3] - T[:3, 3]))
        rot_err = float(np.linalg.norm(rotation_error(T_target[:3, :3], T[:3, :3])))
        return IKResult(
            q, pos_err, rot_err, bool(pos_err < pos_tol and rot_err < rot_tol)
        )

    def solve(
        self,
        T_target: NDArray,
        q_seed: NDArray,
        n_seeds: int = 24,
        seed: int = 0,
    ) -> IKResult:
        """IK from ``q_seed`` and random seeds; the converged solution farthest from the
        joint limits (in shares of each joint's range).

        For goals far from the current pose, where following a Cartesian path would drag
        the arm into its limits; move there in joint space.
        """
        rng = np.random.default_rng(seed)
        span = self.q_max - self.q_min
        seeds = [np.asarray(q_seed, float)] + [
            self.q_min + 0.1 * span + rng.random(7) * 0.8 * span for _ in range(n_seeds)
        ]
        best, best_margin = None, -np.inf
        for q0 in seeds:
            sol = self.ik(T_target, q0, max_iters=300)
            if not sol.converged:
                continue
            margin = float(
                np.min(np.minimum(sol.q - self.q_min, self.q_max - sol.q) / span)
            )
            if margin > best_margin:
                best, best_margin = sol, margin
        return best if best is not None else self.ik(T_target, q_seed)
