"""The robot interface program policies are written against, and motion primitives.

A deployment target (Isaac Lab, MuJoCo standing in for the real world, the real arm)
implements :class:`RobotInterface`: hold arm joint targets and a gripper command for one
control period. Everything above that (Cartesian interpolation, inverse kinematics,
gripper timing) is here, so a policy produces the same command stream wherever it runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from r2s2r.robots.franka import PandaKinematics


@dataclass
class CommandLog:
    """Every command sent, for comparing deployments."""

    t: list[float] = field(default_factory=list)
    q: list[list[float]] = field(default_factory=list)
    gripper_closed: list[bool] = field(default_factory=list)

    def as_dict(self) -> dict[str, list]:
        """JSON-friendly form."""
        return {"t": self.t, "q": self.q, "gripper_closed": self.gripper_closed}


class RobotInterface(ABC):
    """A Panda arm with a two-finger gripper, driven by joint position targets."""

    control_dt: float = 0.02

    def __init__(self, kinematics: PandaKinematics | None = None) -> None:
        self.kin = kinematics or PandaKinematics()
        self.log = CommandLog()
        self._q_cmd: NDArray[np.float64] | None = None
        self._gripper_closed = False
        self._t = 0.0

    # -------------------------------------------------- implemented per target
    @abstractmethod
    def joint_positions(self) -> NDArray[np.float64]:
        """Measured arm joints (7,)."""

    @abstractmethod
    def gripper_width(self) -> float:
        """Measured finger opening in meters."""

    @abstractmethod
    def _hold(self, q: NDArray[np.float64], gripper_closed: bool) -> None:
        """Track these targets for one ``control_dt``."""

    # ---------------------------------------------------------------- commands
    @property
    def q_command(self) -> NDArray[np.float64]:
        """Last commanded arm joints (the measured ones before any command)."""
        if self._q_cmd is None:
            self._q_cmd = self.joint_positions().copy()
        return self._q_cmd

    def step(self, q: NDArray, gripper_closed: bool | None = None) -> None:
        """Send one command and advance one control period."""
        self._q_cmd = np.asarray(q, dtype=float).copy()
        if gripper_closed is not None:
            self._gripper_closed = gripper_closed
        self._hold(self._q_cmd, self._gripper_closed)
        self._t += self.control_dt
        self.log.t.append(round(self._t, 6))
        self.log.q.append([float(v) for v in self._q_cmd])
        self.log.gripper_closed.append(self._gripper_closed)

    def tcp_pose(self) -> NDArray[np.float64]:
        """Commanded ``T_base_tcp``."""
        return self.kin.fk(self.q_command)

    def wait(self, seconds: float) -> None:
        """Hold the current command."""
        for _ in range(max(1, int(round(seconds / self.control_dt)))):
            self.step(self.q_command)

    def set_gripper(self, closed: bool, seconds: float = 0.8) -> None:
        """Open or close, holding the arm still while the fingers move."""
        self._gripper_closed = closed
        self.wait(seconds)

    def move_joints(
        self, q_goal: NDArray, speed: float = 0.6, settle: float = 0.3
    ) -> None:
        """Joint-space move (smoothstep timing), ``speed`` in rad/s for the joint that
        moves most."""
        q_start = self.q_command
        q_goal = np.asarray(q_goal, dtype=float)
        duration = max(float(np.max(np.abs(q_goal - q_start))) / speed, self.control_dt)
        n = int(np.ceil(duration / self.control_dt))
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
        duration = max(dist / speed, angle / angular_speed, self.control_dt)
        n = int(np.ceil(duration / self.control_dt))
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
            sol = self.kin.ik(T, q)
            q, pos_error = sol.q, sol.pos_error
            self.step(q)
        self.wait(settle)
        return pos_error
