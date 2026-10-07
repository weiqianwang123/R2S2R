"""What r2s2r needs to know about a robot, one :class:`RobotSpec` per embodiment.

A spec names the robot's models (a MuJoCo MJCF for masking, the viewer and the
testbed; an Isaac Lab articulation for settling, replaying and picking) and its arms:
which of the models' joints each has, how its gripper's joints follow its opening, and
where its tool centre point is. Its base frame is the MuJoCo world, the model's base
link and the Isaac environment origin alike, so poses need no conversion between them.

Models load lazily, so specs import without MuJoCo, Isaac Lab or the assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from r2s2r.paths import CACHE_DIR

# mujoco_menagerie, fetched by scripts/setup/fetch_mujoco_assets.sh.
MENAGERIE_DIR = CACHE_DIR / "mujoco_menagerie"
# robotiq/isaacsim_assets, fetched by scripts/setup/fetch_robotiq_isaac.sh.
ROBOTIQ_ISAAC_DIR = CACHE_DIR / "robotiq_isaacsim_assets"
FOLLOWER_MODES = ("equality", "simulate")


@dataclass(frozen=True)
class GripperSpec:
    """A gripper's opening, from 0 (open) to 1 (closed), in MuJoCo and in Isaac Lab.

    In MuJoCo the ``driver`` joint goes from ``open`` to ``closed``, driven by
    ``actuator`` from ``ctrl[0]`` to ``ctrl[1]``; the other gripper joints follow it
    either linearly, by the MJCF's ``<equality joint polycoef>`` constraints
    (``followers="equality"``), or by a table solved once by simulating the actuator at
    a few openings (``followers="simulate"``, for a closed linkage).

    In Isaac Lab ``isaac_driver`` carries the only gripper actuator (a drive on a mimic
    joint fights its constraint). ``isaac_joints`` gives every coupled joint's
    (open, closed) position, set linearly with the opening; None sets the driver alone,
    open at its lower soft limit and closed at its upper one, and lets the linkage
    follow in simulation.

    ``speed`` (m/s) is how fast the real gripper's opening changes, as its controller
    moves it; a simulation commanding a new opening moves its target there at that
    speed (None: at once, as fast as its actuator).
    """

    driver: str
    open: float
    closed: float
    followers: str
    actuator: str
    ctrl: tuple[float, float]
    isaac_driver: str
    isaac_joints: Mapping[str, tuple[float, float]] | None = None
    speed: float | None = None

    def __post_init__(self) -> None:
        if self.followers not in FOLLOWER_MODES:
            raise ValueError(
                f"followers must be one of {FOLLOWER_MODES}, not {self.followers!r}"
            )

    def driver_at(self, level: float) -> float:
        """The MuJoCo driver joint's position at opening ``level`` (clipped to 0-1)."""
        return self.open + min(max(float(level), 0.0), 1.0) * (self.closed - self.open)

    def level_of(self, driver: float) -> float:
        """The opening (clipped to 0-1) at the MuJoCo driver joint position
        ``driver``."""
        level = (float(driver) - self.open) / (self.closed - self.open)
        return min(max(level, 0.0), 1.0)

    def ctrl_at(self, level: float) -> float:
        """The MuJoCo actuator's control at opening ``level``."""
        lo, hi = self.ctrl
        return lo + float(level) * (hi - lo)


def _no_position_control(mjspec: Any) -> None:
    """For MJCFs whose arm actuators already hold joint positions."""
    del mjspec


@dataclass(frozen=True)
class ArmSpec:
    """One arm and the gripper on it.

    ``joints`` are its joints in the MJCF, in the order captures record them, and
    ``isaac_joints`` the same joints in the Isaac Lab articulation; ``home_q`` is where
    they rest. The tool centre point is ``tcp_offset`` along ``tcp_body``'s z axis,
    between the fingertips when closed; ``max_opening`` is the widest object the gripper
    takes.
    """

    joints: tuple[str, ...]
    home_q: tuple[float, ...]
    isaac_joints: tuple[str, ...]
    gripper: GripperSpec
    tcp_body: str
    tcp_offset: float
    max_opening: float

    def __post_init__(self) -> None:
        if not len(self.joints) == len(self.home_q) == len(self.isaac_joints):
            raise ValueError(
                f"arm {self.joints}: home_q and isaac_joints differ in length"
            )

    @property
    def gripper_ctrl_rate(self) -> float:
        """How fast (per second) the gripper's MuJoCo control moves for its opening to
        change at the real gripper's speed; infinite (at once) without one."""
        gripper = self.gripper
        if gripper.speed is None:
            return float("inf")
        return abs(gripper.ctrl[1] - gripper.ctrl[0]) * gripper.speed / self.max_opening


@dataclass(frozen=True)
class RobotSpec:
    """One embodiment (``Capture.embodiment`` names it), of one arm or more.

    ``mjcf()`` gives the robot alone as a fresh ``mujoco.MjSpec``, base at the origin,
    visual geoms in group 2. ``arms`` are its arms in the order captures record them:
    a capture's joint positions are every arm's joints in turn (:attr:`arm_joints`),
    its gripper position one opening per arm, a number for a one-armed robot
    (:meth:`gripper_levels`). ``position_control(mjspec)`` makes the arms' actuators
    hold joint positions for the testbed's physics.

    ``isaac_cfg()`` gives the Isaac Lab ``ArticulationCfg`` (at the origin, the arms on
    a stiff joint-position PD, gravity off; imports Isaac Lab, so call it in an Isaac
    process only). ``isaac_post_spawn(prim_path)``, when given, fixes the spawned robot
    up before the simulation starts. Isaac cameras render nothing nearer than
    ``isaac_camera_near`` (m), where a USD models parts the real cameras never show.
    """

    name: str
    mjcf: Callable[[], Any]
    arms: tuple[ArmSpec, ...]
    isaac_cfg: Callable[[], Any]
    position_control: Callable[[Any], None] = _no_position_control
    isaac_post_spawn: Callable[[str], None] | None = None
    isaac_camera_near: float = 0.02

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError(f"{self.name} has no arm")

    @property
    def arm_joints(self) -> tuple[str, ...]:
        """Every arm's joints, arm after arm."""
        return tuple(j for arm in self.arms for j in arm.joints)

    @property
    def home_q(self) -> tuple[float, ...]:
        """Every arm's home, arm after arm."""
        return tuple(q for arm in self.arms for q in arm.home_q)

    @property
    def isaac_arm_joints(self) -> tuple[str, ...]:
        """Every arm's Isaac Lab joints, arm after arm."""
        return tuple(j for arm in self.arms for j in arm.isaac_joints)

    @property
    def dof(self) -> int:
        """Number of arm joints, of every arm."""
        return len(self.arm_joints)

    @property
    def arm(self) -> ArmSpec:
        """The arm of a one-armed robot (raises for more)."""
        if len(self.arms) != 1:
            raise ValueError(f"{self.name} has {len(self.arms)} arms, not one")
        return self.arms[0]

    def arm_slice(self, arm: int) -> slice:
        """Where arm ``arm``'s joints are in the robot's (:attr:`arm_joints`)."""
        start = sum(len(a.joints) for a in self.arms[:arm])
        return slice(start, start + len(self.arms[arm].joints))

    def gripper_levels(self, position: ArrayLike) -> NDArray[np.float64]:
        """Every gripper's opening (0 open, 1 closed), from a gripper position as a
        capture records it: a number for a one-armed robot, else one per arm."""
        levels = np.atleast_1d(np.asarray(position, float)).ravel()
        if len(levels) != len(self.arms):
            raise ValueError(
                f"{self.name} has {len(self.arms)} grippers, not {len(levels)}"
            )
        return levels

    def gripper_position(self, levels: ArrayLike) -> float | NDArray[np.float64]:
        """Every gripper's opening as a capture records it (:meth:`gripper_levels`)."""
        each = self.gripper_levels(levels)
        return float(each[0]) if len(self.arms) == 1 else each
