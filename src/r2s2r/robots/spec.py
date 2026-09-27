"""What r2s2r needs to know about a robot, one :class:`RobotSpec` per embodiment.

A spec names the robot's models (a MuJoCo MJCF for masking, the viewer and the
testbed; an Isaac Lab articulation for settling, replaying and picking), which of
their joints are the arm, how the gripper's joints follow its opening, and where the
tool centre point is. Its base frame is the MuJoCo world, the model's base link and the
Isaac environment origin alike, so poses need no conversion between them.

Models load lazily, so specs import without MuJoCo, Isaac Lab or the assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from r2s2r.paths import CACHE_DIR

# mujoco_menagerie, fetched by scripts/setup/fetch_mujoco_assets.sh.
MENAGERIE_DIR = CACHE_DIR / "mujoco_menagerie"
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
    """

    driver: str
    open: float
    closed: float
    followers: str
    actuator: str
    ctrl: tuple[float, float]
    isaac_driver: str
    isaac_joints: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.followers not in FOLLOWER_MODES:
            raise ValueError(
                f"followers must be one of {FOLLOWER_MODES}, not {self.followers!r}"
            )


def _no_position_control(mjspec: Any) -> None:
    """For MJCFs whose arm actuators already hold joint positions."""
    del mjspec


@dataclass(frozen=True)
class RobotSpec:
    """One embodiment (``Capture.embodiment`` names it).

    ``mjcf()`` gives the robot alone as a fresh ``mujoco.MjSpec``, base at the origin,
    visual geoms in group 2. ``arm_joints`` are its arm joints in the order captures
    record them. The tool centre point is ``tcp_offset`` along ``tcp_body``'s z axis,
    between the fingertips when closed; ``max_opening`` is the widest object the gripper
    takes. ``position_control(mjspec)`` makes the arm's actuators hold joint positions
    for the testbed's physics.

    ``isaac_cfg()`` gives the Isaac Lab ``ArticulationCfg`` (at the origin, the arm on a
    stiff joint-position PD, gravity off; imports Isaac Lab, so call it in an Isaac
    process only); ``isaac_arm_joints`` are its arm joints in capture order.
    ``isaac_post_spawn(prim_path)``, when given, fixes the spawned robot up before the
    simulation starts. Isaac cameras render nothing nearer than ``isaac_camera_near``
    (m), where a USD models parts the real cameras never show.
    """

    name: str
    mjcf: Callable[[], Any]
    arm_joints: tuple[str, ...]
    home_q: tuple[float, ...]
    gripper: GripperSpec
    tcp_body: str
    tcp_offset: float
    max_opening: float
    isaac_cfg: Callable[[], Any]
    isaac_arm_joints: tuple[str, ...]
    position_control: Callable[[Any], None] = _no_position_control
    isaac_post_spawn: Callable[[str], None] | None = None
    isaac_camera_near: float = 0.02

    def __post_init__(self) -> None:
        if not len(self.arm_joints) == len(self.home_q) == len(self.isaac_arm_joints):
            raise ValueError(
                f"{self.name}: arm_joints, home_q and isaac_arm_joints differ in length"
            )

    @property
    def dof(self) -> int:
        """Number of arm joints."""
        return len(self.arm_joints)
