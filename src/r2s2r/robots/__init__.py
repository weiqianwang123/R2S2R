"""The robots r2s2r supports, by embodiment name (``Capture.embodiment``).

Each is a :class:`~r2s2r.robots.spec.RobotSpec` in its own module; masking, the viewer,
Isaac Lab and the testbed all take the spec, so a new robot is a new module added here.
"""

from __future__ import annotations

from r2s2r.robots.droid_franka import DROID_FRANKA
from r2s2r.robots.franka_panda import FRANKA_PANDA
from r2s2r.robots.spec import GripperSpec, RobotSpec
from r2s2r.robots.ur5e_2f140 import UR5E_2F140

ROBOTS: dict[str, RobotSpec] = {
    robot.name: robot for robot in (FRANKA_PANDA, DROID_FRANKA, UR5E_2F140)
}


def get_robot(name: str) -> RobotSpec:
    """The spec of embodiment ``name``."""
    if name not in ROBOTS:
        raise ValueError(
            f"unknown robot {name!r} (known: {', '.join(sorted(ROBOTS))}); add a "
            "module for it under r2s2r/robots/ and register it in ROBOTS"
        )
    return ROBOTS[name]


__all__ = ["ROBOTS", "GripperSpec", "RobotSpec", "get_robot"]
