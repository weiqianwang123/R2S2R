"""``franka_panda``: the Franka Panda with the Franka Hand.

MuJoCo: mujoco_menagerie's ``panda.xml``. Isaac Lab: ``FRANKA_PANDA_HIGH_PD_CFG``
(stiff arm PD, gravity off).
"""

from __future__ import annotations

from typing import Any

from r2s2r.robots.spec import MENAGERIE_DIR, GripperSpec, RobotSpec

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ISAAC_ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
# Hand above the table, pointing down, out of the exterior cameras' way.
HOME_Q = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
HAND_TCP = 0.1034  # hand frame -> fingertip pad centre, along z
MAX_WIDTH = 0.08
FINGER_OPEN = MAX_WIDTH / 2  # each finger's slide


def mjcf() -> Any:
    """The robot alone."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    return mujoco.MjSpec.from_file(
        str(MENAGERIE_DIR / "franka_emika_panda" / "panda.xml")
    )


def isaac_cfg() -> Any:
    """Isaac Lab's Panda with the Franka Hand and a stiff PD."""
    # pylint: disable=import-outside-toplevel
    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

    return FRANKA_PANDA_HIGH_PD_CFG.copy()


FRANKA_PANDA = RobotSpec(
    name="franka_panda",
    mjcf=mjcf,
    arm_joints=ARM_JOINTS,
    home_q=HOME_Q,
    gripper=GripperSpec(
        # finger_joint1 = finger_joint2 by an equality; the actuator drives both
        # through a tendon, 255 open.
        driver="finger_joint2",
        open=FINGER_OPEN,
        closed=0.0,
        followers="equality",
        actuator="actuator8",
        ctrl=(255.0, 0.0),
        isaac_driver="panda_finger_joint1",
        isaac_joints={
            "panda_finger_joint1": (FINGER_OPEN, 0.0),
            "panda_finger_joint2": (FINGER_OPEN, 0.0),
        },
    ),
    tcp_body="hand",
    tcp_offset=HAND_TCP,
    max_opening=MAX_WIDTH,
    isaac_cfg=isaac_cfg,
    isaac_arm_joints=ISAAC_ARM_JOINTS,
)
