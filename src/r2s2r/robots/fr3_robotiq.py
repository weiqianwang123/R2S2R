"""``fr3_robotiq``: the Franka Research 3 with a Robotiq 2F-85 on DROID's coupling.

MuJoCo: mujoco_menagerie's ``franka_fr3/fr3.xml`` with the Robotiq attached to
``fr3_link7`` exactly as :mod:`~r2s2r.robots.droid_franka` attaches it to the Panda's
link7 (:func:`~r2s2r.robots.droid_franka.attach_robotiq`).

Isaac Lab (2.3) has no FR3 asset. The FR3 shares the Panda's kinematics (the same DH
parameters), so this uses Isaac's Panda with Robotiq's own 2F-85 like
``droid_franka`` (:func:`~r2s2r.robots.droid_franka.fit_robotiq`), and the joint limits
set to the FR3's (:data:`JOINT_LIMITS`, the MJCF's ranges). Isaac's link meshes and
inertias are therefore the Panda's.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from r2s2r.robots import droid_franka
from r2s2r.robots.franka_panda import ISAAC_ARM_JOINTS
from r2s2r.robots.spec import MENAGERIE_DIR, RobotSpec

ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
# DROID's reset pose (droid/robot_env.py).
HOME_Q = (0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0)
# The arm joints' (lower, upper) limits (rad), from the MJCF's ranges; a test holds them
# to the model.
JOINT_LIMITS = (
    (-2.7437, 2.7437),
    (-1.7837, 1.7837),
    (-2.9007, 2.9007),
    (-3.0421, -0.1518),
    (-2.8065, 2.8065),
    (0.5445, 4.5169),
    (-3.0159, 3.0159),
)


def mjcf() -> Any:
    """The FR3 with the Robotiq on its flange."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    return droid_franka.attach_robotiq(
        mujoco.MjSpec.from_file(str(MENAGERIE_DIR / "franka_fr3" / "fr3.xml")),
        "fr3_link7",
    )


def isaac_post_spawn(robot_prim: str) -> None:
    """Fit the Robotiq as on DROID (:func:`~r2s2r.robots.droid_franka.fit_robotiq`)
    and give Isaac's Panda joints the FR3's limits."""
    # pylint: disable=import-outside-toplevel
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdPhysics

    droid_franka.fit_robotiq(robot_prim)
    stage = sim_utils.get_current_stage()
    joints = {
        prim.GetName(): UsdPhysics.RevoluteJoint(prim)
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim))
        if prim.GetName() in ISAAC_ARM_JOINTS and prim.IsA(UsdPhysics.RevoluteJoint)
    }
    if len(joints) != len(ISAAC_ARM_JOINTS):
        raise ValueError(f"{robot_prim} lacks Panda joints: has {sorted(joints)}")
    for name, (lower, upper) in zip(ISAAC_ARM_JOINTS, JOINT_LIMITS):
        joints[name].GetLowerLimitAttr().Set(float(np.degrees(lower)))
        joints[name].GetUpperLimitAttr().Set(float(np.degrees(upper)))


FR3_ROBOTIQ = RobotSpec(
    name="fr3_robotiq",
    mjcf=mjcf,
    arm_joints=ARM_JOINTS,
    home_q=HOME_Q,
    gripper=droid_franka.DROID_FRANKA.gripper,
    tcp_body=droid_franka.DROID_FRANKA.tcp_body,
    tcp_offset=droid_franka.DROID_FRANKA.tcp_offset,
    max_opening=droid_franka.DROID_FRANKA.max_opening,
    isaac_cfg=droid_franka.isaac_cfg,
    isaac_arm_joints=ISAAC_ARM_JOINTS,
    isaac_post_spawn=isaac_post_spawn,
)
