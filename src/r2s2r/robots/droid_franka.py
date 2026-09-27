"""``droid_franka``: the Franka Panda with a Robotiq 2F-85 on the flange, as on DROID.

MuJoCo: mujoco_menagerie's ``panda_nohand.xml`` with its ``2f85.xml`` attached to
link7, turned about the flange axis as fitted to the fingers in the wrist-camera images
of a DROID episode (IRIS). Isaac Lab: ``FRANKA_ROBOTIQ_GRIPPER_CFG``, whose Robotiq is
mounted like the Franka Hand (10.7 cm out, turned -45 deg); :func:`mount_robotiq`
re-mounts it where the MuJoCo model has it, 1.1 cm further out and turned +45 deg from
that, or the fingers show up 45 degrees off in every wrist-camera render.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from r2s2r.robots.franka_panda import ARM_JOINTS, HOME_Q, ISAAC_ARM_JOINTS
from r2s2r.robots.spec import MENAGERIE_DIR, GripperSpec, RobotSpec

FLANGE_OFFSET = 0.107  # link7 -> flange, along z
ROBOTIQ_YAW = np.pi / 2  # the Robotiq's mount about the flange z axis
PREFIX = "gripper/"  # of the Robotiq's names in the MJCF
# link7 -> the Robotiq's base: this far along z, no rotation (the MJCF's flange offset,
# mount and base_mount -> base; a test holds it to the model).
ROBOTIQ_BASE_Z = 0.1178
# The pads' centre along the Robotiq base's z when closed (0.131 m when open), from the
# MJCF's pad geoms.
ROBOTIQ_TCP = 0.144
# Joints of Isaac Sim's franka.usd (Robotiq variant), under the robot prim: link7 to
# the hand frame, and the hand frame to the Robotiq base.
ISAAC_HAND_JOINT = "panda_link7/panda_hand_joint"
ISAAC_ROBOTIQ_JOINT = "Robotiq_2F_85_edit/Robotiq_2F_85/base_link/AssemblerFixedJoint"


def mjcf() -> Any:
    """The Panda with the Robotiq on its flange."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    spec = mujoco.MjSpec.from_file(
        str(MENAGERIE_DIR / "franka_emika_panda" / "panda_nohand.xml")
    )
    gripper = mujoco.MjSpec.from_file(str(MENAGERIE_DIR / "robotiq_2f85" / "2f85.xml"))
    frame = spec.body("link7").add_frame(
        pos=[0.0, 0.0, FLANGE_OFFSET],
        quat=[np.cos(ROBOTIQ_YAW / 2), 0.0, 0.0, np.sin(ROBOTIQ_YAW / 2)],
    )
    frame.attach_body(gripper.body("base_mount"), PREFIX, "")
    return spec


def isaac_cfg() -> Any:
    """Isaac Lab's Panda with the Robotiq 2F-85 (stiff arm PD, gravity off)."""
    # pylint: disable=import-outside-toplevel
    from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

    return FRANKA_ROBOTIQ_GRIPPER_CFG.copy()


def mount_robotiq(robot_prim: str) -> None:
    """Re-mount the Robotiq base :data:`ROBOTIQ_BASE_Z` out along link7's z axis."""
    # pylint: disable=import-outside-toplevel
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdPhysics
    from scipy.spatial.transform import Rotation

    from r2s2r.transforms import invert, make_transform

    stage = sim_utils.get_current_stage()
    hand = UsdPhysics.Joint(stage.GetPrimAtPath(f"{robot_prim}/{ISAAC_HAND_JOINT}"))
    mount = UsdPhysics.Joint(stage.GetPrimAtPath(f"{robot_prim}/{ISAAC_ROBOTIQ_JOINT}"))
    if not (hand and mount):
        raise ValueError(f"{robot_prim} is not Isaac's Franka + Robotiq asset")

    def frame(pos: Any, rot: Any) -> np.ndarray:
        x, y, z = rot.GetImaginary()
        R = Rotation.from_quat([x, y, z, rot.GetReal()]).as_matrix()
        return make_transform(R, np.array(pos, float))

    # The hand joint's body0 -> body1 (link7 -> hand), from its two local frames.
    T_link7_hand = frame(
        hand.GetLocalPos0Attr().Get(), hand.GetLocalRot0Attr().Get()
    ) @ invert(frame(hand.GetLocalPos1Attr().Get(), hand.GetLocalRot1Attr().Get()))
    T_hand_base = invert(T_link7_hand) @ make_transform(
        np.eye(3), [0.0, 0.0, ROBOTIQ_BASE_Z]
    )
    x, y, z, w = Rotation.from_matrix(T_hand_base[:3, :3]).as_quat()
    mount.GetLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in T_hand_base[:3, 3])))
    mount.GetLocalRot0Attr().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    mount.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mount.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


DROID_FRANKA = RobotSpec(
    name="droid_franka",
    mjcf=mjcf,
    arm_joints=ARM_JOINTS,
    home_q=HOME_Q,
    gripper=GripperSpec(
        # A four-bar linkage closed by equality constraints: the other joints come
        # from simulating the actuator (0 open, 255 closed).
        driver=f"{PREFIX}right_driver_joint",
        open=0.0,
        closed=0.8,
        followers="simulate",
        actuator=f"{PREFIX}fingers_actuator",
        ctrl=(0.0, 255.0),
        # Isaac's 2F-85 is a closed loop with passive joints: set the driver only.
        isaac_driver="finger_joint",
    ),
    tcp_body=f"{PREFIX}base",
    tcp_offset=ROBOTIQ_TCP,
    max_opening=0.085,
    isaac_cfg=isaac_cfg,
    isaac_arm_joints=ISAAC_ARM_JOINTS,
    isaac_post_spawn=mount_robotiq,
)
