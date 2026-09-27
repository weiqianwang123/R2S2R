"""``ur5e_2f140``: a UR5e with a Robotiq 2F-140, physcoder's robot.

Both models are physcoder's, used by path (``paths.PHYSCODER_ASSETS``) and never copied:
the MJCF ``mujoco/robots/universal_robots/ur5e_2f140.xml`` (its ``wrist`` camera is the
real wrist camera) and the USD ``isaaclab/robots/universal_robots/ur5e_robotiq2f140/``.
Their kinematics agree, and base_link is the origin of both.

The USD's articulation root is a fixed joint to the world at the origin, so the robot
stays there whatever its spawn pose: the scene must have one environment, at the origin.
physcoder drives the arm by torque (zero stiffness); here it gets a stiff position PD,
with physcoder's gravity off, 36 position iterations and effort limits, but no
self-collisions. Cameras clip at 7 cm as physcoder's do, or the wrist camera sees the
USD's camera adapter.
"""

from __future__ import annotations

from typing import Any

from r2s2r.paths import PHYSCODER_ASSETS
from r2s2r.robots.spec import GripperSpec, RobotSpec

MJCF_PATH = (
    PHYSCODER_ASSETS / "mujoco" / "robots" / "universal_robots" / "ur5e_2f140.xml"
)
USD_PATH = (
    PHYSCODER_ASSETS
    / "isaaclab"
    / "robots"
    / "universal_robots"
    / "ur5e_robotiq2f140"
    / "ur5e_robotiq2f140.usd"
)
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HOME_Q = (0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0)  # physcoder's default
FINGER_CLOSED = 0.785398  # finger_joint, rad
# The pads' centre along wrist_3_link's z when closed (0.197 m when open), from the
# MJCF's pad geoms. Placing it at the grasp point keeps the closing pads above it.
TCP_OFFSET = 0.222
# The MJCF's actuator defaults: a position servo per joint size (class ``size3`` for
# the shoulder and elbow, ``size1`` for the wrist).
SERVO_CLASS = {joint: "size1" if "wrist" in joint else "size3" for joint in ARM_JOINTS}
# The USD has the wrist camera's RealSense adapter plate 3-4 cm in front of the lens
# (the MJCF has not); physcoder's Isaac cameras, like the real one, clip at 7 cm.
CAMERA_NEAR = 0.07
# The gripper's Isaac articulation joints, open and closed. Measured in Isaac Sim 5.1:
# finger_joint driven to FINGER_CLOSED reads 0.7853 on it and four joints and -0.7853
# on right_inner_knuckle_joint (the USD's mimic gearings differ from the MJCF's signs);
# driven to 0, all read 0.0115 in size (the drive's steady-state error). Set directly,
# these put every gripper body within 0.1 mm of the MJCF's. left/right_inner_finger_joint
# close the loops and are not articulation joints.
ISAAC_GRIPPER = {
    "finger_joint": (0.0, FINGER_CLOSED),
    "right_outer_knuckle_joint": (0.0, FINGER_CLOSED),
    "left_inner_knuckle_joint": (0.0, FINGER_CLOSED),
    "right_inner_knuckle_joint": (0.0, -FINGER_CLOSED),
    "left_inner_finger_pad_joint": (0.0, FINGER_CLOSED),
    "right_inner_finger_pad_joint": (0.0, FINGER_CLOSED),
}


def mjcf() -> Any:
    """physcoder's MJCF of the robot alone."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    if not MJCF_PATH.exists():
        raise FileNotFoundError(
            f"{MJCF_PATH} not found: fetch physcoder's assets, or point "
            "PHYSCODER_ASSETS_DIR or PHYSCODER_ROOT at them"
        )
    return mujoco.MjSpec.from_file(str(MJCF_PATH))


def position_control(mjspec: Any) -> None:
    """Turn the arm's torque motors into position servos with the gains of the MJCF's
    defaults (the gripper already has a position actuator)."""
    for joint in ARM_JOINTS:
        servo = mjspec.find_default(SERVO_CLASS[joint]).actuator
        act = mjspec.actuator(joint)
        act.gaintype, act.biastype = servo.gaintype, servo.biastype
        act.gainprm, act.biasprm = servo.gainprm, servo.biasprm
        act.forcerange, act.forcelimited = servo.forcerange, servo.forcelimited
        act.ctrlrange = mjspec.joint(joint).range


def isaac_cfg() -> Any:
    """physcoder's USD with a stiff arm PD (the MJCF servo gains) and the gripper
    actuator on finger_joint only."""
    # pylint: disable=import-outside-toplevel
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg

    if not USD_PATH.exists():
        raise FileNotFoundError(f"{USD_PATH} not found (physcoder's Isaac assets)")
    big, wrist = list(ARM_JOINTS[:3]), list(ARM_JOINTS[3:])
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=36,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                **dict(zip(ARM_JOINTS, HOME_Q)),
                **{joint: 0.0 for joint in ISAAC_GRIPPER},
            }
        ),
        actuators={
            "shoulder_elbow": ImplicitActuatorCfg(
                joint_names_expr=big,
                stiffness=2000.0,
                damping=400.0,
                effort_limit_sim=150.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=wrist,
                stiffness=500.0,
                damping=100.0,
                effort_limit_sim=28.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=17.0,
                damping=5.0,
                effort_limit_sim=60.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


UR5E_2F140 = RobotSpec(
    name="ur5e_2f140",
    mjcf=mjcf,
    arm_joints=ARM_JOINTS,
    home_q=HOME_Q,
    gripper=GripperSpec(
        driver="finger_joint",
        open=0.0,
        closed=FINGER_CLOSED,
        followers="equality",
        actuator="gripper",
        ctrl=(0.0, FINGER_CLOSED),
        isaac_driver="finger_joint",
        isaac_joints=ISAAC_GRIPPER,
    ),
    tcp_body="wrist_3_link",
    tcp_offset=TCP_OFFSET,
    max_opening=0.128,
    isaac_cfg=isaac_cfg,
    isaac_arm_joints=ARM_JOINTS,
    position_control=position_control,
    isaac_camera_near=CAMERA_NEAR,
)
