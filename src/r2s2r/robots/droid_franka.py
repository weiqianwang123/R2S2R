"""``droid_franka``: the Franka Panda with a Robotiq 2F-85 on the flange, as on DROID.

MuJoCo: mujoco_menagerie's ``panda_nohand.xml`` with its ``2f85.xml`` attached to
link7, turned about the flange axis as fitted to the fingers in the wrist-camera images
of a DROID episode (IRIS). Isaac Lab: the Panda of ``FRANKA_ROBOTIQ_GRIPPER_CFG`` with
its Robotiq swapped by :func:`fit_robotiq` for Robotiq's own 2F-85
(``robotiq/isaacsim_assets``, its ``Physx_parallel_grip`` build), mounted where the
MuJoCo model has it, its pads given a rubber's friction. Isaac's own is mounted like
the Franka Hand (10.7 cm out, turned -45 deg), so its fingers show up 45 degrees off in
every wrist-camera render; and with PhysX on the CPU its linkage gives way: the left
pad tilts 8 degrees, and a 60 g knife pinched by its handle was lifted 2 times in 6,
against 6 in 6 with Robotiq's.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from r2s2r.robots.franka_panda import ARM_JOINTS, HOME_Q, ISAAC_ARM_JOINTS
from r2s2r.robots.spec import (
    MENAGERIE_DIR,
    ROBOTIQ_ISAAC_DIR,
    GripperSpec,
    RobotSpec,
)
from r2s2r.transforms import (
    invert,
    make_transform,
    matrix_to_pos_quat,
    pos_quat_to_matrix,
)

FLANGE_OFFSET = 0.107  # link7 -> flange, along z
ROBOTIQ_YAW = np.pi / 2  # the Robotiq's mount about the flange z axis
PREFIX = "gripper/"  # of the Robotiq's names in the MJCF
# link7 -> the Robotiq's base: this far along z, no rotation (the MJCF's flange offset,
# mount and base_mount -> base; a test holds it to the model).
ROBOTIQ_BASE_Z = 0.1178
# The pads' centre along the Robotiq base's z when closed (0.131 m when open), from the
# MJCF's pad geoms.
ROBOTIQ_TCP = 0.144
# Under the robot prim of Isaac Sim's franka.usd (Robotiq variant): the joint from link7
# to the hand frame, and Isaac's Robotiq, which :func:`fit_robotiq` turns off.
ISAAC_HAND_JOINT = "panda_link7/panda_hand_joint"
ISAAC_ROBOTIQ = "Robotiq_2F_85_edit"
# Robotiq's 2F-85 for Isaac (scripts/setup/fetch_robotiq_isaac.sh): the finger linkage
# as mimic joints of finger_joint, a tree, so every joint stays in the articulation.
ROBOTIQ_ISAAC_USD = (
    ROBOTIQ_ISAAC_DIR
    / "grippers/Robotiq_2F_85/configuration"
    / "Robotiq_2F_85_config_physics_parallel_grip.usda"
)
ROBOTIQ_PRIM = "Robotiq_2F_85"  # where fit_robotiq puts it, under the robot prim
# The Robotiq's drive (finger_joint) in Isaac Lab. Isaac Lab's (stiffness 17,
# damping 0.02, effort limit 1650) cannot open the closed linkage again: closed in the
# air it reopens only half way, holding a 58 mm block it does not let go, and it drifts
# shut while the arm moves. Stiffer, it opens fully; its torque capped near the real
# 2F-85's grip force (up to 235 N), it does not crush into light objects.
ISAAC_GRIPPER_DRIVE = {"stiffness": 100.0, "damping": 5.0, "effort_limit_sim": 20.0}
# The pads of Robotiq's 2F-85. The asset gives them no physics material, so they would
# take the scene's default (0.5, averaged with an object's).
ISAAC_PAD_LINKS = ("left_fingertip", "right_fingertip")
# The pads' friction, as silicone on wood or plastic; the higher of it and an object's
# holds, as in MuJoCo, whose 2F-85 pads have 0.7 and 0.6.
PAD_FRICTION = 1.0


def attach_robotiq(spec: Any, link7: str) -> Any:
    """``spec`` (an arm's ``mujoco.MjSpec``) with the Robotiq 2F-85 on the flange of its
    body ``link7``, as on DROID: :data:`FLANGE_OFFSET` out, turned :data:`ROBOTIQ_YAW`,
    its names prefixed :data:`PREFIX`."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    gripper = mujoco.MjSpec.from_file(str(MENAGERIE_DIR / "robotiq_2f85" / "2f85.xml"))
    frame = spec.body(link7).add_frame(
        pos=[0.0, 0.0, FLANGE_OFFSET],
        quat=[np.cos(ROBOTIQ_YAW / 2), 0.0, 0.0, np.sin(ROBOTIQ_YAW / 2)],
    )
    frame.attach_body(gripper.body("base_mount"), PREFIX, "")
    return spec


def mjcf() -> Any:
    """The Panda with the Robotiq on its flange."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    return attach_robotiq(
        mujoco.MjSpec.from_file(
            str(MENAGERIE_DIR / "franka_emika_panda" / "panda_nohand.xml")
        ),
        "link7",
    )


def isaac_cfg() -> Any:
    """Isaac Lab's Panda with the Robotiq 2F-85 (stiff arm PD, gravity off), the
    gripper's drive :data:`ISAAC_GRIPPER_DRIVE`, no self-collisions."""
    # pylint: disable=import-outside-toplevel
    from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

    cfg = FRANKA_ROBOTIQ_GRIPPER_CFG.copy()
    # Isaac Lab's config collides the robot's links with each other, and the
    # gripper's touch the hand's: held at DROID's home, the wrist stood 32 mrad
    # (Isaac's Robotiq) or 60 mrad (Robotiq's) from its target, the flange 2-4
    # mm off, where the real arm stands on it. Robotiq's own asset turns
    # self-collisions off too.
    cfg.spawn.articulation_props.enabled_self_collisions = False
    drive = copy.deepcopy(cfg.actuators["gripper_drive"])
    for name, value in ISAAC_GRIPPER_DRIVE.items():
        setattr(drive, name, value)
    cfg.actuators = {**cfg.actuators, "gripper_drive": drive}
    return cfg


def fit_robotiq(robot_prim: str) -> None:
    """Swap the spawned robot's Robotiq for Robotiq's (:data:`ROBOTIQ_ISAAC_USD`): its
    own articulation root and weld to the world removed, so its joints join the robot's,
    welded instead to the hand frame :data:`ROBOTIQ_BASE_Z` out along link7's z axis,
    its bodies given the robot's rigid-body settings; then :func:`grip_pads`."""
    # pylint: disable=import-outside-toplevel
    import isaaclab.sim as sim_utils
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    if not ROBOTIQ_ISAAC_USD.exists():
        raise FileNotFoundError(
            f"{ROBOTIQ_ISAAC_USD}: run scripts/setup/fetch_robotiq_isaac.sh"
        )
    stage = sim_utils.get_current_stage()
    hand = UsdPhysics.Joint(stage.GetPrimAtPath(f"{robot_prim}/{ISAAC_HAND_JOINT}"))
    isaacs = stage.GetPrimAtPath(f"{robot_prim}/{ISAAC_ROBOTIQ}")
    if not (hand and isaacs):
        raise ValueError(f"{robot_prim} is not Isaac's Franka + Robotiq asset")
    isaacs.SetActive(False)

    def frame(pos: Any, rot: Any) -> np.ndarray:
        return pos_quat_to_matrix(
            np.array(pos, float), [rot.GetReal(), *rot.GetImaginary()]
        )

    # The hand joint's body0 -> body1 (link7 -> hand), from its two local frames.
    T_link7_hand = frame(
        hand.GetLocalPos0Attr().Get(), hand.GetLocalRot0Attr().Get()
    ) @ invert(frame(hand.GetLocalPos1Attr().Get(), hand.GetLocalRot1Attr().Get()))
    T_hand_base = invert(T_link7_hand) @ make_transform(
        np.eye(3), [0.0, 0.0, ROBOTIQ_BASE_Z]
    )
    hand_body = hand.GetBody1Rel().GetTargets()[0]
    cache = UsdGeom.XformCache()
    T_world_robot, T_world_hand = (
        np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
        for path in (robot_prim, hand_body)
    )
    holder = UsdGeom.Xform.Define(stage, f"{robot_prim}/{ROBOTIQ_PRIM}")
    holder.GetPrim().GetReferences().AddReference(str(ROBOTIQ_ISAAC_USD))
    # Placed where the weld holds it, so the first step does not snap it there.
    T_robot_base = invert(T_world_robot) @ T_world_hand @ T_hand_base
    holder.AddTransformOp().Set(Gf.Matrix4d(T_robot_base.T.tolist()))
    prims = list(Usd.PrimRange(holder.GetPrim()))
    for prim in prims:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    for prim in prims:
        if prim.GetName() == "root_joint":
            prim.SetActive(False)
    (base,) = (
        p.GetPath()
        for p in prims
        if p.GetName() == "base_link" and p.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    weld = UsdPhysics.FixedJoint.Define(stage, holder.GetPath().AppendChild("mount"))
    weld.CreateBody0Rel().SetTargets([hand_body])
    weld.CreateBody1Rel().SetTargets([base])
    pos, (w, x, y, z) = matrix_to_pos_quat(T_hand_base)
    weld.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in pos)))
    weld.CreateLocalRot0Attr().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    weld.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    weld.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # The robot's rigid-body settings (gravity off, as the arm's PD expects)
    # reach only what was spawned with it, not the bodies added here.
    sim_utils.modify_rigid_body_properties(
        str(holder.GetPath()), isaac_cfg().spawn.rigid_props
    )
    grip_pads(robot_prim)


def grip_pads(robot_prim: str) -> None:
    """Bind a physics material of :data:`PAD_FRICTION`, combined by maximum, to the
    Robotiq's :data:`ISAAC_PAD_LINKS` (and so to every collider under them)."""
    # pylint: disable=import-outside-toplevel
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdShade

    stage = sim_utils.get_current_stage()
    links = [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim))
        if prim.GetName() in ISAAC_PAD_LINKS
    ]
    if len(links) != len(ISAAC_PAD_LINKS):
        raise ValueError(f"{robot_prim} lacks the Robotiq's {ISAAC_PAD_LINKS}")
    material = UsdShade.Material(
        sim_utils.spawn_rigid_body_material(
            f"{robot_prim}/PadMaterial",
            sim_utils.RigidBodyMaterialCfg(
                static_friction=PAD_FRICTION,
                dynamic_friction=PAD_FRICTION,
                friction_combine_mode="max",
            ),
        )
    )
    for link in links:
        UsdShade.MaterialBindingAPI.Apply(link).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, "physics"
        )


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
    isaac_post_spawn=fit_robotiq,
)
