"""``dual_x5``: RoboDojo's robot, two ARX X5 arms with their parallel grippers.

Each arm is RoboDojo's X5A URDF (``Assets/Robots/x5`` of its Hugging Face dataset,
fetched by ``scripts/setup/fetch_robodojo_x5.sh``), placed as RoboDojo's ``dual_x5``
robot places them: 0.6 m apart, side by side, facing the same way. The base frame is
midway between the arms' bases, on the plane they stand on (RoboDojo's table top), x
the way they face, z up; the left arm (RoboDojo's first, its ``left_`` observations)
on +y. :data:`T_ROBODOJO_BASE` is the base in RoboDojo's world frame.

:func:`urdf` writes both arms as one URDF, their links and joints prefixed ``left_``
and ``right_`` under a ``base`` link, the wrist cameras' glTF meshes as their convex
hulls (neither MuJoCo nor Isaac Lab's importer reads glTF). Both simulators load it:
MuJoCo directly (:func:`mjcf`), Isaac Lab through its URDF importer (:func:`isaac_cfg`,
the links then coloured as the URDF has them, :func:`isaac_post_spawn`).

Both drive it as RoboDojo does (its ``robot_config/x5.py``): the arm joints on
:data:`ARM_DRIVE`, the fingers on :data:`FINGER_DRIVE`, gravity off. Each finger slides
from 0 (closed, the pads 0.8 mm apart) to 44 mm (open); closed, the fingers are driven
to 0, where RoboDojo drives them 10 mm past their stop (squeezing 23 N harder each).
The arms collide with what is around them, not with themselves or each other
(RoboDojo's do): the links' convex hulls, all either simulator takes, overlap where the
links do not (a third of random poses had one arm's links 2 mm into each other).
"""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from r2s2r.paths import CACHE_DIR
from r2s2r.robots.spec import ArmSpec, GripperSpec, RobotSpec
from r2s2r.transforms import make_transform

X5_DIR = CACHE_DIR / "robodojo_x5"  # fetched: X5A.urdf, meshes/
DUAL_DIR = CACHE_DIR / "dual_x5"  # made here: the URDF, the hulls, Isaac's USD
ARMS = {"left": 0.3, "right": -0.3}  # each arm's base, along the base frame's y (m)
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGERS = ("joint7", "joint8")  # joint8 follows joint7
FINGER_OPEN = 0.044  # each finger's slide (m)
# The pads' centre along link6's x axis (the grasp's approach), from the fingers'
# meshes: their inner faces reach from 0.143 to 0.158 m.
PAD_CENTRE = 0.150


class Drive(NamedTuple):
    """A joint's PD drive: its stiffness (N m/rad, N/m), damping (N m s/rad, N s/m),
    the most it pushes with (N m, N) and, in Isaac Lab only, the fastest it moves
    (rad/s; None: no limit)."""

    stiffness: float
    damping: float
    effort: float
    velocity: float | None = None


# RoboDojo's drives, and the arm joints' armature (kg m^2).
ARM_DRIVE = Drive(4400.0, 40.0, 100.0, 5.0)
FINGER_DRIVE = Drive(2300.0, 100.0, 100.0)
ARMATURE = 0.01
# The base in RoboDojo's world frame: its arms stand at (-0.3, -0.45, 0.765) (left)
# and (0.3, -0.45, 0.765), turned 90 degrees about z.
T_ROBODOJO_BASE = make_transform(
    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    [0.0, -0.45, 0.765],
)
VISUAL_GROUP, COLLISION_GROUP = 2, 3  # as in every robot's MJCF


def urdf() -> Path:
    """The two arms as one URDF (``DUAL_DIR/dual_x5.urdf``, written when missing or
    out of date), its meshes by absolute path; the wrist cameras, which the X5A URDF
    leaves uncoloured, the colour of the wrist (link6) they are on, as RoboDojo
    renders them."""
    source = X5_DIR / "X5A.urdf"
    if not source.exists():
        raise FileNotFoundError(f"{source}: run scripts/setup/fetch_robodojo_x5.sh")
    DUAL_DIR.mkdir(parents=True, exist_ok=True)
    arm = ET.parse(source).getroot()
    wrist = arm.find("link[@name='link6']/visual/material")
    for visual in arm.iterfind("link/visual"):
        if wrist is not None and visual.find("material") is None:
            visual.append(copy.deepcopy(wrist))
    robot = ET.Element("robot", name="dual_x5")
    ET.SubElement(robot, "link", name="base")
    for side, y in ARMS.items():
        mount = ET.SubElement(robot, "joint", name=f"{side}_mount", type="fixed")
        ET.SubElement(mount, "parent", link="base")
        ET.SubElement(mount, "child", link=f"{side}_base_link")
        ET.SubElement(mount, "origin", xyz=f"0 {y} 0", rpy="0 0 0")
        for element in arm:
            element = ET.fromstring(ET.tostring(element))
            element.set("name", f"{side}_{element.get('name')}")
            for end in (*element.findall("parent"), *element.findall("child")):
                end.set("link", f"{side}_{end.get('link')}")
            for mesh in element.iter("mesh"):
                mesh.set("filename", str(_mesh(mesh.get("filename", ""))))
            robot.append(element)
    ET.indent(robot)
    text = ET.tostring(robot, encoding="unicode") + "\n"
    path = DUAL_DIR / "dual_x5.urdf"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        part = path.with_suffix(f".{os.getpid()}.part")
        part.write_text(text, encoding="utf-8")
        part.replace(path)
    return path


def _mesh(name: str) -> Path:
    """The absolute path of the URDF's mesh ``name``; a glTF one as its convex hull
    in STL, made once."""
    path = X5_DIR / name
    if path.suffix.lower() != ".glb":
        return path.resolve()
    hull = DUAL_DIR / f"{path.stem}_hull.stl"
    if not hull.exists():
        # pylint: disable=import-outside-toplevel
        import trimesh

        mesh = trimesh.load(path, force="mesh")
        assert isinstance(mesh, trimesh.Trimesh), f"{path} is not a single mesh"
        part = hull.with_suffix(f".{os.getpid()}.part.stl")
        mesh.convex_hull.export(part)
        part.replace(hull)
    return hull


def mjcf() -> Any:
    """Both arms on position servos, each one's joint8 following its joint7."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    root = ET.parse(urdf()).getroot()
    extension = ET.Element("mujoco")
    ET.SubElement(
        extension,
        "compiler",
        strippath="false",
        discardvisual="false",
        fusestatic="false",
        balanceinertia="true",
    )
    root.insert(0, extension)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.group = VISUAL_GROUP
        else:  # touches what has the default contype, not the robot
            geom.group, geom.contype, geom.conaffinity = COLLISION_GROUP, 2, 1
    for side in ARMS:
        for joint in ARM_JOINTS:
            _servo(spec, f"{side}_{joint}", f"{side}_{joint}", ARM_DRIVE)
            spec.joint(f"{side}_{joint}").armature = ARMATURE
        driver, follower = (f"{side}_{f}" for f in FINGERS)
        _servo(spec, f"{side}_gripper", driver, FINGER_DRIVE)
        spec.add_equality(  # follower = driver (polycoef 0 1 0 0 0)
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=follower,
            name2=driver,
            data=[0.0, 1.0] + [0.0] * 9,
        )
        # The TCP: between the pads, z the approach (link6's x), y the fingers' slide.
        spec.body(f"{side}_link6").add_body(
            name=f"{side}_tcp",
            pos=[PAD_CENTRE, 0.0, 0.0],
            quat=[np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0],
        )
    return spec


def _servo(spec: Any, name: str, joint: str, drive: Drive) -> None:
    """A position servo ``name`` on ``joint`` with ``drive``'s gains and effort limit,
    its control within the joint's range."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    act = spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=drive.stiffness, kv=drive.damping)
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-drive.effort, drive.effort]
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = spec.joint(joint).range


def isaac_cfg() -> Any:
    """The URDF in Isaac Lab, its drives RoboDojo's, the robot not colliding with
    itself."""
    # pylint: disable=import-outside-toplevel
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg

    def actuator(joints: list[str], drive: Drive, **extra: Any) -> Any:
        return ImplicitActuatorCfg(
            joint_names_expr=joints,
            stiffness=drive.stiffness,
            damping=drive.damping,
            effort_limit_sim=drive.effort,
            velocity_limit_sim=drive.velocity,
            **extra,
        )

    arm = [f"{side}_{j}" for side in ARMS for j in ARM_JOINTS]
    fingers = [f"{side}_{f}" for side in ARMS for f in FINGERS]
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf()),
            usd_dir=str(DUAL_DIR / "usd"),
            fix_base=True,
            merge_fixed_joints=True,
            joint_drive=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={**{j: 0.0 for j in arm}, **{f: FINGER_OPEN for f in fingers}}
        ),
        actuators={
            "arms": actuator(arm, ARM_DRIVE, armature=ARMATURE),
            "fingers": actuator(fingers, FINGER_DRIVE),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def isaac_post_spawn(robot_prim: str) -> None:
    """Give each link the URDF's colour: Isaac's importer binds every STL mesh a white
    material of its own, which hides it."""
    # pylint: disable=import-outside-toplevel
    import isaaclab.sim as sim_utils

    colours = _colours()
    stage = sim_utils.get_current_stage()
    for prim in stage.GetPrimAtPath(robot_prim).GetChildren():
        # The base holds both arms' base links (fixed joints merged).
        name = prim.GetName()
        link = "base_link" if name == "base" else name.split("_", 1)[-1]
        if link not in colours:
            continue
        colour, rgb = colours[link]
        material = f"{robot_prim}/Looks/x5_{colour}"
        if not stage.GetPrimAtPath(material):
            sim_utils.spawn_preview_surface(
                material, sim_utils.PreviewSurfaceCfg(diffuse_color=rgb)
            )
        sim_utils.bind_visual_material(str(prim.GetPath()), material)


def _colours() -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Each X5A link's colour: its material's name and RGB."""
    colours = {}
    for link in ET.parse(X5_DIR / "X5A.urdf").getroot().iter("link"):
        material = link.find("visual/material")
        color = link.find("visual/material/color")
        if material is not None and color is not None:
            r, g, b = (float(v) for v in color.get("rgba", "0 0 0 1").split()[:3])
            colours[link.get("name", "")] = (material.get("name", ""), (r, g, b))
    return colours


def _arm(side: str) -> ArmSpec:
    """The ``side`` arm, at RoboDojo's home (every joint at 0), its fingers driven
    together."""
    driver, follower = (f"{side}_{f}" for f in FINGERS)
    return ArmSpec(
        joints=tuple(f"{side}_{j}" for j in ARM_JOINTS),
        home_q=(0.0,) * len(ARM_JOINTS),
        isaac_joints=tuple(f"{side}_{j}" for j in ARM_JOINTS),
        gripper=GripperSpec(
            driver=driver,
            open=FINGER_OPEN,
            closed=0.0,
            followers="equality",
            actuator=f"{side}_gripper",
            ctrl=(FINGER_OPEN, 0.0),
            isaac_driver=driver,
            isaac_joints={driver: (FINGER_OPEN, 0.0), follower: (FINGER_OPEN, 0.0)},
        ),
        tcp_body=f"{side}_tcp",
        tcp_offset=0.0,
        max_opening=2 * FINGER_OPEN,
    )


DUAL_X5 = RobotSpec(
    name="dual_x5",
    mjcf=mjcf,
    arms=(_arm("left"), _arm("right")),
    isaac_cfg=isaac_cfg,
    isaac_post_spawn=isaac_post_spawn,
)
