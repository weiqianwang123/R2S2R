"""``dual_yam``: ABC-Sim's robot, two i2rt YAM arms with their parallel grippers.

ABC-Sim (amazon-far/abc) simulates the bimanual YAM in its ``yam_bimanual_empty.xml``,
both arms inlined with their station; i2rt's meshes (MIT) come in its mesh package.
``scripts/setup/fetch_abc_yam.sh`` fetches both into :data:`YAM_DIR`. :func:`mjcf` is
that XML with the station (floor, gate, walls, table, their cameras and lights) taken
out and the arms re-rooted: the base frame is midway between the arms' bases, on the
plane they stand on (ABC's table top), x the way they face, z up; the left arm (ABC's
``left_`` joints and observations) on +y, the arms 0.62 m apart. :data:`T_ABC_BASE` is
the base in ABC's world frame as its robot file stands the arms (z 0.75); most of
its task scenes stand them 1 cm higher, at 0.76 (6 of 25 at 0.75): a world placing a
scene against ABC's reads the scene's. Everything else is ABC's: bodies, joints, sites
and actuators keep their names, the position servos their gains and force limits, the
joints their armature and friction loss, the links their gravity compensation, the
pads their friction, the arms their collisions with each other, and the wrist cameras
(``left``, ``right``) their mounts. So an arm moves here as it does in ABC.

Each finger slides from 0 (closed, the pads 0.8 mm apart) to 47.5 mm (open, 95 mm
apart); the right finger mirrors the driven left one by ABC's joint equality, written
with the driver second so that the gripper poser reads it. ABC commands the gripper
from 0 (closed) to 1 (open), the reverse of the opening here. Each arm's TCP,
``{side}_tcp``, is ABC's ``{side}_grasp_site``: 134.7 mm along link6's z (the approach),
its y the fingers' slide. The home pose is ABC's ``config.init_q``, where its episodes
start.

:func:`urdf` writes one arm as a URDF for planners (CDG's cuRobo): the links
``base_link, link1..link6, left_finger, right_finger``, joint frames at the MJCF's body
origins, the links' meshes as visual and collision geometry, inertials the MJCF's. The
YAM has no Isaac Lab model here (no USD): :func:`isaac_cfg` says so.
"""

from __future__ import annotations

import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from r2s2r.paths import CACHE_DIR
from r2s2r.robots.spec import ArmSpec, GripperSpec, RobotSpec
from r2s2r.transforms import make_transform

# Fetched: yam_bimanual_empty.xml, assets/*.stl, LICENSE; made here: yam_arm.urdf.
YAM_DIR = CACHE_DIR / "i2rt_yam"
MJCF_PATH = YAM_DIR / "yam_bimanual_empty.xml"
URDF_PATH = YAM_DIR / "yam_arm.urdf"
ARMS = {"left": 0.31, "right": -0.31}  # each arm's base, along the base frame's y (m)
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGERS = ("left_finger", "right_finger")  # right_finger mirrors left_finger
# The URDF's velocity limits: i2rt's own URDF gives the arm joints 10 rad/s; the
# fingers' is the 47.5 mm slide in a tenth of a second.
JOINT_VELOCITY = 10.0
FINGER_VELOCITY = 0.5
FINGER_OPEN = 0.0475  # the driven finger's slide (m)
PAD_GAP = 0.095  # the pads' inner faces apart, open (m)
GRASP_OFFSET = 0.1347  # ABC's grasp site along link6's z (m)
# ABC's config.init_q (the real teleop pose) per arm, its gripper open.
HOME_Q = {
    "left": (-0.20656902, 0.47283894, 0.99431604, -0.7043946, -0.30842298, -0.32864118),
    "right": (0.20160982, 0.39005876, 1.1182956, -0.8726253, 0.13332571, 0.42629892),
}
# The base in ABC's world frame: its arms stand at (0.2525, +-0.31, 0.75), facing +x.
T_ABC_BASE = make_transform(np.eye(3), [0.2525, 0.0, 0.75])
# ABC's bodies (after the side prefix): the arm's base, its links and its fingers, by
# URDF link name.
BODIES = {
    "base_link": "arm",
    **{f"link{i}": f"link_{i}" for i in range(1, 7)},
    **{finger: f"link_{finger}" for finger in FINGERS},
}


def abc_xml() -> ET.Element:
    """ABC's ``yam_bimanual_empty.xml``, parsed."""
    if not MJCF_PATH.exists():
        raise FileNotFoundError(f"{MJCF_PATH}: run scripts/setup/fetch_abc_yam.sh")
    return ET.parse(MJCF_PATH).getroot()


def abc_spec(root: ET.Element | None = None) -> Any:
    """ABC's XML (:func:`abc_xml` unless given) as a fresh ``MjSpec``, its meshes the
    fetched ones (the XML names them by ABC's layout)."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    text = ET.tostring(abc_xml() if root is None else root, encoding="unicode")
    spec = mujoco.MjSpec.from_string(text)
    spec.meshdir = str(YAM_DIR / "assets")
    return spec


def mjcf() -> Any:
    """ABC's two arms alone, at the base frame, each gripper's right finger following
    its left."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    # The station, the keyframe (its qpos is the station's model's) and the equalities
    # go in the XML: MjSpec.delete leaves a deleted keyframe in the compiled model.
    root = abc_xml()
    root.set("model", "dual_yam")
    for tag in ("keyframe", "equality"):
        root.remove(root.findall(tag)[0])
    worldbody = root.findall("worldbody")[0]
    for element in list(worldbody):
        if element.get("name") not in {f"{side}_arm" for side in ARMS}:
            worldbody.remove(element)
    asset = root.findall("asset")[0]
    for element in list(asset):
        if element.tag == "texture" or element.get("name") in (
            "groundplane",
            "base_visual_gate",
        ):
            asset.remove(element)
    spec = abc_spec(root)
    for side, y in ARMS.items():
        spec.body(f"{side}_arm").pos = [0.0, y, 0.0]
        driver, follower = (f"{side}_{finger}" for finger in FINGERS)
        spec.add_equality(  # follower = -driver (ABC's polycoef 0 -1 0 0 0)
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=follower,
            name2=driver,
            data=[0.0, -1.0] + [0.0] * 9,
        )
        grasp = spec.site(f"{side}_grasp_site")
        spec.body(f"{side}_link_6").add_body(
            name=f"{side}_tcp", pos=grasp.pos, quat=grasp.quat
        )
    return spec


def urdf() -> Path:
    """One arm as a URDF (:data:`URDF_PATH`, written when missing or out of date), from
    the MJCF: joint frames at the body origins, axes the MJCF's, the meshes by path
    relative to :data:`YAM_DIR`, a finger's effort its gripper actuator's. ABC's XML
    has no velocity limits; the URDF's are i2rt's own (:data:`JOINT_VELOCITY`,
    :data:`FINGER_VELOCITY`), which a planner reading the file takes."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    spec = mjcf()
    robot = ET.Element("robot", name="yam_arm")
    for link, name in BODIES.items():
        body = spec.body(f"left_{name}")
        element = ET.SubElement(robot, "link", name=link)
        if link != "base_link":
            inertial = ET.SubElement(element, "inertial")
            _origin(inertial, body.ipos, body.iquat)
            ET.SubElement(inertial, "mass", value=_num(body.mass))
            ixx, iyy, izz = (_num(i) for i in body.inertia)
            ET.SubElement(
                inertial,
                "inertia",
                ixx=ixx,
                iyy=iyy,
                izz=izz,
                ixy="0",
                ixz="0",
                iyz="0",
            )
        for geom in body.geoms:
            if not geom.meshname:
                continue
            for tag in ("visual", "collision"):
                shape = ET.SubElement(element, tag)
                _origin(shape, geom.pos, geom.quat)
                mesh = ET.SubElement(ET.SubElement(shape, "geometry"), "mesh")
                mesh.set("filename", f"assets/{spec.mesh(geom.meshname).file}")
                if tag == "visual":
                    material = ET.SubElement(shape, "material", name=geom.material)
                    rgba = spec.material(geom.material).rgba
                    ET.SubElement(material, "color", rgba=" ".join(map(_num, rgba)))
    links = {name: link for link, name in BODIES.items()}
    for link, name in BODIES.items():
        if link == "base_link":
            continue
        body = spec.body(f"left_{name}")
        joint = body.joints[0]
        slide = joint.type == mujoco.mjtJoint.mjJNT_SLIDE
        joint_name = link.replace("link", "joint")  # joint1..6; the fingers their own
        element = ET.SubElement(
            robot, "joint", name=joint_name, type="prismatic" if slide else "revolute"
        )
        parent = body.parent.name.removeprefix("left_")
        ET.SubElement(element, "parent", link=links[parent])
        ET.SubElement(element, "child", link=link)
        _origin(element, body.pos, body.quat)
        ET.SubElement(element, "axis", xyz=" ".join(map(_num, joint.axis)))
        actuator = "left_gripper" if link in FINGERS else f"left_{joint_name}"
        ET.SubElement(
            element,
            "limit",
            lower=_num(joint.range[0]),
            upper=_num(joint.range[1]),
            effort=_num(spec.actuator(actuator).forcerange[1]),
            velocity=_num(FINGER_VELOCITY if slide else JOINT_VELOCITY),
        )
    ET.indent(robot)
    text = ET.tostring(robot, encoding="unicode") + "\n"
    if not URDF_PATH.exists() or URDF_PATH.read_text(encoding="utf-8") != text:
        part = URDF_PATH.with_suffix(f".{os.getpid()}.part")
        part.write_text(text, encoding="utf-8")
        part.replace(URDF_PATH)
    return URDF_PATH


def _origin(element: ET.Element, pos: Any, quat: Any) -> None:
    """A URDF ``<origin>`` from a MuJoCo position and (w, x, y, z) quaternion."""
    with warnings.catch_warnings():  # a gimbal lock leaves the angles valid
        warnings.simplefilter("ignore")
        rpy = Rotation.from_quat(np.roll(quat, -1)).as_euler("xyz")
    ET.SubElement(
        element, "origin", xyz=" ".join(map(_num, pos)), rpy=" ".join(map(_num, rpy))
    )


def _num(value: Any) -> str:
    return f"{float(value):.9g}"


def isaac_cfg() -> Any:
    """No Isaac Lab model: the YAM is simulated in MuJoCo only."""
    raise NotImplementedError(
        "dual_yam has no Isaac Lab articulation (no USD of the i2rt YAM); simulate "
        "it in MuJoCo"
    )


def _arm(side: str) -> ArmSpec:
    """The ``side`` arm at ABC's start pose, its right finger mirroring its left."""
    driver, follower = (f"{side}_{finger}" for finger in FINGERS)
    return ArmSpec(
        joints=tuple(f"{side}_{j}" for j in ARM_JOINTS),
        home_q=HOME_Q[side],
        isaac_joints=tuple(f"{side}_{j}" for j in ARM_JOINTS),
        gripper=GripperSpec(
            driver=driver,
            open=FINGER_OPEN,
            closed=0.0,
            followers="equality",
            actuator=f"{side}_gripper",
            ctrl=(FINGER_OPEN, 0.0),
            isaac_driver=driver,
            isaac_joints={driver: (FINGER_OPEN, 0.0), follower: (-FINGER_OPEN, 0.0)},
        ),
        tcp_body=f"{side}_tcp",
        tcp_offset=0.0,
        max_opening=PAD_GAP,
    )


DUAL_YAM = RobotSpec(
    name="dual_yam",
    mjcf=mjcf,
    arms=(_arm("left"), _arm("right")),
    isaac_cfg=isaac_cfg,
)
