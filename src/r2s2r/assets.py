"""URDF assets: writing an object's URDF (one link, or an articulated object's links
and joints), making it simulation-ready, reading its geometry, and where an object
rests.

:func:`make_sim_ready` writes one ``<name>_r2s2r.urdf`` per object:

- an object resting on the support gets a flat base (on its root link). An object seen
  standing still on the support must also stand in simulation, but generated meshes
  round off the edges an object stands on, and the contact patch left can be a
  fraction of the real footprint (a tall box then topples).

Its meshes must be at their true size, with no ``<mesh scale>`` (Isaac Sim 5.1's
importer turns a scaled mesh into an unscaled prototype plus a scaled instance, and
left a 1 m collider beside the right one); :func:`~r2s2r.tools.objects.assemble`
scales the meshes before it writes the URDF.

The readers place every link where its joints put it: at given joint positions (an
articulated object's :attr:`~r2s2r.structs.ObjectSpec.joints`), else at zero.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from r2s2r.structs import ObjectSpec
from r2s2r.transforms import make_transform, rotation_to_quat, transform_points

SIM_READY_SUFFIX = "_r2s2r"
RESTING_BASE = "r2s2r_resting_base"
ROOT_LINK = "base"  # an object's root link: a rigid object's only one
VISUAL = "visual"  # the file name of an object's (root link's) visual mesh
BASE_HEIGHT = 0.005  # m: the flat base's thickness
CONTACT_TOLERANCE = 0.01  # m: lowest point this close to the support: resting


@dataclass
class UrdfLink:
    """A link to write: its visual mesh and collision meshes (paths relative to the
    URDF), and ``mass`` (kg) at ``com`` with the 3x3 ``inertia`` about it, in the
    link's axes."""

    name: str
    visual: str
    mass: float
    com: ArrayLike
    inertia: ArrayLike
    collisions: list[str]


@dataclass
class UrdfJoint:
    """A joint to write: ``kind`` ``revolute`` or ``prismatic``, its frame at ``xyz`` in
    the parent link's frame (axes aligned with it), moving along or about ``axis``
    within ``limits`` (rad or m)."""

    name: str
    kind: str
    parent: str
    child: str
    xyz: ArrayLike
    axis: ArrayLike
    limits: tuple[float, float]


JOINT_KINDS = ("revolute", "prismatic")


@dataclass
class VisualMesh:
    """A visual mesh in the URDF root link frame, with its base-colour texture, and
    the link it belongs to."""

    mesh: trimesh.Trimesh
    texture: Path | None
    link: str = ROOT_LINK


def make_sim_ready(urdf_path: str | Path, T_support_obj: NDArray) -> Path:
    """Write ``<name>_r2s2r.urdf`` next to ``urdf_path`` (see the module doc); a
    ValueError for a scaled mesh.

    ``T_support_obj`` places the object in the support frame (z up, the support
    surface at z = 0); it decides whether the object rests on the support.
    """
    urdf_path = Path(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for mesh_el in root.iter("mesh"):
        if not np.allclose(urdf_vector(mesh_el, "scale", "1 1 1"), 1.0):
            raise ValueError(
                f"{urdf_path}: mesh {mesh_el.attrib.get('filename')} is scaled; "
                "write it at its true size"
            )
    _add_resting_base(root, urdf_path, np.asarray(T_support_obj, float))
    out_path = urdf_path.with_name(f"{urdf_path.stem}{SIM_READY_SUFFIX}.urdf")
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    return out_path


def write_object_urdf(
    path: Path, name: str, links: list[UrdfLink], joints: Sequence[UrdfJoint] = ()
) -> None:
    """An object's URDF: ``links`` (the first is the root) joined by ``joints``."""

    def numbers(values: ArrayLike, fmt: str = ".6f") -> str:
        return " ".join(f"{v:{fmt}}" for v in np.asarray(values, float))

    robot = ET.Element("robot", {"name": name})
    for spec in links:
        inertia = np.asarray(spec.inertia, float)
        link = ET.SubElement(robot, "link", {"name": spec.name})
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": numbers(spec.com), "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": f"{spec.mass:.6f}"})
        ET.SubElement(
            inertial,
            "inertia",
            {
                k: f"{inertia[i, j]:.8e}"
                for k, (i, j) in {
                    "ixx": (0, 0),
                    "ixy": (0, 1),
                    "ixz": (0, 2),
                    "iyy": (1, 1),
                    "iyz": (1, 2),
                    "izz": (2, 2),
                }.items()
            },
        )
        visual = ET.SubElement(link, "visual")
        ET.SubElement(
            ET.SubElement(visual, "geometry"), "mesh", {"filename": spec.visual}
        )
        for k, rel in enumerate(spec.collisions):
            col = ET.SubElement(link, "collision", {"name": f"hull_{k}"})
            ET.SubElement(ET.SubElement(col, "geometry"), "mesh", {"filename": rel})
    for j in joints:
        if j.kind not in JOINT_KINDS:
            raise ValueError(f"joint {j.name}: kind must be one of {JOINT_KINDS}")
        joint = ET.SubElement(robot, "joint", {"name": j.name, "type": j.kind})
        ET.SubElement(joint, "parent", {"link": j.parent})
        ET.SubElement(joint, "child", {"link": j.child})
        ET.SubElement(joint, "origin", {"xyz": numbers(j.xyz), "rpy": "0 0 0"})
        ET.SubElement(joint, "axis", {"xyz": numbers(j.axis)})
        lower, upper = j.limits
        ET.SubElement(
            joint,
            "limit",
            {
                "lower": f"{lower:.6f}",
                "upper": f"{upper:.6f}",
                "effort": "100",
                "velocity": "10",
            },
        )
    ET.indent(robot)
    ET.ElementTree(robot).write(path, xml_declaration=True, encoding="utf-8")


# ----------------------------------------------------------------------- readers
def urdf_visual_meshes(
    urdf_path: str | Path,
    joints: dict[str, float] | None = None,
    links: list[str] | None = None,
) -> list[VisualMesh]:
    """Every visual mesh (of ``links`` only, when given), posed in the root link frame
    with the joints at ``joints`` (zero for those it leaves out)."""
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    poses = _link_poses(root, joints or {})
    out = []
    for link in root.iter("link"):
        if links is not None and link.attrib["name"] not in links:
            continue
        for visual in link.iter("visual"):
            mesh_el = visual.find("geometry/mesh")
            if mesh_el is None:
                continue
            path = urdf_path.parent / mesh_el.attrib["filename"]
            mesh = trimesh.load(path, force="mesh")
            assert isinstance(mesh, trimesh.Trimesh), f"{path} is not a single mesh"
            mesh.apply_transform(
                np.diag([*urdf_vector(mesh_el, "scale", "1 1 1"), 1.0])
            )
            mesh.apply_transform(poses[link.attrib["name"]] @ urdf_origin(visual))
            out.append(VisualMesh(mesh, base_color_texture(path), link.attrib["name"]))
    if not out:
        raise ValueError(f"no visual meshes in {urdf_path}")
    return out


def urdf_joints(
    urdf_path: str | Path, joints: dict[str, float] | None = None
) -> list[dict[str, Any]]:
    """The URDF's moving joints, posed with its joints at ``joints`` (zero for those
    it leaves out): name, type, parent and child links, a point on the axis
    (``origin``) and its direction (``axis``) in the root link frame, limits, and
    the joint's position."""
    root = ET.parse(Path(urdf_path)).getroot()
    joints = joints or {}
    poses = _link_poses(root, joints)
    out = []
    for joint in root.iter("joint"):
        kind = joint.attrib.get("type", "fixed")
        if kind == "fixed":
            continue
        frame = poses[_link(joint, "parent")] @ urdf_origin(joint)
        limit = joint.find("limit")
        out.append(
            {
                "name": joint.attrib["name"],
                "type": kind,
                "parent": _link(joint, "parent"),
                "child": _link(joint, "child"),
                "origin": frame[:3, 3].tolist(),
                "axis": (
                    frame[:3, :3] @ urdf_vector(joint.find("axis"), "xyz", "1 0 0")
                ).tolist(),
                "limits": [
                    float(urdf_vector(limit, "lower", "0")[0]),
                    float(urdf_vector(limit, "upper", "0")[0]),
                ],
                "position": float(joints.get(joint.attrib["name"], 0.0)),
            }
        )
    return out


def urdf_visual_points(
    urdf_path: str | Path,
    n: int,
    seed: int = 0,
    joints: dict[str, float] | None = None,
) -> NDArray[np.float64]:
    """``n`` surface samples per visual mesh, in the root link frame (joints as for
    :func:`urdf_visual_meshes`)."""
    return np.concatenate(
        [
            trimesh.sample.sample_surface(v.mesh, n, seed=seed)[0]
            for v in urdf_visual_meshes(urdf_path, joints)
        ]
    )


def object_points(obj: ObjectSpec, n: int = 3000) -> NDArray[np.float64]:
    """``n`` surface samples per visual mesh of a scene object, as it is placed (its
    joints as recorded), in the robot base frame."""
    return transform_points(
        obj.T_base_obj, urdf_visual_points(obj.asset_path, n, joints=obj.joints)
    )


# ------------------------------------------------------------------ preparation
def _add_resting_base(
    root: ET.Element, urdf_path: Path, T_support_obj: NDArray
) -> None:
    """A convex prism: the root link's visual mesh's cross-section :data:`BASE_HEIGHT`
    above its lowest point (support frame), extruded down to that point; added to the
    root link as an extra collision.

    Skipped for objects not resting on the support.
    """
    link = root.find("link")
    if link is None:
        return
    for old in link.findall("collision"):
        if old.attrib.get("name") == RESTING_BASE:
            link.remove(old)
    try:
        visuals = urdf_visual_meshes(urdf_path, links=[link.attrib["name"]])
    except ValueError:  # nothing to take the footprint from
        return
    mesh = trimesh.util.concatenate([v.mesh for v in visuals])
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.apply_transform(T_support_obj)
    z0 = float(mesh.bounds[0, 2])
    if abs(z0) > CONTACT_TOLERANCE:
        return
    section = mesh.section(
        plane_origin=[0.0, 0.0, z0 + BASE_HEIGHT], plane_normal=[0.0, 0.0, 1.0]
    )
    if section is None or len(section.vertices) < 3:
        return
    outline = np.asarray(section.vertices)[:, :2]
    base = trimesh.convex.convex_hull(
        np.concatenate(
            [
                np.c_[outline, np.full(len(outline), z0)],
                np.c_[outline, np.full(len(outline), z0 + BASE_HEIGHT)],
            ]
        )
    )
    base.apply_transform(np.linalg.inv(T_support_obj))  # back to the link frame
    base_file = f"{urdf_path.stem}{SIM_READY_SUFFIX}_base.obj"
    base.export(urdf_path.parent / base_file)
    collision = ET.SubElement(link, "collision", {"name": RESTING_BASE})
    ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", {"filename": base_file})


def bottom_offset(
    points: NDArray, up: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The bottom centre of ``points`` (object frame) along the object-frame direction
    ``up``, as a pose in the object frame: position, and the (w, x, y, z) quaternion of
    the frame turned by the least rotation that takes z onto ``up`` (the identity for
    an upright object). physcoder's ``bottom_offset``."""
    rotation, _ = Rotation.align_vectors([np.asarray(up, float)], [[0.0, 0.0, 1.0]])
    R = rotation.as_matrix()
    local = np.asarray(points, float) @ R  # in the bottom frame's axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    pos = R @ np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    quat = rotation_to_quat(R)
    return pos, quat if quat[0] >= 0 else -quat


# ---------------------------------------------------------------------- helpers
def export_visual(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Write ``mesh`` to the OBJ file ``path``, its material and texture beside it
    named after it (``<stem>.mtl``, ``<stem>.png``): trimesh's own names, the same for
    every mesh, let a directory's last mesh overwrite the others' textures."""
    path = Path(path)
    mesh = mesh.copy()
    material = getattr(mesh.visual, "material", None)
    if material is not None:
        # A glTF's PBR material becomes a simple one on export, unnamed: name that.
        if hasattr(material, "to_simple"):
            material = material.to_simple()
        material.name = path.stem
        setattr(mesh.visual, "material", material)  # a textured mesh's visual
    text, files = trimesh.exchange.obj.export_obj(  # type: ignore[no-untyped-call]
        mesh, return_texture=True, mtl_name=f"{path.stem}.mtl"
    )
    path.write_text(text, encoding="utf-8")
    for name, data in files.items():
        (path.parent / name).write_bytes(data)
    return path


def base_color_texture(mesh_path: Path) -> Path | None:
    """The ``map_Kd`` image of an OBJ's first material, if any."""
    if mesh_path.suffix.lower() != ".obj":
        return None
    for line in mesh_path.read_text(errors="ignore").splitlines():
        if line.startswith("mtllib"):
            mtl = mesh_path.parent / line.split(maxsplit=1)[1].strip()
            if not mtl.exists():
                return None
            for mline in mtl.read_text(errors="ignore").splitlines():
                if mline.strip().startswith("map_Kd"):
                    texture = mtl.parent / mline.split(maxsplit=1)[1].strip()
                    return texture if texture.exists() else None
    return None


def _link_poses(
    root: ET.Element, joints: dict[str, float]
) -> dict[str, NDArray[np.float64]]:
    """``T_root_link`` of every link, its joints at ``joints`` (zero for the others)."""
    by_child = {_link(j, "child"): j for j in root.iter("joint")}
    poses = {}
    for link in root.iter("link"):
        name = link.attrib["name"]
        T = np.eye(4)
        while name in by_child:
            joint = by_child[name]
            T = urdf_origin(joint) @ _joint_motion(joint, joints) @ T
            name = _link(joint, "parent")
        poses[link.attrib["name"]] = T
    return poses


def _joint_motion(joint: ET.Element, joints: dict[str, float]) -> NDArray[np.float64]:
    """How a URDF joint at its position in ``joints`` (zero if not given) moves its
    child (:func:`joint_transform`)."""
    return joint_transform(
        joint.attrib.get("type", "fixed"),
        urdf_vector(joint.find("axis"), "xyz", "1 0 0"),
        float(joints.get(joint.attrib["name"], 0.0)),
    )


def joint_transform(kind: str, axis: ArrayLike, q: float) -> NDArray[np.float64]:
    """A joint of ``kind`` at ``q`` moving its child about or along the unit ``axis``: a
    turn (revolute), a slide (prismatic), or nothing (fixed)."""
    axis = np.asarray(axis, float)
    if kind in ("revolute", "continuous"):
        return make_transform(Rotation.from_rotvec(axis * q).as_matrix(), np.zeros(3))
    if kind == "prismatic":
        return make_transform(np.eye(3), axis * q)
    return np.eye(4)


def _link(joint: ET.Element, tag: str) -> str:
    el = joint.find(tag)
    assert el is not None, f"joint {joint.attrib.get('name')} has no <{tag}>"
    return el.attrib["link"]


def urdf_vector(el: ET.Element | None, key: str, default: str) -> NDArray[np.float64]:
    """An element's attribute ``key`` as numbers (``default`` when it has none)."""
    return np.array(
        (default if el is None else el.attrib.get(key, default)).split(), float
    )


def urdf_origin(el: ET.Element) -> NDArray[np.float64]:
    """The pose its ``<origin>`` gives an element (the identity without one)."""
    origin = el.find("origin")
    rpy = urdf_vector(origin, "rpy", "0 0 0")
    return make_transform(
        Rotation.from_euler("xyz", rpy).as_matrix(), urdf_vector(origin, "xyz", "0 0 0")
    )
