"""URDF assets: writing an object's URDF, making it simulation-ready, reading its
geometry, and where an object rests.

:func:`make_sim_ready` writes one ``<name>_r2s2r.urdf`` per object:

- every ``<mesh scale>`` is baked into its mesh (Isaac Sim 5.1's importer turns a
  scaled mesh into an unscaled prototype plus a scaled instance, and left a 1 m
  collider beside the right one);
- a rigid object resting on the support gets a flat base. An object seen standing
  still on the support must also stand in simulation, but generated meshes round off
  the edges an object stands on, and the contact patch left can be a fraction of the
  real footprint (a tall box then topples).

Objects are rigid: the readers place every link where its joints' origins put it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from r2s2r.structs import ObjectSpec
from r2s2r.transforms import make_transform, rotation_to_quat, transform_points

SIM_READY_SUFFIX = "_r2s2r"
RESTING_BASE = "r2s2r_resting_base"


@dataclass
class SimReadyConfig:
    """Knobs of :func:`make_sim_ready` (metres)."""

    base_height: float = 0.005  # the flat base's thickness
    contact_tolerance: float = 0.01  # lowest point this close to the support: resting


@dataclass
class VisualMesh:
    """A visual mesh in the URDF root link frame, with its base-colour texture."""

    mesh: trimesh.Trimesh
    texture: Path | None


def make_sim_ready(
    urdf_path: str | Path,
    T_support_obj: NDArray,
    config: SimReadyConfig | None = None,
) -> Path:
    """Write ``<name>_r2s2r.urdf`` next to ``urdf_path`` (see the module doc).

    ``T_support_obj`` places the object in the support frame (z up, the support
    surface at z = 0); it decides whether the object rests on the support.
    """
    cfg = config or SimReadyConfig()
    urdf_path = Path(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    _bake_mesh_scales(root, urdf_path.parent)
    _add_resting_base(root, urdf_path, np.asarray(T_support_obj, float), cfg)
    out_path = urdf_path.with_name(f"{urdf_path.stem}{SIM_READY_SUFFIX}.urdf")
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    return out_path


def write_object_urdf(
    path: Path,
    name: str,
    mass: float,
    com: ArrayLike,
    inertia: ArrayLike,
    collisions: list[Path] | list[str],
) -> None:
    """One link: the visual mesh ``visual.obj`` beside ``path``, the collision meshes
    ``collisions`` (relative to it), and ``mass`` (kg) at ``com`` with the 3x3
    ``inertia`` about it, in the link's axes."""
    inertia = np.asarray(inertia, float)
    robot = ET.Element("robot", {"name": name})
    link = ET.SubElement(robot, "link", {"name": "base"})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(
        inertial,
        "origin",
        {"xyz": " ".join(f"{v:.6f}" for v in np.asarray(com, float)), "rpy": "0 0 0"},
    )
    ET.SubElement(inertial, "mass", {"value": f"{mass:.6f}"})
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
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", {"filename": "visual.obj"})
    for k, rel in enumerate(collisions):
        col = ET.SubElement(link, "collision", {"name": f"hull_{k}"})
        ET.SubElement(ET.SubElement(col, "geometry"), "mesh", {"filename": str(rel)})
    ET.indent(robot)
    ET.ElementTree(robot).write(path, xml_declaration=True, encoding="utf-8")


# ----------------------------------------------------------------------- readers
def urdf_visual_meshes(urdf_path: str | Path) -> list[VisualMesh]:
    """Every visual mesh, posed in the root link frame."""
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    poses = _link_poses(root)
    out = []
    for link in root.iter("link"):
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
            out.append(VisualMesh(mesh, base_color_texture(path)))
    if not out:
        raise ValueError(f"no visual meshes in {urdf_path}")
    return out


def urdf_visual_points(
    urdf_path: str | Path, n: int, seed: int = 0
) -> NDArray[np.float64]:
    """``n`` surface samples per visual mesh, in the root link frame."""
    return np.concatenate(
        [
            trimesh.sample.sample_surface(v.mesh, n, seed=seed)[0]
            for v in urdf_visual_meshes(urdf_path)
        ]
    )


def object_points(obj: ObjectSpec, n: int = 3000) -> NDArray[np.float64]:
    """``n`` surface samples per visual mesh of a scene object, in the robot base
    frame."""
    return transform_points(obj.T_base_obj, urdf_visual_points(obj.asset_path, n))


# ------------------------------------------------------------------ preparation
def _bake_mesh_scales(root: ET.Element, asset_dir: Path) -> None:
    for mesh_el in root.iter("mesh"):
        scale = urdf_vector(mesh_el, "scale", "1 1 1")
        mesh_el.attrib.pop("scale", None)
        if np.allclose(scale, 1.0):
            continue
        rel = Path(mesh_el.attrib["filename"])
        baked_rel = rel.parent / f"{rel.stem}{SIM_READY_SUFFIX}.obj"
        baked, source = asset_dir / baked_rel, asset_dir / rel
        if not baked.exists() or baked.stat().st_mtime < source.stat().st_mtime:
            mesh = trimesh.load(source, force="mesh", process=False)
            assert isinstance(mesh, trimesh.Trimesh), f"{rel} is not a single mesh"
            mesh.apply_transform(np.diag([*scale, 1.0]))
            mesh.export(baked)
        mesh_el.attrib["filename"] = str(baked_rel)


def _add_resting_base(
    root: ET.Element, urdf_path: Path, T_support_obj: NDArray, cfg: SimReadyConfig
) -> None:
    """A convex prism: the visual mesh's cross-section ``base_height`` above its lowest
    point (support frame), extruded down to that point; added to the root link as an
    extra collision.

    Skipped for objects not resting on the support.
    """
    link = root.find("link")
    if link is None:
        return
    for old in link.findall("collision"):
        if old.attrib.get("name") == RESTING_BASE:
            link.remove(old)
    try:
        visuals = urdf_visual_meshes(urdf_path)
    except ValueError:  # nothing to take the footprint from
        return
    mesh = trimesh.util.concatenate([v.mesh for v in visuals])
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.apply_transform(T_support_obj)
    z0 = float(mesh.bounds[0, 2])
    if abs(z0) > cfg.contact_tolerance:
        return
    section = mesh.section(
        plane_origin=[0.0, 0.0, z0 + cfg.base_height], plane_normal=[0.0, 0.0, 1.0]
    )
    if section is None or len(section.vertices) < 3:
        return
    outline = np.asarray(section.vertices)[:, :2]
    base = trimesh.convex.convex_hull(
        np.concatenate(
            [
                np.c_[outline, np.full(len(outline), z0)],
                np.c_[outline, np.full(len(outline), z0 + cfg.base_height)],
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


def _link_poses(root: ET.Element) -> dict[str, NDArray[np.float64]]:
    """``T_root_link`` of every link, from the joints' origins."""
    joints = {_link(j, "child"): j for j in root.iter("joint")}
    poses = {}
    for link in root.iter("link"):
        name = link.attrib["name"]
        T = np.eye(4)
        while name in joints:
            T = urdf_origin(joints[name]) @ T
            name = _link(joints[name], "parent")
        poses[link.attrib["name"]] = T
    return poses


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
