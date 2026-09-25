"""URDF assets: making a backend's URDF simulation-ready, and reading its geometry.

:func:`make_sim_ready` writes one ``<name>_r2s2r.urdf`` per object:

- every ``<mesh scale>`` is baked into its mesh (Isaac Sim 5.1's importer turns a
  scaled mesh into an unscaled prototype plus a scaled instance, and left a 1 m
  collider beside the right one);
- a rigid object resting on the support gets a flat base. An object seen standing
  still on the support must also stand in simulation, but generated meshes round off
  the edges an object stands on, and the contact patch left can be a fraction of the
  real footprint (a tall box then topples).

The readers compose link poses at zero joint positions, so they also serve
articulated objects.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.transforms import make_transform

SIM_READY_SUFFIX = "_r2s2r"
RESTING_BASE = "r2s2r_resting_base"
MOVABLE_JOINTS = ("revolute", "continuous", "prismatic")


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
    articulated: bool,
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
    if not articulated:  # articulated objects keep the backend's collision geometry
        _add_resting_base(root, urdf_path, np.asarray(T_support_obj, float), cfg)
    out_path = urdf_path.with_name(f"{urdf_path.stem}{SIM_READY_SUFFIX}.urdf")
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    return out_path


# ----------------------------------------------------------------------- readers
def urdf_link_poses(root: ET.Element) -> dict[str, NDArray[np.float64]]:
    """``T_root_link`` of every link at zero joint positions."""
    joints = {_link(j, "child"): j for j in root.iter("joint")}
    poses = {}
    for link in root.iter("link"):
        name = link.attrib["name"]
        T = np.eye(4)
        while name in joints:
            T = _origin(joints[name]) @ T
            name = _link(joints[name], "parent")
        poses[link.attrib["name"]] = T
    return poses


def urdf_visual_meshes(urdf_path: str | Path) -> list[VisualMesh]:
    """Every visual mesh, posed in the root link frame at zero joint positions."""
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    poses = urdf_link_poses(root)
    out = []
    for link in root.iter("link"):
        for visual in link.iter("visual"):
            mesh_el = visual.find("geometry/mesh")
            if mesh_el is None:
                continue
            path = urdf_path.parent / mesh_el.attrib["filename"]
            mesh = trimesh.load(path, force="mesh")
            assert isinstance(mesh, trimesh.Trimesh), f"{path} is not a single mesh"
            mesh.apply_transform(np.diag([*_vector(mesh_el, "scale", "1 1 1"), 1.0]))
            mesh.apply_transform(poses[link.attrib["name"]] @ _origin(visual))
            out.append(VisualMesh(mesh, _base_color_texture(path)))
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


def urdf_movable_joints(urdf_path: str | Path) -> list[dict[str, Any]]:
    """Movable joints with their axis and pivot in the root link frame.

    Each entry: ``name``, ``type``, ``lower``/``upper`` (None when unlimited),
    ``axis`` (unit vector) and ``origin`` (the joint frame's position).
    """
    root = ET.parse(urdf_path).getroot()
    poses = urdf_link_poses(root)
    out = []
    for joint in root.iter("joint"):
        kind = joint.attrib.get("type")
        if kind not in MOVABLE_JOINTS:
            continue
        T = poses[_link(joint, "child")]  # the joint frame at zero position
        axis = _vector(joint.find("axis"), "xyz", "1 0 0")
        lower, upper = _limits(joint)
        out.append(
            {
                "name": joint.attrib["name"],
                "type": kind,
                "lower": lower,
                "upper": upper,
                "axis": (T[:3, :3] @ axis / np.linalg.norm(axis)).tolist(),
                "origin": T[:3, 3].tolist(),
            }
        )
    return out


# ------------------------------------------------------------------ preparation
def _bake_mesh_scales(root: ET.Element, asset_dir: Path) -> None:
    for mesh_el in root.iter("mesh"):
        scale = _vector(mesh_el, "scale", "1 1 1")
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


# ---------------------------------------------------------------------- helpers
def _base_color_texture(mesh_path: Path) -> Path | None:
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


def _limits(joint: ET.Element) -> tuple[float | None, float | None]:
    limit = joint.find("limit")
    if limit is None or joint.attrib.get("type") == "continuous":
        return None, None
    return float(limit.attrib.get("lower", 0.0)), float(limit.attrib.get("upper", 0.0))


def _link(joint: ET.Element, tag: str) -> str:
    el = joint.find(tag)
    assert el is not None, f"joint {joint.attrib.get('name')} has no <{tag}>"
    return el.attrib["link"]


def _vector(el: ET.Element | None, key: str, default: str) -> NDArray[np.float64]:
    return np.array(
        (default if el is None else el.attrib.get(key, default)).split(), float
    )


def _origin(el: ET.Element) -> NDArray[np.float64]:
    origin = el.find("origin")
    rpy = _vector(origin, "rpy", "0 0 0")
    return make_transform(
        Rotation.from_euler("xyz", rpy).as_matrix(), _vector(origin, "xyz", "0 0 0")
    )
