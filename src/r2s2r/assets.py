"""Asset preparation shared by the simulator builders."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.transforms import make_transform

BAKED_SUFFIX = "_r2s2r"


def bake_mesh_scales(urdf_path: str | Path) -> Path:
    """Write ``<name>_r2s2r.urdf`` with every ``<mesh scale>`` baked into its mesh.

    SimFoundry's collision hulls are unit-size meshes scaled in the URDF. Isaac Sim
    5.1's URDF importer turns a scaled mesh into an unscaled prototype plus a scaled
    instance, and the converted asset ended up with a 1 m collider beside the correct
    one, which shoves objects away as soon as physics steps. Baked copies have no scale
    attributes, so every simulator sees true-size meshes. Meshes whose scale is already
    1 (the textured visual meshes) are untouched.
    """
    urdf_path = Path(urdf_path)
    out_path = urdf_path.with_name(f"{urdf_path.stem}{BAKED_SUFFIX}.urdf")
    if out_path.exists() and out_path.stat().st_mtime >= urdf_path.stat().st_mtime:
        return out_path
    tree = ET.parse(urdf_path)
    for mesh_el in tree.getroot().iter("mesh"):
        scale = np.array(mesh_el.attrib.get("scale", "1 1 1").split(), float)
        if np.allclose(scale, 1.0):
            mesh_el.attrib.pop("scale", None)
            continue
        rel = Path(mesh_el.attrib["filename"])
        mesh = trimesh.load(urdf_path.parent / rel, force="mesh", process=False)
        assert isinstance(mesh, trimesh.Trimesh), f"{rel} is not a single mesh"
        mesh.apply_transform(np.diag([*scale, 1.0]))
        baked_rel = rel.parent / f"{rel.stem}{BAKED_SUFFIX}.obj"
        mesh.export(urdf_path.parent / baked_rel)
        mesh_el.attrib["filename"] = str(baked_rel)
        del mesh_el.attrib["scale"]
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    return out_path


def urdf_visual_points(
    urdf_path: str | Path, n: int, seed: int = 0
) -> NDArray[np.float64]:
    """Surface samples of a URDF's visual meshes, in the URDF root link frame."""
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    chunks = []
    for visual in root.iter("visual"):
        mesh_el = visual.find("geometry/mesh")
        if mesh_el is None:
            continue
        mesh = trimesh.load(urdf_path.parent / mesh_el.attrib["filename"], force="mesh")
        scale = np.array(mesh_el.attrib.get("scale", "1 1 1").split(), float)
        origin = visual.find("origin")
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        if origin is not None:
            xyz = np.array(origin.attrib.get("xyz", "0 0 0").split(), float)
            rpy = np.array(origin.attrib.get("rpy", "0 0 0").split(), float)
        pts = trimesh.sample.sample_surface(mesh, n, seed=seed)[0] * scale
        T = make_transform(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)
        chunks.append(pts @ T[:3, :3].T + T[:3, 3])
    if not chunks:
        raise ValueError(f"no visual meshes in {urdf_path}")
    return np.concatenate(chunks)
