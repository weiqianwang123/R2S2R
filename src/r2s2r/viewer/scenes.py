"""Scenes and meshes as GLB for the viewer, in the robot base frame."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from r2s2r.assets import ROOT_LINK, urdf_joints, urdf_visual_meshes
from r2s2r.structs import SCENE_FILENAME, SUPPORT_THICKNESS, SceneSpec
from r2s2r.tools.geometry import UP_ROTATIONS, load_mesh, load_support
from r2s2r.tools.objects import (
    articulation_problems,
    read_objects_file,
    render_preview,
)
from r2s2r.tools.segment import slug
from r2s2r.transforms import transform_points

SUPPORT_COLOR = [150, 170, 200, 110]


def mesh_glb(path: Path) -> bytes:
    """Any mesh file (with its texture) as a GLB."""
    loaded: Any = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.Scene(loaded)
    return bytes(loaded.export(file_type="glb"))


def mesh_preview(path: Path, up: str) -> bytes:
    """Four views of a mesh turned ``up``-axis up, as PNG."""
    if up not in UP_ROTATIONS:
        raise FileNotFoundError(f"no up axis {up!r}")
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "mesh.obj"
        if path.suffix.lower() == ".obj":
            obj = path
        else:
            load_mesh(path).export(obj)
        png = Path(tmp) / "preview.png"
        render_preview(obj, png, up)
        return png.read_bytes()


def scene_glb(path: Path, only: str | None = None) -> bytes:
    """A scene directory (``scene.json``), an objects file or a support file as a GLB of
    its objects (or of the object ``only``) and support slab, posed in the base frame.

    Each link of an articulated object is its own node (``<object>__<link>__<k>``),
    and the scene's extras list its joints (:func:`_joints`), so that the page can
    move them."""
    scene: Any = trimesh.Scene()
    joints: list[dict[str, Any]] = []
    extent: Any
    if path.is_dir() or path.name == SCENE_FILENAME:
        spec = SceneSpec.load(path if path.is_dir() else path.parent)
        T_support, extent = spec.T_base_support, spec.support_extent
        for obj in spec.objects:
            if only not in (None, obj.name):
                continue
            nodes: dict[str, list[str]] = {}
            for visual in urdf_visual_meshes(obj.asset_path, obj.joints):
                mesh = visual.mesh.copy()
                mesh.apply_transform(obj.T_base_obj)
                named = nodes.setdefault(visual.link, [])
                named.append(f"{obj.name}__{visual.link}__{len(named)}")
                scene.add_geometry(mesh, node_name=named[-1])
            if obj.joints:
                posed = urdf_joints(obj.asset_path, obj.joints)
                joints += _joints(obj.name, posed, obj.T_base_obj, nodes)
    else:
        objects: list[dict[str, Any]] = []
        if "objects" in json.loads(path.read_text(encoding="utf-8")):
            spec_file = read_objects_file(path)
            T_support, extent = spec_file.T_base_support, spec_file.extent
            objects = spec_file.objects
        else:
            T_support, extent = load_support(path)
        for entry in objects:
            if only not in (None, entry["name"]):
                continue
            name = slug(entry["name"])  # as assembly names it
            T = np.asarray(entry["T_base_obj"], float)
            nodes = {}
            links = [(ROOT_LINK, entry["mesh"])] + [
                (p["name"], p["mesh"]) for p in entry["parts"]
            ]
            for link, mesh_path in links:  # parts: as recorded, like the mesh
                loaded = load_mesh(mesh_path)
                loaded.apply_transform(np.diag([*entry["scale"], 1.0]))
                loaded.apply_transform(T)
                nodes[link] = [f"{name}__{link}__0"]
                scene.add_geometry(loaded, node_name=nodes[link][0])
            if entry["joints"] and not articulation_problems(entry):
                joints += _joints(name, entry["joints"], T, nodes)
    if extent and only is None:
        slab = trimesh.creation.box(extents=[extent[0], extent[1], SUPPORT_THICKNESS])
        slab.apply_translation([0, 0, -SUPPORT_THICKNESS / 2])
        slab.apply_transform(T_support)
        slab.visual.face_colors = SUPPORT_COLOR
        scene.add_geometry(slab, node_name="support")
    scene.metadata["joints"] = joints
    return bytes(scene.export(file_type="glb"))


def _joints(
    obj: str,
    joints: list[dict[str, Any]],
    T_base_obj: NDArray[np.float64],
    nodes: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """An articulated object's joints (as recorded, in its frame) in the base frame,
    parents first, each with the nodes it moves (its child link and everything below)
    and one node of its parent link (whose motion moves the joint)."""
    below: dict[str, list[dict[str, Any]]] = {}
    for joint in joints:
        below.setdefault(joint["parent"], []).append(joint)

    def subtree(link: str) -> list[str]:
        return [link] + [l for j in below.get(link, []) for l in subtree(j["child"])]

    ordered, todo = [], [ROOT_LINK]  # breadth first from the root: parents first
    while todo:
        ordered += below.get(todo[0], [])
        todo = todo[1:] + [j["child"] for j in below.get(todo[0], [])]
    out = []
    for joint in ordered:
        axis = T_base_obj[:3, :3] @ np.asarray(joint["axis"], float)
        out.append(
            {
                "object": obj,
                "name": joint["name"],
                "type": joint["type"],
                "origin": transform_points(
                    T_base_obj, np.asarray([joint["origin"]], float)
                )[0].tolist(),
                "axis": (axis / np.linalg.norm(axis)).tolist(),
                "limits": [float(v) for v in joint["limits"]],
                "position": float(joint["position"]),
                "nodes": [n for link in subtree(joint["child"]) for n in nodes[link]],
                "parent_node": (
                    None if joint["parent"] == ROOT_LINK else nodes[joint["parent"]][0]
                ),
            }
        )
    return out
