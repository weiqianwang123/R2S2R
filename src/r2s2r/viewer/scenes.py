"""Scenes and meshes as GLB for the viewer, in the robot base frame."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from r2s2r.assets import urdf_visual_meshes
from r2s2r.structs import SCENE_FILENAME, SUPPORT_THICKNESS, SceneSpec
from r2s2r.tools.geometry import UP_ROTATIONS, load_mesh, load_support
from r2s2r.tools.objects import read_objects_file, render_preview

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


def scene_glb(path: Path) -> tuple[bytes, dict[str, Any]]:
    """A scene directory (``scene.json``), an objects file or a support file as a GLB of
    its objects and support slab, posed in the base frame; and what it holds."""
    scene: Any = trimesh.Scene()
    extent: Any
    info: dict[str, Any] = {"objects": []}
    if path.is_dir() or path.name == SCENE_FILENAME:
        spec = SceneSpec.load(path if path.is_dir() else path.parent)
        T_support, extent = spec.T_base_support, spec.support_extent
        for obj in spec.objects:
            for k, visual in enumerate(urdf_visual_meshes(obj.asset_path, obj.joints)):
                mesh = visual.mesh.copy()
                mesh.apply_transform(obj.T_base_obj)
                scene.add_geometry(mesh, node_name=f"{obj.name}_{k}")
            info["objects"].append(
                {
                    "name": obj.name,
                    "mass": obj.mass,
                    "position": obj.T_base_obj[:3, 3].round(4).tolist(),
                }
            )
    else:
        objects: list[dict[str, Any]] = []
        if "objects" in json.loads(path.read_text(encoding="utf-8")):
            spec_file = read_objects_file(path)
            T_support, extent = spec_file.T_base_support, spec_file.extent
            objects = spec_file.objects
        else:
            T_support, extent = load_support(path)
        for entry in objects:
            T = np.asarray(entry["T_base_obj"], float)
            meshes = [entry["mesh"]] + [p["mesh"] for p in entry["parts"]]
            for k, mesh_path in enumerate(meshes):  # parts: as recorded, like the mesh
                loaded = load_mesh(mesh_path)
                loaded.apply_transform(np.diag([*entry["scale"], 1.0]))
                loaded.apply_transform(T)
                scene.add_geometry(loaded, node_name=f"{entry['name']}_{k}")
            info["objects"].append(
                {"name": entry["name"], "position": T[:3, 3].round(4).tolist()}
            )
    if extent:
        slab = trimesh.creation.box(extents=[extent[0], extent[1], SUPPORT_THICKNESS])
        slab.apply_translation([0, 0, -SUPPORT_THICKNESS / 2])
        slab.apply_transform(T_support)
        slab.visual.face_colors = SUPPORT_COLOR
        scene.add_geometry(slab, node_name="support")
        info["support"] = {"extent": list(extent)}
    return bytes(scene.export(file_type="glb")), info
