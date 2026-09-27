"""Scenes and meshes as GLB for the viewer, in the robot base frame."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from r2s2r.agentic.geometry import load_mesh
from r2s2r.agentic.objects import render_preview
from r2s2r.assets import urdf_visual_meshes
from r2s2r.structs import SCENE_FILENAME, SceneSpec

SUPPORT_THICKNESS = 0.02
SUPPORT_COLOR = [150, 170, 200, 110]


def mesh_glb(path: Path) -> bytes:
    """Any mesh file (with its texture) as a GLB."""
    loaded: Any = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.Scene(loaded)
    return bytes(loaded.export(file_type="glb"))


def mesh_preview(path: Path) -> bytes:
    """Four views of a generated (y-up) mesh, as PNG."""
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "mesh.obj"
        if path.suffix.lower() == ".obj":
            obj = path
        else:
            load_mesh(path).export(obj)
        png = Path(tmp) / "preview.png"
        render_preview(obj, png)
        return png.read_bytes()


def scene_glb(path: Path) -> tuple[bytes, dict[str, Any]]:
    """A scene directory (``scene.json``) or an objects file as a GLB of its objects and
    support slab, posed in the base frame; and what it holds."""
    scene: Any = trimesh.Scene()
    T_support: np.ndarray | None
    extent: Any
    info: dict[str, Any] = {"objects": []}
    if path.is_dir() or path.name == SCENE_FILENAME:
        spec = SceneSpec.load(path if path.is_dir() else path.parent)
        T_support, extent = spec.T_base_support, spec.support_extent
        for obj in spec.objects:
            for k, visual in enumerate(urdf_visual_meshes(obj.asset_path)):
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
        spec_json = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent
        support = spec_json.get(
            "support", spec_json if "T_base_support" in spec_json else None
        )
        if isinstance(support, str):
            support = json.loads((base / support).read_text(encoding="utf-8"))
        T_support = np.asarray(support["T_base_support"], float) if support else None
        extent = support.get("extent") if support else None
        for obj in spec_json.get("objects", []):
            mesh_path = Path(obj["mesh"])
            mesh_path = mesh_path if mesh_path.is_absolute() else base / mesh_path
            loaded = load_mesh(mesh_path)
            scale = np.broadcast_to(np.asarray(obj.get("scale", 1.0), float), (3,))
            loaded.apply_transform(np.diag([*scale, 1.0]))
            T = np.asarray(obj["T_base_obj"], float)
            loaded.apply_transform(T)
            scene.add_geometry(loaded, node_name=str(obj["name"]))
            info["objects"].append(
                {"name": obj["name"], "position": T[:3, 3].round(4).tolist()}
            )
    if T_support is not None and extent:
        slab = trimesh.creation.box(extents=[extent[0], extent[1], SUPPORT_THICKNESS])
        slab.apply_translation([0, 0, -SUPPORT_THICKNESS / 2])
        slab.apply_transform(T_support)
        slab.visual.face_colors = SUPPORT_COLOR
        scene.add_geometry(slab, node_name="support")
        info["support"] = {"extent": list(extent)}
    return bytes(scene.export(file_type="glb")), info
