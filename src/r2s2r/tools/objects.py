"""Tools on objects: generating a mesh from one image (Hunyuan3D-2.1), and assembling
placed meshes into a simulation-ready scene.

An objects file (JSON) describes a scene the way the agent builds it::

    {
      "support": "path/to/support.json" | {"T_base_support": 4x4, "extent": [x, y]},
      "reference_frame": "ext2@0",            # optional: whose robot state to keep
      "objects": [
        {"name": "mug", "category": "mug", "mesh": "path/to/mesh.glb",
         "scale": 0.052 | [sx, sy, sz],        # applied to the mesh file's coordinates
         "T_base_obj": 4x4,                    # then this rigid pose (base frame)
         "up": "y",                            # the mesh file's up axis (default z)
         "mass": 0.3, "friction": 0.6,         # kg; optional until assembly
         "collision": ["hull_0.obj", ...]}     # optional: convex parts, as the mesh
      ]
    }

Relative paths are relative to the objects file. ``up`` only says how to show the mesh
file on its own (upright); ``T_base_obj`` places it. Collision parts are in the mesh
file's coordinates (``scale`` applies to them too); an object without them gets its own
at assembly.
"""

from __future__ import annotations

import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from r2s2r.assets import VisualMesh, base_color_texture, make_sim_ready
from r2s2r.mjrender import CameraRenderer, add_camera, add_mesh, mujoco
from r2s2r.paths import ENV_MESH, ENV_SIMFOUNDRY, SIMFOUNDRY_DIR
from r2s2r.pipeline.workspace import Workspace
from r2s2r.structs import FrameRecord, ObjectSpec, SceneSpec
from r2s2r.tools.envjobs import run_env_job
from r2s2r.tools.geometry import UP_ROTATIONS, load_mesh, load_support
from r2s2r.tools.segment import slug
from r2s2r.transforms import (
    intrinsics_matrix,
    invert,
    is_rigid,
    look_at,
    make_transform,
    transform_points,
)

logger = logging.getLogger(__name__)

HUNYUAN_REPO = SIMFOUNDRY_DIR / "deps" / "Hunyuan3D-2.1"
DEFAULT_DENSITY = 500.0  # kg/m^3, when an object has no mass


# ----------------------------------------------------------------------- generate
def generate(
    ws: Workspace,
    images: dict[str, str],
    out_dir: str | Path,
    seed: int = 1,
    low_vram: bool = True,
) -> dict[str, Any]:
    """A textured mesh per image (RGBA, object on a transparent background) with
    Hunyuan3D-2.1, in ``out_dir/<name>/``: ``mesh.glb``, ``mesh.obj`` (+ texture),
    ``preview.png`` (four views, turned z-up) and its size in the mesh's units."""
    out_dir = Path(out_dir).resolve()
    items = [
        {
            "name": name,
            "image": str(Path(image).resolve()),
            "out": str(out_dir / name / "mesh.glb"),
            "seed": seed,
        }
        for name, image in images.items()
    ]
    result = run_env_job(
        "hunyuan_job.py",
        {"repo": str(HUNYUAN_REPO), "low_vram": low_vram, "items": items},
        ws.root / "cache" / "jobs",
        ENV_MESH,
    )
    summary = {}
    for item in result["results"]:
        name = item["name"]
        if not item["ok"]:
            summary[name] = {"ok": False, "error": item["error"]}
            continue
        mesh = load_mesh(item["out"])
        obj = Path(item["out"]).with_suffix(".obj")
        mesh.export(obj)
        preview = obj.with_name("preview.png")
        render_preview(obj, preview)
        upright = np.asarray(mesh.vertices) @ UP_ROTATIONS["y"].T
        summary[name] = {
            "ok": True,
            "glb": item["out"],
            "obj": str(obj),
            "preview": str(preview),
            "seconds": item.get("seconds"),
            "faces": int(len(mesh.faces)),
            "size_zup": np.round(np.ptp(upright, axis=0), 3).tolist(),
            "watertight": bool(mesh.is_watertight),
        }
    return summary


def render_preview(
    obj_path: Path, out_path: Path, up: str = "y", size: int = 384
) -> None:
    """Four views (front, right, back, left; from 25 degrees above) of a mesh turned
    ``up``-axis up, with its texture."""
    mesh = trimesh.load(str(obj_path), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.apply_transform(make_transform(UP_ROTATIONS[up], np.zeros(3)))
    centre = mesh.bounds.mean(0)
    mesh.apply_translation(-centre)
    radius = float(np.linalg.norm(mesh.extents)) / 2
    spec = mujoco.MjSpec()
    add_camera(spec, (size, size))
    spec.visual.headlight.ambient = [0.55, 0.55, 0.55]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    body = spec.worldbody.add_body(name="object")
    add_mesh(spec, body, "object", VisualMesh(mesh, base_color_texture(obj_path)))
    model = spec.compile()
    data = mujoco.MjData(model)
    camera = CameraRenderer(model, data, (size, size))
    f = size / (2 * np.tan(np.radians(20)))
    K = intrinsics_matrix(f, f, (size - 1) / 2, (size - 1) / 2)
    tiles = []
    try:
        for azimuth, label in (
            (-90, "front (-y)"),
            (0, "+x"),
            (90, "back (+y)"),
            (180, "-x"),
        ):
            az, el = np.radians(azimuth), np.radians(25)
            direction = [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
            T = look_at(3.0 * radius * np.array(direction), np.zeros(3))
            rgb = camera.render(K, size, size, T)["rgb"].copy()
            cv2.putText(
                rgb, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1
            )
            tiles.append(rgb)
    finally:
        camera.close()
    ext = mesh.extents
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.putText(
        sheet,
        f"z-up size {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} (mesh units)",
        (8, 2 * size - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        1,
    )
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------- assemble
def assemble(
    ws: Workspace,
    objects_path: str | Path,
    out_dir: str | Path,
    collision: str = "coacd",
    max_hulls: int = 16,
) -> dict[str, Any]:
    """An objects file -> a scene (``scene.json``) of simulation-ready objects.

    Per object, in ``out_dir/objects/<name>/``: the mesh scaled into ``visual.obj``;
    collision parts: the object's own when it has them, else ``collision``'s (``coacd``:
    CoACD's convex decomposition, ``hull``: one convex hull, ``none``: none, for looking
    only); inertia of the convex hull at the given mass (else ``DEFAULT_DENSITY``); a
    URDF, made simulation-ready (a flat base for an object resting on the support). The
    scene's provenance names the run's method.
    """
    if collision not in ("coacd", "hull", "none"):
        raise ValueError(f"unknown collision {collision!r} (coacd, hull, none)")
    objects_path = Path(objects_path).resolve()
    spec = json.loads(objects_path.read_text(encoding="utf-8"))
    base = objects_path.parent  # what the relative paths in it start from
    out_dir = Path(out_dir).resolve()
    support_src = spec["support"]
    if isinstance(support_src, str):
        support_src = str(_resolve(base, support_src))
    T_base_support, extent = load_support(support_src)
    names = [o["name"] for o in spec["objects"]]
    if len(set(names)) != len(names):
        raise ValueError(f"object names must be unique: {names}")

    objects, report = [], {}
    for obj in spec["objects"]:
        name = slug(obj["name"])
        T_base_obj = np.asarray(obj["T_base_obj"], float)
        if not is_rigid(T_base_obj):
            raise ValueError(
                f"{name}: T_base_obj must be a rigid transform (put scale in 'scale')"
            )
        obj_dir = out_dir / "objects" / name
        if obj_dir.exists():
            shutil.rmtree(obj_dir)
        obj_dir.mkdir(parents=True)
        mesh = load_mesh(_resolve(base, obj["mesh"]))
        scaling = np.diag([*np.broadcast_to(obj.get("scale", 1.0), (3,)), 1.0])
        mesh.apply_transform(scaling)
        mesh.export(obj_dir / "visual.obj")
        if obj.get("collision"):
            hulls = _given_hulls(
                [_resolve(base, h) for h in obj["collision"]], scaling, obj_dir
            )
        else:
            hulls = _collision(ws, mesh, obj_dir, collision, max_hulls)
        hull = mesh.convex_hull
        mass = obj.get("mass")
        mass = float(mass) if mass else DEFAULT_DENSITY * float(hull.volume)
        urdf = obj_dir / f"{name}.urdf"
        _write_urdf(urdf, name, hull, mass, [h.relative_to(obj_dir) for h in hulls])
        asset = make_sim_ready(urdf, invert(T_base_support) @ T_base_obj)
        objects.append(
            ObjectSpec(
                name=name,
                category=obj.get("category", name),
                asset_path=str(asset),
                T_base_obj=T_base_obj,
                mass=mass,
                friction=obj.get("friction"),
            )
        )
        hull_volume = sum(float(load_mesh(h).volume) for h in hulls)
        report[name] = {
            "size_m": np.round(mesh.extents, 4).tolist(),
            "mass_kg": round(mass, 4),
            "hulls": len(hulls),
            "hull_volume_share": (
                round(hull_volume / float(hull.volume), 3) if hulls else None
            ),
            "lowest_point_above_support_m": round(
                float(
                    transform_points(
                        invert(T_base_support) @ T_base_obj, mesh.vertices
                    )[:, 2].min()
                ),
                4,
            ),
        }

    ref = spec.get("reference_frame")
    frame = ws.frame(ref) if ref else _default_reference(ws)
    scene = SceneSpec(
        name=ws.capture.name,
        embodiment=ws.capture.embodiment,
        objects=objects,
        T_base_support=T_base_support,
        cameras=ws.capture.cameras,
        reference_camera=frame.camera,
        reference_step=frame.step,
        joint_positions=frame.joint_positions,
        provenance={
            "method": ws.read_run()["method"],
            "objects_file": str(objects_path),
            "collision": collision,
        },
        support_extent=extent,
    )
    scene.save(out_dir)
    return {"scene": str(out_dir / "scene.json"), "objects": report}


def _resolve(base: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def _default_reference(ws: Workspace) -> FrameRecord:
    frames = ws.reconstructable()
    static = [f for f in frames if ws.capture.cameras[f.camera].is_static]
    return (static or frames)[0]


def _given_hulls(sources: list[Path], scaling: np.ndarray, obj_dir: Path) -> list[Path]:
    """An object's own collision parts, scaled like its mesh, in ``collision/``."""
    out = obj_dir / "collision"
    out.mkdir()
    hulls = []
    for k, source in enumerate(sources):
        hull = load_mesh(source)
        hull.apply_transform(scaling)
        hulls.append(out / f"hull_{k}.obj")
        hull.export(hulls[-1])
    return hulls


def _collision(
    ws: Workspace, mesh: trimesh.Trimesh, obj_dir: Path, method: str, max_hulls: int
) -> list[Path]:
    out = obj_dir / "collision"
    if method == "none":
        return []
    out.mkdir()
    if method == "coacd":
        plain = obj_dir / "collision_source.obj"
        trimesh.Trimesh(mesh.vertices, mesh.faces).export(plain)
        try:
            result = run_env_job(
                "coacd_job.py",
                {"mesh": str(plain), "out_dir": str(out), "max_hulls": max_hulls},
                ws.root / "cache" / "jobs",
                ENV_SIMFOUNDRY,
            )
            plain.unlink()
            return [Path(h) for h in result["hulls"]]
        except RuntimeError as exc:
            logger.warning(
                "CoACD failed for %s (%s); using its convex hull", obj_dir, exc
            )
            plain.unlink()
    path = out / "hull_0.obj"
    mesh.convex_hull.export(path)
    return [path]


def _write_urdf(
    path: Path, name: str, hull: trimesh.Trimesh, mass: float, hulls: list[Path]
) -> None:
    """One link: the visual mesh, the collision parts, the inertia of ``hull`` (uniform
    density) at ``mass``."""
    inertia = np.asarray(hull.moment_inertia) * mass / float(hull.mass)
    com = np.asarray(hull.center_mass)
    robot = ET.Element("robot", {"name": name})
    link = ET.SubElement(robot, "link", {"name": "base"})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(
        inertial, "origin", {"xyz": " ".join(f"{v:.6f}" for v in com), "rpy": "0 0 0"}
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
    for k, rel in enumerate(hulls):
        col = ET.SubElement(link, "collision", {"name": f"hull_{k}"})
        ET.SubElement(ET.SubElement(col, "geometry"), "mesh", {"filename": str(rel)})
    ET.indent(robot)
    ET.ElementTree(robot).write(path, xml_declaration=True, encoding="utf-8")
