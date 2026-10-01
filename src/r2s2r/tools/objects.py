"""Tools on objects: generating a mesh from one image (Hunyuan3D-2.1), and assembling
placed meshes into a simulation-ready scene.

An objects file (JSON) describes a scene the way a method's stage 3 leaves it::

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
file's coordinates (``scale`` applies to them too) and belong to ``mesh``; without them
(and for an articulated object's parts) assembly makes its own.

An articulated object (a box and its lid, a cabinet and its door) also has the parts
that move, and the joints that move them; ``mesh`` is then the part that does not::

        "parts": [{"name": "lid", "mesh": "lid.obj"}],   # as the mesh: file coordinates
        "joints": [{"name": "hinge", "type": "revolute", # or "prismatic"
                    "parent": "base", "child": "lid",    # "base": the object's mesh
                    "origin": [x, y, z],                 # a point on the axis, and
                    "axis": [x, y, z],                   # its direction (object frame)
                    "limits": [0.0, 1.9],                # rad or m
                    "position": 0.0}]                    # as recorded

Everything is as recorded: the parts' meshes where they were, the joints' origins and
axes in the object's frame (metres, after ``scale``), at their recorded positions.
Assembly turns that into a URDF whose joints are zero where they are at their origins.
Part and joint names are lower case, digits and underscores; ``base`` and ``visual``
are taken.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray

from r2s2r.assets import (
    JOINT_KINDS,
    ROOT_LINK,
    UrdfJoint,
    UrdfLink,
    VisualMesh,
    base_color_texture,
    joint_transform,
    make_sim_ready,
    write_object_urdf,
)
from r2s2r.mjrender import CameraRenderer, add_camera, add_mesh, mujoco
from r2s2r.paths import ENV_MESH, ENV_SIMFOUNDRY, SIMFOUNDRY_DIR
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
from r2s2r.workspace import Workspace

logger = logging.getLogger(__name__)

HUNYUAN_REPO = SIMFOUNDRY_DIR / "deps" / "Hunyuan3D-2.1"
DEFAULT_DENSITY = 500.0  # kg/m^3, when an object has no mass
VISUAL = "visual"  # the file name of an object's (root link's) visual mesh


# ----------------------------------------------------------------------- generate
def generate(
    ws: Workspace,
    images: dict[str, str],
    out_dir: str | Path,
    seed: int = 1,
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
        {"repo": str(HUNYUAN_REPO), "items": items},
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
@dataclass
class ObjectsFile:
    """An objects file, read: its support, and its objects as written but for their
    ``mesh``, ``collision`` parts and ``parts``' meshes (absolute paths), ``scale``
    (three values), and ``parts`` and ``joints`` (lists, empty for a rigid object)."""

    T_base_support: NDArray[np.float64]
    extent: tuple[float, float] | None
    objects: list[dict[str, Any]]
    reference_frame: str | None


def read_objects_file(path: str | Path) -> ObjectsFile:
    """The objects file at ``path`` (see the module doc)."""
    path = Path(path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent  # what the relative paths in it start from
    support = spec["support"]
    T_base_support, extent = load_support(
        _resolve(base, support) if isinstance(support, str) else support
    )
    objects = [
        {
            **obj,
            "mesh": _resolve(base, obj["mesh"]),
            "scale": np.broadcast_to(np.asarray(obj.get("scale", 1.0), float), (3,)),
            "collision": [_resolve(base, h) for h in obj.get("collision") or []],
            "parts": [
                {**part, "mesh": _resolve(base, part["mesh"])}
                for part in obj.get("parts") or []
            ],
            "joints": obj.get("joints") or [],
        }
        for obj in spec["objects"]
    ]
    return ObjectsFile(T_base_support, extent, objects, spec.get("reference_frame"))


def articulation_problems(obj: dict[str, Any]) -> list[str]:
    """What is wrong with an object's ``parts`` and ``joints`` (see the module doc)."""
    parts = [p.get("name") for p in obj.get("parts") or []]
    joints = obj.get("joints") or []
    name = obj.get("name", "?")
    if not parts and not joints:
        return []
    problems = []
    for kind, names in (("part", parts), ("joint", [j.get("name") for j in joints])):
        bad = [n for n in names if not isinstance(n, str) or slug(n) != n]
        if bad:
            problems.append(
                f"{name}: {kind} names must be lower case, digits and underscores: "
                f"{bad}"
            )
        if len(set(map(str, names))) != len(names):
            problems.append(f"{name}: {kind} names must be unique")
    if {ROOT_LINK, VISUAL} & set(map(str, parts)):
        problems.append(f"{name}: no part may be called {ROOT_LINK!r} or {VISUAL!r}")
    children = [j.get("child") for j in joints]
    if sorted(map(str, children)) != sorted(map(str, parts)):
        problems.append(f"{name}: every part must be the child of exactly one joint")
    for joint in joints:
        label = f"{name}: joint {joint.get('name')}"
        if joint.get("type") not in JOINT_KINDS:
            problems.append(f"{label}: type must be one of {JOINT_KINDS}")
        if joint.get("parent") not in (ROOT_LINK, *parts):
            problems.append(f"{label}: parent must be {ROOT_LINK!r} or a part")
        try:
            origin = np.asarray(joint["origin"], float)
            axis = np.asarray(joint["axis"], float)
            lower, upper = (float(v) for v in joint["limits"])
            position = float(joint["position"])
        except (KeyError, TypeError, ValueError):
            problems.append(
                f"{label}: needs origin and axis (3 numbers each), limits (2) and "
                "position"
            )
            continue
        if origin.shape != (3,) or axis.shape != (3,) or np.linalg.norm(axis) < 1e-9:
            problems.append(f"{label}: origin and axis must be 3 numbers, axis not 0")
        if not lower < upper:
            problems.append(f"{label}: limits must be [lower, upper], lower < upper")
        elif not lower - 1e-6 <= position <= upper + 1e-6:
            problems.append(f"{label}: position {position} is outside its limits")
    if not problems:
        try:
            link_frames(joints)
        except ValueError as exc:
            problems.append(f"{name}: {exc}")
    return problems


def link_frames(
    joints: list[dict[str, Any]],
) -> tuple[dict[str, NDArray[np.float64]], list[UrdfJoint], dict[str, float]]:
    """An articulated object's links as recorded: each link's frame in the object frame
    (the root at the identity), the URDF joints that give them, and the joints'
    recorded positions.

    A joint's frame sits at its origin, turned like its parent link; at the joint's
    position it moves its child's frame from there. At zero the child's frame is the
    joint's frame, which is how the URDF puts it.
    """
    frames = {ROOT_LINK: np.eye(4)}
    urdf, positions = [], {}
    pending = list(joints)
    while pending:
        ready = [j for j in pending if j["parent"] in frames]
        if not ready:
            raise ValueError(
                f"joints must form a tree from {ROOT_LINK!r}: "
                f"{[j.get('name') for j in pending]} hang from nothing"
            )
        for joint in ready:
            pending.remove(joint)
            parent = frames[joint["parent"]]
            R = parent[:3, :3]
            at_origin = make_transform(R, np.asarray(joint["origin"], float))
            axis = np.asarray(joint["axis"], float)
            axis = R.T @ (axis / np.linalg.norm(axis))  # in the joint's frame
            q = float(joint["position"])
            frames[joint["child"]] = at_origin @ joint_transform(joint["type"], axis, q)
            urdf.append(
                UrdfJoint(
                    name=joint["name"],
                    kind=joint["type"],
                    parent=joint["parent"],
                    child=joint["child"],
                    xyz=(invert(parent) @ at_origin)[:3, 3],
                    axis=axis,
                    limits=(float(joint["limits"][0]), float(joint["limits"][1])),
                )
            )
            positions[joint["name"]] = q
    return frames, urdf, positions


def assemble(
    ws: Workspace,
    objects_path: str | Path,
    out_dir: str | Path,
    collision: str = "coacd",
    max_hulls: int = 16,
) -> dict[str, Any]:
    """An objects file -> a scene (``scene.json``) of simulation-ready objects.

    Per object, in ``out_dir/objects/<name>/``: the mesh scaled into ``visual.obj``
    (an articulated object's parts into ``<part>.obj``, in their links' frames);
    collision parts: the object's own when it has them, else ``collision``'s (``coacd``:
    CoACD's convex decomposition, ``hull``: one convex hull, ``none``: none, for looking
    only), per link; inertia of the convex hull at the given mass (else
    ``DEFAULT_DENSITY``), shared among the links by convex volume; a URDF, made
    simulation-ready (a flat base for an object resting on the support). The
    scene's provenance names the run's method and where the collision parts came from:
    ``"given"`` or ``collision``, per object when they differ.
    """
    if collision not in ("coacd", "hull", "none"):
        raise ValueError(f"unknown collision {collision!r} (coacd, hull, none)")
    objects_path = Path(objects_path).resolve()
    spec = read_objects_file(objects_path)
    T_base_support = spec.T_base_support
    out_dir = Path(out_dir).resolve()
    names = [o["name"] for o in spec.objects]
    if len(set(names)) != len(names):
        raise ValueError(f"object names must be unique: {names}")

    objects, report, sources = [], {}, {}
    for obj in spec.objects:
        name = slug(obj["name"])
        T_base_obj = np.asarray(obj["T_base_obj"], float)
        if not is_rigid(T_base_obj):
            raise ValueError(
                f"{name}: T_base_obj must be a rigid transform (put scale in 'scale')"
            )
        problems = articulation_problems(obj)
        if problems:
            raise ValueError("; ".join(problems))
        frames, joints, positions = link_frames(obj["joints"])
        obj_dir = out_dir / "objects" / name
        if obj_dir.exists():
            shutil.rmtree(obj_dir)
        obj_dir.mkdir(parents=True)
        scaling = np.diag([*obj["scale"], 1.0])
        sources[name] = "given" if obj["collision"] else collision
        seen = {}  # every link's mesh as recorded, scaled, in the object frame
        for link, source in [(ROOT_LINK, obj["mesh"])] + [
            (p["name"], p["mesh"]) for p in obj["parts"]
        ]:
            seen[link] = load_mesh(source)
            seen[link].apply_transform(scaling)
        meshes, hulls_of = {}, {}  # in the links' frames
        for link, mesh in seen.items():
            meshes[link] = mesh.copy()
            meshes[link].apply_transform(invert(frames[link]))
            stem = _stem(link)
            meshes[link].export(obj_dir / f"{stem}.obj")
            if link == ROOT_LINK and obj["collision"]:
                hulls_of[link] = _given_hulls(obj["collision"], scaling, obj_dir)
            else:
                out = obj_dir / (
                    "collision" if link == ROOT_LINK else f"collision_{stem}"
                )
                hulls_of[link] = _collision(ws, meshes[link], out, collision, max_hulls)
        volumes = {link: float(m.convex_hull.volume) for link, m in meshes.items()}
        mass = obj.get("mass")
        mass = float(mass) if mass else DEFAULT_DENSITY * sum(volumes.values())
        links = []
        for link, mesh in meshes.items():
            hull = mesh.convex_hull
            share = mass * volumes[link] / sum(volumes.values())
            links.append(
                UrdfLink(
                    name=link,
                    visual=f"{_stem(link)}.obj",
                    mass=share,
                    com=hull.center_mass,
                    inertia=np.asarray(hull.moment_inertia) * share / float(hull.mass),
                    collisions=[str(h.relative_to(obj_dir)) for h in hulls_of[link]],
                )
            )
        urdf = obj_dir / f"{name}.urdf"
        write_object_urdf(urdf, name, links, joints)
        asset = make_sim_ready(urdf, invert(T_base_support) @ T_base_obj)
        objects.append(
            ObjectSpec(
                name=name,
                category=obj.get("category", name),
                asset_path=str(asset),
                T_base_obj=T_base_obj,
                mass=mass,
                friction=obj.get("friction"),
                joints=positions or None,
            )
        )
        hulls = [h for link in hulls_of.values() for h in link]
        whole = cast(trimesh.Trimesh, trimesh.util.concatenate(list(seen.values())))
        hull_volume = sum(float(load_mesh(h).volume) for h in hulls)
        report[name] = {
            "size_m": np.round(whole.extents, 4).tolist(),
            "mass_kg": round(mass, 4),
            "hulls": len(hulls),
            "hull_volume_share": (  # of the links' convex hulls
                round(hull_volume / sum(volumes.values()), 3) if hulls else None
            ),
            "lowest_point_above_support_m": round(
                float(
                    transform_points(
                        invert(T_base_support) @ T_base_obj, whole.vertices
                    )[:, 2].min()
                ),
                4,
            ),
            **(
                {
                    "joints": {
                        j.name: {
                            "type": j.kind,
                            "limits": list(j.limits),
                            "position": positions[j.name],
                        }
                        for j in joints
                    }
                }
                if joints
                else {}
            ),
        }

    ref = spec.reference_frame
    frame = ws.frame(ref) if ref else _default_reference(ws)
    kinds = set(sources.values())
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
            "collision": sources if len(kinds) > 1 else next(iter(kinds), collision),
        },
        support_extent=spec.extent,
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
    ws: Workspace, mesh: trimesh.Trimesh, out: Path, method: str, max_hulls: int
) -> list[Path]:
    """Convex collision parts of ``mesh`` in ``out/`` (see :func:`assemble`)."""
    if method == "none":
        return []
    out.mkdir()
    if method == "coacd":
        plain = out / "source.obj"
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
            logger.warning("CoACD failed for %s (%s); using its convex hull", out, exc)
            plain.unlink()
    path = out / "hull_0.obj"
    mesh.convex_hull.export(path)
    return [path]


def _stem(link: str) -> str:
    """The file name (without suffix) of a link's visual mesh."""
    return VISUAL if link == ROOT_LINK else link
