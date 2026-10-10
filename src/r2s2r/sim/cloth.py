"""Cloths in Newton (its VBD solver, ``scripts/tools/cloth_job.py``, in the env
:data:`~r2s2r.paths.ENV_NEWTON`), beside MuJoCo: settling (:func:`settle`) and moving
with a robot (:class:`ClothSim`).

Neither Isaac Lab 2.3 nor MuJoCo 3.3 simulates a cloth well (Isaac's deformable surfaces,
MuJoCo's flexes: stiff cloth only at a tenth of the time step), so both hold a cloth as
it lies, a surface to see that nothing touches, and Newton moves it. What a cloth meets
is what collides in the scene as MuJoCo builds it (:mod:`r2s2r.sim.mjscene`): every
body's convex colliders (the robot's, the objects') and the support's plane.

Settling, after either simulator has let the bodies come to rest: the free bodies within
reach of a cloth (:data:`REACH` of what it spans, down to the support) move with the
cloths, each pushing the other (a towel's weight on a cup, a bowl standing on a napkin),
touching the other objects and the support as well; the robot (as the capture has it at
the start of the static period) and the articulated objects are held still. A body that
moved takes Newton's pose; one that stayed within :data:`RESTING_M` and
:data:`RESTING_DEG` stays where the simulator left it. The settled surface replaces the
cloth's visual mesh, its vertices (and UVs, texture) as they were.

Beside a running session (:class:`ClothSim`), every body is held in Newton and moved
where MuJoCo has it, so the robot can push, drag, pinch and lift a cloth; the cloth
pushes nothing back (the bodies are MuJoCo's to move), and MuJoCo draws it where Newton
has it.

A cloth's material becomes Newton's (:func:`newton_material`): a membrane and a plate of
its Young's modulus E, Poisson's ratio nu and thickness t, stretching with the Lame
parameters mu = E t / 2 (1 + nu) and lambda = E t nu / (1 - nu^2) (plane stress, N/m),
bending with the plate's stiffness D = E t^3 / 12 (1 - nu^2) (N m) back toward flat,
its mass spread evenly over its area; its contacts half its thickness out. It touches
the other cloths, and itself. It must start on or above the support: one hanging over
the table's edge would start inside the support's plane.
"""

from __future__ import annotations

import base64
import json
import shutil
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.assets import VISUAL, export_visual, urdf_visual_meshes
from r2s2r.mjrender import geom_mesh, mujoco
from r2s2r.paths import ENV_NEWTON
from r2s2r.sim.mjscene import DEFAULT_FRICTION, Session
from r2s2r.sim.mujoco import held_session
from r2s2r.structs import Capture, ObjectSpec, SceneSpec
from r2s2r.tools.envjobs import JOBS_DIR, env_python, run_env_job, simfoundry_env
from r2s2r.transforms import (
    invert,
    make_transform,
    matrix_to_pos_quat,
    quat_wxyz_to_xyzw,
    transform_points,
)

MIN_RADIUS = 0.001  # m: a cloth's contact radius, half its thickness, at least this
# A body that moved less than this while settling stays where the simulator left it:
# Newton's own rest differs from MuJoCo's or PhysX's by about half as much (a banana
# 0.6 mm and 0.5 degrees, a cloth nowhere near it).
RESTING_M, RESTING_DEG = 0.001, 1.0
REACH = 0.02  # m: a free body this near what a cloth spans moves with it, settling
JOB = "cloth_job.py"
CLOSE_SECONDS = 30  # Newton has this long to stop when asked


# ------------------------------------------------------------------------- settle
def settle(
    out_dir: str | Path, capture_dir: str | Path, seconds: float
) -> dict[str, Any] | None:
    """Let the cloths of the settled scene in ``out_dir`` (``scene.json``,
    ``settle.json``, as a simulator left them) come to rest for ``seconds``, the free
    bodies within their reach with them; each cloth's settled surface and URDF go to
    ``out_dir/objects/<name>/``, and the scene and the settle report (how far each
    cloth's furthest point moved; ``by_cloth``: how far each body moved with them) are
    rewritten. Returns the report, or None when the scene has no cloth."""
    out_dir = Path(out_dir).resolve()
    scene = SceneSpec.load(out_dir)
    cloths = [obj for obj in scene.objects if obj.cloth]
    if not cloths:
        return None
    to_support = invert(np.asarray(scene.T_base_support, float))
    surfaces, placed = _placed(cloths, to_support)
    for obj in cloths:
        assert obj.cloth is not None
        below = -float(placed[obj.name][:, 2].min())
        if below > obj.cloth["thickness"]:
            raise ValueError(
                f"{obj.name} starts {below * 1000:.0f} mm below the support: a cloth "
                "must lie on or above it"
            )
    session = held_session(scene, Capture.load(capture_dir))
    try:
        m = session.model
        free = {  # each free body (MuJoCo's id) and its object
            m.body(obj.name).id: obj
            for obj in scene.objects
            if not obj.cloth and not obj.joints
        }
        objects = {  # every object's links: what a free body may rest on
            b
            for name, ids in session.bodies.items()
            if not session.objects[name].cloth
            for b in ids
        }
        ids, arrays = scene_bodies(m, session.data, to_support, objects)
    finally:
        session.close()
    arrays["body_dynamic"] = np.isin(ids, sorted(free)) & within_reach(
        arrays, list(placed.values())
    )
    work = out_dir / "cloth"
    work.mkdir(parents=True, exist_ok=True)
    write_input(work / "input.npz", arrays, cloths, placed, surfaces)
    result = run_env_job(
        JOB,
        {
            "input": str(work / "input.npz"),
            "output": str(work / "output.npz"),
            "seconds": seconds,
            "cloths": [newton_material(obj, surfaces[obj.name]) for obj in cloths],
        },
        work,
        ENV_NEWTON,
    )
    settled = np.load(result["output"])
    report = json.loads((out_dir / "settle.json").read_text(encoding="utf-8"))
    report["cloth"] = {
        "device": result["device"],
        "seconds_wall": result["seconds_wall"],
    }
    changed = {}
    for i, (obj, job) in enumerate(zip(cloths, result["cloths"])):
        assert job["name"] == obj.name, (job["name"], obj.name)
        after = settled[f"cloth{i}_vertices"]
        changed[obj.name] = write_cloth(
            obj, surfaces[obj.name], to_support, after, out_dir
        )
        report["objects"][obj.name] = {
            **cloth_movement(placed[obj.name], after),
            "speed_m_s": job["speed_m_s"],
        }
    to_base = invert(to_support)
    for entry in report["objects"].values():
        entry.pop("by_cloth", None)  # an earlier run's
    for k, body in enumerate(ids):
        if not arrays["body_dynamic"][k]:
            continue
        obj = free[body]
        T0, T1 = obj.T_base_obj, to_base @ _pose_matrix(settled["body_pose"][k])
        moved = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        turned = float(
            np.degrees(Rotation.from_matrix(T0[:3, :3].T @ T1[:3, :3]).magnitude())
        )
        if moved < RESTING_M and turned < RESTING_DEG:
            continue
        changed[obj.name] = replace(obj, T_base_obj=T1, usd=None)
        report["objects"].setdefault(obj.name, {})["by_cloth"] = {
            "moved_m": round(moved, 4),
            "turned_deg": round(turned, 2),
        }
    scene = replace(
        scene,
        objects=[changed.get(obj.name, obj) for obj in scene.objects],
        provenance={**scene.provenance, "settle": report},
    )
    scene.save(out_dir)
    (out_dir / "settle.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


# ----------------------------------------------------------------- beside MuJoCo
class ClothSim:
    """The cloths of a MuJoCo session's scene, in Newton beside it (the module doc):
    :meth:`step` after the session has stepped, :meth:`show` before it renders.
    Close it when done."""

    def __init__(self, session: Session, work_dir: str | Path) -> None:
        self.session = session
        scene = session.scene
        self.cloths = [obj for obj in scene.objects if obj.cloth]
        if not self.cloths:
            raise ValueError(f"scene {scene.name} has no cloth")
        self.to_support = invert(np.asarray(scene.T_base_support, float))
        self.ids, arrays = scene_bodies(session.model, session.data, self.to_support)
        work = Path(work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        surfaces, placed = _placed(self.cloths, self.to_support)
        write_input(work / "serve.npz", arrays, self.cloths, placed, surfaces)
        # pylint: disable-next=consider-using-with
        self._log = open(work / "cloth_sim.log", "wb")
        try:
            # pylint: disable-next=consider-using-with
            self._proc = subprocess.Popen(
                [env_python(ENV_NEWTON), str(JOBS_DIR / JOB), "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                cwd=work,
                env=simfoundry_env(),
            )
            self.device = self._ask(
                {
                    "op": "init",
                    "input": str(work / "serve.npz"),
                    "cloths": [
                        newton_material(obj, surfaces[obj.name]) for obj in self.cloths
                    ],
                }
            )["device"]
        except BaseException:
            self.close()
            raise

    def step(self, seconds: float) -> None:
        """``seconds`` of the cloths' physics, the bodies moving steadily from where
        they were to where the session has them now."""
        mujoco.mj_kinematics(self.session.model, self.session.data)  # after mj_step
        poses = body_poses(self.session.data, self.ids, self.to_support)
        self._ask({"op": "step", "seconds": seconds, "body_pose": _encode(poses)})

    def vertices(self) -> dict[str, NDArray[np.float64]]:
        """Each cloth's vertices (its surface's, in order), in the base frame."""
        answer = self._ask({"op": "cloths"})
        to_base = invert(self.to_support)
        return {
            obj.name: transform_points(to_base, _decode(v))
            for obj, v in zip(self.cloths, answer["vertices"])
        }

    def show(self) -> None:
        """Draw every cloth in the session's renders where Newton has it."""
        for name, vertices in self.vertices().items():
            self.session.show_surface(name, vertices)

    def close(self) -> None:
        """Stop Newton (asked, else killed)."""
        proc = getattr(self, "_proc", None)
        try:
            if proc is not None and proc.poll() is None:
                try:
                    self._ask({"op": "close"})
                    proc.wait(timeout=CLOSE_SECONDS)
                except (RuntimeError, subprocess.TimeoutExpired):
                    proc.kill()
                    proc.wait()
        finally:
            self._log.close()

    def _ask(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        body = json.dumps(request).encode()
        try:
            self._proc.stdin.write(struct.pack("<I", len(body)) + body)
            self._proc.stdin.flush()
            head = self._proc.stdout.read(4)
            if len(head) < 4:
                raise EOFError("Newton stopped")
            answer: dict[str, Any] = json.loads(
                self._proc.stdout.read(struct.unpack("<I", head)[0])
            )
        except (BrokenPipeError, EOFError, ValueError) as exc:
            raise RuntimeError(f"cloth simulation failed: {self._tail()}") from exc
        if "error" in answer:
            raise RuntimeError(f"cloth simulation: {answer['error']}; {self._tail()}")
        return answer

    def _tail(self) -> str:
        self._log.flush()
        text = Path(self._log.name).read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-20:])


# ------------------------------------------------------------------- the scene
def scene_bodies(
    model: Any, data: Any, to_support: NDArray, resting: set[int] | None = None
) -> tuple[list[int], dict[str, NDArray]]:
    """Every body of a compiled MuJoCo scene with colliders but the support's plane
    (the scene's one plane), as the Newton job takes them (its module doc): their
    MuJoCo ids, and the arrays (``body_*``, ``hull_*``, ``support_friction``; no body
    dynamic). A hull is a collider as MuJoCo collides it (a mesh as its convex hull),
    in its body's frame, and touches the cloths; ``resting`` bodies' hulls (the
    objects', what a body may stand on) touch the other bodies too."""
    hulls: dict[int, list[tuple[trimesh.Trimesh, float]]] = {}
    support = None
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        friction = float(model.geom_friction[g, 0])
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            support = friction
            continue
        mesh = geom_mesh(model, g)  # in its body's frame
        if mesh is not None:
            hulls.setdefault(int(model.geom_bodyid[g]), []).append(
                (mesh.convex_hull, friction)
            )
    if support is None:
        raise ValueError("the scene has no support plane")
    ids = sorted(hulls)
    every = [(h, f, k) for k, b in enumerate(ids) for h, f in hulls[b]]
    inertia = []
    for b in ids:
        R = Rotation.from_quat(quat_wxyz_to_xyzw(model.body_iquat[b])).as_matrix()
        inertia.append(R @ np.diag(model.body_inertia[b]) @ R.T)
    arrays = {
        "body_pose": body_poses(data, ids, to_support),
        "body_mass": np.array([model.body_mass[b] for b in ids], float),
        "body_com": np.array([model.body_ipos[b] for b in ids], float),
        "body_inertia": np.array(inertia, float).reshape((-1, 3, 3)),
        "body_dynamic": np.zeros(len(ids), bool),
        "hull_vertices": np.concatenate([np.asarray(h.vertices) for h, _, _ in every]),
        "hull_faces": np.concatenate([np.asarray(h.faces) for h, _, _ in every]),
        "hull_vertex_counts": np.array([len(h.vertices) for h, _, _ in every]),
        "hull_face_counts": np.array([len(h.faces) for h, _, _ in every]),
        "hull_body": np.array([k for _, _, k in every]),
        "hull_friction": np.array([f for _, f, _ in every], float),
        "hull_bodies": np.isin([ids[k] for _, _, k in every], sorted(resting or set())),
        "support_friction": np.array(support),
    }
    return ids, arrays


def within_reach(
    bodies: dict[str, NDArray], cloths: list[NDArray]
) -> NDArray[np.bool_]:
    """Which bodies (the arrays of :func:`scene_bodies`) come within :data:`REACH` of
    what a cloth (its vertices, the support's frame) spans, from its top down to the
    support: the box of each against the other."""
    count = len(bodies["body_pose"])
    lo, hi = np.full((count, 3), np.inf), np.full((count, 3), -np.inf)
    v0 = 0
    for body, n in zip(bodies["hull_body"], bodies["hull_vertex_counts"]):
        points = transform_points(
            _pose_matrix(bodies["body_pose"][body]),
            bodies["hull_vertices"][v0 : v0 + n],
        )
        v0 += n
        lo[body] = np.minimum(lo[body], points.min(axis=0))
        hi[body] = np.maximum(hi[body], points.max(axis=0))
    near = np.zeros(count, bool)
    for cloth in cloths:
        c_lo, c_hi = cloth.min(axis=0) - REACH, cloth.max(axis=0) + REACH
        c_lo[2] = -np.inf  # it may fall to the support
        near |= np.all((lo <= c_hi) & (hi >= c_lo), axis=1)
    return near


def body_poses(data: Any, ids: list[int], to_support: NDArray) -> NDArray[np.float64]:
    """Bodies ``ids``' poses in the support's frame: x, y, z and a quaternion x, y, z,
    w (Warp's order)."""
    out = []
    for b in ids:
        T = to_support @ make_transform(data.xmat[b].reshape(3, 3), data.xpos[b])
        pos, quat = matrix_to_pos_quat(T)
        out.append(np.r_[pos, quat_wxyz_to_xyzw(quat)])
    return np.array(out, float).reshape(-1, 7)


def _pose_matrix(pose: NDArray) -> NDArray[np.float64]:
    """A pose of :func:`body_poses`'s as a transform."""
    return make_transform(Rotation.from_quat(pose[3:]).as_matrix(), pose[:3])


def _visual(obj: ObjectSpec) -> trimesh.Trimesh:
    return urdf_visual_meshes(obj.asset_path)[0].mesh  # a cloth has one


def _placed(
    cloths: list[ObjectSpec], to_support: NDArray
) -> tuple[dict[str, trimesh.Trimesh], dict[str, NDArray[np.float64]]]:
    """Each cloth's visual mesh, and its vertices where it lies (the support's
    frame), by name."""
    surfaces = {obj.name: _visual(obj) for obj in cloths}
    placed = {
        obj.name: transform_points(
            to_support @ obj.T_base_obj, surfaces[obj.name].vertices
        )
        for obj in cloths
    }
    return surfaces, placed


def write_input(
    path: Path,
    bodies: dict[str, NDArray],
    cloths: list[ObjectSpec],
    placed: dict[str, NDArray],
    surfaces: dict[str, trimesh.Trimesh],
) -> None:
    """The Newton job's input NPZ: the bodies' arrays and each cloth's surface where
    it lies (``placed``, the support's frame) with its visual mesh's faces
    (``surfaces``)."""
    arrays: dict[str, Any] = dict(bodies)
    for i, obj in enumerate(cloths):
        arrays[f"cloth{i}_vertices"] = placed[obj.name]
        arrays[f"cloth{i}_faces"] = np.asarray(surfaces[obj.name].faces)
    np.savez(path, **arrays)


def newton_material(obj: ObjectSpec, surface: trimesh.Trimesh) -> dict[str, Any]:
    """Cloth ``obj`` (its visual mesh ``surface``) as the Newton job takes it (see the
    module doc): its name, its mass per area (kg/m^2), its membrane's Lame parameters
    (``tri_ke`` mu, ``tri_ka`` lambda; N/m), its plate's bending stiffness (N m), its
    contact radius (m) and its friction."""
    assert obj.cloth is not None and obj.mass is not None
    E, nu, t = (
        float(obj.cloth[k]) for k in ("youngs_modulus", "poissons_ratio", "thickness")
    )
    return {
        "name": obj.name,
        "density": float(obj.mass) / float(surface.area),
        "tri_ke": E * t / (2 * (1 + nu)),
        "tri_ka": E * t * nu / (1 - nu**2),
        "bending": E * t**3 / (12 * (1 - nu**2)),
        "radius": max(t / 2, MIN_RADIUS),
        "friction": DEFAULT_FRICTION if obj.friction is None else float(obj.friction),
    }


def write_cloth(
    obj: ObjectSpec,
    surface: trimesh.Trimesh,
    to_support: NDArray[np.float64],
    after: NDArray[np.float64],
    out_dir: Path,
) -> ObjectSpec:
    """Cloth ``obj`` with its visual mesh ``surface`` where it came to rest (``after``,
    the support's frame), in ``out_dir/objects/<name>/``: the mesh with those vertices
    as ``visual.obj`` (its texture beside it) and its URDF. Its Isaac USD, made for the
    surface as it lay, is dropped."""
    obj_dir = out_dir / "objects" / obj.name
    source = Path(obj.asset_path).resolve()
    mesh = surface.copy()
    if obj_dir.exists() and obj_dir != source.parent:
        shutil.rmtree(obj_dir)  # what a simulator left: the cloth as it lay
    obj_dir.mkdir(parents=True, exist_ok=True)
    mesh.vertices = transform_points(invert(to_support @ obj.T_base_obj), after)
    export_visual(mesh, obj_dir / f"{VISUAL}.obj")
    urdf = obj_dir / source.name
    if urdf != source:  # its visual is the one beside it
        shutil.copy(source, urdf)
    return replace(obj, asset_path=str(urdf), usd=None)


def cloth_movement(
    before: NDArray[np.float64], after: NDArray[np.float64]
) -> dict[str, float]:
    """How far a cloth's furthest point moved, its points on average, and how far it
    dropped (along the support's normal, the frame's z) on average."""
    moved = np.linalg.norm(after - before, axis=1)
    return {
        "moved_m": round(float(moved.max()), 4),
        "mean_moved_m": round(float(moved.mean()), 4),
        "dropped_m": round(float(np.mean(before[:, 2] - after[:, 2])), 4),
    }


def _encode(a: NDArray) -> dict[str, Any]:
    a = np.ascontiguousarray(a, float)
    return {"array": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape)}


def _decode(d: dict[str, Any]) -> NDArray[np.float64]:
    return np.frombuffer(base64.b64decode(d["array"]), float).reshape(d["shape"])
