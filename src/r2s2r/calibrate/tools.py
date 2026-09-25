"""The fixed tools of joint calibration: a workspace per object, and the evaluation of a
candidate joint model against the video.

An evaluation fits the candidate's joints to the video and runs the check of
:mod:`r2s2r.reconstruct.joints`. At the worst steps it then shows what is left
unexplained, with the joints at their fitted positions: geometry the cameras see that
the model lacks, and model geometry where the cameras see past it. Each comes as a box
in the object and base frames and as an image. The agent runs these tools through
``r2s2r joints-eval`` and ``r2s2r urdf-info``; the orchestrator reruns them itself
before accepting anything.
"""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Collection

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.assets import (
    urdf_link_collisions,
    urdf_link_poses,
    urdf_link_visuals,
    urdf_movable_joints,
)
from r2s2r.io.rgbd import load_depth_steps
from r2s2r.reconstruct.joints import (
    JointFitConfig,
    fit_joints,
    held_trajectory,
    neighbourhood,
)
from r2s2r.reconstruct.render import SceneRenderer
from r2s2r.structs import CAPTURE_FILENAME, Capture, DepthView, ObjectSpec, SceneSpec
from r2s2r.transforms import backproject

WORKSPACE_FILE = "workspace.json"
LEDGER_FILE = "ledger.jsonl"
_SKY = 20.0  # metres; rendered depth beyond this is nothing (the far plane)
# A link's collision meshes match its visual ones when their boxes agree this well.
COLLISION_TOLERANCE = 0.01
# A list of numbers, or of such lists: printed on one line.
_FLAT_LIST = re.compile(r"\[(?:[^\[\]{}]|\[[^\[\]{}]*\])*\]")


def dumps(data: Any) -> str:
    """JSON for people and agents to read: indented, with lists of numbers kept on one
    line."""
    return _FLAT_LIST.sub(
        lambda m: " ".join(m.group(0).split()).replace("[ ", "[").replace(" ]", "]"),
        json.dumps(data, indent=1),
    )


@dataclass
class Workspace:
    """Where one object is calibrated.

    ``model/`` is a copy of the object's asset directory; candidates are written next to
    its URDF, sharing its meshes.
    """

    root: Path
    scene_dir: str  # the scene being calibrated
    capture_dir: str  # the video it was reconstructed from
    object_name: str
    urdf: str  # the backend's URDF, copied into model/
    steps_file: str  # the video's depth views (r2s2r.io.rgbd.save_depth_steps)
    rest_steps: list[int]  # the static period's steps

    @property
    def model_dir(self) -> Path:
        """The object's asset directory, where candidates go."""
        return self.root / "model"

    @classmethod
    def load(cls, root: str | Path) -> Workspace:
        """Read ``workspace.json`` from ``root``."""
        root = Path(root).resolve()
        d = json.loads((root / WORKSPACE_FILE).read_text(encoding="utf-8"))
        return cls(root=root, **d)

    def save(self) -> None:
        """Write ``workspace.json``."""
        d = asdict(self)
        d.pop("root")
        (self.root / WORKSPACE_FILE).write_text(json.dumps(d, indent=2), "utf-8")


def create_workspace(
    root: str | Path,
    scene_dir: str | Path,
    capture_dir: str | Path,
    object_name: str,
    steps_file: str | Path,
    rest_steps: Collection[int],
) -> Workspace:
    """A fresh workspace for ``object_name`` of the scene saved in ``scene_dir``."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    scene = SceneSpec.load(scene_dir)
    obj = next(o for o in scene.objects if o.name == object_name)
    source = Path(obj.asset_path)
    if (root / "model").exists():
        shutil.rmtree(root / "model")
    shutil.copytree(
        source.parent, root / "model", ignore=shutil.ignore_patterns("*_fit.urdf")
    )
    ws = Workspace(
        root=root,
        scene_dir=str(Path(scene_dir).resolve()),
        capture_dir=str(Path(capture_dir).resolve()),
        object_name=object_name,
        urdf=str(root / "model" / source.name),
        steps_file=str(Path(steps_file).resolve()),
        rest_steps=sorted(int(t) for t in rest_steps),
    )
    ws.save()
    return ws


def evaluate(
    ws: Workspace,
    urdf: str | Path,
    out_dir: str | Path,
    config: JointFitConfig | None = None,
    evidence_steps: int = 4,
) -> dict[str, Any]:
    """Fit the joints of ``urdf`` (a model of the workspace's object) to the video,
    check the fit, and show where the worst steps are unexplained.

    Writes ``out_dir/report.json`` (the summary returned, plus the whole fit: the
    trajectories and per-step residuals) and one image per evidence step. ``score_mm``
    ranks models of the same object: lower explains the video better.
    """
    urdf = Path(urdf).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(config or JointFitConfig(), view_scale=1.0)  # steps come scaled
    scene = SceneSpec.load(ws.scene_dir)
    index = next(i for i, o in enumerate(scene.objects) if o.name == ws.object_name)
    obj = replace(scene.objects[index], asset_path=str(urdf), articulated=True)
    candidate = replace(
        scene, objects=[obj if i == index else o for i, o in enumerate(scene.objects)]
    )
    steps = load_depth_steps(ws.steps_file)
    fitted, report = fit_joints(
        candidate, steps, ws.rest_steps, cfg, objects=[obj.name]
    )
    entry = report[obj.name]
    check = entry["check"]
    info = urdf_info(urdf, obj.T_base_obj)
    joints = {}
    for joint in info["joints"]:
        fit = entry["joints"].get(joint["name"], {})
        joints[joint["name"]] = {
            **{k: v for k, v in joint.items() if k != "name"},
            **{k: fit.get(k) for k in ("observed_steps", "moved", "observed_range")},
        }
    summary: dict[str, Any] = {
        "urdf": str(urdf),
        "score_mm": check.get("score_mm"),
        "consistent": check["consistent"],
        "unexplained_steps": check.get("unexplained_steps", []),
        "residual_mm": check.get("residual_mm"),
        "joints": joints,
        # Links whose collision meshes do not match their visual ones: not sim-ready.
        "collision_mismatch": {
            link: v["collision_box_off_m"]
            for link, v in info["links"].items()
            if v["collision_box_off_m"] is None
            or v["collision_box_off_m"] > COLLISION_TOLERANCE
        },
        "fitted_urdf": fitted.objects[index].asset_path,
    }
    summary["evidence"] = _evidence(
        candidate, index, entry, steps, ws, out_dir, cfg, evidence_steps
    )
    path = out_dir / "report.json"
    path.write_text(json.dumps({**summary, "fit": entry}, indent=2), "utf-8")
    summary["report"] = str(path)
    return summary


def urdf_info(urdf: str | Path, T_base_obj: NDArray | None = None) -> dict[str, Any]:
    """Every link's bounding box and every joint's axis and origin, in the object frame
    (the URDF's root link) and, given the object's pose, the base frame."""
    T = np.eye(4) if T_base_obj is None else np.asarray(T_base_obj, float)
    root = ET.parse(urdf).getroot()
    poses = urdf_link_poses(root)
    collisions = urdf_link_collisions(urdf)
    links: dict[str, dict[str, Any]] = {}
    for link, meshes in urdf_link_visuals(urdf).items():
        pts = _to(poses[link], np.concatenate([m.mesh.vertices for m in meshes]))
        links[link] = {"box_obj": _box(pts), "box_base": _box(_to(T, pts))}
        # How far the collision meshes' box is from the visual one (None: no meshes).
        links[link]["collision_box_off_m"] = None
        if link in collisions:
            col = np.concatenate([m.vertices for m in collisions[link]])
            links[link]["collision_box_off_m"] = _box_offset(_to(poses[link], col), pts)
    movable = {j["name"]: j for j in urdf_movable_joints(urdf)}
    joints = []
    for el in root.iter("joint"):
        name = el.attrib.get("name", "")
        row: dict[str, Any] = {"name": name, "type": el.attrib.get("type")}
        for tag in ("parent", "child"):
            link_el = el.find(tag)
            row[tag] = None if link_el is None else link_el.attrib.get("link")
        if name in movable:
            j = movable[name]
            axis, origin = np.asarray(j["axis"]), np.asarray(j["origin"])
            row |= {
                "axis_obj": _round(axis),
                "axis_base": _round(T[:3, :3] @ axis),
                "origin_obj": _round(origin),
                "origin_base": _round(_to(T, origin[None])[0]),
                "limits": (
                    None
                    if j["lower"] is None
                    else _round(np.r_[j["lower"], j["upper"]], 4)
                ),
            }
        joints.append(row)
    return {"T_base_obj": _round(T), "links": links, "joints": joints}


def _evidence(
    scene: SceneSpec,
    index: int,
    entry: dict[str, Any],
    steps: dict[int, list[DepthView]],
    ws: Workspace,
    out_dir: Path,
    cfg: JointFitConfig,
    count: int,
) -> list[dict[str, Any]]:
    """What the worst steps (unexplained ones first) leave unexplained in each view,
    with an image per step (a row per view).

    What was already unexplained in the static period (the model's shape, not its
    joints) is left out of the boxes and drawn dimmed.
    """
    check = entry["check"]
    by_step = {int(t): r for t, r in check.get("residual_by_step_mm", {}).items()}
    worst = sorted(check.get("unexplained_steps") or by_step, key=lambda t: -by_step[t])
    held = {
        name: held_trajectory({int(t): q for t, q in e["trajectory"].items()})
        for name, e in entry["joints"].items()
    }
    obj = scene.objects[index]
    near = neighbourhood(obj)
    has_capture = (Path(ws.capture_dir) / CAPTURE_FILENAME).exists()
    capture = Capture.load(ws.capture_dir) if has_capture else None
    renderer = SceneRenderer(scene, cfg.max_size)
    voxel = cfg.explained

    def unexplained(t: int) -> list[tuple[DepthView, NDArray, NDArray, NDArray]]:
        renderer.pose(index, obj.T_base_obj, {n: h[t] for n, h in held.items()})
        out = []
        for v in steps[t]:
            model = renderer.render(v, index)["depth"]
            region = (v.depth > 0) & (near(v, v.depth) | near(v, model))
            seen = region & (v.depth < model - cfg.explained)
            unseen = region & (model < v.depth - cfg.explained)
            out.append((v, model, seen, unseen))
        return out

    out = []
    try:
        for k, other in enumerate(scene.objects):
            renderer.pose(k, other.T_base_obj)
        static: dict[str, list[NDArray]] = {"seen": [], "unseen": []}
        rest = [t for t in ws.rest_steps if t in steps]
        for t in rest[:: max(1, len(rest) // 8)]:
            for v, model, seen, unseen in unexplained(t):
                static["seen"].append(
                    _voxels(_obj_points(v, v.depth, seen, obj), voxel)
                )
                static["unseen"].append(
                    _voxels(_obj_points(v, model, unseen, obj), voxel)
                )
        known = {k: _grown(np.concatenate(c) if c else []) for k, c in static.items()}
        for t in worst[:count]:
            rows, views = [], {}
            for v, model, seen, unseen in unexplained(t):
                new_seen = _drop_known(v, v.depth, seen, obj, known["seen"], voxel)
                new_unseen = _drop_known(v, model, unseen, obj, known["unseen"], voxel)
                rows.append(
                    _row(capture, v, model, (seen, new_seen), (unseen, new_unseen))
                )
                views[v.camera] = {
                    "seen_not_in_model": _points(v, v.depth, new_seen, seen, obj),
                    "in_model_not_seen": _points(v, model, new_unseen, unseen, obj),
                }
            image = out_dir / f"step_{t}.png"
            if rows:
                cv2.imwrite(str(image), np.vstack(rows))
            out.append(
                {
                    "step": t,
                    "residual_mm": by_step[t],
                    "joint_positions": {n: round(h[t], 4) for n, h in held.items()},
                    "views": views,
                    "image": str(image),
                }
            )
    finally:
        renderer.close()
    return out


def _obj_points(
    view: DepthView, depth: NDArray, mask: NDArray, obj: ObjectSpec
) -> NDArray:
    """Object-frame points of the masked pixels (in row-major pixel order)."""
    T = np.linalg.inv(obj.T_base_obj) @ view.T_base_cam
    return _to(T, backproject(np.where(mask, depth, 0.0), view.K))


def _voxels(pts: NDArray, size: float) -> NDArray[np.int64]:
    """Codes of the voxels (edge ``size``) the points fall in."""
    ijk = np.floor(np.asarray(pts).reshape(-1, 3) / size).astype(np.int64) + 1024
    return (ijk[:, 0] << 22) | (ijk[:, 1] << 11) | ijk[:, 2]


def _grown(codes: Any) -> NDArray[np.int64]:
    """The voxels and their 26 neighbours."""
    codes = np.unique(np.asarray(codes, np.int64))
    ijk = np.stack([codes >> 22, (codes >> 11) & 2047, codes & 2047], axis=1)
    steps = np.array(
        [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    )
    ijk = (ijk[:, None, :] + steps[None]).reshape(-1, 3)
    return np.unique((ijk[:, 0] << 22) | (ijk[:, 1] << 11) | ijk[:, 2])


def _drop_known(
    view: DepthView,
    depth: NDArray,
    mask: NDArray,
    obj: ObjectSpec,
    known: NDArray[np.int64],
    voxel: float,
) -> NDArray:
    """``mask`` without the pixels whose points lie in ``known`` voxels."""
    if not mask.any() or known.size == 0:
        return mask
    fresh = ~np.isin(_voxels(_obj_points(view, depth, mask, obj), voxel), known)
    out = np.zeros_like(mask)
    out[np.nonzero(mask)] = fresh
    return out


def _points(
    view: DepthView,
    depth: NDArray,
    mask: NDArray,
    before: NDArray,
    obj: ObjectSpec,
) -> dict[str, Any]:
    """How many pixels, how many more were so in the static period too, and the box
    (2nd-98th percentile) of the points of the new ones."""
    out: dict[str, Any] = {
        "pixels": int(mask.sum()),
        "pixels_as_in_static_period": int(before.sum() - mask.sum()),
    }
    if mask.any():
        pts = _obj_points(view, depth, mask, obj)
        out["box_obj"] = _box(pts, percentile=2.0)
        out["box_base"] = _box(_to(obj.T_base_obj, pts), percentile=2.0)
    return out


def _row(
    capture: Capture | None,
    view: DepthView,
    model: NDArray,
    seen: tuple[NDArray, NDArray],
    unseen: tuple[NDArray, NDArray],
    width: int = 480,
) -> NDArray[np.uint8]:
    """Photo, measured depth, model depth and what is unexplained, side by side;
    ``seen`` and ``unseen`` are (all, new) masks."""
    h, w = view.depth.shape
    frames = [] if capture is None else capture.frames
    frame = next(
        (f for f in frames if f.step == view.step and f.camera == view.camera), None
    )
    bgr = None
    if capture is not None and frame is not None:
        bgr = cv2.imread(str(capture.root / frame.left_image))
    photo = np.zeros((h, w, 3), np.uint8) if bgr is None else cv2.resize(bgr, (w, h))
    shown = seen[0] | unseen[0]
    both = np.concatenate(
        [view.depth[shown & (view.depth > 0)], model[shown & (model < _SKY)]]
    )
    lo, hi = np.percentile(both, [2, 98]) if both.size else (0.0, 2.0)
    lo, hi = max(lo - 0.1, 0.0), hi + 0.1
    overlay = (photo * 0.4).astype(np.uint8)
    overlay[seen[0]] = (0, 0, 110)  # BGR dark red: as in the static period
    overlay[unseen[0]] = (110, 30, 0)  # dark blue
    overlay[seen[1]] = (0, 0, 255)  # red
    overlay[unseen[1]] = (255, 64, 0)  # blue
    size = (width, int(round(h * width / w)))
    tiles = [
        (photo, f"{view.camera}, step {view.step}"),
        (_colour(view.depth, lo, hi), f"measured depth, {lo:.2f}-{hi:.2f} m"),
        (_colour(model, lo, hi), "model, joints as fitted"),
        (overlay, "red: seen, not in model; blue: model, not seen"),
    ]
    return np.hstack(
        [
            _label(cv2.resize(image, size, interpolation=cv2.INTER_NEAREST), text)
            for image, text in tiles
        ]
    )


def _colour(depth: NDArray, lo: float, hi: float) -> NDArray[np.uint8]:
    x = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = np.asarray(
        cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO), np.uint8
    )
    out[(depth <= 0) | (depth >= _SKY)] = 0
    return out


def _label(image: NDArray, text: str) -> NDArray[np.uint8]:
    out = np.ascontiguousarray(image, dtype=np.uint8).copy()
    cv2.rectangle(out, (0, 0), (8 + 7 * len(text), 20), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return out


def _to(T: NDArray, pts: NDArray) -> NDArray:
    return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]


def _box_offset(a: NDArray, b: NDArray) -> float:
    """The largest distance between the faces of two point sets' boxes."""
    offset = np.r_[a.min(axis=0) - b.min(axis=0), a.max(axis=0) - b.max(axis=0)]
    return round(float(np.abs(offset).max()), 4)


def _box(pts: NDArray, percentile: float = 0.0) -> list[list[float]]:
    lo = np.percentile(pts, percentile, axis=0)
    hi = np.percentile(pts, 100.0 - percentile, axis=0)
    return [_round(lo), _round(hi)]


def _round(x: NDArray, digits: int = 3) -> Any:
    return np.round(np.asarray(x, float), digits).tolist()
