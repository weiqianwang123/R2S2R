"""Fit articulated objects' joints to the motion in the video, and check the fit.

A static view shows a joint's type and axis at best; how far a drawer slides or a
lid swings shows only when it moves. For every sampled step of the video (the robot
removed from the depth), each movable joint is set to the position whose render best
explains that step's views, the other objects and joints held still:

- only the pixels the joint changes count (where the renders over the searched
  positions differ and depth was measured), so the rest of the object's modelling
  error does not drown the signal;
- only views where the joint changes a fair share of the image, and that some
  position explains, count: in a sliver at the image's edge, or where the model is off
  from a view's angle (up close, a few percent of size are many pixels), model error
  passes for motion;
- a step whose views do not pin the position down is left unobserved rather than
  guessed: no view counts (the part out of view, occluded, too close to the camera),
  or the best position is hardly better than positions far from it;
- only positions reachable from zero without the moving part passing into its parent
  are searched: otherwise a part could "explain" the views by hiding inside its
  parent, or by jumping through it.

This gives a trajectory q(t) per joint and the range it covers. A check then scores
the object's whole neighbourhood (whatever is within its own size of it, measured or
rendered) at every step, with the joints following the fitted trajectory. A joint
model that cannot produce the motion seen (the wrong axis, type or part) leaves steps
explained far worse than the static period; the backend's limits are only widened
when no step is.

Nothing here depends on the kind of object or joint.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Collection

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

from r2s2r.assets import (
    urdf_link_poses,
    urdf_link_visuals,
    urdf_movable_joints,
    urdf_visual_meshes,
)
from r2s2r.reconstruct.render import SceneRenderer
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec
from r2s2r.transforms import backproject, invert

FITTED_SUFFIX = "_fit"


@dataclass
class JointFitConfig:
    """Knobs of :func:`fit_joints` (metres and radians)."""

    residual_clip: float = 0.05  # bounds each pixel's depth residual (truncated mean)
    changed: float = 0.01  # a pixel the joint changes: renders differ by this much
    # A view counts at a step only if the joint changes at least this share of its
    # pixels with depth, and some position explains it (a mean residual below
    # `explained`). In a sliver at the image's edge, or where the model is off from a
    # view's angle (up close, a few percent of size are many pixels), model error
    # passes for motion.
    min_view_share: float = 0.01
    explained: float = 0.025
    # The best position must beat every position min_motion away by at least this
    # many pixels going from clipped to exact; otherwise the step is unobserved.
    min_evidence: int = 200
    # Per joint type (see _MIN_MOTION): positions closer than this are not told
    # apart, and a joint farther than this from zero has moved.
    min_motion: dict[str, float] | None = None
    # A step is unexplained when the object's neighbourhood fits this many times
    # worse than in the static period, taken to fit no better than min_residual (no
    # depth sensor is exact).
    unexplained_ratio: float = 2.0
    min_residual: float = 0.001
    # A position is infeasible when this much more of the moving part lies inside
    # its parent's convex hull than at zero.
    max_penetration: float = 0.1
    coarse: int = 25  # grid over the search range
    fine: int = 11  # grid around the best coarse position
    view_scale: float = 0.5  # render and compare at this share of the resolution
    max_size: tuple[int, int] = (2560, 1600)


_MIN_MOTION = {
    "prismatic": 0.01,
    "revolute": np.deg2rad(5.0),
    "continuous": np.deg2rad(5.0),
}


def fit_joints(
    scene: SceneSpec,
    steps: dict[int, list[DepthView]],
    rest_steps: Collection[int] = (),
    config: JointFitConfig | None = None,
    out_suffix: str = FITTED_SUFFIX,
    objects: Collection[str] | None = None,
) -> tuple[SceneSpec, dict[str, Any]]:
    """Fit the joints of every articulated object (or only those named in ``objects``)
    to ``steps`` (step -> views).

    ``rest_steps`` are known to be still (the static period): the check's baseline, else
    the steps where no joint moved. An object whose joints moved and whose check passes
    gets a URDF (``<name><out_suffix>.urdf``) with limits widened to the observed range.
    Returns the scene and, per object, ``joints`` (per joint) and ``check``.
    """
    cfg = config or JointFitConfig()
    min_motion = {**_MIN_MOTION, **(cfg.min_motion or {})}
    views = {
        t: [scale_view(v, cfg.view_scale) for v in vs]
        for t, vs in sorted(steps.items())
    }
    renderer = SceneRenderer(scene, cfg.max_size)
    fitted = list(scene.objects)
    report: dict[str, Any] = {}
    try:
        for k, other in enumerate(scene.objects):
            renderer.pose(k, other.T_base_obj)
        for i, obj in enumerate(scene.objects):
            if not obj.articulated or (objects is not None and obj.name not in objects):
                continue
            joints = urdf_movable_joints(obj.asset_path)
            motion = {
                j["name"]: float(min_motion.get(j["type"], min_motion["prismatic"]))
                for j in joints
            }
            trajectories = {}
            for joint in joints:
                search = _reachable(
                    obj,
                    joint,
                    _search_range(obj, joint),
                    cfg.max_penetration,
                    _sweep_step(joint),
                )
                trajectories[joint["name"]] = _fit_joint(
                    renderer, i, obj, joint, search, views, motion[joint["name"]], cfg
                )
                renderer.pose(i, obj.T_base_obj)  # back to rest for the next joint
            entries = {
                j["name"]: _summarize(j, trajectories[j["name"]], motion[j["name"]])
                for j in joints
            }
            check = _check(
                renderer, i, obj, trajectories, motion, views, rest_steps, cfg
            )
            renderer.pose(i, obj.T_base_obj)
            report[obj.name] = {"joints": entries, "check": check}
            limits = {n: e["limits"] for n, e in entries.items() if e["moved"]}
            if limits and check["consistent"]:
                path = _write_limits(Path(obj.asset_path), limits, out_suffix)
                fitted[i] = replace(obj, asset_path=str(path))
    finally:
        renderer.close()
    return replace(scene, objects=fitted), report


def _search_range(obj: ObjectSpec, joint: dict[str, Any]) -> tuple[float, float]:
    """Positions to search: up to the object's own size along a prismatic axis either
    way, a full turn for rotations, always covering the current limits."""
    if joint["type"] == "prismatic":
        root = ET.parse(obj.asset_path).getroot()
        poses = urdf_link_poses(root)
        axis = np.asarray(joint["axis"])
        pts = np.concatenate(
            [
                v.mesh.vertices @ poses[link][:3, :3].T + poses[link][:3, 3]
                for link, meshes in urdf_link_visuals(obj.asset_path).items()
                for v in meshes
            ]
        )
        extent = float(np.ptp(pts @ axis))
        lo, hi = -extent, extent
    else:
        lo, hi = -np.pi, np.pi
    if joint["lower"] is not None:
        lo, hi = min(lo, joint["lower"]), max(hi, joint["upper"])
    return lo, hi


def _reachable(
    obj: ObjectSpec,
    joint: dict[str, Any],
    search: tuple[float, float],
    max_penetration: float,
    step: float,
) -> tuple[float, float]:
    """The interval around zero the joint sweeps through without the moving part
    entering its parent's convex hull (``max_penetration`` more of it than at zero)."""
    root = ET.parse(obj.asset_path).getroot()
    visuals = urdf_link_visuals(obj.asset_path)
    if joint["parent"] not in visuals or joint["child"] not in visuals:
        return search
    rest = urdf_link_poses(root)
    parent = np.concatenate([v.mesh.vertices for v in visuals[joint["parent"]]])
    parent = parent @ rest[joint["parent"]][:3, :3].T + rest[joint["parent"]][:3, 3]
    hull = Delaunay(parent)
    child = np.concatenate(
        [
            trimesh.sample.sample_surface(v.mesh, 400, seed=0)[0]
            for v in visuals[joint["child"]]
        ]
    )

    def inside(q: float) -> float:
        T = urdf_link_poses(root, {joint["name"]: q})[joint["child"]]
        return float(np.mean(hull.find_simplex(child @ T[:3, :3].T + T[:3, 3]) >= 0))

    limit = inside(0.0) + max_penetration
    bounds = []
    for end, sign in ((search[0], -1.0), (search[1], 1.0)):
        q = 0.0
        while (
            sign * (q + sign * step) <= sign * end and inside(q + sign * step) <= limit
        ):
            q += sign * step
        bounds.append(q)
    return bounds[0], bounds[1]


def _sweep_step(joint: dict[str, Any]) -> float:
    return 0.005 if joint["type"] == "prismatic" else np.deg2rad(2.0)


def _fit_joint(
    renderer: SceneRenderer,
    index: int,
    obj: ObjectSpec,
    joint: dict[str, Any],
    search: tuple[float, float],
    steps: dict[int, list[DepthView]],
    motion: float,
    cfg: JointFitConfig,
) -> dict[int, float | None]:
    """The joint's position at every step (None: unobserved)."""
    name = joint["name"]
    coarse = [float(q) for q in np.unique(np.r_[np.linspace(*search, cfg.coarse), 0.0])]
    spacing = (search[1] - search[0]) / (cfg.coarse - 1)
    out: dict[int, float | None] = {}
    for step, views in steps.items():

        def render(q: float, views: list[DepthView] = views) -> list[NDArray]:
            renderer.pose(index, obj.T_base_obj, {name: q})
            return [renderer.render(v, index)["depth"] for v in views]

        out[step] = _fit_step(render, views, coarse, spacing, search, motion, cfg)
    return out


def _fit_step(
    render: Callable[[float], list[NDArray]],
    views: list[DepthView],
    coarse: list[float],
    spacing: float,
    search: tuple[float, float],
    motion: float,
    cfg: JointFitConfig,
) -> float | None:
    """The position whose renders best explain one step's views, searched over the
    ``coarse`` grid and then ``spacing`` around its best, if the views pin it down to
    within ``motion``."""
    renders = {q: render(q) for q in coarse}
    changed = [
        (np.ptp(np.stack([renders[q][n] for q in coarse]), axis=0) > cfg.changed)
        & (v.depth > 0)
        for n, v in enumerate(views)
    ]

    def errors(depths: list[NDArray]) -> list[NDArray]:
        return [
            np.minimum(np.abs(v.depth[c] - d[c]), cfg.residual_clip)
            for v, c, d in zip(views, changed, depths)
        ]

    errs = {q: errors(d) for q, d in renders.items()}
    kept = [
        n
        for n, c in enumerate(changed)
        if c.sum() >= cfg.min_view_share * c.size
        and min(float(np.mean(e[n])) for e in errs.values()) < cfg.explained
    ]
    if not kept:
        return None
    pixels = int(sum(changed[n].sum() for n in kept))

    def residual(e: list[NDArray]) -> float:
        # A truncated mean: a median over the changed pixels would ignore the
        # minority any one position gets wrong, which is exactly what decides it.
        return float(np.mean(np.concatenate([e[n] for n in kept])))

    scores = {q: residual(e) for q, e in errs.items()}
    best = min(scores, key=scores.__getitem__)
    for q in np.linspace(best - spacing, best + spacing, cfg.fine):
        if search[0] <= q <= search[1] and float(q) not in scores:
            scores[float(q)] = residual(errors(render(float(q))))
    best = min(scores, key=scores.__getitem__)
    rivals = [r for q, r in scores.items() if abs(q - best) >= motion]
    evidence = pixels * (min(rivals) - scores[best]) if rivals else np.inf
    return best if evidence >= cfg.min_evidence * cfg.residual_clip else None


def _summarize(
    joint: dict[str, Any], trajectory: dict[int, float | None], motion: float
) -> dict[str, Any]:
    observed = [q for q in trajectory.values() if q is not None]
    moved = [q for q in observed if abs(q) >= motion]
    entry: dict[str, Any] = {
        "type": joint["type"],
        "trajectory": {
            str(t): None if q is None else round(q, 4) for t, q in trajectory.items()
        },
        "observed_steps": len(observed),
        "moved": bool(moved),
    }
    if moved:
        lower = joint["lower"] if joint["lower"] is not None else min(moved)
        upper = joint["upper"] if joint["upper"] is not None else max(moved)
        entry["observed_range"] = [round(min(moved), 4), round(max(moved), 4)]
        entry["limits"] = [round(min(lower, *moved), 4), round(max(upper, *moved), 4)]
    return entry


def _check(
    renderer: SceneRenderer,
    index: int,
    obj: ObjectSpec,
    trajectories: dict[str, dict[int, float | None]],
    motion: dict[str, float],
    steps: dict[int, list[DepthView]],
    rest_steps: Collection[int],
    cfg: JointFitConfig,
) -> dict[str, Any]:
    """How well the object's neighbourhood is explained at every step with the joints on
    their fitted trajectories (unobserved steps hold the last position), against the
    static period and against the joints frozen at zero."""
    held = {name: held_trajectory(t) for name, t in trajectories.items()}
    near = neighbourhood(obj)
    fitted: dict[int, float] = {}
    frozen: dict[int, float] = {}
    for t, views in steps.items():
        renderer.pose(index, obj.T_base_obj, {n: h[t] for n, h in held.items()})
        moving = [renderer.render(v, index)["depth"] for v in views]
        renderer.pose(index, obj.T_base_obj)
        still = [renderer.render(v, index)["depth"] for v in views]
        regions = [
            (v.depth > 0) & (near(v, v.depth) | near(v, a) | near(v, b))
            for v, a, b in zip(views, moving, still)
        ]
        if sum(r.sum() for r in regions) < cfg.min_view_share * regions[0].size:
            continue
        fitted[t], frozen[t] = (
            float(
                np.mean(
                    np.concatenate(
                        [
                            np.minimum(np.abs(v.depth[r] - d[r]), cfg.residual_clip)
                            for v, r, d in zip(views, regions, depths)
                        ]
                    )
                )
            )
            for depths in (moving, still)
        )
    if not fitted:
        return {"steps_seen": 0, "consistent": False}
    rest = [t for t in fitted if t in rest_steps] or [
        t for t in fitted if all(abs(h[t]) < motion[n] for n, h in held.items())
    ]
    baseline = max(
        float(np.median([fitted[t] for t in rest or fitted])), cfg.min_residual
    )
    later = [t for t in fitted if t not in rest]
    unexplained = [t for t, r in fitted.items() if r > cfg.unexplained_ratio * baseline]

    def mm(values: list[float]) -> float | None:
        return round(float(np.mean(values)) * 1000, 1) if values else None

    return {
        "steps_seen": len(fitted),
        # Mean over every step seen: ranks joint models of the same object.
        "score_mm": mm(list(fitted.values())),
        "residual_mm": {
            "static_period": round(baseline * 1000, 1),
            "rest_of_video": mm([fitted[t] for t in later]),
            "rest_of_video_frozen": mm([frozen[t] for t in later]),
        },
        "residual_by_step_mm": {str(t): round(r * 1000, 1) for t, r in fitted.items()},
        "unexplained_steps": unexplained,
        "consistent": not unexplained,
    }


def held_trajectory(trajectory: dict[int, float | None]) -> dict[int, float]:
    """The trajectory with every unobserved step at the last observed position (the
    first one before any; zero if none)."""
    observed = [q for q in trajectory.values() if q is not None]
    last = observed[0] if observed else 0.0
    out = {}
    for t in sorted(trajectory):
        q = trajectory[t]
        if q is not None:
            last = q
        out[t] = last
    return out


def neighbourhood(obj: ObjectSpec) -> Callable[[DepthView, NDArray], NDArray]:
    """``near(view, depth)``: which pixels' points (at ``depth``) lie within the
    object's own size of it (its bounding box at rest, grown by its largest extent)."""
    pts = np.concatenate([v.mesh.vertices for v in urdf_visual_meshes(obj.asset_path)])
    grow = float(np.max(np.ptp(pts, axis=0)))
    lo, hi = pts.min(axis=0) - grow, pts.max(axis=0) + grow
    T_obj_base = invert(obj.T_base_obj)

    def near(view: DepthView, depth: NDArray) -> NDArray:
        valid = depth > 0
        T = T_obj_base @ view.T_base_cam
        p = backproject(np.where(valid, depth, 0.0), view.K) @ T[:3, :3].T + T[:3, 3]
        out = np.zeros(depth.shape, bool)
        out[np.nonzero(valid)] = np.all((p >= lo) & (p <= hi), axis=1)
        return out

    return near


def _write_limits(urdf: Path, limits: dict[str, list[float]], suffix: str) -> Path:
    tree = ET.parse(urdf)
    for joint in tree.getroot().iter("joint"):
        name = joint.attrib.get("name", "")
        if name not in limits:
            continue
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit", {"effort": "5", "velocity": "5"})
        limit.attrib["lower"], limit.attrib["upper"] = (
            repr(float(v)) for v in limits[name]
        )
    out = urdf.with_name(f"{urdf.stem}{suffix}.urdf")
    tree.write(out, xml_declaration=True, encoding="utf-8")
    return out


def scale_view(view: DepthView, scale: float) -> DepthView:
    """The view's depth at ``scale`` of its resolution (the fit needs no colour)."""
    if scale == 1.0:
        return replace(view, image=None)
    h, w = view.depth.shape
    size = (int(round(w * scale)), int(round(h * scale)))
    K = view.K.copy()
    K[:2, :2] *= scale
    K[:2, 2] = (K[:2, 2] + 0.5) * scale - 0.5
    depth = cv2.resize(view.depth, size, interpolation=cv2.INTER_NEAREST)
    return replace(view, depth=np.asarray(depth, np.float64), K=K, image=None)
