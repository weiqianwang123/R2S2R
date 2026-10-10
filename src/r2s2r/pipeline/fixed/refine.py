"""Multi-view refinement of a reconstructed scene from calibrated metric depth.

A single-frame reconstruction places objects from one view, so small or thin objects
can end up centimetres off, and it reconstructs only the support plane, not its
outline. This module fuses metric depth from several calibrated views (the candidate
frames' depth, from every camera of the run) in the robot base frame and

1. refits the support plane to the points near it, inside its outline and outside
   the objects' footprints, and levels every view onto it: turns and lifts the
   view's points so that their own fit of the plane lies on it. SimFoundry's plane
   can be a degree or two off the depth, and so can a view's calibration (DROID's
   wrist views: up to 3 degrees and 1.5 cm); either lifts part of the table above
   the objects' height threshold, where it merges with the objects into one cluster;
2. fits the support surface's outline: the connected region of on-plane points
   around the objects, summarised as a rectangle (centre, size, yaw);
3. re-registers every object mesh to the fused points above the support by
   matching top-down footprints (yaw about the support normal plus in-plane
   shift), then puts objects that rested on the support back in contact with it.

Points above the support are clustered, and each cluster is matched to at most one
object: the one whose surface and the cluster's points lie closest on average, so a
flat object does not take a tall one's points.

Mesh shape, scale and tilt are kept; only where the objects stand changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from scipy.spatial.transform import Rotation

from r2s2r.assets import object_points, urdf_visual_points
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec
from r2s2r.tools.geometry import (
    MAX_DEPTH,
    cluster,
    fit_plane,
    fit_support_outline,
    planar,
    register_footprint,
    view_points,
)
from r2s2r.transforms import invert, make_transform, tilt_deg, transform_points


@dataclass
class RefineConfig:
    """Tuning knobs (metres unless noted)."""

    max_depth: float = MAX_DEPTH
    plane_band: float = 0.03  # points this close to a plane, in its outline, refit it
    # Points per view for the plane fits; a view with fewer there has no say in the
    # support's plane and is not levelled onto it.
    plane_points: int = 1000
    footprint_margin: float = 0.03  # objects' footprints grown by this: not in the fits
    support_band: float = 0.008  # |height| counted as on the support
    support_cell: float = 0.01  # occupancy grid for the outline
    object_min_height: float = 0.004  # above the support
    object_max_height: float = 0.4
    outline_margin: float = 0.05  # object points may overhang the outline
    cluster_voxel: float = 0.01
    min_cluster_points: int = 30
    # An object matches a cluster whose points and its surface lie this close on
    # average (each way); the matching is one-to-one.
    max_match_cost: float = 0.05
    match_points: int = 2000  # of the model and of each cluster, for matching
    model_points: int = 4000
    min_iou_gain: float = 0.02  # keep SimFoundry's pose unless the fit improves
    contact_tolerance: float = 0.03  # objects this close to the support rest on it


@dataclass
class Observation:
    """What the fused depth shows: the support's plane and outline, and the clusters
    of points above it, matched one-to-one to the scene's objects."""

    T_base_support: NDArray[np.float64]  # on the refit plane, at the outline's centre
    size: tuple[float, float]  # the outline's rectangle
    # Each view's own fit of the plane, as a frame in the support frame (the view's
    # points were moved by its inverse).
    view_planes: list[NDArray[np.float64]]
    above: NDArray[np.float64]  # points above the support, in the frame above
    groups: list  # clusters: index arrays into ``above``
    matches: dict[int, int]  # object index -> cluster index


def observe(
    scene: SceneSpec, views: list[DepthView], cfg: RefineConfig | None = None
) -> Observation:
    """Refit the support plane to ``views`` (each view levelled onto it) and its
    outline, cluster the points above it, and match the clusters to the objects
    one-to-one, by how close each cluster's points and the object's surface lie."""
    cfg = cfg or RefineConfig()
    view_pts = [view_points(v, max_depth=cfg.max_depth)[0] for v in views]
    pts = np.concatenate(view_pts)
    obj_pts = [object_points(o, cfg.match_points) for o in scene.objects]
    middle = (  # of the objects: the support is the plane region around it
        np.mean([o.T_base_obj[:3, 3] for o in scene.objects], axis=0)
        if scene.objects
        else scene.T_base_support[:3, 3]
    )

    # 1. SimFoundry's plane refit to the views, then each view levelled onto it.
    T_base_sf, size = _support_frame(pts, scene.T_base_support, middle, cfg)
    T_sf_plane = _refit_plane(view_pts, T_base_sf, size, obj_pts, cfg)
    T_base_plane, size = _support_frame(pts, T_base_sf @ T_sf_plane, middle, cfg)
    view_planes = [
        _refit_plane([p], T_base_plane, size, obj_pts, cfg) for p in view_pts
    ]
    pts = np.concatenate(
        [
            transform_points(T_base_plane @ invert(T) @ invert(T_base_plane), p)
            for T, p in zip(view_planes, view_pts)
        ]
    )
    # 2. The outline of the levelled points: the support frame at its centre.
    T_base_support, size = _support_frame(pts, T_base_plane, middle, cfg)
    T_support_plane = invert(T_base_support) @ T_base_plane
    view_planes = [T_support_plane @ T @ invert(T_support_plane) for T in view_planes]
    pts_s = transform_points(invert(T_base_support), pts)
    half = np.array(size) / 2 + cfg.outline_margin

    # 3. Object points: above the support, over its outline.
    above = pts_s[
        (pts_s[:, 2] > cfg.object_min_height)
        & (pts_s[:, 2] < cfg.object_max_height)
        & np.all(np.abs(pts_s[:, :2]) < half, axis=1)
    ]
    groups = cluster(above, cfg.cluster_voxel, cfg.min_cluster_points)
    rng = np.random.default_rng(0)
    clusters = [
        cKDTree(above[rng.choice(g, min(cfg.match_points, len(g)), replace=False)])
        for g in groups
    ]
    cost = np.full((len(scene.objects), max(len(groups), 1)), 1e6)
    for i, model in enumerate(obj_pts):
        surface = cKDTree(transform_points(invert(T_base_support), model))
        for j, tree in enumerate(clusters):
            cost[i, j] = 0.5 * (
                surface.query(tree.data)[0].mean() + tree.query(surface.data)[0].mean()
            )
    matches = match_clusters(cost, cfg.max_match_cost) if groups else {}
    return Observation(T_base_support, size, view_planes, above, groups, matches)


def _support_frame(
    pts: NDArray[np.float64],
    T_base_plane: NDArray[np.float64],
    middle: NDArray[np.float64],
    cfg: RefineConfig,
) -> tuple[NDArray[np.float64], tuple[float, float]]:
    """The support's outline on a plane (a frame with z along its normal): the frame
    moved to the centre of the outline's rectangle and turned with it, and the
    rectangle's size. The outline is the on-plane region of ``pts`` nearest
    ``middle``."""
    to_plane = invert(T_base_plane)
    centre, size, yaw = fit_support_outline(
        transform_points(to_plane, pts),
        transform_points(to_plane, middle[None])[0, :2],
        cfg.support_band,
        cfg.support_cell,
    )
    return T_base_plane @ planar(yaw, *centre), size


def _refit_plane(
    view_pts: list[NDArray[np.float64]],
    T_base_support: NDArray[np.float64],
    size: tuple[float, float],
    obj_pts: list[NDArray[np.float64]],
    cfg: RefineConfig,
) -> NDArray[np.float64]:
    """The plane fit (RANSAC) to the views' points within ``plane_band`` of the
    support, inside its outline (``size``, centred on the frame) and outside the
    objects' footprints (their points ``obj_pts``' boxes on the support, grown by
    ``footprint_margin``): ``plane_points`` of them from every view that has as many,
    so each view has the same say however close it is. As a frame in the support
    frame: z along its normal, turned the least it can, its origin where the support's
    normal through the support's origin meets it. The identity when no view has
    enough points there."""
    rng = np.random.default_rng(0)
    to_support = invert(T_base_support)
    footprints = [transform_points(to_support, p)[:, :2] for p in obj_pts]
    chosen = []
    for pts in view_pts:
        pts_s = transform_points(to_support, pts)
        keep = (np.abs(pts_s[:, 2]) < cfg.plane_band) & np.all(
            np.abs(pts_s[:, :2]) < np.array(size) / 2, axis=1
        )
        for xy in footprints:
            lo, hi = xy.min(0) - cfg.footprint_margin, xy.max(0) + cfg.footprint_margin
            keep &= ~np.all((pts_s[:, :2] > lo) & (pts_s[:, :2] < hi), axis=1)
        near = pts_s[keep]
        if len(near) >= cfg.plane_points:
            chosen.append(near[rng.choice(len(near), cfg.plane_points, replace=False)])
    if not chosen:
        return np.eye(4)
    n, d, _ = fit_plane(np.concatenate(chosen))
    if n[2] < 0:  # the normal stays on the objects' side
        n, d = -n, -d
    axis = np.cross([0.0, 0.0, 1.0], n)
    sine = float(np.linalg.norm(axis))  # of the tilt
    turn = Rotation.from_rotvec(axis * np.arctan2(sine, n[2]) / max(sine, 1e-12))
    return make_transform(turn.as_matrix(), [0.0, 0.0, -d / n[2]])


def match_clusters(cost: NDArray[np.float64], max_cost: float) -> dict[int, int]:
    """One-to-one object -> cluster matches of least total cost, among the pairs under
    ``max_cost``.

    Gating first matters: otherwise an object that fits no cluster can
    take the cluster of one that does, leaving both unmatched.
    """
    feasible = cost < max_cost
    rows, cols = linear_sum_assignment(np.where(feasible, cost, 1e6))
    return {int(r): int(c) for r, c in zip(rows, cols) if feasible[r, c]}


def refine_scene(
    scene: SceneSpec, views: list[DepthView], cfg: RefineConfig | None = None
) -> tuple[SceneSpec, dict[str, Any]]:
    """Refit the support plane, its outline and the object placements from
    ``views``."""
    cfg = cfg or RefineConfig()
    obs = observe(scene, views, cfg)
    T_old_new = invert(scene.T_base_support) @ obs.T_base_support
    report: dict[str, Any] = {
        "views": [  # each view's own plane against the support
            {"camera": v.camera, "step": v.step, **_tilt_lift(T)}
            for v, T in zip(views, obs.view_planes)
        ],
        "support": {  # against SimFoundry's
            "size": list(obs.size),
            "yaw_deg": float(np.rad2deg(np.arctan2(T_old_new[1, 0], T_old_new[0, 0]))),
            **_tilt_lift(T_old_new),
        },
        "clusters": [int(len(g)) for g in obs.groups],
        "objects": {},
    }
    T_support_objs = [invert(obs.T_base_support) @ o.T_base_obj for o in scene.objects]

    new_objects: list[ObjectSpec] = []
    for i, (obj, T_so) in enumerate(zip(scene.objects, T_support_objs)):
        entry: dict[str, Any] = {"matched": i in obs.matches}
        report["objects"][obj.name] = entry
        if i not in obs.matches:
            new_objects.append(obj)
            continue
        observed = obs.above[obs.groups[obs.matches[i]]]
        model_obj = urdf_visual_points(
            obj.asset_path, cfg.model_points, joints=obj.joints
        )
        model = transform_points(T_so, model_obj)
        T_fix, before, after = register_footprint(model, observed)
        if after < before + cfg.min_iou_gain:
            T_fix = np.eye(4)  # already as good as the depth can tell
        T_new = T_fix @ T_so
        moved = transform_points(T_new, model_obj)
        if abs(model[:, 2].min()) < cfg.contact_tolerance:
            T_new[2, 3] -= moved[:, 2].min()  # rest on the support
        entry.update(
            observed_points=int(len(observed)),
            footprint_iou_before=before,
            footprint_iou_after=after,
            applied=bool(not np.allclose(T_fix, np.eye(4))),
            shift_m=float(np.linalg.norm(T_new[:3, 3] - T_so[:3, 3])),
            yaw_change_deg=float(np.rad2deg(np.arctan2(T_fix[1, 0], T_fix[0, 0]))),
        )
        new_objects.append(replace(obj, T_base_obj=obs.T_base_support @ T_new))

    refined = replace(
        scene,
        objects=new_objects,
        T_base_support=obs.T_base_support,
        support_extent=obs.size,
        provenance={**scene.provenance, "refinement": report},
    )
    return refined, report


def _tilt_lift(T_plane: NDArray[np.float64]) -> dict[str, float]:
    """How a plane (a frame, z along its normal) lies against the frame it is given in:
    its tilt, and its height above the origin."""
    return {
        "tilt_deg": tilt_deg(T_plane[:3, 2]),
        "lift_m": float(T_plane[2, 3]),
    }
