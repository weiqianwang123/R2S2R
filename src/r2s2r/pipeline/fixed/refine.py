"""Multi-view refinement of a reconstructed scene from calibrated metric depth.

A single-frame reconstruction places objects from one view, so small or thin objects
can end up centimetres off, and it reconstructs only the support plane, not its
outline. This module fuses metric depth from several calibrated views (e.g. both
DROID exterior cameras at the reference step) in the robot base frame and

1. fits the support surface's outline: the connected region of on-plane points
   around the objects, summarised as a rectangle (centre, size, yaw);
2. re-registers every object mesh to the fused points above the support by
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

from r2s2r.assets import urdf_visual_points
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec
from r2s2r.tools.geometry import (
    cluster,
    fit_support_outline,
    planar,
    register_footprint,
    view_points,
)
from r2s2r.transforms import invert, transform_points


@dataclass
class RefineConfig:
    """Tuning knobs (metres unless noted)."""

    max_depth: float = 2.5
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
    """What the fused depth shows: the support's outline, and the clusters of points
    above it, matched one-to-one to the scene's objects."""

    T_base_support: NDArray[np.float64]  # at the outline's centre, turned with it
    size: tuple[float, float]  # the outline's rectangle
    yaw: float  # its turn from SimFoundry's support frame
    above: NDArray[np.float64]  # points above the support, in the frame above
    groups: list  # clusters: index arrays into ``above``
    matches: dict[int, int]  # object index -> cluster index


def observe(
    scene: SceneSpec, views: list[DepthView], cfg: RefineConfig | None = None
) -> Observation:
    """Fit the support's outline to ``views``, cluster the points above it, and match
    the clusters to the objects one-to-one, by how close each cluster's points and the
    object's surface lie."""
    cfg = cfg or RefineConfig()
    pts = np.concatenate([view_points(v, max_depth=cfg.max_depth)[0] for v in views])
    T_support_base = invert(scene.T_base_support)
    pts_s = transform_points(T_support_base, pts)
    obj_xy = (
        transform_points(
            T_support_base, np.array([o.T_base_obj[:3, 3] for o in scene.objects])
        )[:, :2]
        if scene.objects
        else np.zeros((1, 2))
    )

    # 1. Support outline -> new support frame at the rectangle centre.
    centre, size, yaw = fit_support_outline(
        pts_s, obj_xy.mean(0), cfg.support_band, cfg.support_cell
    )
    T_base_support = scene.T_base_support @ planar(yaw, *centre)
    pts_s = transform_points(invert(T_base_support), pts)
    half = np.array(size) / 2 + cfg.outline_margin

    # 2. Object points: above the support, over its outline.
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
    for i, obj in enumerate(scene.objects):
        T = invert(T_base_support) @ obj.T_base_obj
        model = urdf_visual_points(obj.asset_path, cfg.match_points)
        surface = cKDTree(transform_points(T, model))
        for j, tree in enumerate(clusters):
            cost[i, j] = 0.5 * (
                surface.query(tree.data)[0].mean() + tree.query(surface.data)[0].mean()
            )
    matches = match_clusters(cost, cfg.max_match_cost) if groups else {}
    return Observation(T_base_support, size, yaw, above, groups, matches)


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
    """Refit the support outline and object placements from ``views``."""
    cfg = cfg or RefineConfig()
    obs = observe(scene, views, cfg)
    report: dict[str, Any] = {
        "views": [{"camera": v.camera, "step": v.step} for v in views],
        "support": {"size": list(obs.size), "yaw_deg": float(np.rad2deg(obs.yaw))},
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
        model_obj = urdf_visual_points(obj.asset_path, cfg.model_points)
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
