"""Drop objects the measured depth says are not there.

A backend can reconstruct objects that do not exist. SimFoundry finds objects one at a
time in an image it edits: each one found is erased and the gap inpainted, and a smudge
left where an object was erased can be found as another object. Such a ghost has no
points of its own in the measured depth, and the cameras see through it: where its
surface should be, they measure what lies behind it.

An object is dropped when both hold:

1. no cluster of measured points above the support is matched to it (the one-to-one
   matching refinement uses, :func:`r2s2r.reconstruct.refine.observe`);
2. rendered with the whole scene into every view, most of its visible surface lies
   more than ``see_through`` in front of the measured depth.

An object seen by no view, or too flat to rise above the depth noise, is kept: there is
no evidence either way. So is one standing over measured points that no object explains:
that is a real object whose reconstructed shape disagrees with the depth (a mug generated
twice as tall from an overhead view), and keeping it wrong beats losing it. Objects that
rested on a dropped one settle on what is beneath them, the support if nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r2s2r.assets import urdf_visual_points
from r2s2r.reconstruct.refine import Observation, RefineConfig, observe
from r2s2r.reconstruct.render import SceneRenderer
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec
from r2s2r.transforms import invert


@dataclass
class ExistenceConfig:
    """Knobs of :func:`drop_unseen` (metres)."""

    see_through: float = 0.02  # a surface this far in front of the depth is not there
    min_share: float = 0.5  # of an object's visible pixels, seen through, to drop it
    min_pixels: int = 500  # visible pixels over all views; fewer is no evidence
    model_points: int = 4000
    # An unmatched cluster with at least this share of its points inside an object's
    # footprint (grown by footprint_margin) counts as that object's own points.
    min_share_under: float = 0.5
    footprint_margin: float = 0.02
    max_size: tuple[int, int] = (2560, 1600)
    refine: RefineConfig = field(default_factory=RefineConfig)


def drop_unseen(
    scene: SceneSpec, views: list[DepthView], config: ExistenceConfig | None = None
) -> tuple[SceneSpec, dict[str, Any]]:
    """``scene`` without the objects ``views`` see through and no points match; see the
    module doc."""
    cfg = config or ExistenceConfig()
    obs = observe(scene, views, cfg.refine)
    report: dict[str, Any] = {"objects": {}, "settled": {}}
    unmatched = [i for i in range(len(scene.objects)) if i not in obs.matches]
    shares = _see_through(scene, views, unmatched, cfg) if unmatched else {}
    free = [g for j, g in enumerate(obs.groups) if j not in obs.matches.values()]
    dropped = set()
    for i, obj in enumerate(scene.objects):
        entry: dict[str, Any] = {"matched": i in obs.matches}
        if i in shares:
            pixels, share = shares[i]
            entry.update(pixels=pixels, seen_through=round(share, 3))
            if pixels >= cfg.min_pixels and share > cfg.min_share:
                under = _points_under(obs, obj, free, cfg)
                if under:
                    entry["unexplained_points_under"] = under
                else:
                    dropped.add(i)
        entry["dropped"] = i in dropped
        report["objects"][obj.name] = entry
    if not dropped:
        return scene, report

    # Heights in the support frame: which objects rested on a dropped one.
    boxes = [_box(scene, obj.asset_path, obj.T_base_obj, cfg) for obj in scene.objects]
    contact = cfg.refine.contact_tolerance
    objects = []
    for i, obj in enumerate(scene.objects):
        if i in dropped:
            continue
        lo = boxes[i][0]
        on_dropped = any(
            _overlap(boxes[i], boxes[k]) and abs(boxes[k][1][2] - lo[2]) < contact
            for k in dropped
        )
        if on_dropped:
            below = [
                boxes[k][1][2]
                for k in range(len(scene.objects))
                if k not in dropped
                and k != i
                and _overlap(boxes[i], boxes[k])
                and boxes[k][1][2] < lo[2] + contact
            ]
            fall = lo[2] - max(below, default=0.0)
            if fall > 0:
                T = invert(scene.T_base_support) @ obj.T_base_obj
                T[2, 3] -= fall
                obj = replace(obj, T_base_obj=scene.T_base_support @ T)
                report["settled"][obj.name] = round(float(fall), 4)
        objects.append(obj)
    kept = replace(
        scene,
        objects=objects,
        provenance={**scene.provenance, "existence": report},
    )
    return kept, report


def _points_under(
    obs: Observation, obj: ObjectSpec, groups: list, cfg: ExistenceConfig
) -> int:
    """How many points of unmatched clusters stand within ``obj``'s footprint."""
    T = invert(obs.T_base_support) @ obj.T_base_obj
    pts = urdf_visual_points(obj.asset_path, cfg.model_points) @ T[:3, :3].T + T[:3, 3]
    lo = pts[:, :2].min(axis=0) - cfg.footprint_margin
    hi = pts[:, :2].max(axis=0) + cfg.footprint_margin
    count = 0
    for g in groups:
        xy = obs.above[g, :2]
        if np.all((xy >= lo) & (xy <= hi), axis=1).mean() >= cfg.min_share_under:
            count += len(g)
    return count


def _see_through(
    scene: SceneSpec, views: list[DepthView], indices: list[int], cfg: ExistenceConfig
) -> dict[int, tuple[int, float]]:
    """For each object in ``indices``: its visible pixels with measured depth, over all
    views, and the share of them the depth sees through."""
    renderer = SceneRenderer(scene, cfg.max_size)
    counts = {i: [0, 0] for i in indices}
    try:
        for k, obj in enumerate(scene.objects):
            renderer.pose(k, obj.T_base_obj)
        for view in views:
            for i in indices:
                out = renderer.render(view, i)
                visible = out["mask"] & (view.depth > 0)
                behind = view.depth[visible] - out["depth"][visible] > cfg.see_through
                counts[i][0] += int(visible.sum())
                counts[i][1] += int(behind.sum())
    finally:
        renderer.close()
    return {i: (n, seen / n if n else 0.0) for i, (n, seen) in counts.items()}


def _box(
    scene: SceneSpec, asset_path: str, T_base_obj: NDArray, cfg: ExistenceConfig
) -> tuple[NDArray, NDArray]:
    """The object's axis-aligned box in the support frame (low and high corners)."""
    T = invert(scene.T_base_support) @ T_base_obj
    pts = urdf_visual_points(asset_path, cfg.model_points) @ T[:3, :3].T + T[:3, 3]
    return pts.min(axis=0), pts.max(axis=0)


def _overlap(a: tuple[NDArray, NDArray], b: tuple[NDArray, NDArray]) -> bool:
    """Whether two boxes' footprints on the support overlap."""
    return bool(np.all(a[0][:2] < b[1][:2]) and np.all(b[0][:2] < a[1][:2]))
