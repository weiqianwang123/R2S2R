"""Multi-view refinement of a reconstructed scene from calibrated metric depth.

A single-frame backend places objects from one view, so small or thin objects
can end up centimetres off, and it reconstructs only the support plane, not its
outline. This module fuses metric depth from several calibrated views (e.g. both
DROID exterior cameras at the reference step) in the robot base frame and

1. fits the support surface's outline: the connected region of on-plane points
   around the objects, summarised as a rectangle (centre, size, yaw);
2. re-registers every object mesh to the fused points above the support by
   matching top-down footprints (yaw about the support normal plus in-plane
   shift), then puts objects that rested on the support back in contact with it.

Mesh shape, scale and tilt are kept; only where the objects stand changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from r2s2r.assets import urdf_visual_points
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec
from r2s2r.transforms import backproject, invert, make_transform

logger = logging.getLogger(__name__)


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
    max_match_distance: float = 0.25
    model_points: int = 4000
    min_iou_gain: float = 0.02  # keep the backend's pose unless the fit improves
    contact_tolerance: float = 0.03  # objects this close to the support rest on it


def view_points(view: DepthView, max_depth: float) -> NDArray[np.float64]:
    """Base-frame points of all valid pixels."""
    T = view.T_base_cam
    return backproject(view.depth, view.K, max_depth) @ T[:3, :3].T + T[:3, 3]


def to_frame(T_base_frame: NDArray[np.float64], pts_base: NDArray[np.float64]):
    """Express base-frame points in another frame."""
    T = invert(T_base_frame)
    return pts_base @ T[:3, :3].T + T[:3, 3]


def fit_support_outline(
    pts_s: NDArray[np.float64],
    seed_xy: NDArray[np.float64],
    band: float,
    cell: float,
) -> tuple[NDArray[np.float64], tuple[float, float], float]:
    """Rectangle of the on-plane region connected to ``seed_xy``.

    ``pts_s`` are in the support frame (z = height above the plane). Returns the
    rectangle centre (x, y), its size and its yaw (rad) in that frame.
    """
    on = pts_s[np.abs(pts_s[:, 2]) < band, :2]
    if len(on) < 50:
        raise ValueError("too few points on the support plane")
    lo = np.minimum(on.min(0), seed_xy) - cell
    ij = np.floor((on - lo) / cell).astype(int)
    occupied_cells = np.zeros(ij.max(0) + 2, np.uint8)
    occupied_cells[ij[:, 0], ij[:, 1]] = 1
    # Objects hide the support under them; closing bridges those holes.
    closed = cv2.morphologyEx(
        occupied_cells, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    n_labels, labels = cv2.connectedComponents(closed, connectivity=8)
    seed = np.floor((seed_xy - lo) / cell).astype(int)
    occupied = np.argwhere(labels > 0)
    nearest = occupied[np.argmin(np.linalg.norm(occupied - seed, axis=1))]
    in_region = labels[ij[:, 0], ij[:, 1]] == labels[tuple(nearest)]
    logger.debug(
        "support: %d components, %d points in region", n_labels - 1, in_region.sum()
    )
    # Fit the points themselves: cell centres of a rotated outline overshoot by a cell.
    (cx, cy), (w, h), angle = cv2.minAreaRect(on[in_region].astype(np.float32))
    return np.array([cx, cy]), (float(w), float(h)), float(np.deg2rad(angle))


def cluster(pts: NDArray[np.float64], voxel: float, min_points: int) -> list:
    """Connected components of occupied voxels (26-connectivity)."""
    if len(pts) == 0:
        return []
    ijk = np.floor((pts - pts.min(0)) / voxel).astype(int)
    grid = np.zeros(ijk.max(0) + 1, bool)
    grid[tuple(ijk.T)] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    point_labels = labels[tuple(ijk.T)]
    groups = [np.flatnonzero(point_labels == k) for k in range(1, n + 1)]
    return [g for g in groups if len(g) >= min_points]


def _planar(yaw: float, dx: float, dy: float) -> NDArray[np.float64]:
    rot = Rotation.from_euler("z", yaw).as_matrix()
    return make_transform(rot, [dx, dy, 0.0])


def _occupancy(
    xy: NDArray[np.float64],
    lo: NDArray[np.float64],
    shape: tuple[int, int],
    cell: float,
) -> NDArray[np.float32]:
    """Binary footprint image of 2D points (one-cell dilation fills gaps)."""
    grid = np.zeros(shape, np.uint8)
    ij = np.floor((xy - lo) / cell).astype(int)
    ok = np.all((ij >= 0) & (ij < shape), axis=1)
    grid[ij[ok, 0], ij[ok, 1]] = 1
    return cv2.dilate(grid, np.ones((3, 3), np.uint8)).astype(np.float32)


def register_footprint(
    model: NDArray[np.float64],
    observed: NDArray[np.float64],
    flip_gain: float = 0.8,
    yaw_window_deg: float = 45.0,
    cell: float = 0.003,
) -> tuple[NDArray[np.float64], float, float]:
    """Yaw + in-plane shift putting the model's top-down footprint on the observed one.

    Both point sets (support frame) are rasterised to binary footprints, so how
    densely a surface happens to be sampled (side walls facing a camera, say)
    cannot pull the result the way it pulls a point-to-point fit. For every yaw
    on a grid (1 degree, then 0.25 degree) the shift maximising the overlap comes
    from an FFT cross-correlation, and candidates are scored by footprint IoU.
    Yaws within ``yaw_window_deg`` of the backend's win unless a turn by
    90/180/270 degrees has less than ``flip_gain`` times their non-overlap
    (1 - IoU): nearly symmetric objects (a mug without a clearly seen handle)
    otherwise flip on noise.

    Returns the 4x4 correction (applied on the left), and the IoU before and
    after.
    """
    obs_xy, model_xy = observed[:, :2], model[:, :2]
    centre = model_xy.mean(0)
    reach = np.ptp(model_xy, 0).max() + np.ptp(obs_xy, 0).max()
    lo = obs_xy.mean(0) - reach
    size = int(np.ceil(2 * reach / cell))
    shape = (size, size)
    obs = _occupancy(obs_xy, lo, shape, cell)
    obs_area = float(obs.sum())
    obs_fft = np.fft.rfft2(obs)

    Candidate = tuple[float, float, NDArray[np.float64]]  # (IoU, yaw, shift)

    def iou(grid: NDArray[np.float32], overlap: float) -> float:
        return overlap / (obs_area + float(grid.sum()) - overlap)

    def evaluate(yaw: float) -> Candidate:
        c, s_ = np.cos(yaw), np.sin(yaw)
        turned = (model_xy - centre) @ np.array([[c, -s_], [s_, c]]).T + centre
        # Start centred on the observations; the correlation finds the rest.
        start = obs_xy.mean(0) - turned.mean(0)
        grid = _occupancy(turned + start, lo, shape, cell)
        corr = np.fft.irfft2(obs_fft * np.conj(np.fft.rfft2(grid)), s=shape)
        k = np.unravel_index(int(np.argmax(corr)), shape)
        shift = np.array(
            [
                k[0] if k[0] < shape[0] // 2 else k[0] - shape[0],
                k[1] if k[1] < shape[1] // 2 else k[1] - shape[1],
            ]
        )
        return iou(grid, float(corr[k])), yaw, start + shift * cell

    def search(yaws: NDArray[np.float64]) -> Candidate:
        coarse = max((evaluate(y) for y in yaws), key=lambda r: r[0])
        fine = np.deg2rad(np.arange(-1.0, 1.01, 0.25)) + coarse[1]
        return max((evaluate(y) for y in fine), key=lambda r: r[0])

    grid0 = _occupancy(model_xy, lo, shape, cell)
    before = iou(grid0, float((grid0 * obs).sum()))
    window = np.deg2rad(np.arange(-yaw_window_deg, yaw_window_deg + 0.5, 1.0))
    near = search(window)
    turned_best = max(
        (
            search(window[np.abs(window) <= np.deg2rad(10.0)] + np.deg2rad(turn))
            for turn in (90.0, 180.0, 270.0)
        ),
        key=lambda r: r[0],
    )
    use_turned = 1.0 - turned_best[0] < flip_gain * (1.0 - near[0])
    after, yaw, t = turned_best if use_turned else near
    T = _planar(0, *(centre + t)) @ _planar(yaw, 0, 0) @ _planar(0, *(-centre))
    return T, before, after


def refine_scene(
    scene: SceneSpec, views: list[DepthView], cfg: RefineConfig | None = None
) -> tuple[SceneSpec, dict[str, Any]]:
    """Refit the support outline and object placements from ``views``."""
    cfg = cfg or RefineConfig()
    pts = np.concatenate([view_points(v, cfg.max_depth) for v in views])
    pts_s = to_frame(scene.T_base_support, pts)
    obj_xy = (
        np.array(
            [
                to_frame(scene.T_base_support, o.T_base_obj[:3, 3][None])[0, :2]
                for o in scene.objects
            ]
        )
        if scene.objects
        else np.zeros((1, 2))
    )

    # 1. Support outline -> new support frame at the rectangle centre.
    centre, size, yaw = fit_support_outline(
        pts_s, obj_xy.mean(0), cfg.support_band, cfg.support_cell
    )
    T_old_new = _planar(yaw, *centre)
    T_base_support = scene.T_base_support @ T_old_new
    pts_s = to_frame(T_base_support, pts)
    half = np.array(size) / 2 + cfg.outline_margin

    # 2. Object points: above the support, over its outline.
    above = pts_s[
        (pts_s[:, 2] > cfg.object_min_height)
        & (pts_s[:, 2] < cfg.object_max_height)
        & np.all(np.abs(pts_s[:, :2]) < half, axis=1)
    ]
    groups = cluster(above, cfg.cluster_voxel, cfg.min_cluster_points)
    report: dict[str, Any] = {
        "views": [{"camera": v.camera, "step": v.step} for v in views],
        "support": {"size": list(size), "yaw_deg": float(np.rad2deg(yaw))},
        "clusters": [int(len(g)) for g in groups],
        "objects": {},
    }

    T_support_objs = [invert(T_base_support) @ o.T_base_obj for o in scene.objects]
    cost = np.full((len(scene.objects), max(len(groups), 1)), 1e6)
    for i, T_so in enumerate(T_support_objs):
        for j, g in enumerate(groups):
            cost[i, j] = np.linalg.norm(above[g, :2].mean(0) - T_so[:2, 3])
    rows, cols = linear_sum_assignment(cost) if groups else ([], [])
    matches = {r: c for r, c in zip(rows, cols) if cost[r, c] < cfg.max_match_distance}

    new_objects: list[ObjectSpec] = []
    for i, (obj, T_so) in enumerate(zip(scene.objects, T_support_objs)):
        entry: dict[str, Any] = {"matched": i in matches}
        report["objects"][obj.name] = entry
        if i not in matches:
            new_objects.append(obj)
            continue
        observed = above[groups[matches[i]]]
        model_obj = urdf_visual_points(obj.asset_path, cfg.model_points)
        model = model_obj @ T_so[:3, :3].T + T_so[:3, 3]
        T_fix, before, after = register_footprint(model, observed)
        if after < before + cfg.min_iou_gain:
            T_fix = np.eye(4)  # already as good as the depth can tell
        T_new = T_fix @ T_so
        moved = model_obj @ T_new[:3, :3].T + T_new[:3, 3]
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
        new_objects.append(replace(obj, T_base_obj=T_base_support @ T_new))

    refined = replace(
        scene,
        objects=new_objects,
        T_base_support=T_base_support,
        support_extent=size,
        provenance={**scene.provenance, "refinement": report},
    )
    return refined, report
