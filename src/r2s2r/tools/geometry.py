"""Geometry for the tools and the methods: points from calibrated depth, the support
plane and its outline, clusters of points, top-down footprint registration, and fitting
a generated mesh to what several calibrated views see of an object.

Everything is in the robot base frame, in metres; the support frame has z along the
support's normal (towards the cameras) and its origin on the support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial.transform import Rotation

from r2s2r.assets import VisualMesh
from r2s2r.mjrender import CameraRenderer, add_camera, add_mesh, mujoco
from r2s2r.pipeline.workspace import Workspace, load_mask
from r2s2r.structs import DepthView, FrameRecord
from r2s2r.transforms import (
    backproject,
    invert,
    make_transform,
    project_points,
    transform_points,
)

logger = logging.getLogger(__name__)

MAX_DEPTH = 2.5
# Generated meshes (Hunyuan3D, glTF) are y-up; this turns them z-up.
UP_ROTATIONS = {
    "y": Rotation.from_euler("x", 90, degrees=True).as_matrix(),
    "z": np.eye(3),
    "-y": Rotation.from_euler("x", -90, degrees=True).as_matrix(),
    "x": Rotation.from_euler("y", -90, degrees=True).as_matrix(),
}
FIT_WIDTH = 640  # render width while fitting


# ------------------------------------------------------------------------- points
def view_points(
    view: DepthView,
    mask: NDArray[np.bool_] | None = None,
    erode_px: int = 0,
    stride: int = 1,
    max_depth: float = MAX_DEPTH,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Base-frame points of a view's depth, and the pixels they come from: every
    ``stride``-th pixel with ``0 < depth < max_depth`` inside ``mask`` shrunk by
    ``erode_px`` (depth at object edges bleeds into the background).

    The pixels are in the points' order: ``view.image[pixels]`` are their colours.
    """
    keep = (view.depth > 0) & (view.depth < max_depth)
    if mask is not None:
        m = mask.astype(np.uint8)
        if erode_px > 0:
            m = np.asarray(cv2.erode(m, np.ones((2 * erode_px + 1,) * 2, np.uint8)))
        keep &= m > 0
    if stride > 1:
        grid = np.zeros_like(keep)
        grid[::stride, ::stride] = True
        keep &= grid
    pts_cam = backproject(np.where(keep, view.depth, 0.0), view.K)
    return transform_points(view.T_base_cam, pts_cam), keep


def parse_masks(pairs: list[str] | None) -> dict[str, str]:
    """``["ext2@0=path.png", ...]`` -> ``{"ext2@0": "path.png"}``."""
    out = {}
    for pair in pairs or []:
        fid, sep, path = pair.partition("=")
        if not sep:
            raise ValueError(f"mask {pair!r} is not FRAME=PATH")
        out[fid] = path
    return out


def points(
    ws: Workspace,
    frame_ids: list[str],
    masks: dict[str, str],
    out_path: str | Path,
    support_file: str | Path | None = None,
    stride: int = 2,
    erode_px: int = 3,
) -> dict[str, Any]:
    """Fused points of the frames (inside their masks, where given) as a coloured PLY,
    and where they lie."""
    all_pts, all_rgb, per_frame = [], [], {}
    for fid in dict.fromkeys([*frame_ids, *masks]):
        frame = ws.frame(fid)
        cam = ws.capture.cameras[frame.camera]
        mask = load_mask(masks[fid], (cam.height, cam.width)) if fid in masks else None
        view = ws.depth_view(frame)
        pts, pixels = view_points(
            view, mask, erode_px if mask is not None else 0, stride
        )
        assert view.image is not None
        per_frame[fid] = int(len(pts))
        all_pts.append(pts)
        all_rgb.append(view.image[pixels])
    pts, rgb = np.concatenate(all_pts), np.concatenate(all_rgb)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(out_path, pts, rgb)
    summary: dict[str, Any] = {"file": str(out_path), "points": per_frame}
    if len(pts):
        summary["base"] = _extent(pts)
        if support_file is not None:
            T_base_support, _ = load_support(support_file)
            summary["support_frame"] = _extent(
                transform_points(invert(T_base_support), pts)
            )
    return summary


def _write_ply(path: Path, pts: NDArray[np.float64], rgb: NDArray[np.uint8]) -> None:
    """A binary PLY of coloured points."""
    row = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ]
    )
    data = np.empty(len(pts), row)
    for k, name in enumerate("xyz"):
        data[name] = pts[:, k]
    for k, name in enumerate("rgb"):
        data[name] = rgb[:, k]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def _extent(pts: NDArray[np.float64]) -> dict[str, Any]:
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    return {
        "median": np.round(np.median(pts, axis=0), 4).tolist(),
        "p2": np.round(lo, 4).tolist(),
        "p98": np.round(hi, 4).tolist(),
    }


def cluster(pts: NDArray[np.float64], voxel: float, min_points: int) -> list:
    """Connected components of occupied voxels (26-connectivity): index arrays into
    ``pts``, those of at least ``min_points``."""
    if len(pts) == 0:
        return []
    ijk = np.floor((pts - pts.min(0)) / voxel).astype(int)
    grid = np.zeros(ijk.max(0) + 1, bool)
    grid[tuple(ijk.T)] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    point_labels = labels[tuple(ijk.T)]
    groups = [np.flatnonzero(point_labels == k) for k in range(1, n + 1)]
    return [g for g in groups if len(g) >= min_points]


# ------------------------------------------------------------------------ support
def fit_plane(
    pts: NDArray[np.float64],
    threshold: float = 0.005,
    iterations: int = 500,
    seed: int = 0,
) -> tuple[NDArray[np.float64], float, NDArray[np.bool_]]:
    """RANSAC plane ``n . p + d = 0`` (unit ``n``), refined on its inliers."""
    rng = np.random.default_rng(seed)
    sample = pts[rng.choice(len(pts), min(len(pts), 50_000), replace=False)]
    best, best_count = (np.array([0.0, 0.0, 1.0]), 0.0), -1
    for _ in range(iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -float(n @ a)
        count = int((np.abs(sample @ n + d) < threshold).sum())
        if count > best_count:
            best, best_count = (n, d), count
    n, d = best
    inliers = np.abs(pts @ n + d) < threshold
    centroid = pts[inliers].mean(0)
    _, _, vt = np.linalg.svd(pts[inliers] - centroid, full_matrices=False)
    n = vt[2]
    d = -float(n @ centroid)
    return n, d, np.abs(pts @ n + d) < threshold


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


def planar(yaw: float, dx: float, dy: float) -> NDArray[np.float64]:
    """A turn by ``yaw`` about z, then a shift by (``dx``, ``dy``) in the plane."""
    return make_transform(Rotation.from_euler("z", yaw).as_matrix(), [dx, dy, 0.0])


def support(
    ws: Workspace,
    masks: dict[str, str],
    out_path: str | Path,
    frame_ids: list[str] | None = None,
    threshold: float = 0.005,
) -> dict[str, Any]:
    """The support plane through the masked pixels of several frames (or whole frames),
    its outline as a rectangle, and the support frame at the rectangle's centre.

    Writes ``out_path`` (JSON: ``T_base_support``, ``extent``, fit numbers) and an
    overlay per frame (``<stem>_<frame>.png``): the rectangle and a 10 cm grid on it.
    """
    ids = list(dict.fromkeys([*(frame_ids or []), *masks]))
    chunks: list[NDArray[np.float64]] = []
    owners: list[NDArray[np.int64]] = []
    centres: list[NDArray[np.float64]] = []
    for fid in ids:
        frame = ws.frame(fid)
        cam = ws.capture.cameras[frame.camera]
        mask = load_mask(masks[fid], (cam.height, cam.width)) if fid in masks else None
        pts, _ = view_points(ws.depth_view(frame), mask, erode_px=2, stride=2)
        chunks.append(pts)
        owners.append(np.full(len(pts), len(owners)))
        centres.append(frame.T_base_cam[:3, 3])
    pts, owner = np.concatenate(chunks), np.concatenate(owners)
    if len(pts) < 100:
        raise ValueError("too few depth points in the masks to fit a plane")
    n, d, inliers = fit_plane(pts, threshold)
    if n @ np.mean(centres, axis=0) + d < 0:  # the normal points to the cameras
        n, d = -n, -d
    x = np.array([1.0, 0.0, 0.0]) - n[0] * n
    if np.linalg.norm(x) < 0.3:
        x = np.array([0.0, 1.0, 0.0]) - n[1] * n
    x /= np.linalg.norm(x)
    origin = pts[inliers].mean(0)
    origin -= (n @ origin + d) * n
    T0 = make_transform(np.column_stack([x, np.cross(n, x), n]), origin)
    pts_s = transform_points(invert(T0), pts[inliers])
    centre, size, yaw = fit_support_outline(pts_s, np.zeros(2), threshold, 0.01)
    T_base_support = T0 @ planar(yaw, *centre)
    residual = np.abs(pts[inliers] @ n + d)
    summary: dict[str, Any] = {
        "T_base_support": np.round(T_base_support, 6).tolist(),
        "extent": [round(size[0], 4), round(size[1], 4)],
        "normal_base": np.round(n, 4).tolist(),
        "tilt_deg": round(float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))), 2),
        "height_at_base_origin_m": round(
            float(-d / n[2]) if abs(n[2]) > 1e-6 else 0, 4
        ),
        "inlier_rms_m": round(float(np.sqrt(np.mean(residual**2))), 5),
        "inlier_share": round(float(inliers.mean()), 3),
        "frames": {
            fid: {
                "points": int((owner == k).sum()),
                "inlier_share": (
                    round(float(inliers[owner == k].mean()), 3)
                    if (owner == k).any()
                    else None
                ),
            }
            for k, fid in enumerate(ids)
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    overlays = []
    for fid in ids:
        path = out_path.with_name(f"{out_path.stem}_{fid}.png")
        _support_overlay(ws, ws.frame(fid), T_base_support, size, path)
        overlays.append(str(path))
    summary["overlays"] = overlays
    return summary


def load_support(
    source: str | Path | dict[str, Any],
) -> tuple[NDArray[np.float64], tuple[float, float] | None]:
    """``T_base_support`` and extent from a support JSON (or its dict)."""
    d = (
        source
        if isinstance(source, dict)
        else json.loads(Path(source).read_text(encoding="utf-8"))
    )
    extent = d.get("extent")
    return np.asarray(d["T_base_support"], float), (
        None if extent is None else (float(extent[0]), float(extent[1]))
    )


def _support_overlay(
    ws: Workspace,
    frame: FrameRecord,
    T_base_support: NDArray[np.float64],
    size: tuple[float, float],
    path: Path,
) -> None:
    image = ws.image(frame)
    cam = ws.capture.cameras[frame.camera]
    hx, hy = size[0] / 2, size[1] / 2

    def draw(
        a: tuple[float, float], b: tuple[float, float], color: Any, w: int
    ) -> None:
        seg = np.linspace(a, b, 40)
        pts = transform_points(T_base_support, np.c_[seg, np.zeros(len(seg))])
        uv = project_points(pts, frame.T_base_cam, cam.K)
        ok = np.all(np.isfinite(uv), axis=1)
        for p, q, good in zip(uv[:-1], uv[1:], ok[:-1] & ok[1:]):
            if good:
                cv2.line(
                    image,
                    tuple(int(v) for v in np.clip(p, -1e4, 1e4)),
                    tuple(int(v) for v in np.clip(q, -1e4, 1e4)),
                    color,
                    w,
                    cv2.LINE_AA,
                )

    for g in np.arange(-np.floor(hx / 0.1) * 0.1, hx, 0.1):
        draw((g, -hy), (g, hy), (0, 200, 255), 1)
    for g in np.arange(-np.floor(hy / 0.1) * 0.1, hy, 0.1):
        draw((-hx, g), (hx, g), (0, 200, 255), 1)
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)]
    for a, b in zip(corners[:-1], corners[1:]):
        draw(a, b, (255, 0, 255), 3)
    draw((0, 0), (0.1, 0), (255, 0, 0), 3)  # support x
    draw((0, 0), (0, 0.1), (0, 255, 0), 3)  # support y
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------- footprints
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
    Yaws within ``yaw_window_deg`` of the model's win unless a turn by
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
    T = planar(0, *(centre + t)) @ planar(yaw, 0, 0) @ planar(0, *(-centre))
    return T, before, after


# -------------------------------------------------------------------------- meshes
def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """A mesh file as one mesh (a glTF scene's parts concatenated)."""
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} holds no mesh")
    return mesh


class SilhouetteRenderer:
    """Renders one mesh (fixed at the origin) from calibrated cameras.

    A uniformly scaled mesh looks like the unscaled one from a camera whose position is
    scaled down by the same factor, so scale changes need no recompiling.
    """

    def __init__(self, mesh: trimesh.Trimesh, max_size: tuple[int, int]) -> None:
        spec = mujoco.MjSpec()
        add_camera(spec, max_size)
        body = spec.worldbody.add_body(name="object")
        add_mesh(spec, body, "object", VisualMesh(mesh, None))
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.camera = CameraRenderer(self.model, self.data, max_size)

    def render(
        self,
        K: NDArray,
        width: int,
        height: int,
        T_obj_cam: NDArray,
        scale: float,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Silhouette and planar depth of the mesh scaled by ``scale``, seen from
        ``T_obj_cam`` (object frame)."""
        T = T_obj_cam.copy()
        T[:3, 3] /= scale
        out = self.camera.render(K, width, height, T)
        mask = out["geom"] >= 0
        return mask, np.where(mask, out["depth"] * scale, 0.0)

    def close(self) -> None:
        """Free the GL context."""
        self.camera.close()


class _View:
    """One frame's evidence about an object, at the fitting resolution."""

    def __init__(self, ws: Workspace, fid: str, mask_path: str) -> None:
        frame = ws.frame(fid)
        cam = ws.capture.cameras[frame.camera]
        self.fid, self.frame = fid, frame
        self.full = (cam.height, cam.width)
        self.factor = min(1.0, FIT_WIDTH / cam.width)
        self.size = (
            int(round(cam.width * self.factor)),
            int(round(cam.height * self.factor)),
        )
        self.K = cam.K.copy()
        self.K[:2] *= self.factor
        self.K[:2, 2] += 0.5 * self.factor - 0.5
        self.mask_full = load_mask(mask_path, self.full)
        small = (self.size[1], self.size[0])
        self.mask = load_mask(mask_path, small)
        depth = ws.depth_view(frame).depth
        self.depth = cv2.resize(
            depth.astype(np.float32), self.size, interpolation=cv2.INTER_NEAREST
        )
        robot = ws.robot_mask(frame).astype(np.uint8)
        self.robot = cv2.resize(robot, self.size, interpolation=cv2.INTER_NEAREST) > 0


def fit(
    ws: Workspace,
    mesh_path: str | Path,
    support_source: str | Path | dict[str, Any],
    masks: dict[str, str],
    out_dir: str | Path,
    up: str = "y",
    scale: float | None = None,
    yaw_deg: float | None = None,
    rest: bool = True,
    refine: bool = True,
) -> dict[str, Any]:
    """Scale and place a mesh so that it matches the object's masks and depth in every
    given frame.

    The mesh is turned ``up``-axis to the support normal. Scale starts from the observed
    height (or footprint, for flat objects), yaw and position from matching top-down
    footprints of the mesh and the fused masked depth; then scale, yaw and position (and
    height, unless the object ``rest``s on the support) are searched to maximise the
    mean silhouette IoU over the frames (robot pixels ignored).

    Writes ``out_dir/fit.json`` and an overlay per frame (green: the mask; red: the
    fitted mesh). ``T_base_obj`` maps the mesh file's coordinates, scaled by ``scale``,
    to the base frame.
    """
    if not masks:
        raise ValueError("fit needs the object's mask in at least one frame")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T_base_support, _ = load_support(support_source)
    raw = load_mesh(mesh_path)
    R_up = UP_ROTATIONS[up]
    upright = np.asarray(raw.vertices) @ R_up.T
    lo, hi = upright.min(0), upright.max(0)
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    canon = trimesh.Trimesh(upright - centre, np.asarray(raw.faces), process=False)
    model_pts = np.asarray(trimesh.sample.sample_surface(canon, 4000, seed=0)[0])
    model_h = float(canon.bounds[1, 2])

    views = [_View(ws, fid, path) for fid, path in masks.items()]
    observed = _object_points(ws, views, T_base_support)
    if len(observed) < 20:
        raise ValueError("too few depth points inside the masks")
    obs_h = float(np.percentile(observed[:, 2], 98))
    obs_long = _long_side(observed[:, :2])
    model_long = _long_side(model_pts[:, :2])
    if scale is None:
        scale = obs_h / model_h if obs_h >= 0.03 else obs_long / model_long
        scale_source = "height" if obs_h >= 0.03 else "footprint"
    else:
        scale_source = "given"

    # Yaw and position from the footprints.
    xy0 = np.median(observed[:, :2], axis=0)
    z0 = 0.0 if rest else obs_h - scale * model_h
    if yaw_deg is None:
        placed = model_pts * scale + [*xy0, 0.0]
        T_fix, _, iou_fp = register_footprint(
            placed, observed, yaw_window_deg=180.0, cell=0.003
        )
        yaw0 = float(np.arctan2(T_fix[1, 0], T_fix[0, 0]))
        xy_init = (T_fix @ np.array([*xy0, 0.0, 1.0]))[:2]
    else:
        yaw0, xy_init, iou_fp = float(np.radians(yaw_deg)), xy0, None
    params = np.array([np.log(scale), yaw0, xy_init[0], xy_init[1], z0])

    cams = [ws.capture.cameras[v.frame.camera] for v in views]
    renderer = SilhouetteRenderer(
        canon, (max(c.width for c in cams), max(c.height for c in cams))
    )
    try:

        def pose(p: NDArray[np.float64]) -> NDArray[np.float64]:
            R = Rotation.from_euler("z", p[1]).as_matrix()
            return T_base_support @ make_transform(R, [p[2], p[3], p[4]])

        def score(p: NDArray[np.float64]) -> tuple[float, list[dict[str, Any]]]:
            T_base_canon = pose(p)
            rows = []
            for v in views:
                T_obj_cam = invert(T_base_canon) @ v.frame.T_base_cam
                sil, depth = renderer.render(
                    v.K, *v.size, T_obj_cam, float(np.exp(p[0]))
                )
                rows.append(_compare(v, sil, depth))
            return float(np.mean([r["iou"] for r in rows])), rows

        initial, _ = score(params)
        if refine:
            steps = np.array([0.1, np.radians(10.0), 0.01, 0.01, 0.01])
            free = [True, yaw_deg is None, True, True, not rest]
            params = pattern_search(lambda p: score(p)[0], params, steps, free)
        final, rows = score(params)

        s = float(np.exp(params[0]))
        T_base_canon = pose(params)
        T_canon = make_transform(R_up, -s * centre)  # scaled mesh file -> canon
        T_base_obj = T_base_canon @ T_canon
        for v, row in zip(views, rows):
            T_obj_cam = invert(T_base_canon) @ v.frame.T_base_cam
            cam = ws.capture.cameras[v.frame.camera]
            sil, _ = renderer.render(cam.K, cam.width, cam.height, T_obj_cam, s)
            path = out_dir / f"{v.fid}.png"
            _fit_overlay(ws.image(v.frame), v.mask_full, sil, row, path)
            row["overlay"] = str(path)
    finally:
        renderer.close()

    size = (canon.bounds[1] - canon.bounds[0]) * s
    summary: dict[str, Any] = {
        "mesh": str(Path(mesh_path).resolve()),
        "up": up,
        "scale": round(s, 5),
        "T_base_obj": np.round(T_base_obj, 6).tolist(),
        "yaw_deg": round(float(np.degrees(params[1])), 2),
        "rests_on_support": rest,
        "size_m": np.round(size, 4).tolist(),
        "observed_height_m": round(obs_h, 4),
        "scale_from": scale_source,
        "footprint_iou_init": None if iou_fp is None else round(float(iou_fp), 3),
        "mean_iou_init": round(initial, 3),
        "mean_iou": round(final, 3),
        "frames": {v.fid: row for v, row in zip(views, rows)},
    }
    (out_dir / "fit.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _object_points(
    ws: Workspace, views: list[_View], T_base_support: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Masked depth above the support (support frame), stray pixels dropped."""
    T_support_base = invert(T_base_support)
    chunks = [
        transform_points(
            T_support_base, view_points(ws.depth_view(v.frame), v.mask_full, 3, 2)[0]
        )
        for v in views
    ]
    pts = np.concatenate(chunks)
    pts = pts[(pts[:, 2] > 0.003) & (pts[:, 2] < 0.6)]
    groups = cluster(pts, 0.01, 10)
    if not groups:
        return pts
    biggest = max(len(g) for g in groups)
    keep = np.concatenate([g for g in groups if len(g) >= 0.2 * biggest])
    return pts[keep]


def _long_side(xy: NDArray[np.float64]) -> float:
    (_, _), (w, h), _ = cv2.minAreaRect(xy.astype(np.float32))
    return float(max(w, h, 1e-4))


def _compare(
    v: _View, sil: NDArray[np.bool_], depth: NDArray[np.float64]
) -> dict[str, Any]:
    care = ~v.robot
    mask, sil_c = v.mask & care, sil & care
    union = float((mask | sil_c).sum())
    inter = float((mask & sil_c).sum())
    both = mask & sil_c & (v.depth > 0)
    return {
        "iou": round(inter / union, 4) if union else 0.0,
        "mask_missed": round(1 - inter / max(1.0, float(mask.sum())), 3),
        "render_outside": round(1 - inter / max(1.0, float(sil_c.sum())), 3),
        # Median of rendered minus measured depth where both see the object: > 0, the
        # mesh surface lies behind the measured one.
        "depth_offset_m": (
            round(float(np.median(depth[both] - v.depth[both])), 4)
            if both.sum() >= 20
            else None
        ),
    }


def pattern_search(
    f: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    steps: NDArray[np.float64],
    free: list[bool],
    levels: int = 4,
    max_evals: int = 400,
) -> NDArray[np.float64]:
    """Coordinate pattern search maximising ``f``, halving the steps when stuck."""
    best, evals = f(x), 1
    steps = steps.copy()
    for _ in range(levels):
        improved = True
        while improved and evals < max_evals:
            improved = False
            for i in range(len(x)):
                if not free[i]:
                    continue
                for sign in (1.0, -1.0):
                    trial = x.copy()
                    trial[i] += sign * steps[i]
                    value = f(trial)
                    evals += 1
                    if value > best + 1e-4:
                        x, best, improved = trial, value, True
                        break
        steps /= 2
    return x


def _fit_overlay(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    sil: NDArray[np.bool_],
    row: dict[str, Any],
    path: Path,
) -> None:
    out = image.copy()
    for m, color in ((mask, (0, 255, 0)), (sil, (255, 0, 0))):
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(out, contours, -1, color, 2)
    text = f"IoU {row['iou']:.2f}"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
