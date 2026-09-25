"""An object-free SceneSpec straight from stereo depth, for checking alignment.

Before a backend has produced objects, this places the robot, the calibrated cameras and
the support plane, so a render from the real camera poses can be compared with the real
frames. The plane is fitted to stage-2 depth of one frame, restricted to the height
where the gripper first closed (DROID tasks grasp from the support surface), which keeps
the floor out of the fit. Other surfaces at that height (e.g. the robot's own mounting
table) can still pull the plane's origin sideways, so only its height and normal are
reliable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from r2s2r.structs import Capture, FrameRecord, SceneSpec
from r2s2r.transforms import make_transform


def backproject(
    depth: NDArray[np.float64], K: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Camera-frame points (N x 3) of all pixels with positive depth."""
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def fit_support_plane(
    points_base: NDArray[np.float64],
    z_range: tuple[float, float],
    max_radius: float = 1.2,
    iterations: int = 500,
    inlier_dist: float = 0.01,
    seed: int = 0,
) -> NDArray[np.float64]:
    """RANSAC for the largest near-horizontal plane within a height band.

    Returns ``T_base_support``: origin at the inlier centroid, z along the upward
    normal, x along the projection of the base x axis.
    """
    pts = points_base[
        (points_base[:, 2] > z_range[0])
        & (points_base[:, 2] < z_range[1])
        & (np.linalg.norm(points_base[:, :2], axis=1) < max_radius)
    ]
    if len(pts) < 100:
        raise ValueError(f"only {len(pts)} points in the support height band")
    rng = np.random.default_rng(seed)
    best: NDArray[np.bool_] | None = None
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < 0.95:  # keep near-horizontal planes only
            continue
        inliers = np.abs((pts - a) @ n) < inlier_dist
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None:
        raise ValueError("no horizontal plane found")
    inl = pts[best]
    centroid = inl.mean(axis=0)
    normal = np.linalg.svd(inl - centroid)[2][-1]
    normal = normal if normal[2] > 0 else -normal
    x_axis = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    x_axis /= np.linalg.norm(x_axis)
    rot = np.stack([x_axis, np.cross(normal, x_axis), normal], axis=1)
    return make_transform(rot, centroid)


def preview_scene(
    capture: Capture,
    frame: FrameRecord,
    depth: NDArray[np.float64],
    K: NDArray[np.float64],
    grasp_z: float,
    band: tuple[float, float] = (-0.25, -0.10),
) -> SceneSpec:
    """Object-free scene with the plane fitted in ``grasp_z`` + ``band``.

    DROID's ``cartesian_position`` sits about 0.17 m above the Robotiq fingertips (IRIS
    episode: end effector at -0.008 m, grasped marker at -0.179 m), so the support
    surface is well below the end-effector height at the first grasp.
    """
    pts_cam = backproject(depth, K)
    T = frame.T_base_cam
    pts_base = pts_cam @ T[:3, :3].T + T[:3, 3]
    T_base_support = fit_support_plane(pts_base, (grasp_z + band[0], grasp_z + band[1]))
    return SceneSpec(
        name=f"{capture.name}_preview",
        embodiment=capture.embodiment,
        objects=[],
        T_base_support=T_base_support,
        cameras=capture.cameras,
        reference_camera=frame.camera,
        reference_step=frame.step,
        joint_positions=frame.joint_positions,
        provenance={"backend": "preview", "support_from": "stereo depth RANSAC"},
    )
