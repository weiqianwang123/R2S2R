"""Rigid-transform helpers.

Conventions used throughout r2s2r:

- ``T_a_b`` is a 4x4 homogeneous matrix mapping points expressed in frame ``b``
  into frame ``a`` (so ``T_a_c = T_a_b @ T_b_c``).
- Camera frames follow OpenCV: +z forward, +x right, +y down.
- Quaternions are ``(w, x, y, z)`` unless a function name says ``xyzw``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


def make_transform(rotation: ArrayLike, translation: ArrayLike) -> NDArray[np.float64]:
    """Build a 4x4 transform from a 3x3 rotation and a 3-vector."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def invert(T: ArrayLike) -> NDArray[np.float64]:
    """Invert a rigid transform without a general matrix inverse."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    return make_transform(R.T, -R.T @ T[:3, 3])


def is_rigid(T: ArrayLike, atol: float = 1e-4) -> bool:
    """Whether ``T`` is a 4x4 rigid transform (a rotation, no scale or mirroring)."""
    try:
        T = np.asarray(T, dtype=np.float64)
    except (TypeError, ValueError):  # not numbers, or ragged
        return False
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    return bool(np.allclose(R @ R.T, np.eye(3), atol=atol) and np.linalg.det(R) > 0)


def transform_points(T_a_b: ArrayLike, points_b: ArrayLike) -> NDArray[np.float64]:
    """Nx3 points in frame ``b`` expressed in frame ``a``."""
    T = np.asarray(T_a_b, dtype=np.float64)
    return np.asarray(points_b, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]


def look_at(eye: ArrayLike, target: ArrayLike) -> NDArray[np.float64]:
    """``T_base_cam`` of an OpenCV camera at ``eye`` looking at ``target``, level (its x
    axis horizontal; z is up)."""
    eye, target = np.asarray(eye, dtype=np.float64), np.asarray(
        target, dtype=np.float64
    )
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return make_transform(np.column_stack([right, np.cross(fwd, right), fwd]), eye)


def quat_xyzw_to_wxyz(quat: ArrayLike) -> NDArray[np.float64]:
    """Reorder a quaternion from (x, y, z, w) to (w, x, y, z)."""
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    return np.array([w, x, y, z])


def quat_wxyz_to_xyzw(quat: ArrayLike) -> NDArray[np.float64]:
    """Reorder a quaternion from (w, x, y, z) to (x, y, z, w)."""
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array([x, y, z, w])


def pos_quat_to_matrix(pos: ArrayLike, quat_wxyz: ArrayLike) -> NDArray[np.float64]:
    """Build a transform from a position and a (w, x, y, z) quaternion."""
    rot = Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).as_matrix()
    return make_transform(rot, pos)


def matrix_to_pos_quat(
    T: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a transform into a position and a (w, x, y, z) quaternion."""
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), rotation_to_quat(T[:3, :3])


def rotation_to_quat(R: ArrayLike) -> NDArray[np.float64]:
    """The (w, x, y, z) quaternion of a 3x3 rotation (MuJoCo's order too)."""
    return quat_xyzw_to_wxyz(Rotation.from_matrix(np.asarray(R, np.float64)).as_quat())


def intrinsics_matrix(
    fx: float, fy: float, cx: float, cy: float
) -> NDArray[np.float64]:
    """Build a 3x3 pinhole intrinsics matrix."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def project_points(
    points_base: ArrayLike, T_base_cam: ArrayLike, K: ArrayLike
) -> NDArray[np.float64]:
    """Project Nx3 base-frame points to Nx2 pixels (NaN behind the camera)."""
    pts = np.atleast_2d(np.asarray(points_base, dtype=np.float64))
    pts_cam = transform_points(invert(T_base_cam), pts)
    uvw = pts_cam @ np.asarray(K, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
    uv[pts_cam[:, 2] <= 0] = np.nan
    return uv


def backproject(
    depth: ArrayLike, K: ArrayLike, max_depth: float = np.inf
) -> NDArray[np.float64]:
    """Camera-frame points (N x 3) of the pixels with 0 < depth < ``max_depth`` (planar
    depth, OpenCV camera)."""
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    v, u = np.nonzero((depth > 0) & (depth < max_depth))
    z = depth[v, u]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
