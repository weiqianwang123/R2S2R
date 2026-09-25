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


def pose6d_to_matrix(pose: ArrayLike) -> NDArray[np.float64]:
    """Convert DROID's ``[x, y, z, rx, ry, rz]`` (extrinsic xyz Euler) to 4x4."""
    pose = np.asarray(pose, dtype=np.float64)
    assert pose.shape == (6,), f"expected a 6-vector, got {pose.shape}"
    return make_transform(Rotation.from_euler("xyz", pose[3:]).as_matrix(), pose[:3])


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
    quat = quat_xyzw_to_wxyz(Rotation.from_matrix(T[:3, :3]).as_quat())
    return T[:3, 3].copy(), quat


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
    T_cam_base = invert(T_base_cam)
    pts_cam = pts @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
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
