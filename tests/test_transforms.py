"""Tests for transforms.py."""

import numpy as np
from scipy.spatial.transform import Rotation

from r2s2r.transforms import (
    intrinsics_matrix,
    invert,
    make_transform,
    matrix_to_pos_quat,
    pos_quat_to_matrix,
    pose6d_to_matrix,
    project_points,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)


def test_invert_roundtrip():
    """Invert() undoes a random rigid transform."""
    T = make_transform(
        Rotation.from_euler("xyz", [0.3, -1.2, 2.0]).as_matrix(), [1.0, -2.0, 3.0]
    )
    assert np.allclose(invert(T) @ T, np.eye(4))


def test_pose6d_matches_droid_convention():
    """DROID poses are extrinsic xyz Euler angles plus a translation."""
    pose = [0.1, 0.2, 0.3, 0.4, -0.5, 0.6]
    T = pose6d_to_matrix(pose)
    assert np.allclose(T[:3, 3], pose[:3])
    assert np.allclose(T[:3, :3], Rotation.from_euler("xyz", pose[3:]).as_matrix())


def test_quaternion_conventions():
    """Quaternion helpers keep (w, x, y, z) as the package convention."""
    q_xyzw = Rotation.from_euler("zyx", [0.7, 0.1, -0.4]).as_quat()
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    assert np.isclose(q_wxyz[0], q_xyzw[3])
    assert np.allclose(quat_wxyz_to_xyzw(q_wxyz), q_xyzw)
    T = pos_quat_to_matrix([1.0, 2.0, 3.0], q_wxyz)
    pos, quat = matrix_to_pos_quat(T)
    assert np.allclose(pos, [1.0, 2.0, 3.0])
    assert np.isclose(abs(np.dot(quat, q_wxyz)), 1.0)


def test_project_points():
    """A point on the optical axis lands on the principal point."""
    K = intrinsics_matrix(100.0, 100.0, 50.0, 40.0)
    T_base_cam = make_transform(np.eye(3), [0.0, 0.0, -1.0])
    uv = project_points([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]], T_base_cam, K)
    assert np.allclose(uv[0], [50.0, 40.0])
    assert np.isnan(uv[1]).all()  # behind the camera
