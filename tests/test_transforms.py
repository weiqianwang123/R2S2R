"""Tests for transforms.py."""

import numpy as np
from scipy.spatial.transform import Rotation

from r2s2r.transforms import (
    intrinsics_matrix,
    invert,
    is_rigid,
    look_at,
    make_transform,
    matrix_to_pos_quat,
    pos_quat_to_matrix,
    project_points,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    rotation_to_quat,
    transform_points,
)


def test_invert_roundtrip():
    """Invert() undoes a random rigid transform."""
    T = make_transform(
        Rotation.from_euler("xyz", [0.3, -1.2, 2.0]).as_matrix(), [1.0, -2.0, 3.0]
    )
    assert np.allclose(invert(T) @ T, np.eye(4))


def test_is_rigid_and_transform_points():
    """Rotations with translations are rigid; scale, mirroring, a bad shape, a bottom
    row other than 0 0 0 1 or a translation not finite are not. Points map like the
    homogeneous product."""
    T = make_transform(Rotation.from_euler("z", 0.7).as_matrix(), [1.0, 2.0, 3.0])
    assert is_rigid(T) and is_rigid(T.tolist())
    assert not is_rigid(np.diag([2.0, 1.0, 1.0, 1.0]))
    assert not is_rigid(np.diag([-1.0, 1.0, 1.0, 1.0]))
    assert not is_rigid(np.eye(3)) and not is_rigid([[1, 2], [3]])
    for row in ([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 1.0]):
        assert not is_rigid(np.r_[T[:3], [row]])
    for bad in (np.nan, np.inf):
        U = T.copy()
        U[0, 3] = bad
        assert not is_rigid(U)
    pts = np.array([[0.1, 0.2, 0.3], [1.0, -1.0, 0.5]])
    homogeneous = (T @ np.c_[pts, np.ones(2)].T).T[:, :3]
    assert np.allclose(transform_points(T, pts), homogeneous)


def test_look_at_points_the_optical_axis_at_the_target():
    """The camera's z runs to the target, its x stays level, its y points down."""
    T = look_at([1.0, 0.5, 0.8], [0.2, 0.0, 0.1])
    fwd = np.array([0.2, 0.0, 0.1]) - [1.0, 0.5, 0.8]
    assert is_rigid(T)
    assert np.allclose(T[:3, 2], fwd / np.linalg.norm(fwd))
    assert abs(T[2, 0]) < 1e-12 and T[2, 1] < 0
    assert np.allclose(T[:3, 3], [1.0, 0.5, 0.8])


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
    assert np.allclose(rotation_to_quat(T[:3, :3]), quat)


def test_project_points():
    """A point on the optical axis lands on the principal point."""
    K = intrinsics_matrix(100.0, 100.0, 50.0, 40.0)
    T_base_cam = make_transform(np.eye(3), [0.0, 0.0, -1.0])
    uv = project_points([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]], T_base_cam, K)
    assert np.allclose(uv[0], [50.0, 40.0])
    assert np.isnan(uv[1]).all()  # behind the camera
