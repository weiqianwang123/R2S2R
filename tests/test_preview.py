"""Tests for preview.py."""

import numpy as np

from r2s2r.preview import backproject, fit_support_plane
from r2s2r.transforms import intrinsics_matrix


def test_fit_support_plane_ignores_floor():
    """The height band keeps the (larger) floor out of the table fit."""
    rng = np.random.default_rng(0)
    table = np.c_[rng.uniform(0.3, 0.8, (2000, 2)), np.full(2000, -0.03)]
    floor = np.c_[rng.uniform(-1.0, 1.0, (8000, 2)), np.full(8000, -0.8)]
    clutter = rng.uniform([0.3, 0.3, -0.03], [0.8, 0.8, 0.2], (500, 3))
    pts = np.vstack([table, floor, clutter])
    T = fit_support_plane(pts, z_range=(-0.25, 0.05))
    assert abs(T[2, 3] + 0.03) < 0.005
    assert T[2, 2] > 0.999  # normal points up
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)


def test_backproject():
    """Pixels at the principal point back-project onto the optical axis."""
    K = intrinsics_matrix(10.0, 10.0, 2.0, 1.0)
    depth = np.zeros((3, 5))
    depth[1, 2] = 2.0
    assert np.allclose(backproject(depth, K), [[0.0, 0.0, 2.0]])
