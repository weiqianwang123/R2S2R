"""Tests for io/rgbd.py's depth-step cache."""

import numpy as np

from r2s2r.io.rgbd import load_depth_steps, save_depth_steps
from r2s2r.structs import DepthView


def test_depth_steps_round_trip(tmp_path):
    """Views come back grouped by step, depth to the millimetre, colour dropped."""
    rng = np.random.default_rng(0)
    K, T = np.eye(3), np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    steps = {
        0: [DepthView("a", 0, rng.uniform(0.2, 2.0, (4, 5)), K, T)],
        7: [
            DepthView("a", 7, np.zeros((4, 5)), K, T, np.zeros((4, 5, 3), np.uint8)),
            DepthView("b", 7, np.full((4, 5), 1.2345), 2 * K, T),
        ],
    }
    save_depth_steps(tmp_path / "steps.npz", steps)
    back = load_depth_steps(tmp_path / "steps.npz")
    assert sorted(back) == [0, 7] and [v.camera for v in back[7]] == ["a", "b"]
    assert np.allclose(back[0][0].depth, steps[0][0].depth, atol=5e-4)
    assert np.allclose(back[7][1].depth, 1.2345, atol=5e-4)
    assert np.allclose(back[7][1].K, 2 * K)
    assert np.allclose(back[7][0].T_base_cam, T) and back[7][0].image is None
