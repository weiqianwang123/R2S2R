"""Tests for io/rgbd.py."""

import numpy as np

from r2s2r.io.rgbd import read_depth, write_depth


def test_depth_png_round_trip(tmp_path):
    """Depth comes back to the millimetre; out-of-range values are clipped."""
    depth = np.array([[0.0, 0.2345], [1.5, 70.0]])
    write_depth(tmp_path / "d.png", depth)
    back = read_depth(tmp_path / "d.png")
    assert back.dtype == np.float32
    assert np.allclose(back, [[0.0, 0.2345], [1.5, 65.535]], atol=5e-4)
