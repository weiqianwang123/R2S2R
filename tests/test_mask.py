"""Tests for robots/mask.py's occlusion rule (rendering stubbed out)."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.robots.mask import NO_ROBOT, MaskConfig, RobotMasker  # noqa: E402


def _masker(robot_depth):
    masker = RobotMasker.__new__(RobotMasker)  # no model: rendering is stubbed
    masker.config = MaskConfig(dilate_frac=0.02, margin=0.03)
    masker.robot_depth = lambda *args: robot_depth
    return masker


def test_robot_pixels_are_removed_objects_in_front_kept():
    """Behind or on the robot: removed.

    Clearly in front of it: kept.
    """
    robot = np.full((60, 80), NO_ROBOT, np.float32)
    robot[20:40, 30:50] = 1.0  # the arm, 1 m away
    observed = np.full((60, 80), 1.5, np.float32)  # the table behind it
    observed[20:40, 30:50] = 1.0  # the arm itself
    observed[25:30, 30:35] = 0.8  # an object held in front of the arm
    depth, mask = _masker(robot).mask_depth(
        observed, np.eye(3), np.eye(4), np.zeros(7), 0.0
    )
    assert np.all(depth[32:38, 40:48] == 0)  # the arm
    assert np.all(depth[25:30, 30:35] == 0.8)  # in front: kept
    assert mask[20, 29] and not mask[0, 0]  # silhouette grown, far pixels untouched
    assert np.all(depth[0:10] == 1.5)
