"""Tests for io/droid.py."""

import cv2
import numpy as np
from conftest import CLOSE_STEP, SERIALS, SIZE

from r2s2r.io.droid import (
    ROLES,
    align_steps_to_video,
    load_droid_episode,
    static_step_range,
)
from r2s2r.structs import Capture
from r2s2r.transforms import pose6d_to_matrix


def test_static_step_range():
    """The static part ends when the gripper first closes."""
    assert static_step_range(np.array([0.0, 0.0, 0.3, 0.8]), 0.05) == (0, 2)
    assert static_step_range(np.zeros(5), 0.05) == (0, 5)


def test_align_steps_to_video_removes_latency():
    """A constant video latency must not shift steps onto the next frame."""
    steps = np.arange(10) * 66
    assert np.array_equal(align_steps_to_video(steps, steps + 41), np.arange(10))
    # A dropped frame: later steps still land on the frame closest in time.
    video = np.delete(steps + 41, 5)
    assert align_steps_to_video(steps, video)[6] == 5


def test_load_droid_episode(droid_episode, tmp_path):
    """Frames, calibration and robot state end up in a reloadable capture; by default
    from one exterior camera and the wrist camera."""
    episode, calib = droid_episode
    out = tmp_path / "capture"
    capture = load_droid_episode(episode, out, calib_dir=calib, stride=2)

    assert capture.instruction == "put the block in the bowl"
    assert capture.static_steps == (0, CLOSE_STEP)
    assert set(capture.cameras) == {SERIALS["ext1"], SERIALS["wrist"]}
    assert len(capture.frames) == 2 * len(range(0, CLOSE_STEP, 2))

    ext1 = capture.camera_by_role("ext1")
    assert ext1.is_static and ext1.T_base_cam is not None
    # ext1 has an improved calibration, ext2 falls back to trajectory.h5.
    assert np.allclose(ext1.T_base_cam, pose6d_to_matrix([1.0, 0.1, 0.5, np.pi, 0, 0]))
    every = load_droid_episode(episode, tmp_path / "all", calib, roles=ROLES, stride=2)
    assert set(every.cameras) == set(SERIALS.values())
    assert every.metadata["calibration_source"][SERIALS["ext2"]] == "trajectory.h5"
    assert np.isclose(ext1.stereo_baseline, 0.12)
    wrist = capture.camera_by_role("wrist")
    assert not wrist.is_static and wrist.T_base_cam is None
    assert (wrist.width, wrist.height) == SIZE["wrist"]

    frame = capture.frames_of(ext1.serial)[0]
    left = cv2.imread(str(out / frame.left_image))
    right = cv2.imread(str(out / frame.right_image))
    assert left.shape[:2] == (SIZE["ext1"][1], SIZE["ext1"][0])
    assert left[..., 2].mean() > 200 and right[..., 0].mean() > 200  # red | blue
    assert np.allclose(frame.joint_positions, np.arange(7.0))

    reloaded = Capture.load(out)
    assert reloaded.name == capture.name
    assert np.allclose(reloaded.cameras[ext1.serial].K, ext1.K)
