"""Tests for structs.py: frame selection across cameras."""

import numpy as np

from r2s2r.structs import CameraSpec, Capture, FrameRecord


def _capture(frames_per_camera, static=(0, 100)):
    cameras, frames = {}, []
    for serial, (role, n) in frames_per_camera.items():
        cameras[serial] = CameraSpec(serial, role, 4, 3, np.eye(3))
        for step in range(n):
            frames.append(
                FrameRecord(
                    step,
                    serial,
                    f"{serial}/{step}.png",
                    None,
                    np.eye(4),
                    np.zeros(7),
                    0.0,
                )
            )
    return Capture("c", "test", "franka_panda", "", cameras, frames, static, root=None)


def test_budget_is_shared_between_cameras():
    """Cameras split the budget; one with few frames leaves the rest to others."""
    capture = _capture({"a": ("ext1", 20), "b": ("wrist", 20), "c": ("ext2", 1)})
    picked = capture.select_frames(max_frames=9)
    counts = {s: sum(f.camera == s for f in picked) for s in "abc"}
    assert counts == {"a": 4, "b": 4, "c": 1}
    steps = [f.step for f in picked if f.camera == "a"]
    assert steps[0] == 0 and steps[-1] == 19  # spread over the whole stream


def test_cameras_by_role_or_serial_and_static_window():
    """Cameras are named either way; frames outside the static window are skipped."""
    capture = _capture({"a": ("ext1", 10), "b": ("wrist", 10)}, static=(2, 6))
    picked = capture.select_frames(["wrist"])
    assert {f.camera for f in picked} == {"b"}
    assert [f.step for f in picked] == [2, 3, 4, 5]
    assert capture.select_frames(["a"], max_frames=1)[0].step == 2
    only_even = capture.select_frames(keep=lambda f: f.step % 2 == 0)
    assert {f.step for f in only_even} == {2, 4}
