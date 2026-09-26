"""Tests for structs.py: frame selection across cameras, static camera poses."""

import json
from dataclasses import asdict

import numpy as np
import pytest

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
                    joint_positions=np.zeros(7),
                    gripper_position=0.0,
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


def test_a_static_cameras_pose_is_given_once(tmp_path):
    """Frames of a static camera take its pose; a moving camera's frames need theirs."""
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.2, 0.6]
    cameras = {
        "ext": CameraSpec("ext", "ext1", 4, 3, np.eye(3), T_base_cam=T),
        "wrist": CameraSpec("wrist", "wrist", 4, 3, np.eye(3), is_static=False),
    }
    frames = [
        {
            "step": 0,
            "camera": "ext",
            "left_image": "0.png",
            "right_image": None,
            "joint_positions": [0.0] * 7,
            "gripper_position": 0.0,
        }
    ]
    payload = {
        "name": "c",
        "source": "rig",
        "embodiment": "franka_panda",
        "instruction": "",
        "cameras": {k: asdict(v) for k, v in cameras.items()},
        "frames": frames,
        "static_steps": [0, 1],
    }
    (tmp_path / "capture.json").write_text(json.dumps(_jsonable(payload)))
    capture = Capture.load(tmp_path)
    (frame,) = capture.frames
    assert np.allclose(frame.T_base_cam, T) and frame.joint_positions.shape == (7,)

    payload["frames"] = [{**frames[0], "camera": "wrist"}]
    (tmp_path / "capture.json").write_text(json.dumps(_jsonable(payload)))
    with pytest.raises(ValueError, match="no T_base_cam"):
        Capture.load(tmp_path)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value
