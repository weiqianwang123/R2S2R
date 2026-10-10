"""Tests for structs.py: frame selection across cameras, static camera poses, depth
images, scenes that move with their assets."""

import json
import shutil
from dataclasses import asdict, replace

import numpy as np
import pytest

from r2s2r.structs import (
    CameraSpec,
    Capture,
    FrameRecord,
    JointDynamics,
    ObjectSpec,
    SceneSpec,
    read_depth,
    write_depth,
)


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
    return Capture("c", "test", "fr3_robotiq", "", cameras, frames, static, root=None)


def test_budget_is_shared_between_cameras():
    """Cameras split the budget; one with few frames leaves the rest to others."""
    capture = _capture({"a": ("ext1", 20), "b": ("wrist", 20), "c": ("ext2", 1)})
    picked = capture.select_frames(max_frames=9)
    counts = {s: sum(f.camera == s for f in picked) for s in "abc"}
    assert counts == {"a": 4, "b": 4, "c": 1}
    steps = [f.step for f in picked if f.camera == "a"]
    assert steps[0] == 0 and steps[-1] == 19  # spread over the whole stream


def test_selection_keeps_to_the_static_window():
    """Frames outside the static window are skipped; ``keep`` filters the others."""
    capture = _capture({"a": ("ext1", 10), "b": ("wrist", 10)}, static=(2, 6))
    picked = capture.select_frames()
    assert [f.step for f in picked if f.camera == "b"] == [2, 3, 4, 5]
    assert [f.step for f in capture.select_frames(max_frames=2)] == [2, 2]
    only_even = capture.select_frames(keep=lambda f: f.step % 2 == 0)
    assert {f.step for f in only_even} == {2, 4}


def test_frames_by_camera_and_step():
    """A frame is found by camera serial or role and step; the static window is
    half-open."""
    capture = _capture({"a": ("ext1", 3), "b": ("wrist", 3)}, static=(1, 3))
    assert capture.frame("wrist", 2) is capture.frame("b", 2)
    assert capture.frame("b", 2).camera == "b" and capture.frame("b", 2).step == 2
    with pytest.raises(KeyError):
        capture.frame("ext1", 7)
    assert [capture.in_static(step) for step in range(4)] == [False, True, True, False]


def test_depth_png_round_trip(tmp_path):
    """Depth comes back to the millimetre; out-of-range values are clipped."""
    depth = np.array([[0.0, 0.2345], [1.5, 70.0]])
    write_depth(tmp_path / "d.png", depth)
    back = read_depth(tmp_path / "d.png")
    assert back.dtype == np.float32
    assert np.allclose(back, [[0.0, 0.2345], [1.5, 65.535]], atol=5e-4)


def test_scene_asset_paths_are_relative_on_disk(tmp_path):
    """A scene is saved with its asset paths relative to it, so a run moves as a whole;
    in memory they are absolute."""
    urdf = tmp_path / "run/s4_scene/scene/objects/box/box.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text("<robot/>")
    usd = tmp_path / "run/s5_settle/scene/objects/box/box.usd"
    scene = SceneSpec(
        "t",
        "fr3_robotiq",
        [ObjectSpec("box", "box", str(urdf), np.eye(4), usd=str(usd))],
        np.eye(4),
        {},
        "c",
        0,
        np.zeros(7),
    )
    scene.save(tmp_path / "run/s4_scene/scene")
    scene.save(tmp_path / "run/s5_settle/scene")  # a later stage refers to them
    path = tmp_path / "run/s5_settle/scene/scene.json"
    saved = json.loads(path.read_text())
    assert saved["objects"][0]["asset_path"] == (
        "../../s4_scene/scene/objects/box/box.urdf"
    )
    assert saved["objects"][0]["usd"] == "objects/box/box.usd"
    shutil.move(tmp_path / "run", tmp_path / "moved")
    loaded = SceneSpec.load(tmp_path / "moved/s5_settle/scene")
    moved = tmp_path / "moved/s4_scene/scene/objects/box/box.urdf"
    assert loaded.objects[0].asset_path == str(moved.resolve())
    assert loaded.objects[0].usd == str(
        (tmp_path / "moved/s5_settle/scene/objects/box/box.usd").resolve()
    )


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
        "embodiment": "fr3_robotiq",
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


def test_a_scene_keeps_its_joints_dynamics(tmp_path):
    """An articulated object's joints' dynamics come back as written; a joint without
    any is free; a scene saved without them loads; a cloth's material comes back."""
    sprung = JointDynamics(damping=0.02, friction=0.05, stiffness=0.3, rest=-0.1)
    box = ObjectSpec(
        "box",
        "box",
        str(tmp_path / "box.urdf"),
        np.eye(4),
        joints={"hinge": -0.8, "slide": 0.0},
        joint_dynamics={"hinge": sprung},
    )
    scene = SceneSpec("t", "fr3_robotiq", [box], np.eye(4), {}, "c", 0, np.zeros(7))
    scene.save(tmp_path / "scene")
    (loaded,) = SceneSpec.load(tmp_path / "scene").objects
    assert loaded.dynamics("hinge") == sprung
    assert loaded.dynamics("slide") == JointDynamics()
    path = tmp_path / "scene/scene.json"
    saved = json.loads(path.read_text())
    saved["objects"][0].pop("joint_dynamics")
    path.write_text(json.dumps(saved))
    assert SceneSpec.load(tmp_path / "scene").objects[0].joint_dynamics is None
    material = {"thickness": 0.002, "youngs_modulus": 5e5, "poissons_ratio": 0.3}
    saved["objects"][0]["cloth"] = material
    path.write_text(json.dumps(saved))
    assert SceneSpec.load(tmp_path / "scene").objects[0].cloth == material


def test_an_objects_ranges_are_kept_and_drawn_from(tmp_path):
    """The ranges come back as written; a draw lies within them, the same for the
    same seed, and leaves what has no range as estimated."""
    box = ObjectSpec(
        "box",
        "box",
        str(tmp_path / "box.urdf"),
        np.eye(4),
        mass=0.3,
        friction=0.6,
        joints={"hinge": 0.0},
        joint_dynamics={"hinge": JointDynamics(damping=0.01, stiffness=0.2)},
        ranges={"mass": (0.2, 0.45), "hinge.friction": (0.0, 0.05)},
    )
    scene = SceneSpec("t", "fr3_robotiq", [box], np.eye(4), {}, "c", 0, np.zeros(7))
    scene.save(tmp_path / "scene")
    (loaded,) = SceneSpec.load(tmp_path / "scene").objects
    assert loaded.ranges == box.ranges
    draws = [loaded.sample(np.random.default_rng(seed)) for seed in range(20)]
    assert all(0.2 <= d.mass <= 0.45 for d in draws)
    assert all(0.0 <= d.dynamics("hinge").friction <= 0.05 for d in draws)
    assert len({round(d.mass, 6) for d in draws}) == 20  # they differ
    assert all(d.friction == 0.6 and d.dynamics("hinge").damping == 0.01 for d in draws)
    assert all(d.dynamics("hinge").stiffness == 0.2 for d in draws)
    again = loaded.sample(np.random.default_rng(3))
    assert (
        again.mass == draws[3].mass and again.joint_dynamics == draws[3].joint_dynamics
    )
    plain = replace(loaded, ranges=None)
    assert plain.sample(np.random.default_rng(0)) is plain
