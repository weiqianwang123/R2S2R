"""Tests for capture/droid.py, and capture/stereo.py's depth for a run's stereo
frames (the FoundationStereo job faked)."""

import json

import cv2
import numpy as np
from conftest import CLOSE_STEP, NUM_STEPS, SERIALS, SIZE
from scipy.spatial.transform import Rotation

from r2s2r.capture import stereo
from r2s2r.capture.droid import (
    ROLES,
    align_steps_to_video,
    load_droid_episode,
    pose6d_to_matrix,
    static_step_range,
)
from r2s2r.pipeline.workspace import Workspace
from r2s2r.structs import Capture, read_depth


def test_pose6d_matches_droid_convention():
    """DROID poses are extrinsic xyz Euler angles plus a translation."""
    pose = [0.1, 0.2, 0.3, 0.4, -0.5, 0.6]
    T = pose6d_to_matrix(pose)
    assert np.allclose(T[:3, 3], pose[:3])
    assert np.allclose(T[:3, :3], Rotation.from_euler("xyz", pose[3:]).as_matrix())


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
    from one exterior camera (ext2) and the wrist camera."""
    episode, calib = droid_episode
    out = tmp_path / "capture"
    default = load_droid_episode(episode, tmp_path / "default", calib, stride=2)
    assert set(default.cameras) == {SERIALS["ext2"], SERIALS["wrist"]}
    capture = load_droid_episode(
        episode, out, calib_dir=calib, roles=("ext1", "wrist"), stride=2
    )

    assert capture.instruction == "put the block in the bowl"
    assert capture.static_steps == (0, CLOSE_STEP)
    assert set(capture.cameras) == {SERIALS["ext1"], SERIALS["wrist"]}
    # Frames over the whole episode; the static period is the part reconstruction uses.
    assert len(capture.frames) == 2 * len(range(0, NUM_STEPS, 2))
    traj = capture.trajectory
    assert traj is not None and list(traj.steps) == list(range(NUM_STEPS))
    assert np.allclose(np.diff(traj.times), 1 / 15)  # no robot timestamps: 15 Hz
    assert np.allclose(traj.joint_positions[3], np.arange(7.0))
    assert traj.gripper_position[CLOSE_STEP] == 0.5

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
    assert reloaded.trajectory is not None
    assert np.allclose(reloaded.trajectory.times, traj.times)
    assert np.allclose(reloaded.cameras[ext1.serial].K, ext1.K)


def test_a_run_gets_stereo_depth_for_its_static_frames(
    droid_episode, tmp_path, monkeypatch
):
    """A new run computes FoundationStereo depth (at half resolution, scaled back) for
    the static frames of its stereo cameras, in its copy of the capture only."""
    episode, calib = droid_episode
    capture = load_droid_episode(episode, tmp_path / "capture", calib, stride=2)
    jobs = []

    def job(script, spec, workdir, env):
        jobs.append((script, spec, workdir, env))
        (tmp_path / "run/cache/jobs/stereo_depth").mkdir(parents=True)
        depth = []
        for k, pair in enumerate(spec["pairs"]):
            h, w = cv2.imread(pair["left"]).shape[:2]
            depth.append(str(tmp_path / f"{k}.npy"))
            np.save(depth[-1], np.full((h // 2, w // 2), 1.25, np.float32))
        return {"depth": depth}

    monkeypatch.setattr(stereo, "run_env_job", job)
    ws = Workspace.create(capture, tmp_path / "run", "fixed")
    assert len(jobs) == 1
    script, spec, workdir, env = jobs[0]
    assert script == "stereo_job.py" and env == "simfoundry"
    assert workdir == (tmp_path / "run/cache/jobs").resolve()
    static = [f for f in capture.frames if f.step < CLOSE_STEP]
    assert len(spec["pairs"]) == len(static) == 8
    assert (
        spec["pairs"][0]["baseline"]
        == capture.cameras[static[0].camera].stereo_baseline
    )
    for frame in ws.capture.frames:
        if frame.step >= CLOSE_STEP:
            assert frame.depth_image is None
            continue
        depth = read_depth(ws.capture.root / frame.depth_image)
        cam = ws.capture.cameras[frame.camera]
        assert depth.shape == (cam.height, cam.width) and np.allclose(depth, 1.25)
    assert "FoundationStereo" in ws.capture.metadata["depth"]
    assert len(ws.reconstructable()) == 8
    rows = json.loads((tmp_path / "run/inputs/frames.json").read_text())
    assert sum(r["depth"] for r in rows) == 8
    assert all(f.depth_image is None for f in Capture.load(capture.root).frames)
    assert stereo.stereo_frames(ws.capture) == []  # nothing left to do
