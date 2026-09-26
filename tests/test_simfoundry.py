"""Tests for reconstruct/simfoundry.py (no SimFoundry run needed)."""

# pylint: disable=protected-access

import json
import shutil

import cv2
import numpy as np
import pytest

from r2s2r.io.droid import load_droid_episode
from r2s2r.reconstruct import make_backend
from r2s2r.reconstruct.simfoundry import (
    FRAME_MAP_FILENAME,
    SimFoundryBackend,
    _frame_selection,
)
from r2s2r.structs import CameraSpec, Capture, FrameRecord, ObjectSpec, SceneSpec
from r2s2r.transforms import (
    intrinsics_matrix,
    make_transform,
    pos_quat_to_matrix,
    quat_wxyz_to_xyzw,
)


@pytest.fixture(name="capture")
def fixture_capture(droid_episode, tmp_path):
    """A capture built from the synthetic episode."""
    episode, calib = droid_episode
    return load_droid_episode(episode, tmp_path / "capture", calib_dir=calib, stride=2)


def _intrinsic_file(path):
    k_line, baseline = path.read_text().splitlines()
    return np.array(k_line.split(), float).reshape(3, 3), float(baseline)


def test_prepare_inputs(capture, tmp_path):
    """Stereo pairs, intrinsics and the frame map land where stage 2 reads them."""
    backend = SimFoundryBackend(cameras=("ext1",), max_frames=3, mask_robot=False)
    s1_dir = backend.prepare_inputs(capture, tmp_path / "work")
    cam = capture.camera_by_role("ext1")

    frame_map = json.loads((s1_dir / FRAME_MAP_FILENAME).read_text())
    assert [m["step"] for m in frame_map] == [0, 4, 6]  # 3 spread over 0, 2, 4, 6
    assert {m["depth"] for m in frame_map} == {"stereo"}
    for m in frame_map:
        assert (s1_dir / f"image_{m['index']}_l.png").exists()
        assert (s1_dir / f"image_{m['index']}_r.png").exists()
        K, baseline = _intrinsic_file(s1_dir / f"image_{m['index']}_intrinsic.txt")
        assert np.allclose(K, cam.K) and np.isclose(baseline, cam.stereo_baseline)
    assert (s1_dir / "intrinsic.txt").exists()


def test_prepare_inputs_mixes_cameras(capture, tmp_path):
    """Exterior and wrist candidates share the budget, each with its own intrinsics."""
    backend = SimFoundryBackend(
        cameras=("ext1", "wrist"), max_frames=4, mask_robot=False
    )
    s1_dir = backend.prepare_inputs(capture, tmp_path / "work")
    frame_map = json.loads((s1_dir / FRAME_MAP_FILENAME).read_text())
    by_camera = {m["camera"] for m in frame_map}
    assert by_camera == {
        capture.camera_by_role("ext1").serial,
        capture.camera_by_role("wrist").serial,
    }
    assert len(frame_map) == 4
    for m in frame_map:
        K, _ = _intrinsic_file(s1_dir / f"image_{m['index']}_intrinsic.txt")
        assert np.allclose(K, capture.cameras[m["camera"]].K)


def test_build_command(capture, tmp_path):
    """The orchestrator runs the submodule in stereo mode with FoundationStereo."""
    backend = make_backend("simfoundry", stages=("2", "3"))
    assert isinstance(backend, SimFoundryBackend)
    cmd, env, cwd = backend.build_command(capture, tmp_path)
    assert cmd[cmd.index("--include") + 1] == "2,3"
    assert cmd[cmd.index("--input-mode") + 1] == "stereo"
    assert "--detect-articulation" not in cmd  # every object stays rigid
    assert "s2_depth.backend=fs" in cmd and "s3_ground.use_fs=true" in cmd
    assert f"scene_name={capture.name}" in cmd
    assert env["PYTHONPATH"].split(":")[0] == str(cwd)


def test_run_requires_gemini(capture, tmp_path, monkeypatch):
    """Stages that call Gemini fail fast without credentials."""
    for key in ("GCLOUD_PROJECT", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="Gemini"):
        SimFoundryBackend(vlm_backend="gemini").run(
            capture, tmp_path, stages=("2", "3")
        )


def test_codex_backend_env(capture, tmp_path):
    """The Codex backend is selected through environment variables."""
    backend = SimFoundryBackend(codex_reasoning="high")
    _, env, _ = backend.build_command(capture, tmp_path)
    assert env["SIMFOUNDRY_VLM_BACKEND"] == "codex"
    assert env["SIMFOUNDRY_CODEX_MODEL"] == "gpt-6-astra"
    assert env["SIMFOUNDRY_CODEX_REASONING"] == "high"


def test_parse_reanchors_to_robot_base(capture, tmp_path):
    """Object poses from SimFoundry's plane frame are moved into the base frame."""
    backend = SimFoundryBackend(cameras=("ext1",), max_frames=3, mask_robot=False)
    backend.prepare_inputs(capture, tmp_path)
    scene_dir = backend.scene_dir(capture, tmp_path)
    idx = 1  # SimFoundry picked the second candidate frame
    frame_map = json.loads((scene_dir / "s1_zed" / FRAME_MAP_FILENAME).read_text())
    ref_step = frame_map[idx]["step"]

    T_world_cam = make_transform(np.eye(3), [0.0, 0.0, 0.7])
    T_world_obj = pos_quat_to_matrix([0.1, 0.2, 0.05], [1.0, 0.0, 0.0, 0.0])
    for sub in ("s3_ground", "s4_frame", "s11_sim", "s12_physics"):
        (scene_dir / sub).mkdir(parents=True)
    (scene_dir / "s3_ground" / "frame_selection.json").write_text(
        json.dumps({"selected_idx": idx, "decided_by": "heuristic"})
    )
    np.save(scene_dir / "s4_frame" / f"image_{idx}_cam2world.npy", T_world_cam)
    (scene_dir / "s11_sim" / "scene_objects_info.json").write_text(
        json.dumps({"0": {"name": "mug_0", "category": "mug", "model": "m0"}})
    )
    urdf = scene_dir / "s11_sim" / "objects" / "mug" / "m0" / "urdf" / "m0.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text(
        '<robot name="m0"><link name="base"><inertial>'
        '<mass value="0.3"/></inertial></link></robot>'
    )
    pos, quat_wxyz = T_world_obj[:3, 3], np.array([1.0, 0.0, 0.0, 0.0])
    (scene_dir / "s12_physics" / "pb_scene_poses.json").write_text(
        json.dumps({"mug_0": [pos.tolist(), quat_wxyz_to_xyzw(quat_wxyz).tolist()]})
    )

    scene = backend.parse(capture, tmp_path)
    ref = next(
        f for f in capture.frames_of(scene.reference_camera) if f.step == ref_step
    )
    T_base_world = ref.T_base_cam @ np.linalg.inv(T_world_cam)
    assert scene.reference_step == ref_step
    assert np.allclose(scene.T_base_support, T_base_world)
    (obj,) = scene.objects
    assert np.allclose(obj.T_base_obj, T_base_world @ T_world_obj)
    assert obj.mass == pytest.approx(0.3)

    scene.save(tmp_path / "scene")
    reloaded = SceneSpec.load(tmp_path / "scene")
    assert np.allclose(reloaded.objects[0].T_base_obj, obj.T_base_obj)


def _rgbd_capture(root):
    """A mono RGB-D camera with three frames of constant depth."""
    (root / "frames").mkdir(parents=True)
    width, height = 128, 96
    frames = []
    for step in range(3):
        cv2.imwrite(
            str(root / f"frames/{step}_rgb.png"),
            np.full((height, width, 3), 50 * step, np.uint8),
        )
        cv2.imwrite(
            str(root / f"frames/{step}_depth.png"),
            np.full((height, width), 1000 + step, np.uint16),
        )
        frames.append(
            FrameRecord(
                step,
                "cam",
                f"frames/{step}_rgb.png",
                None,
                np.eye(4),
                np.zeros(7),
                0.0,
                depth_image=f"frames/{step}_depth.png",
            )
        )
    K = intrinsics_matrix(100.0, 100.0, 63.5, 47.5)
    cam = CameraSpec("cam", "ext1", width, height, K, T_base_cam=np.eye(4))
    return Capture(
        "c", "mujoco", "franka_panda", "", {"cam": cam}, frames, (0, 3), root
    )


class _HalfMasker:
    """Stands in for RobotMasker: the robot covers the left half of every image."""

    def mask_depth(self, depth, *robot_state):  # pylint: disable=unused-argument
        """Zero the left half."""
        mask = np.zeros(depth.shape, bool)
        mask[:, : depth.shape[1] // 2] = True
        out = depth.astype(np.float32, copy=True)
        out[mask] = 0.0
        return out, mask


def test_prepare_inputs_rgbd(tmp_path):
    """Measured depth is written in FoundationStereo's layout at its scale."""
    capture = _rgbd_capture(tmp_path / "capture")
    backend = SimFoundryBackend(cameras=("ext1",), mask_robot=False)
    backend.prepare_inputs(capture, tmp_path / "work")
    scene_dir = backend.scene_dir(capture, tmp_path / "work")
    fs = scene_dir / "s2_fs"
    depth = np.load(fs / "image_1_depth_meter.npy")
    assert depth.shape == (48, 64) and np.allclose(depth, 1.001)
    K = np.load(fs / "image_1_K.npy")
    assert np.isclose(K[0, 0], 50.0) and np.isclose(K[0, 2], 31.5)
    assert np.load(fs / "image_2_rgb.npy").shape == (48, 64, 3)
    s1 = scene_dir / "s1_zed"
    assert (s1 / "image_0_l.png").exists() and not (s1 / "image_0_r.png").exists()
    frame_map = json.loads((s1 / FRAME_MAP_FILENAME).read_text())
    assert {m["depth"] for m in frame_map} == {"rgbd"}


def test_rgbd_depth_is_robot_masked(tmp_path, monkeypatch):
    """The robot's pixels are invalid in the depth SimFoundry reads."""
    capture = _rgbd_capture(tmp_path / "capture")
    backend = SimFoundryBackend(cameras=("ext1",))
    monkeypatch.setattr(backend, "_masker", lambda capture: _HalfMasker())
    backend.prepare_inputs(capture, tmp_path / "work")
    fs = backend.scene_dir(capture, tmp_path / "work") / "s2_fs"
    depth = np.load(fs / "image_0_depth_meter.npy")
    assert np.all(depth[:, :32] == 0) and np.all(depth[:, 32:] > 0)
    assert cv2.imread(str(fs / "image_0_robot_mask.png"), cv2.IMREAD_GRAYSCALE).any()


def test_stage2_depth_is_robot_masked_once(capture, tmp_path, monkeypatch):
    """FoundationStereo depth is masked from a kept raw copy, so reruns are safe."""
    backend = SimFoundryBackend(cameras=("ext1",), max_frames=2)
    monkeypatch.setattr(backend, "_masker", lambda capture: None)
    backend.prepare_inputs(capture, tmp_path)
    fs = backend.scene_dir(capture, tmp_path) / "s2_fs"
    for i in range(2):  # what stage 2 would write
        np.save(fs / f"image_{i}_depth_meter.npy", np.ones((6, 8), np.float32))
        np.save(fs / f"image_{i}_K.npy", np.eye(3))
    monkeypatch.setattr(backend, "_masker", lambda capture: _HalfMasker())
    backend.mask_stage2_outputs(capture, tmp_path)
    backend.mask_stage2_outputs(capture, tmp_path)
    depth = np.load(fs / "image_0_depth_meter.npy")
    assert np.all(depth[:, :4] == 0) and np.all(depth[:, 4:] == 1)
    assert np.all(np.load(fs / "image_0_depth_meter_raw.npy") == 1)


def test_run_skips_stage2_for_rgbd(tmp_path, monkeypatch):
    """Stage 2 (stereo depth) is dropped when every candidate measured depth."""
    capture = _rgbd_capture(tmp_path / "capture")
    backend = SimFoundryBackend(cameras=("ext1",), mask_robot=False)
    backend.prepare_inputs(capture, tmp_path)
    calls = []
    monkeypatch.setattr(
        "r2s2r.reconstruct.simfoundry.subprocess.run",
        lambda cmd, **kwargs: calls.append(cmd),
    )
    backend.run(capture, tmp_path, stages=("2", "3"))
    assert len(calls) == 1
    assert calls[0][calls[0].index("--include") + 1] == "3"


def test_run_masks_between_stage2_and_3(capture, tmp_path, monkeypatch):
    """Stereo captures run stage 2 alone, then mask, then the rest."""
    backend = SimFoundryBackend(cameras=("ext1",), max_frames=2, mask_robot=False)
    backend.prepare_inputs(capture, tmp_path)
    events = []
    monkeypatch.setattr(
        "r2s2r.reconstruct.simfoundry.subprocess.run",
        lambda cmd, **kwargs: events.append(cmd[cmd.index("--include") + 1]),
    )
    monkeypatch.setattr(
        backend, "mask_stage2_outputs", lambda *args: events.append("mask")
    )
    backend.run(capture, tmp_path, stages=("2", "3", "4"))
    assert events == ["2", "mask", "3,4"]
    events.clear()
    backend.run(capture, tmp_path, stages=("2",))  # depth only: masked all the same
    assert events == ["2", "mask"]


def test_an_empty_scene_is_rebuilt_from_other_frames(capture, tmp_path, monkeypatch):
    """No objects from one frame: stages 3-12 rerun pinned to other eligible frames,
    another camera's best first, until one gives objects; stage 1-2 outputs kept."""
    backend = SimFoundryBackend(
        cameras=("ext1", "ext2"), max_frames=4, mask_robot=False
    )
    backend.prepare_inputs(capture, tmp_path)
    scene_dir = backend.scene_dir(capture, tmp_path)
    frames = json.loads((scene_dir / "s1_zed" / FRAME_MAP_FILENAME).read_text())
    ext1 = capture.camera_by_role("ext1").serial
    first, second = [m["index"] for m in frames if m["camera"] == ext1]
    others = [m["index"] for m in frames if m["camera"] != ext1]
    scores = [
        {
            "idx": m["index"],
            "eligible": True,
            "score": 0.9 if m["camera"] == ext1 else 0.1,
        }
        for m in frames
    ]
    scores[others[1]]["score"] = 0.5  # the best frame of the other camera
    scores[others[0]]["eligible"] = False
    for sub in ("s2_fs", "s3_ground", "s5_scene", "s12_physics"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    (scene_dir / "s3_ground" / "frame_selection.json").write_text(
        json.dumps({"selected_idx": first, "scores": scores})
    )
    calls = []
    monkeypatch.setattr(
        "r2s2r.reconstruct.simfoundry.subprocess.run",
        lambda cmd, **kwargs: calls.append(cmd),
    )
    empty = SceneSpec("s", "droid_franka", [], np.eye(4), {}, ext1, 0, np.zeros(7))
    found = SceneSpec(
        "s",
        "droid_franka",
        [ObjectSpec("mug", "mug", "mug.urdf", np.eye(4))],
        np.eye(4),
        {},
        ext1,
        0,
        np.zeros(7),
    )
    parsed = iter([empty, found])
    monkeypatch.setattr(backend, "parse", lambda *args: next(parsed))
    scene = backend.retry_empty(capture, tmp_path)
    assert scene is found
    assert scene.provenance["retried"] == {
        "empty_frames": [first, others[1]],
        "frame": second,
    }
    assert [cmd[-1] for cmd in calls] == [
        f"s3_ground.img_idx={others[1]}",
        f"s3_ground.img_idx={second}",
    ]
    assert calls[0][calls[0].index("--include") + 1] == "3,4,5,6,7,8,10,11,12"
    assert (scene_dir / "s1_zed").exists() and (scene_dir / "s2_fs").exists()
    assert (
        not (scene_dir / "s3_ground").exists() and not (scene_dir / "s5_scene").exists()
    )

    # Pinned, stage 3 writes only that frame's floor info; parse reads the index there.
    (scene_dir / "s3_ground").mkdir()
    (scene_dir / "s3_ground" / f"image_{second}_floor_info.json").write_text("{}")
    assert _frame_selection(scene_dir)["selected_idx"] == second
    shutil.rmtree(scene_dir / "s3_ground")

    # Nothing eligible elsewhere: nothing to retry.
    (scene_dir / "s3_ground").mkdir()
    (scene_dir / "s3_ground" / "frame_selection.json").write_text(
        json.dumps({"selected_idx": first, "scores": scores[:1]})
    )
    assert backend.retry_empty(capture, tmp_path) is None
