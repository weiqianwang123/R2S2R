"""Tests for reconstruct/simfoundry.py (no SimFoundry run needed)."""

import json

import numpy as np
import pytest

from r2s2r.io.droid import load_droid_episode
from r2s2r.reconstruct import make_backend
from r2s2r.reconstruct.simfoundry import FRAME_MAP_FILENAME, SimFoundryBackend
from r2s2r.structs import SceneSpec
from r2s2r.transforms import make_transform, pos_quat_to_matrix, quat_wxyz_to_xyzw


@pytest.fixture(name="capture")
def fixture_capture(droid_episode, tmp_path):
    """A capture built from the synthetic episode."""
    episode, calib = droid_episode
    return load_droid_episode(episode, tmp_path / "capture", calib_dir=calib, stride=2)


def test_prepare_inputs(capture, tmp_path):
    """Stereo pairs, intrinsics and the frame map land where stage 2 reads them."""
    backend = SimFoundryBackend(camera_role="ext1", max_frames=3)
    s1_dir = backend.prepare_inputs(capture, tmp_path / "work")
    cam = capture.camera_by_role("ext1")

    frame_map = json.loads((s1_dir / FRAME_MAP_FILENAME).read_text())
    assert [m["step"] for m in frame_map] == [0, 4, 6]  # 3 spread over 0, 2, 4, 6
    for m in frame_map:
        assert (s1_dir / f"image_{m['index']}_l.png").exists()
        assert (s1_dir / f"image_{m['index']}_r.png").exists()
    k_line, baseline = (s1_dir / "intrinsic.txt").read_text().splitlines()
    assert np.allclose(np.array(k_line.split(), float).reshape(3, 3), cam.K)
    assert np.isclose(float(baseline), cam.stereo_baseline)


def test_build_command(capture, tmp_path):
    """The orchestrator runs the submodule in stereo mode with FoundationStereo."""
    backend = make_backend("simfoundry", stages=("2", "3"))
    assert isinstance(backend, SimFoundryBackend)
    cmd, env, cwd = backend.build_command(capture, tmp_path)
    assert cmd[cmd.index("--include") + 1] == "2,3"
    assert cmd[cmd.index("--input-mode") + 1] == "stereo"
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
    backend = SimFoundryBackend(camera_role="ext1", max_frames=3)
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
