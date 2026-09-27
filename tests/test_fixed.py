"""Tests for the fixed method (pipeline/fixed/): SimFoundry's candidates, its command,
the stage products from fake SimFoundry outputs (a fake orchestrator; the refinement
steps and previews faked where they need a GL context), the empty-scene retry, and the
CLI that makes the method."""

import json
import os
import sys
from dataclasses import replace

import cv2
import numpy as np
import pytest
import trimesh
from conftest import RGBD_K, rgbd_capture

from r2s2r import cli
from r2s2r.pipeline import fixed
from r2s2r.pipeline.fixed import FixedMethod
from r2s2r.pipeline.fixed import simfoundry as sf
from r2s2r.pipeline.fixed.simfoundry import SimFoundryConfig
from r2s2r.pipeline.run import run
from r2s2r.structs import SceneSpec
from r2s2r.transforms import (
    invert,
    look_at,
    make_transform,
    matrix_to_pos_quat,
    quat_wxyz_to_xyzw,
)
from r2s2r.workspace import Workspace

TARGET = (0.5, 0.05, 0.0)
CAMERAS = {
    "c1": ("ext1", look_at((0.95, 0.35, 0.45), TARGET)),
    "c2": ("wrist", look_at((0.45, -0.4, 0.4), TARGET)),
}
# SimFoundry's world in the robot base frame (its support plane's frame): not the
# identity, so the tests see which way the poses are carried into the base frame.
_YAW = np.radians(30)
T_BASE_WORLD = make_transform(
    [[np.cos(_YAW), -np.sin(_YAW), 0], [np.sin(_YAW), np.cos(_YAW), 0], [0, 0, 1]],
    [0.2, -0.1, 0.0],
)
# What the fake SimFoundry finds, in the robot base frame.
OBJECTS = [
    ("red mug", "m0", make_transform(np.eye(3), [0.5, 0.0, 0.04])),
    ("red mug", "m1", make_transform(np.eye(3), [0.6, 0.1, 0.04])),
    ("box", "b0", make_transform(np.eye(3), [0.4, -0.1, 0.03])),
]


class _HalfMasker:
    """Stands in for the robot masker: the robot covers the left half of every image."""

    def mask_depth(self, depth, *robot_state):  # pylint: disable=unused-argument
        """Zero the left half."""
        mask = np.zeros(depth.shape, bool)
        mask[:, : depth.shape[1] // 2] = True
        return np.where(mask, 0.0, depth).astype(np.float32), mask

    def close(self):
        """Nothing to free."""


@pytest.fixture(name="capture")
def fixture_capture(tmp_path, monkeypatch):
    """A two-camera RGB-D capture; runs on it cut out a fake robot."""
    monkeypatch.setattr(Workspace, "masker", property(lambda self: _HalfMasker()))
    return rgbd_capture(tmp_path / "capture", CAMERAS)


def _write_object(urdf_dir, model):
    """A SimFoundry stage 11 object: one link, a textured visual mesh, a scaled
    collision hull."""
    visual = urdf_dir / "shape" / "visual" / f"{model}.obj"
    for d in (visual.parent, urdf_dir / "shape" / "collision", urdf_dir / "material"):
        d.mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=(0.08, 0.06, 0.08)).export(file_type="obj")
    visual.write_text(f"mtllib {model}.mtl\nusemtl material_0\n{box}")
    visual.with_suffix(".mtl").write_text(
        f"newmtl material_0\nKd 1 1 1\nmap_Kd ../../material/{model}_Kd.png\n"
    )
    cv2.imwrite(str(urdf_dir / "material" / f"{model}_Kd.png"), np.zeros((4, 4, 3)))
    trimesh.creation.box(extents=(1, 1, 1)).export(
        urdf_dir / "shape" / "collision" / f"{model}_collision_0.obj"
    )
    (urdf_dir / f"{model}.urdf").write_text(
        f'<robot name="{model}"><link name="link"><inertial><mass value="0.3"/>'
        "</inertial><visual><geometry>"
        f'<mesh filename="shape/visual/{model}.obj" scale="1. 1. 1."/></geometry>'
        '<origin xyz="0 0 0" rpy="0 0 0"/></visual><collision><geometry>'
        f'<mesh filename="shape/collision/{model}_collision_0.obj" '
        'scale="0.05 0.04 0.03"/></geometry></collision></link></robot>'
    )


def _simfoundry_outputs(run_, idx, pinned, objects):
    """What SimFoundry stages 3-12 leave, having rebuilt from candidate ``idx``."""
    scene_dir = run_.scene_dir
    ground = scene_dir / "s3_ground"
    ground.mkdir(parents=True)
    if pinned:
        (ground / f"image_{idx}_floor_info.json").write_text("{}")
    else:
        scores = [
            {"idx": i, "eligible": True, "score": 1.0 - 0.1 * i} for i in range(4)
        ]
        (ground / "frame_selection.json").write_text(
            json.dumps(
                {
                    "selected_idx": idx,
                    "decided_by": "codex",
                    "vlm_note": "sees every object",
                    "support": {"description": "a wooden table"},
                    "scores": scores,
                }
            )
        )
    frame = run_.frame_of(idx)
    (scene_dir / "s4_frame").mkdir()
    T_world_base = invert(T_BASE_WORLD)
    np.save(
        scene_dir / "s4_frame" / f"image_{idx}_cam2world.npy",
        T_world_base @ frame.T_base_cam,
    )
    info, poses = {}, {}
    for k, (category, model, T) in enumerate(objects):
        urdf_dir = scene_dir / "s11_sim" / "objects" / category / model / "urdf"
        _write_object(urdf_dir, model)
        info[str(k)] = {
            "name": f"iter_{k}",
            "category": category,
            "model": model,
            "friction": 0.4,
        }
        pos, quat = matrix_to_pos_quat(T_world_base @ T)
        poses[f"iter_{k}"] = [pos.tolist(), quat_wxyz_to_xyzw(quat).tolist()]
    (scene_dir / "s11_sim").mkdir(exist_ok=True)
    (scene_dir / "s11_sim" / "scene_objects_info.json").write_text(json.dumps(info))
    (scene_dir / "s12_physics").mkdir()
    (scene_dir / "s12_physics" / "pb_scene_poses.json").write_text(json.dumps(poses))


@pytest.fixture(name="fake_simfoundry")
def fixture_fake_simfoundry(monkeypatch):
    """SimFoundry's orchestrator replaced by a script that prints stage lines; its
    outputs are written when the command is made (from ``plan``: the candidate each
    call rebuilds from and the objects it finds). Returns the plan and the calls."""
    plan = {"runs": [(2, OBJECTS)]}
    calls = []

    def command(self, stages, extra=()):
        calls.append((stages, extra, (self.scene_dir / "s3_ground").exists()))
        idx, objects = plan["runs"][len(calls) - 1]
        _simfoundry_outputs(self, idx, bool(extra), objects)
        script = "print('[Stage 3] Segment ground plane'); print('[Stage 3] cmd: x')"
        return [sys.executable, "-c", script], dict(os.environ)

    monkeypatch.setattr(sf.SimFoundry, "command", command)
    return plan, calls


def _no_gl_refinement(monkeypatch):
    """Stage 3's checks and previews without GL: nothing dropped or turned, the
    support outline 0.8 x 0.6 m."""
    monkeypatch.setattr(fixed, "drop_unseen", lambda s, v: (s, {"objects": {}}))
    monkeypatch.setattr(
        fixed, "check_orientations", lambda s, v, vlm, image_dir: (s, {})
    )
    monkeypatch.setattr(
        fixed,
        "refine_scene",
        lambda s, v: (replace(s, support_extent=(0.8, 0.6)), {"views": len(v)}),
    )
    monkeypatch.setattr(
        fixed, "render_preview", lambda mesh, out, up: out.write_bytes(b"png")
    )


# ----------------------------------------------------------------------- inputs
def test_candidates_are_scaled_like_foundation_stereo(capture, tmp_path, monkeypatch):
    """Candidates of every camera, in the run's order; at most CANDIDATE_WIDTH wide,
    K in the pixel-centre convention; the robot cut out; written where SimFoundry
    stages 1 and 2 leave them."""
    ws = Workspace.create(capture, tmp_path / "run", "fixed", ["wrist", "ext1"])
    frames = sf.candidates(ws.capture, 4)
    assert [ws.frame_id(f) for f in frames] == [
        "wrist@0",
        "wrist@4",
        "ext1@0",
        "ext1@4",
    ]
    full = sf.candidate_view(ws, frames[0])
    assert full.depth.shape == (240, 320) and np.allclose(full.K, RGBD_K)
    monkeypatch.setattr(sf, "CANDIDATE_WIDTH", 160)
    half = sf.candidate_view(ws, frames[0])
    assert half.depth.shape == (120, 160) and half.image.shape == (120, 160, 3)
    assert np.isclose(half.K[0, 0], 150.0) and np.isclose(half.K[0, 2], 79.5)
    assert np.all(half.depth[:, :80] == 0) and np.allclose(half.depth[:, 80:], 0.8)

    run_ = sf.SimFoundry(
        ws.capture, tmp_path / "sf", SimFoundryConfig(), tmp_path / "log"
    )
    run_.prepare(ws, frames)
    s1, fs = run_.scene_dir / "s1_zed", run_.scene_dir / "s2_fs"
    assert cv2.imread(str(s1 / "image_3_l.png")).shape == (240, 320, 3)
    assert np.load(fs / "image_3_depth_meter.npy").shape == (120, 160)
    assert np.load(fs / "image_3_rgb.npy").shape == (120, 160, 3)
    assert np.allclose(np.load(fs / "image_3_K.npy"), half.K)
    assert ws.frame_id(run_.frame_of(2)) == "ext1@0"


def test_command_runs_the_submodule_with_codex(capture, tmp_path):
    """Stages 3-12 but 9, in the submodule; frame selection by Codex with the task;
    every VLM call through Codex."""
    cfg = SimFoundryConfig(codex_reasoning="high", overrides=["a=b"])
    capture = replace(capture, instruction='put the "café" mug away')
    run_ = sf.SimFoundry(capture, tmp_path / "sf", cfg, tmp_path / "log")
    cmd, env = run_.command(sf.STAGES, ("s3_ground.img_idx=2",))
    assert cmd[cmd.index("--include") + 1] == "3,4,5,6,7,8,10,11,12"
    assert f"root_dir={(tmp_path / 'sf').resolve()}" in cmd
    assert "s3_ground.frame_selection.mode=codex" in cmd
    # As Hydra reads it back: quotes escaped, the rest as written.
    assert 's3_ground.frame_selection.task="put the \\"café\\" mug away"' in cmd
    assert cmd[-2:] == ["a=b", "s3_ground.img_idx=2"]
    assert env["SIMFOUNDRY_VLM_BACKEND"] == "codex"
    assert (
        env["SIMFOUNDRY_CODEX_REASONING"] == "high" and env["PYTHONUNBUFFERED"] == "1"
    )
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(sf.SIMFOUNDRY_DIR)
    hybrid = sf.SimFoundry(
        capture, tmp_path, SimFoundryConfig(frame_selection="hybrid"), tmp_path / "l"
    )
    assert not any("frame_selection.task" in c for c in hybrid.command(sf.STAGES)[0])


# ----------------------------------------------------------------------- stages
def test_stages_2_3_4_from_simfoundry_outputs(
    capture, tmp_path, fake_simfoundry, monkeypatch
):
    """Stage 2 writes the frames, the support and SimFoundry's scene as it made it;
    stage 3 exports every object's mesh, texture and hulls; stage 4 assembles them
    with those hulls, SimFoundry's masses and frictions."""
    _, calls = fake_simfoundry
    _no_gl_refinement(monkeypatch)
    log = run(capture.root, tmp_path / "run", FixedMethod(), ("2", "3", "4"))
    root = tmp_path / "run"
    assert [log["stages"][k]["status"] for k in "234"] == ["done"] * 3
    assert log["stages"]["6"] == {"status": "skipped"} and "5" not in log["stages"]
    assert calls == [(sf.STAGES, (), False)]
    s2 = root / "s2_frames"
    out = json.loads((s2 / "output.json").read_text())
    assert out["frames"] == [
        "ext1@0",
        "ext1@1",
        "ext1@2",
        "ext1@3",
        "ext1@4",
        "wrist@0",
        "wrist@1",
        "wrist@2",
        "wrist@3",
        "wrist@4",
    ]
    assert out["selected"] == "ext1@2" and out["decided_by"] == "codex"
    assert out["frame_notes"] == {"ext1@2": "sees every object"}
    assert out["support"] == {"file": "support.json", "description": "a wooden table"}
    assert "retried" not in out and log["stages"]["2"]["selected"] == "ext1@2"
    support = json.loads((s2 / "support.json").read_text())
    assert np.allclose(support["T_base_support"], T_BASE_WORLD)
    assert "[Stage 3] Segment ground plane" in (s2 / "simfoundry.log").read_text()
    parsed = SceneSpec.load(s2 / "parsed")
    assert [o.name for o in parsed.objects] == ["red_mug", "red_mug_2", "box"]
    assert parsed.objects[0].asset_path.endswith(
        "m0/urdf/m0.urdf"
    )  # as SimFoundry made it
    assert np.allclose(parsed.objects[1].T_base_obj, OBJECTS[1][2])

    s3 = root / "s3_objects"
    spec = json.loads((s3 / "objects.json").read_text())
    assert (
        spec["support"]["extent"] == [0.8, 0.6] and spec["reference_frame"] == "ext1@2"
    )
    mug = spec["objects"][0]
    assert mug["mesh"] == "red_mug/m0.obj" and mug["up"] == "z" and mug["scale"] == 1.0
    assert (mug["mass"], mug["friction"]) == (0.3, 0.4)
    assert np.allclose(spec["support"]["T_base_support"], T_BASE_WORLD)
    assert np.allclose(spec["objects"][2]["T_base_obj"], OBJECTS[2][2])
    assert "map_Kd m0_Kd.png" in (s3 / "red_mug" / "m0.mtl").read_text()
    assert (s3 / "red_mug" / "m0_Kd.png").exists() and (
        s3 / "red_mug" / "preview.png"
    ).exists()
    (hull,) = mug["collision"]
    assert np.allclose(trimesh.load(s3 / hull).extents, (0.05, 0.04, 0.03))
    report = json.loads((s3 / "refinement.json").read_text())
    assert set(report) == {"existence", "orientation", "refinement"}

    s4 = root / "s4_scene"
    scene = SceneSpec.load(s4 / "scene")
    assert scene.provenance["method"] == "fixed"
    assert scene.provenance["collision"] == "given"  # every object brought its hulls
    assert np.allclose(scene.T_base_support, T_BASE_WORLD)
    assert scene.reference_camera == "c1" and scene.reference_step == 2
    obj = scene.objects[0]
    assert (obj.mass, obj.friction) == (0.3, 0.4)
    hulls = list((s4 / "scene" / "objects" / "red_mug" / "collision").glob("*.obj"))
    assert len(hulls) == 1  # SimFoundry's, no decomposition
    assert np.allclose(trimesh.load(hulls[0]).extents, (0.05, 0.04, 0.03))
    physics = json.loads((s4 / "output.json").read_text())["objects"]
    assert physics["box"] == {
        "mass": 0.3,
        "friction": 0.4,
        "why": "SimFoundry stage 11 estimate",
    }


def test_an_empty_scene_is_rebuilt_from_another_frame(
    capture, tmp_path, fake_simfoundry
):
    """No objects from the selected frame: SimFoundry 3-12 again, pinned to the best
    other frame, from a clean slate; the frames that gave nothing are recorded."""
    plan, calls = fake_simfoundry
    plan["runs"] = [(0, []), (1, OBJECTS[:1])]
    run(capture.root, tmp_path / "run", FixedMethod(), ("2",))
    assert calls == [
        (sf.STAGES, (), False),
        (sf.STAGES, ("s3_ground.img_idx=1",), False),  # s3-s12 wiped first
    ]
    out = json.loads((tmp_path / "run/s2_frames/output.json").read_text())
    assert out["selected"] == "ext1@1" and out["decided_by"] == "pinned"
    assert out["retried"] == ["ext1@0"]
    assert out["frame_notes"]["ext1@0"] == "SimFoundry found no objects from this frame"

    plan["runs"] = [(0, [])]
    calls.clear()
    with pytest.raises(RuntimeError, match="no objects"):
        run(
            tmp_path / "run",
            None,
            FixedMethod(SimFoundryConfig(retry_frames=0)),
            ("2",),
            True,
        )
    assert len(calls) == 1


# -------------------------------------------------------------------------- CLI
def test_cli_makes_the_fixed_method(tmp_path, monkeypatch):
    """``r2s2r run --method fixed`` with its options."""
    made = []
    monkeypatch.setattr(cli, "run", lambda *args: made.append(args) or {})
    cli.main(
        [
            "run",
            str(tmp_path),
            "--method",
            "fixed",
            "--out",
            str(tmp_path / "run"),
            "--stages",
            "2",
            "3",
            "--frame-selection",
            "hybrid",
            "--max-frames",
            "6",
            "--retry-frames",
            "0",
            "--codex-reasoning",
            "low",
            "--override",
            "a=b",
            "--override",
            "c=d",
        ]
    )
    assert len(made) == 1
    source, out, method, stages, force, cameras = made[0]
    assert isinstance(method, FixedMethod) and stages == ("2", "3")
    assert (source, out, force, cameras) == (tmp_path, tmp_path / "run", False, None)
    assert method.config == SimFoundryConfig("hybrid", 6, 0, "low", ["a=b", "c=d"])
