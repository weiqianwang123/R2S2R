"""Tests for r2s2r.agentic: the workspace, the geometry tools, assembly, the runner's
checks (a fake Codex) and its repository guard."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from r2s2r.agentic import runner
from r2s2r.agentic.geometry import fit_plane, parse_masks, pattern_search
from r2s2r.agentic.objects import assemble
from r2s2r.agentic.workspace import Workspace
from r2s2r.io.rgbd import write_depth
from r2s2r.structs import (
    CameraSpec,
    Capture,
    FrameRecord,
    ObjectSpec,
    SceneSpec,
)
from r2s2r.transforms import intrinsics_matrix, make_transform

K = intrinsics_matrix(300.0, 300.0, 159.5, 119.5)
SIZE = (320, 240)
BOX = (0.06, 0.04, 0.08)  # the true object


def _look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float] = (0.5, 0.05, 0.04),
) -> np.ndarray:
    at, to = np.asarray(eye, float), np.asarray(target, float)
    fwd = (to - at) / np.linalg.norm(to - at)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return make_transform(np.column_stack([right, np.cross(fwd, right), fwd]), at)


CAMERAS = {
    "c1": ("ext1", _look_at((0.95, 0.35, 0.45))),
    "c2": ("wrist", _look_at((0.45, -0.4, 0.4))),
}
TRUE_POSE = make_transform(
    Rotation.from_euler("z", 30, degrees=True).as_matrix(), [0.5, 0.05, 0.0]
)


def _capture(tmp_path, depth_of=None, image_of=None):
    """Two cameras (one static, one on the wrist), five static steps each."""
    root = tmp_path / "capture"
    cameras, frames = {}, []
    for serial, (role, T) in CAMERAS.items():
        static = role != "wrist"
        cameras[serial] = CameraSpec(
            serial, role, *SIZE, K, is_static=static, T_base_cam=T if static else None
        )
        (root / serial).mkdir(parents=True)
        for step in range(5):
            rgb = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
            depth = np.full((SIZE[1], SIZE[0]), 0.8)
            if image_of is not None:
                rgb, depth = image_of[serial], depth_of[serial]
            cv2.imwrite(str(root / f"{serial}/{step}.png"), rgb[..., ::-1])
            write_depth(root / f"{serial}/{step}_d.png", depth)
            frames.append(
                FrameRecord(
                    step,
                    serial,
                    f"{serial}/{step}.png",
                    None,
                    T,
                    np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]),
                    0.0,
                    f"{serial}/{step}_d.png",
                )
            )
    capture = Capture(
        "t",
        "test",
        "franka_panda",
        "pick",
        cameras,
        frames,
        (0, 5),
        root,
        metadata={"truth": "hidden", "depth": "measured"},
    )
    capture.save()
    return capture


def _box_urdf(tmp_path, extents, name="box"):
    box = trimesh.creation.box(extents=extents)
    box.apply_translation([0, 0, extents[2] / 2])
    box.export(tmp_path / f"{name}.obj")
    (tmp_path / f"{name}.urdf").write_text(
        f'<robot name="{name}"><link name="base"><visual><geometry>'
        f'<mesh filename="{name}.obj"/></geometry></visual></link></robot>'
    )
    return tmp_path / f"{name}.urdf"


def test_workspace_hides_metadata_and_names_frames(tmp_path):
    """The agent's copy of the capture drops its metadata but depth's origin."""
    ws = Workspace.create(_capture(tmp_path), tmp_path / "ws")
    assert ws.capture.metadata == {"depth": "measured"}  # "truth" is dropped
    assert ws.frame("wrist@3").camera == "c2"
    assert ws.frame_id(ws.frame("ext1@0")) == "ext1@0"
    assert len(ws.reconstructable()) == 10
    rows = json.loads((tmp_path / "ws/inputs/frames.json").read_text())
    assert {r["id"] for r in rows} >= {"ext1@4", "wrist@0"}
    assert (tmp_path / "ws/inputs/sheets/wrist.png").exists()
    with pytest.raises(KeyError):
        ws.frame("ext1@9")
    assert Workspace.find(tmp_path / "ws" / "inputs").root == ws.root
    assert parse_masks(["ext1@0=a.png"]) == {"ext1@0": "a.png"}

    wrist = Workspace.create(ws.capture, tmp_path / "wrist", cameras=["wrist"])
    assert list(wrist.capture.cameras) == ["c2"]
    assert {f.camera for f in wrist.capture.frames} == {"c2"}
    assert not (tmp_path / "wrist/inputs/capture/c1").exists()  # nothing of the rest
    assert not (tmp_path / "wrist/inputs/sheets/ext1.png").exists()


def test_fit_plane_ignores_outliers():
    """A tilted plane among clutter."""
    rng = np.random.default_rng(1)
    n = np.array([0.1, 0.0, 1.0]) / np.linalg.norm([0.1, 0.0, 1.0])
    xy = rng.uniform(-0.5, 0.5, (3000, 2))
    plane = np.c_[xy, (0.02 - n[0] * xy[:, 0]) / n[2]]
    clutter = rng.uniform(-0.5, 0.5, (800, 3))
    normal, d, inliers = fit_plane(np.concatenate([plane, clutter]))
    assert abs(abs(normal @ n) - 1) < 1e-4
    assert abs(abs(d) - 0.02) < 1e-3
    assert inliers[:3000].mean() > 0.99


def test_pattern_search_climbs_to_the_maximum():
    """A quadratic's maximum; fixed coordinates stay put."""
    target = np.array([0.3, -0.2])

    def f(x):
        return -float(np.sum((x - target) ** 2))

    x = pattern_search(f, np.zeros(2), np.array([0.1, 0.1]), [True, True], levels=8)
    assert np.allclose(x, target, atol=0.01)
    y = pattern_search(f, np.zeros(2), np.array([0.1, 0.1]), [True, False])
    assert y[1] == 0.0


def test_assemble_writes_a_simulation_ready_scene(tmp_path):
    """A scaled mesh becomes a scene of a sim-ready URDF; scale in the pose is
    refused."""
    ws = Workspace.create(_capture(tmp_path), tmp_path / "ws")
    mesh = trimesh.creation.box(extents=(0.12, 0.08, 0.16))  # twice the true size
    mesh.export(tmp_path / "mesh.obj")
    T = TRUE_POSE @ make_transform(np.eye(3), [0, 0, 0.04])  # the box's centre
    objects = {
        "support": {"T_base_support": np.eye(4).tolist(), "extent": [1.0, 1.0]},
        "objects": [
            {
                "name": "Box 1",
                "mesh": "mesh.obj",
                "scale": 0.5,
                "T_base_obj": T.tolist(),
                "mass": 0.2,
                "friction": 0.7,
            }
        ],
    }
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    report = assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "hull")
    scene = SceneSpec.load(tmp_path / "scene")
    (obj,) = scene.objects
    assert obj.name == "box_1" and obj.mass == 0.2 and obj.friction == 0.7
    assert np.allclose(obj.T_base_obj, T)
    assert scene.support_extent == (1.0, 1.0)
    assert scene.reference_camera == "c1"  # the static camera
    assert report["objects"]["box_1"]["size_m"] == pytest.approx(BOX, abs=1e-6)
    assert report["objects"]["box_1"]["lowest_point_above_support_m"] == pytest.approx(
        0.0, abs=1e-6
    )
    root = ET.parse(obj.asset_path).getroot()
    assert float(root.find("link/inertial/mass").attrib["value"]) == pytest.approx(0.2)
    names = [c.attrib.get("name") for c in root.iter("collision")]
    assert "hull_0" in names and "r2s2r_resting_base" in names

    objects["objects"][0]["T_base_obj"] = (2 * np.eye(4)).tolist()
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    with pytest.raises(ValueError, match="rigid"):
        assemble(ws, tmp_path / "objects.json", tmp_path / "scene2", "hull")


@pytest.mark.gl
def test_fit_recovers_scale_yaw_and_position(tmp_path):
    """The mesh is twice the object's size and not turned; two views see the object."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.agentic.geometry import fit
    from r2s2r.reconstruct.render import SceneRenderer
    from r2s2r.structs import DepthView

    truth = SceneSpec(
        "t",
        "franka_panda",
        [ObjectSpec("box", "box", str(_box_urdf(tmp_path, BOX)), TRUE_POSE)],
        np.eye(4),
        {},
        "c1",
        0,
        np.zeros(7),
    )
    renderer = SceneRenderer(truth, SIZE)
    renderer.pose(0, TRUE_POSE)
    images, depths, masks = {}, {}, {}
    for serial, (_, T) in CAMERAS.items():
        out = renderer.render(DepthView(serial, 0, np.zeros(SIZE[::-1]), K, T), 0)
        images[serial], depths[serial] = out["rgb"], out["depth"].astype(float)
        masks[serial] = out["mask"]
    renderer.close()
    ws = Workspace.create(_capture(tmp_path, depths, images), tmp_path / "ws")
    mask_paths = {}
    for serial, (role, _) in CAMERAS.items():
        path = tmp_path / f"mask_{role}.png"
        cv2.imwrite(str(path), masks[serial].astype(np.uint8) * 255)
        mask_paths[f"{role}@2"] = str(path)
    mesh = trimesh.creation.box(extents=(0.12, 0.08, 0.16))
    mesh.export(tmp_path / "mesh.obj")
    support = {"T_base_support": np.eye(4).tolist(), "extent": [1.0, 1.0]}

    result = fit(
        ws, tmp_path / "mesh.obj", support, mask_paths, tmp_path / "fit", up="z"
    )
    ws.close()
    assert result["scale"] == pytest.approx(0.5, abs=0.025)
    assert result["mean_iou"] > 0.9
    T = np.asarray(result["T_base_obj"])
    centre = T[:3, 3]  # the mesh file's origin is the box centre
    assert np.linalg.norm(centre[:2] - TRUE_POSE[:2, 3]) < 0.005
    assert centre[2] == pytest.approx(0.04, abs=0.005)  # resting on the support
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    assert abs((yaw - 30 + 90) % 180 - 90) < 4  # a box looks the same turned 180
    assert (tmp_path / "fit" / "ext1@2.png").exists()


FAKE_CODEX = r"""
import json, pathlib, sys
args = sys.argv[1:]
out = pathlib.Path.cwd()
print(json.dumps({"type": "thread.started", "thread_id": "fake-session"}))
if "resume" in args:  # the second call: finish the stage
    (out / "support.json").write_text(json.dumps(
        {"T_base_support": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "extent": [1, 1]}))
    (out / "output.json").write_text(json.dumps({
        "frames": ["ext1@0", "ext1@4", "wrist@0", "wrist@4"],
        "support": {"file": "support.json"}, "objects": []}))
else:  # the first call: three frames, no support
    (out / "output.json").write_text(json.dumps(
        {"frames": ["ext1@0", "ext1@4", "wrist@9"], "support": {}}))
"""


def test_runner_resumes_until_the_stage_output_is_valid(tmp_path, monkeypatch):
    """A fake Codex writes invalid output first, then valid output when resumed."""
    codex = tmp_path / "codex"
    codex.write_text(f"#!{sys.executable}\n{FAKE_CODEX}")
    codex.chmod(0o755)
    monkeypatch.setattr(runner, "repo_state", lambda: {})
    log = runner.run(
        tmp_path / "ws",
        _capture(tmp_path).root,
        stages=("2",),
        config=runner.AgentConfig(codex_bin=str(codex)),
    )
    assert log["stages"]["2"]["session"] == "fake-session"
    assert log["started"]["2"] > 0  # the viewer's stage clock
    assert log["stages"]["2"]["resumptions"] == 1
    brief = (tmp_path / "ws/s2_frames/BRIEF.md").read_text()
    rules = (tmp_path / "ws/AGENTS.md").read_text()
    assert "{{" not in brief + rules and "franka_panda" in rules
    # Done stages are skipped.
    again = runner.run(tmp_path / "ws", stages=("2",))
    assert again["stages"]["2"]["resumptions"] == 1


def test_repository_guard_sees_changes(tmp_path, monkeypatch):
    """New untracked files, their contents, and tracked edits all count."""
    for name in ("repo", "sub"):
        repo = tmp_path / name
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "a.txt").write_text("a")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "commit",
                "-qm",
                "a",
            ],
            check=True,
        )
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(runner, "DEFAULT_SIMFOUNDRY_DIR", tmp_path / "sub")
    before = runner.repo_state()
    assert not runner.diff_states(before, runner.repo_state())
    (tmp_path / "sub" / "new.txt").write_text("x")
    after = runner.repo_state()
    assert runner.diff_states(before, after)
    (tmp_path / "sub" / "new.txt").write_text("y")  # same status, new contents
    later = runner.repo_state()
    assert runner.diff_states(after, later)
    (tmp_path / "repo" / "a.txt").write_text("b")
    changed = runner.diff_states(later, runner.repo_state())
    assert len(changed) == 1 and changed[0].startswith(str(tmp_path / "repo"))
