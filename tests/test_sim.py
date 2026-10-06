"""Tests for r2s2r.sim with MuJoCo: a scene settled and a recording replayed in it
through the calls every simulator answers, leaving the files Isaac Lab's leave."""

import json

import numpy as np
import pytest
from conftest import hinged_box, rgbd_capture, robot_or_skip

from r2s2r import sim
from r2s2r.structs import SceneSpec
from r2s2r.tools.objects import assemble
from r2s2r.transforms import look_at, make_transform
from r2s2r.workspace import Workspace

pytest.importorskip("mujoco")

TARGET = (0.5, 0.0, 0.04)
CAMERAS = {
    "c1": ("ext1", look_at((0.95, 0.35, 0.45), TARGET)),
    "c2": ("wrist", look_at((0.45, -0.4, 0.4), TARGET)),
}


def _run(tmp_path):
    """A run of the test capture (the Panda at home), and the hinged box assembled
    on its table."""
    robot_or_skip("franka_panda")
    capture = rgbd_capture(tmp_path / "capture", CAMERAS)
    ws = Workspace.create(capture, tmp_path / "run", "agentic", sim="mujoco")
    hinged_box(tmp_path, make_transform(np.eye(3), [0.5, 0.0, 0.0]))
    assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "hull")
    return ws


def test_a_scene_settles_in_mujoco(tmp_path):
    """The objects at rest (the lid, let go half open, falls shut), the report as
    Isaac Lab's, the scene saved without USDs."""
    ws = _run(tmp_path)
    report = sim.settle(tmp_path / "scene", ws.capture.root, tmp_path / "s5", ws.sim)
    assert report["scene"] == str(tmp_path / "s5" / "scene.json")
    box = report["objects"]["box"]
    assert box["moved_m"] < 0.002 and box["joints_moved"]["hinge"] > 0.7
    assert json.loads((tmp_path / "s5" / "settle.json").read_text()) == {
        k: v for k, v in report.items() if k != "scene"
    }
    (obj,) = SceneSpec.load(tmp_path / "s5").objects
    assert obj.usd is None and obj.joints["hinge"] > -0.1


@pytest.mark.gl
def test_a_recording_replays_in_mujoco(tmp_path):
    """Every frame of the static period rendered and compared: the residuals per
    camera, the renders and the comparison where Isaac Lab's replay leaves them."""
    ws = _run(tmp_path)
    out = tmp_path / "replay"
    summary = sim.replay(tmp_path / "scene", ws.capture.root, out, ws.sim, every=2)
    assert summary["frames"] == len(list((out / "frames").glob("*_sim.png")))
    assert set(summary["median_depth_residual_m"]) == {"ext1", "wrist"}  # by role
    log = json.loads((out / "replay.json").read_text())
    assert max(error for _, error in log["arm_error_rad"]) < 1e-6
    assert (out / "compare" / "compare.json").exists() and summary["sheets"]
