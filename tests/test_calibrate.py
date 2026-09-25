"""Tests for calibrate/: the evaluation tools, and the orchestrator with stand-ins for
the Codex agent (needs GL rendering)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from synthetic_drawer import (  # noqa: E402
    CAMERAS,
    TRUTH,
    drawer_cabinet,
    scene_of,
    video,
)

from r2s2r.calibrate import articulation  # noqa: E402
from r2s2r.calibrate.tools import (  # noqa: E402
    LEDGER_FILE,
    Workspace,
    create_workspace,
    evaluate,
    urdf_info,
)
from r2s2r.io.rgbd import save_depth_steps  # noqa: E402
from r2s2r.reconstruct.joints import JointFitConfig  # noqa: E402
from r2s2r.transforms import make_transform  # noqa: E402

pytestmark = pytest.mark.gl

REST = [0, 1]
SIDEWAYS = '<axis xyz="0 1 0"/>'


def _setup(tmp_path, axis="0 1 0"):
    """The video of the true drawer (sliding along x), and a scene whose model slides
    along ``axis``."""
    (tmp_path / "truth").mkdir()
    steps = video(scene_of(drawer_cabinet(tmp_path / "truth")), TRUTH, CAMERAS)
    (tmp_path / "asset").mkdir()
    scene_of(drawer_cabinet(tmp_path / "asset", axis=axis)).save(tmp_path / "scene")
    return tmp_path / "scene", steps


def _calibrate(tmp_path, agent, axis="0 1 0"):
    scene_dir, steps = _setup(tmp_path, axis)
    config = articulation.CalibrationConfig(fit=JointFitConfig(view_scale=1.0))
    return articulation.calibrate_articulation(
        scene_dir, tmp_path / "no_capture", steps, REST, tmp_path / "cal", agent, config
    )


def _agent(axis, fake_score=None, collisions=True):
    """A stand-in for the agent: one candidate sliding along ``axis``, evaluated with
    the tool (or entered in the ledger with ``fake_score``), and chosen."""

    def run(prompt, workdir, images):
        assert "joints-eval" in prompt and images
        ws = Workspace.load(workdir)
        text = Path(ws.urdf).read_text(encoding="utf-8")
        text = text.replace(SIDEWAYS, f'<axis xyz="{axis}"/>')
        if not collisions:
            text = re.sub(r"<collision>.*?</collision>", "", text)
        candidate = ws.model_dir / "cand_x.urdf"
        candidate.write_text(text, "utf-8")
        if fake_score is None:
            line = articulation.brief(
                evaluate(ws, candidate, ws.root / "evals" / "cand_x")
            )
        else:
            line = {"urdf": str(candidate), "score_mm": fake_score, "consistent": True}
        with open(ws.root / LEDGER_FILE, "a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(line) + "\n")
        choice = {"candidate": "model/cand_x.urdf", "reason": "test"}
        (workdir / "choice.json").write_text(json.dumps(choice), "utf-8")
        return "done"

    return run


def test_evaluation_shows_where_the_model_falls_short(tmp_path):
    """With the drawer modelled sliding sideways, the open steps are unexplained, and
    the evidence finds the surface the model lacks in front of the cabinet."""
    scene_dir, steps = _setup(tmp_path)
    save_depth_steps(tmp_path / "steps.npz", steps)
    ws = create_workspace(
        tmp_path / "ws",
        scene_dir,
        tmp_path / "no_capture",
        "cab",
        tmp_path / "steps.npz",
        REST,
    )
    summary = evaluate(ws, ws.urdf, ws.root / "baseline")
    assert not summary["consistent"] and {3, 4} <= set(summary["unexplained_steps"])
    worst = summary["evidence"][0]
    assert worst["step"] in (3, 4) and Path(worst["image"]).exists()
    seen = [v["seen_not_in_model"] for v in worst["views"].values()]
    # The drawer's front, pulled 12 cm out of the carcass's front face at x = 0.1.
    assert any(s["pixels"] and s["box_obj"][1][0] > 0.15 for s in seen)


def test_a_better_candidate_is_accepted(tmp_path):
    """The agent's candidate slides the drawer out: it passes and replaces the model,
    with its limits widened to the travel seen."""
    scene, report = _calibrate(tmp_path, _agent("1 0 0"))
    entry = report["cab"]
    assert entry["decision"] == "accepted cand_x"
    assert entry["final"]["consistent"] and not entry["baseline"]["consistent"]
    assert entry["final"]["score_mm"] < entry["baseline"]["score_mm"]
    assert scene.objects[0].asset_path.endswith("cand_x_fit.urdf")


def test_a_worse_candidate_is_rejected(tmp_path):
    """A candidate sliding the drawer up does not explain the video: the model stays."""
    scene, report = _calibrate(tmp_path, _agent("0 0 1"))
    assert report["cab"]["decision"].startswith("kept")
    assert Path(scene.objects[0].asset_path).name == "cab.urdf"


def test_a_candidate_without_matching_collisions_is_rejected(tmp_path):
    """The right joint, but links without collision meshes: not sim-ready, not taken."""
    _, report = _calibrate(tmp_path, _agent("1 0 0", collisions=False))
    entry = report["cab"]
    assert entry["decision"].startswith("kept")
    assert entry["verified"]["cand_x"]["consistent"]
    assert set(entry["verified"]["cand_x"]["collision_mismatch"]) == {
        "carcass",
        "drawer",
    }


def test_the_agents_own_numbers_are_not_trusted(tmp_path):
    """A ledger entry claiming a perfect score does not survive re-evaluation."""
    _, report = _calibrate(tmp_path, _agent("0 0 1", fake_score=0.1))
    assert report["cab"]["decision"].startswith("kept")
    assert not report["cab"]["verified"]["cand_x"]["consistent"]


def test_a_model_that_passes_is_kept_without_the_agent(tmp_path):
    """The agent is only called when the check fails."""

    def agent(*args):
        raise AssertionError("the agent should not run")

    scene, report = _calibrate(tmp_path, agent, axis="1 0 0")
    assert report["cab"]["decision"].startswith("kept")
    assert scene.objects[0].asset_path.endswith("cab_fit.urdf")


def test_urdf_info_gives_axes_in_both_frames(tmp_path):
    """The joint axis turns with the object's pose."""
    urdf = drawer_cabinet(tmp_path)
    yaw = make_transform([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [0.5, 0.2, 0.0])
    info = urdf_info(urdf, yaw)
    (joint,) = [j for j in info["joints"] if j["name"] == "slide"]
    assert joint["axis_obj"] == [1.0, 0.0, 0.0]
    assert np.allclose(joint["axis_base"], [0.0, 1.0, 0.0])
    assert info["links"]["drawer"]["box_obj"][1][0] == pytest.approx(0.1)
