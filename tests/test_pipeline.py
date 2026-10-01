"""Tests for r2s2r.pipeline: the run directory, the stage checks, the run loop
(``run.json``, staleness, the shared settling and final replay, with Isaac Lab faked)
and the agentic method (a fake Codex; its repository guard)."""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import trimesh
from conftest import hinged_box, rgbd_capture

from r2s2r.pipeline import run as run_module
from r2s2r.pipeline.agentic import method as agentic
from r2s2r.pipeline.run import StageFailed, is_done, run
from r2s2r.pipeline.stages import VALIDATORS
from r2s2r.structs import ObjectSpec, SceneSpec
from r2s2r.transforms import look_at
from r2s2r.workspace import Workspace

TARGET = (0.5, 0.05, 0.04)  # where the cameras look
CAMERAS = {
    "c1": ("ext1", look_at((0.95, 0.35, 0.45), TARGET)),
    "c2": ("wrist", look_at((0.45, -0.4, 0.4), TARGET)),
}
SUPPORT = {"T_base_support": np.eye(4).tolist(), "extent": [1.0, 1.0]}


def _capture(tmp_path):
    return rgbd_capture(tmp_path / "capture", CAMERAS)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)


def _frames_product(d, frames=("ext1@0", "wrist@4"), support=None):
    _write(d / "support.json", SUPPORT if support is None else support)
    _write(
        d / "output.json", {"frames": list(frames), "support": {"file": "support.json"}}
    )


def _objects_product(d, **obj):
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(d / "mesh.obj")
    box = {"name": "box", "mesh": "mesh.obj", "scale": 1.0, "T_base_obj": np.eye(4)}
    box = {**box, **obj}
    box["T_base_obj"] = np.asarray(box["T_base_obj"]).tolist()
    _write(
        d / "objects.json",
        {"support": "../s2_frames/support.json", "objects": [box]},
    )


def _scene_product(d, ws, extra="output.json"):
    urdf = d / "scene" / "objects" / "box" / "box.urdf"
    urdf.parent.mkdir(parents=True, exist_ok=True)
    urdf.write_text("<robot/>")
    obj = ObjectSpec("box", "box", str(urdf), np.eye(4), mass=0.2)
    frame = ws.frame("ext1@0")
    scene = SceneSpec(
        "t",
        "franka_panda",
        [obj],
        np.eye(4),
        ws.capture.cameras,
        frame.camera,
        frame.step,
        frame.joint_positions,
    )
    scene.save(d / "scene")
    _write(d / extra, {"objects": {"box": {"mass": 0.2}}})


class FakeMethod:
    """Writes valid products for stages 2, 3 and 4, and counts its calls."""

    name = "fake"
    stages = ("2", "3", "4")

    def __init__(self):
        self.calls = []

    def run_stage(self, ws, key, stage_dir):
        """Write the stage's product."""
        self.calls.append(key)
        {"2": _frames_product, "3": _objects_product}.get(
            key, lambda d: _scene_product(d, ws)
        )(stage_dir)
        return {"note": f"did {key}"}

    def check(self, ws, key, stage_dir):  # pylint: disable=unused-argument
        """Nothing beyond the shared checks."""
        return []


# ------------------------------------------------------------------------ the run dir
def test_run_dir_hides_metadata_and_names_frames(tmp_path):
    """The run's copy of the capture drops its metadata but depth's origin; run.json
    says the method and where the capture is."""
    ws = Workspace.create(_capture(tmp_path), tmp_path / "run", "agentic")
    assert ws.capture.metadata == {"depth": "measured"}  # "truth" is dropped
    assert ws.frame("wrist@3").camera == "c2"
    assert ws.frame_id(ws.frame("ext1@0")) == "ext1@0"
    assert len(ws.reconstructable()) == 10
    rows = json.loads((tmp_path / "run/inputs/frames.json").read_text())
    assert {r["id"] for r in rows} >= {"ext1@4", "wrist@0"}
    assert (tmp_path / "run/inputs/sheets/wrist.png").exists()
    with pytest.raises(KeyError):
        ws.frame("ext1@9")
    assert Workspace.find(tmp_path / "run" / "inputs").root == ws.root
    assert ws.read_run() == {
        "method": "agentic",
        "capture": "inputs/capture",
        "stages": {},
    }
    with pytest.raises(FileExistsError):
        Workspace.create(ws.capture, tmp_path / "run", "agentic")

    wrist = Workspace.create(ws.capture, tmp_path / "wrist", "agentic", ["wrist"])
    assert list(wrist.capture.cameras) == ["c2"]
    assert {f.camera for f in wrist.capture.frames} == {"c2"}
    assert not (tmp_path / "wrist/inputs/capture/c1").exists()  # nothing of the rest
    assert not (tmp_path / "wrist/inputs/sheets/ext1.png").exists()


# ----------------------------------------------------------------------- the checks
def test_stage_checks_say_what_is_wrong(tmp_path):
    """Every stage's shared check, on good and bad products."""
    ws = Workspace.create(_capture(tmp_path), tmp_path / "run", "agentic")
    s2, s3, s4 = (ws.root / d for d in ("s2_frames", "s3_objects", "s4_scene"))
    assert VALIDATORS["2"](ws, s2) == ["output.json not written yet"]
    _frames_product(s2)
    assert not VALIDATORS["2"](ws, s2)
    _frames_product(s2, ("ext1@0", "ext1@9", 3), {**SUPPORT, "extent": [0, 1]})
    problems = VALIDATORS["2"](ws, s2)
    assert len(problems) == 3 and "extent" in problems[-1]
    _frames_product(s2, support={"T_base_support": (2 * np.eye(4)).tolist()})
    assert VALIDATORS["2"](ws, s2) == [
        f"support {s2 / 'support.json'}: T_base_support is not a rigid transform"
    ]
    _frames_product(s2, support={"T_base_support": np.eye(4).tolist()})  # no extent
    assert not VALIDATORS["2"](ws, s2)

    s3.mkdir()
    _objects_product(s3)
    assert not VALIDATORS["3"](ws, s3)
    _objects_product(s3, up="sideways", T_base_obj=2 * np.eye(4), mesh="no.obj")
    assert len(VALIDATORS["3"](ws, s3)) == 3
    hinged = s3 / "hinged"
    hinged.mkdir()
    objects, _ = hinged_box(hinged, np.eye(4))
    objects["objects"][0]["parts"][0]["mesh"] = "hinged/no_lid.obj"
    objects["objects"][0]["joints"][0]["position"] = 1.0
    objects["objects"][0]["mesh"] = "hinged/body.obj"
    (s3 / "objects.json").write_text(json.dumps(objects))
    problems = VALIDATORS["3"](ws, s3)
    assert len(problems) == 2
    assert "lid's mesh not found" in problems[0] and "outside its limits" in problems[1]

    _scene_product(s4, ws, extra="notes.md")
    assert VALIDATORS["4"](ws, s4) == ["output.json missing"]
    assert VALIDATORS["6"](ws, s4) == ["report.md missing"]


def test_agentic_checks_more_at_stage_2(tmp_path):
    """The agent must choose 4 to 8 frames with depth, a support with an extent, and
    list the objects."""
    ws = Workspace.create(_capture(tmp_path), tmp_path / "run", "agentic")
    s2 = ws.root / "s2_frames"
    method = agentic.AgenticMethod(agentic.AgenticConfig(codex_bin="codex"))
    _frames_product(s2, support={"T_base_support": np.eye(4).tolist()})
    assert method.check(ws, "2", s2) == [
        "choose 4 to 8 frames, not 2",
        "support support.json: no extent",
        "output.json lists no objects",
    ]
    assert not method.check(ws, "3", s2)


# ------------------------------------------------------------------------- the run
def test_run_records_every_stage_and_redoes_stale_ones(tmp_path):
    """run.json says what ran and how; done stages are skipped until the one before is
    redone."""
    method = FakeMethod()
    log = run(_capture(tmp_path).root, tmp_path / "run", method, ("2", "3"))
    assert method.calls == ["2", "3"]
    assert log["method"] == "fake"
    two = log["stages"]["2"]
    assert two["status"] == "done" and two["note"] == "did 2" and two["started"] > 0
    assert "seconds" in two and log["stages"]["6"] == {"status": "skipped"}
    assert "4" not in log["stages"] and "final_replay" not in log

    run(tmp_path / "run", None, method, ("2", "3"))  # resumed: nothing to do
    assert method.calls == ["2", "3"]
    time.sleep(0.01)
    (tmp_path / "run/s2_frames/output.json").touch()  # stage 2 redone...
    ws = Workspace.load(tmp_path / "run")
    assert not is_done(ws, method, "3")  # ...so stage 3 is stale
    run(tmp_path / "run", None, method, ("2", "3"))
    assert method.calls == ["2", "3", "3"]
    run(tmp_path / "run", None, method, ("3",), force=True)
    assert method.calls == ["2", "3", "3", "3"]


def test_redoing_a_stage_makes_every_later_one_stale(tmp_path):
    """Not only the next one; and a stage that leaves its old product in place has not
    done its work."""

    class Lazy(FakeMethod):
        """Stage 3 keeps the objects file it finds."""

        def run_stage(self, ws, key, stage_dir):
            return {} if key == "3" else super().run_stage(ws, key, stage_dir)

    method = FakeMethod()
    run(_capture(tmp_path).root, tmp_path / "run", method, ("2", "3", "4"))
    time.sleep(0.01)
    run(tmp_path / "run", None, method, ("2",), force=True)
    ws = Workspace.load(tmp_path / "run")
    assert [is_done(ws, method, k) for k in "234"] == [True, False, False]
    with pytest.raises(StageFailed, match="stage 2's product changed after"):
        run(tmp_path / "run", None, Lazy(), ("3",))
    assert ws.read_run()["stages"]["3"]["status"] == "failed"
    method.calls.clear()
    run(tmp_path / "run", None, method, ("2", "3", "4"))
    assert method.calls == ["3", "4"]


def test_run_refuses_what_does_not_fit(tmp_path):
    """Another method's run, a method's missing stage, other cameras, a second --out."""
    run(_capture(tmp_path).root, tmp_path / "run", FakeMethod(), ("2",))
    other = agentic.AgenticMethod(agentic.AgenticConfig(codex_bin="codex"))
    with pytest.raises(ValueError, match="fake method"):
        run(tmp_path / "run", None, other)
    with pytest.raises(ValueError, match="stages"):
        run(tmp_path / "run", None, FakeMethod(), ("6",))
    with pytest.raises(ValueError, match="cameras"):
        run(
            tmp_path / "capture",
            tmp_path / "run",
            FakeMethod(),
            ("2",),
            False,
            ["ext1"],
        )
    with pytest.raises(ValueError, match="without --out"):
        run(tmp_path / "run", tmp_path / "elsewhere", FakeMethod(), ("2",))
    with pytest.raises(FileNotFoundError):
        run(tmp_path, tmp_path / "run2", FakeMethod())


def test_failed_stages_are_recorded(tmp_path):
    """A stage that raises, or ends without a valid product, is failed in run.json."""

    class Broken(FakeMethod):
        """Stage 2 raises; stage 3 writes nothing."""

        def run_stage(self, ws, key, stage_dir):
            if key == "2":
                raise RuntimeError("no GPU")
            return {}

    with pytest.raises(RuntimeError, match="no GPU"):
        run(_capture(tmp_path).root, tmp_path / "run", Broken(), ("2",))
    entry = Workspace.load(tmp_path / "run").read_run()["stages"]["2"]
    assert entry["status"] == "failed" and entry["error"] == "RuntimeError: no GPU"
    run(tmp_path / "run", None, FakeMethod(), ("2",))
    with pytest.raises(StageFailed, match="objects.json not written yet"):
        run(tmp_path / "run", None, Broken(), ("3",))
    stages = Workspace.load(tmp_path / "run").read_run()["stages"]
    assert stages["2"]["status"] == "done" and stages["3"]["status"] == "failed"


def test_settling_and_final_replay_are_shared(tmp_path, monkeypatch):
    """Stage 5 settles stage 4's scene (its objects referred to, not copied); the final
    scene is replayed once, and again only when it changes."""
    replays = []

    def settle(scene_dir, capture_dir, out_dir):
        assert capture_dir == tmp_path.resolve() / "run/inputs/capture"
        SceneSpec.load(scene_dir).save(out_dir)
        return {"scene": str(out_dir / "scene.json"), "objects": {"box": {}}}

    def replay(scene_dir, capture_dir, out_dir):  # pylint: disable=unused-argument
        replays.append(scene_dir)
        _write(out_dir / "compare/compare.json", {"frames": []})
        return {"compare_dir": str(out_dir / "compare"), "sheets": [], "frames": 0}

    monkeypatch.setattr(run_module.isaac, "settle", settle)
    monkeypatch.setattr(run_module.isaac, "replay", replay)
    log = run(_capture(tmp_path).root, tmp_path / "run", FakeMethod())
    assert [log["stages"][k]["status"] for k in "2345"] == ["done"] * 4
    s5 = tmp_path / "run/s5_settle"
    assert json.loads((s5 / "output.json").read_text())["scene"] == "scene/scene.json"
    saved = json.loads((s5 / "scene/scene.json").read_text())
    assert saved["objects"][0]["asset_path"].startswith("../../s4_scene/scene/objects")
    assert replays == [s5.resolve() / "scene"]
    final = log["final_replay"]
    assert final["stage"] == "5" and final["status"] == "done"
    assert final["compare_dir"] == "s5_settle/final_replay/compare"
    run(tmp_path / "run", None, FakeMethod())
    assert len(replays) == 1  # up to date
    os.utime(s5 / "scene/scene.json")
    time.sleep(0.01)
    (s5 / "output.json").touch()
    run(tmp_path / "run", None, FakeMethod(), ("5",))
    assert len(replays) == 2


# ------------------------------------------------------------------ agentic method
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


def test_agent_is_resumed_until_the_stage_output_is_valid(tmp_path, monkeypatch):
    """A fake Codex writes invalid output first, then valid output when resumed."""
    codex = tmp_path / "codex"
    codex.write_text(f"#!{sys.executable}\n{FAKE_CODEX}")
    codex.chmod(0o755)
    monkeypatch.setattr(agentic, "repo_state", lambda: {})
    method = agentic.AgenticMethod(agentic.AgenticConfig(codex_bin=str(codex)))
    log = run(_capture(tmp_path).root, tmp_path / "run", method, ("2",))
    entry = log["stages"]["2"]
    assert entry["session"] == "fake-session" and entry["resumptions"] == 1
    assert entry["status"] == "done" and entry["started"] > 0
    brief = (tmp_path / "run/s2_frames/BRIEF.md").read_text()
    rules = (tmp_path / "run/AGENTS.md").read_text()
    assert "{{" not in brief + rules and "franka_panda" in rules
    # Done stages are skipped.
    again = run(tmp_path / "run", None, method, ("2",))
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
    monkeypatch.setattr(agentic, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(agentic, "SIMFOUNDRY_DIR", tmp_path / "sub")
    before = agentic.repo_state()
    assert not agentic.diff_states(before, agentic.repo_state())
    (tmp_path / "sub" / "new.txt").write_text("x")
    after = agentic.repo_state()
    assert agentic.diff_states(before, after)
    (tmp_path / "sub" / "new.txt").write_text("y")  # same status, new contents
    later = agentic.repo_state()
    assert agentic.diff_states(after, later)
    (tmp_path / "repo" / "a.txt").write_text("b")
    changed = agentic.diff_states(later, agentic.repo_state())
    assert len(changed) == 1 and changed[0].startswith(str(tmp_path / "repo"))
