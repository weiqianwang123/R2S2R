"""Tests for testbed/evaluate.py: scoring a scene against a capture's ground truth,
from its metadata alone."""

import numpy as np
import pytest
from conftest import box_urdf

from r2s2r import cli
from r2s2r.structs import Capture, ObjectSpec, SceneSpec
from r2s2r.testbed.evaluate import evaluate, summary
from r2s2r.transforms import make_transform

BOX = (0.03, 0.07, 0.12)


def _capture(tmp_path):
    """A capture whose truth is a box and a mug on a table at z = 0.02."""

    def truth(centre, size):
        return {
            "T_base_obj": make_transform(np.eye(3), centre),
            "center": centre,
            "size": size,
            "collision": [],
            "mass": 0.1,
            "friction": 0.5,
            "static": False,
            "rests_on": "table",
        }

    ground_truth = {
        "world": "test",
        "target": "box",
        "support": {
            "T_base_support": make_transform(np.eye(3), [0.4, 0.0, 0.02]),
            "height": 0.02,
            "size": [1.0, 0.8],
            "body": "table",
        },
        "objects": {
            "box": truth([0.5, -0.1, 0.08], list(BOX)),
            "mug": truth([0.5, 0.2, 0.07], [0.08, 0.08, 0.1]),
        },
        "layout": {},
        "source": {},
    }
    capture = Capture(
        "c", "mujoco", "franka_panda", "", {}, [], (0, 1), tmp_path / "capture",
        metadata={"depth": "rendered", "ground_truth": ground_truth},
    )  # fmt: skip
    capture.save()
    return Capture.load(capture.root)


def _scene(tmp_path, offset, support_z=0.02, tilt_deg=0.0):
    tilt = np.deg2rad(tilt_deg)
    R = np.array(
        [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]
    )
    obj = ObjectSpec(
        "thing",
        "box",
        str(box_urdf(tmp_path, BOX)),
        make_transform(np.eye(3), np.add([0.5, -0.1, 0.08 - BOX[2] / 2], offset)),
    )
    return SceneSpec(
        name="s",
        embodiment="franka_panda",
        objects=[obj],
        T_base_support=make_transform(R, [0.3, 0.0, support_z]),
        cameras={},
        reference_camera="c",
        reference_step=0,
        joint_positions=np.zeros(7),
        support_extent=(0.9, 0.7),
    )


def test_objects_match_the_nearest_truth(tmp_path):
    """A box 1 cm off matches the true box with a 1 cm centre error and the same
    size; the mug nothing matched is missed."""
    report = evaluate(_scene(tmp_path, [0.01, 0.0, 0.0]), _capture(tmp_path))
    err = report["objects"]["thing"]
    assert err["ground_truth"] == "box"
    assert err["center_error_m"] == pytest.approx(0.01, abs=5e-4)
    assert err["center_offset_m"] == pytest.approx([0.01, 0.0, 0.0], abs=5e-4)
    assert err["size_m"] == pytest.approx(list(BOX), abs=2e-3)
    assert report["missed"] == ["mug"]
    assert any("missed: mug" in line for line in summary(report))


def test_support_height_where_the_objects_stand(tmp_path):
    """A plane 1 cm high is 1 cm off; a tilted one is off by its slope at the objects
    (their mean y is 0.05 m from its origin), and says how far it is tilted."""
    capture = _capture(tmp_path)
    support = evaluate(_scene(tmp_path, [0, 0, 0], 0.03), capture)["support"]
    assert support["height_error_m"] == pytest.approx(0.01)
    assert support["tilt_deg"] == pytest.approx(0.0)
    assert support["ground_truth_size_m"] == [1.0, 0.8]
    support = evaluate(_scene(tmp_path, [0, 0, 0], 0.02, 2.0), capture)["support"]
    assert support["tilt_deg"] == pytest.approx(2.0)
    assert support["height_error_m"] == pytest.approx(
        0.05 * np.tan(np.deg2rad(2.0)), abs=1e-4
    )


def test_only_the_original_capture_scores(tmp_path):
    """A capture without ground truth (a run's copy) is refused."""
    capture = _capture(tmp_path)
    capture.metadata = {"depth": "rendered"}
    with pytest.raises(ValueError, match="original"):
        evaluate(_scene(tmp_path, [0, 0, 0]), capture)


def test_eval_scores_a_runs_scenes(tmp_path, capsys):
    """``r2s2r eval`` scores the scenes a run's stages left, or a scene itself."""
    capture = _capture(tmp_path)
    scene = _scene(tmp_path, [0, 0, 0])
    run = tmp_path / "run"
    for stage in ("s4_scene", "s6_refine"):
        scene.save(run / stage / "scene")
    for path, scored in (
        (run, ["s4_scene", "s6_refine"]),
        (run / "s4_scene/scene", ["s4_scene"]),
    ):
        cli.main(["eval", str(path), "--capture", str(capture.root)])
        out = capsys.readouterr().out
        assert [line[:-1] for line in out.splitlines() if line.endswith(":")] == [
            str(run / stage / "scene") for stage in scored
        ]
        assert "thing ~ box: centre off by 0.0 cm" in out
    with pytest.raises(SystemExit, match="neither"):
        cli.main(["eval", str(tmp_path / "capture"), "--capture", str(capture.root)])
