"""Tests for testbed/worlds.py and testbed/record.py: the worlds build, settle and hold
their robots, their cameras and ground truth agree with what they render, and a capture
recorded in them (skipped without MuJoCo's assets, physcoder's, or GL rendering)."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.robots import get_robot  # noqa: E402
from r2s2r.robots.mask import NO_ROBOT, RobotMasker  # noqa: E402
from r2s2r.robots.spec import MENAGERIE_DIR  # noqa: E402
from r2s2r.structs import Capture, DepthView, read_depth  # noqa: E402
from r2s2r.testbed import worlds  # noqa: E402
from r2s2r.testbed.record import (  # noqa: E402
    CLOSEST,
    SENSOR_MAX_DEPTH,
    SENSOR_MIN_DEPTH,
    View,
    record_capture,
)
from r2s2r.tools.geometry import view_points  # noqa: E402

FR3 = pytest.mark.skipif(
    not (MENAGERIE_DIR / "franka_fr3").exists() or not worlds.GSO_DIR.exists(),
    reason="MuJoCo assets not fetched (scripts/setup/fetch_mujoco_assets.sh)",
)
PHYSCODER = pytest.mark.skipif(
    not worlds.BOX_BLOCK_XML.exists(), reason="physcoder's assets are not here"
)
TRUTH_KEYS = {"world", "target", "support", "objects", "layout", "source"}
OBJECT_KEYS = {
    "T_base_obj",
    "center",
    "size",
    "collision",
    "mass",
    "friction",
    "static",
    "rests_on",
}


@pytest.fixture(name="fr3", scope="module")
def fixture_fr3():
    """fr3_table, settled."""
    world = worlds.build_world("fr3_table")
    world.reset()
    yield world
    world.close()


@FR3
def test_fr3_objects_rest_where_placed(fr3):
    """The GSO objects settle upright where the layout puts them, on the table, while
    the arm holds its home pose; the ground truth says so with every key."""
    assert np.allclose(fr3.arm_q(), fr3.robot.home_q, atol=1e-3)
    truth = fr3.ground_truth()
    assert set(truth) == TRUTH_KEYS and truth["target"] == "crayon_box"
    support = truth["support"]
    assert support["height"] == pytest.approx(0.0)
    assert support["size"] == pytest.approx(worlds.TABLE_SIZE)
    assert support["T_base_support"][:2, 3] == pytest.approx(worlds.TABLE_CENTER)
    for placed in truth["layout"]["objects"]:
        obj = truth["objects"][placed["name"]]
        assert set(obj) == OBJECT_KEYS and not obj["static"]
        assert np.allclose(obj["T_base_obj"][:2, 3], placed["xy"], atol=5e-3)
        assert obj["T_base_obj"][2, 2] > 0.999
        assert obj["mass"] == pytest.approx(placed["mass"])
        bottom = obj["center"][2] - obj["size"][2] / 2
        assert abs(bottom) < 3e-3 and obj["rests_on"] == "table"


@FR3
@pytest.mark.gl
def test_fr3_depth_backprojects_onto_the_table(fr3):
    """K, T_base_cam and planar depth agree: table pixels land on z = 0."""
    cam = fr3.camera_spec("ext1")
    depth = fr3.render("ext1")["depth"]
    view = DepthView("ext1", 0, depth.astype(float), cam.K, cam.T_base_cam)
    z = view_points(view, max_depth=3.0)[0][:, 2]
    near = z[np.abs(z) < 0.02]
    assert len(near) > 0.3 * len(z)
    assert abs(np.median(near)) < 0.002


@FR3
@pytest.mark.gl
def test_robot_masker_matches_the_world(fr3):
    """Rendering the robot from calibration and joints reproduces what the world's
    cameras see of it."""
    masker = RobotMasker(get_robot("fr3_robotiq"))
    m = fr3.model
    for role in fr3.cameras:
        cam = fr3.camera_spec(role)
        body = fr3.render(role)["body"]
        robot = m.body_rootid[m.jnt_bodyid[m.joint("fr3_joint1").id]]
        truth = (body >= 0) & (m.body_rootid[np.maximum(body, 0)] == robot)
        pred = (
            masker.robot_depth(
                cam.K,
                cam.width,
                cam.height,
                fr3.camera_pose(role),
                fr3.arm_q(),
                fr3.gripper_level(),
            )
            < NO_ROBOT
        )
        union = (truth | pred).sum()
        assert union == 0 or (truth & pred).sum() / union > 0.97, role
    masker.close()


@PHYSCODER
@pytest.mark.parametrize("block", worlds.BLOCK_LAYOUTS)
def test_physcoder_world_holds_its_layout(block):
    """The block comes to rest in a box corner (or on the table beside it) within
    physcoder's ranges; the arm and the open gripper hold; the cameras are the MJCF's
    wrist and the front camera."""
    world = worlds.build_world("physcoder_box_block", seed=3, block=block)
    world.reset()
    truth = world.ground_truth()
    box, blk = truth["objects"]["box"], truth["objects"]["block"]
    assert box["static"] and box["mass"] is None and len(box["collision"]) == 5
    assert not blk["static"] and blk["mass"] == pytest.approx(0.11)
    # The support is the black mat the cameras see (the table's collision box is
    # turned 90 degrees from it).
    support = truth["support"]
    assert support["height"] == pytest.approx(worlds.TABLE_TOP)
    assert support["size"] == pytest.approx([0.81, 1.35], abs=0.01)
    assert support["T_base_support"][:2, 3] == pytest.approx([-0.535, 0.0], abs=0.01)
    lo, hi = worlds.BOX_X, worlds.BOX_Y
    assert lo[0] <= box["T_base_obj"][0, 3] <= lo[1]
    assert hi[0] <= box["T_base_obj"][1, 3] <= hi[1]
    in_box = np.abs((np.linalg.inv(box["T_base_obj"]) @ blk["T_base_obj"])[:2, 3])
    bottom = blk["center"][2] - blk["size"][2] / 2
    if block == "corner":
        assert 0.07 < in_box.min() and in_box.max() < 0.1
        floor = box["T_base_obj"][2, 3] + worlds.BOX_FLOOR
        assert bottom == pytest.approx(floor, abs=2e-3) and blk["rests_on"] == "box"
    else:
        assert in_box[1] > worlds.BOX_OUTER + worlds.BESIDE_GAP[0]
        assert bottom == pytest.approx(worlds.TABLE_TOP, abs=2e-3)
    assert np.allclose(world.arm_q(), world.robot.home_q, atol=1e-3)
    assert world.gripper_level() < 0.01
    wrist, ext1 = world.camera_spec("wrist"), world.camera_spec("ext1")
    assert not wrist.is_static and (wrist.width, wrist.height) == (320, 240)
    assert wrist.K[0, 0] == pytest.approx(389.406, abs=1e-3)
    assert ext1.is_static and ext1.T_base_cam[:3, 3] == pytest.approx(
        worlds.FRONT_CAMERA[:3, 3]
    )
    assert world.clearance(0.05) == pytest.approx(0.05)  # its mount aside
    assert truth["source"]["assets"] == str(worlds.PHYSCODER_ASSETS)
    world.close()


def test_a_view_is_tried_closer():
    """A view far out is tried closer and closer, the nearest at CLOSEST; a near one
    only where it is."""
    view = View("v", np.zeros(3), np.array([0.0, 0.6, 0.8]), 0.8)
    distances = [np.linalg.norm(eye) for eye in view.eyes()]
    assert distances == pytest.approx([0.8, 0.68, CLOSEST])
    view.distance = 0.5
    assert [np.linalg.norm(eye) for eye in view.eyes()] == pytest.approx([0.5])


def test_unknown_worlds_and_parameters_are_named():
    """A wrong world or parameter says what is known."""
    with pytest.raises(ValueError, match="fr3_table"):
        worlds.build_world("kitchen")
    with pytest.raises(ValueError, match="takes no"):
        worlds.build_world("fr3_table", seed=1)


def test_a_builders_own_type_error_is_its_own(monkeypatch):
    """A TypeError inside a builder is not read as a wrong parameter."""

    def broken(seed: int = 0):
        raise TypeError(f"an API changed under seed {seed}")

    monkeypatch.setitem(worlds.WORLDS, "broken", broken)
    with pytest.raises(TypeError, match="an API changed"):
        worlds.build_world("broken", seed=2)
    with pytest.raises(ValueError, match="takes no"):
        worlds.build_world("broken", size=2)


@pytest.mark.gl
@pytest.mark.parametrize(
    "name, params",
    [
        pytest.param("fr3_table", {}, marks=FR3),
        pytest.param("physcoder_box_block", {"block": "corner"}, marks=PHYSCODER),
    ],
)
def test_recorded_capture(tmp_path, name, params):
    """A few frames of each camera, all static, a trajectory row per step, depth only
    within the sensor's range, the wrist moving around the objects with at least one
    wide view mostly support; the ground truth in the metadata rebuilds the world."""
    world = worlds.build_world(name, **params)
    capture = record_capture(world, tmp_path / "cap", every=60)
    world.close()
    capture = Capture.load(capture.root)
    assert {c.role for c in capture.cameras.values()} == {"ext1", "wrist"}
    traj = capture.trajectory
    assert traj is not None and np.array_equal(traj.steps, np.arange(len(traj.steps)))
    assert capture.static_steps == (0, len(traj.steps))
    assert np.allclose(traj.joint_positions[0], world.robot.home_q)
    assert np.max(np.abs(np.diff(traj.joint_positions, axis=0))) < 0.02
    assert np.all(traj.gripper_position < 0.01)  # open
    wrist = capture.frames_of("mj_wrist")
    assert len(wrist) > 3 and len(capture.frames_of("mj_ext1")) == len(wrist)
    poses = np.stack([f.T_base_cam for f in wrist])
    assert np.ptp(poses[:, :3, 3], axis=0).max() > 0.1
    depth = read_depth(capture.root / wrist[len(wrist) // 2].depth_image)
    seen = depth[depth > 0]
    assert (
        seen.size
        and seen.min() >= SENSOR_MIN_DEPTH - 1e-3
        and seen.max() <= SENSOR_MAX_DEPTH
    )
    meta = capture.metadata
    assert set(meta) == {"depth", "world", "ground_truth", "scan"}
    assert meta["world"] == {"name": name, "params": world.params}
    assert set(meta["ground_truth"]) == TRUTH_KEYS
    visited = [v for v in meta["scan"] if "step" in v]
    assert len(visited) >= 4
    assert max(v["support_fraction"] for v in visited) >= 0.4
    json.dumps(meta)  # plain JSON
    rebuilt = worlds.world_from_capture(capture)
    for obj, gt in meta["ground_truth"]["objects"].items():
        assert np.allclose(rebuilt.body_pose(obj), gt["T_base_obj"], atol=1e-6)
    rebuilt.close()


@FR3
def test_a_changed_world_is_refused(tmp_path):
    """A capture whose layout the world no longer has is not picked in."""
    world = worlds.build_world("fr3_table")
    truth = world.ground_truth()
    world.close()
    truth["layout"]["objects"][0]["xy"] = [0.1, 0.1]
    capture = Capture(
        "c", "mujoco", "fr3_robotiq", "", {}, [], (0, 1), Path(tmp_path),
        metadata={"world": {"name": "fr3_table", "params": {}}, "ground_truth": truth},
    )  # fmt: skip
    with pytest.raises(ValueError, match="no longer has"):
        worlds.world_from_capture(capture)
