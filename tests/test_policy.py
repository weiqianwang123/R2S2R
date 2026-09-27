"""Tests for testbed/policy.py and testbed/pick.py: the motion primitives and the pick
program on a kinematic robot, for every robot (skipped without its assets), and
choosing the scene object a world's target is."""

from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.robots import ROBOTS, get_robot, ur5e_2f140  # noqa: E402
from r2s2r.robots.spec import MENAGERIE_DIR  # noqa: E402
from r2s2r.structs import Capture, ObjectSpec, SceneSpec  # noqa: E402
from r2s2r.testbed.pick import match_target  # noqa: E402
from r2s2r.testbed.policy import (  # noqa: E402
    RobotInterface,
    find_object,
    gripper_geometry,
    pick_up,
    plan_top_down_grasp,
    score_lift,
)
from r2s2r.transforms import invert, make_transform  # noqa: E402

BOX = (0.03, 0.07, 0.12)  # thin along x in its own frame
ASSETS = {
    "franka_panda": MENAGERIE_DIR / "franka_emika_panda",
    "droid_franka": MENAGERIE_DIR / "robotiq_2f85",
    "ur5e_2f140": ur5e_2f140.MJCF_PATH,
}
# The closing axis in the TCP frame, and how far the fingertips reach past the TCP.
GRIPPERS = {
    "franka_panda": ((0, 1, 0), 0.009),
    "droid_franka": ((0, 1, 0), 0.019),
    "ur5e_2f140": ((1, 0, 0), 0.036),
}


def _robot(name):
    if not Path(ASSETS[name]).exists():
        pytest.skip(f"{name}'s MJCF is not here ({ASSETS[name]})")
    return get_robot(name)


class KinematicRobot(RobotInterface):
    """Tracks every command exactly; the gripper stops at ``stop`` (its level on the
    object)."""

    def __init__(self, robot, stop: float) -> None:
        super().__init__(robot)
        self.q = np.asarray(robot.home_q) + 0.2
        self.level = 0.0
        self.stop = stop

    def joint_positions(self):
        return self.q.copy()

    def gripper_level(self):
        return self.level

    def _hold(self, q, level):
        self.q = q.copy()
        self.level = min(level, self.stop)


def _scene(tmp_path, robot, yaw_deg):
    """A box standing on the table, below where the robot's home pose points."""
    mesh = trimesh.creation.box(extents=BOX)
    mesh.apply_translation([0, 0, BOX[2] / 2])
    mesh.export(tmp_path / "box.obj")
    (tmp_path / "box.urdf").write_text(
        '<robot name="box"><link name="base"><visual><geometry>'
        '<mesh filename="box.obj"/></geometry></visual></link></robot>'
    )
    probe = KinematicRobot(robot, 1.0)
    probe.model.set(np.asarray(robot.home_q))
    x, y = probe.model.tcp_pose()[:2, 3] + 0.05
    T = make_transform(
        Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix(), [x, y, 0.0]
    )
    obj = ObjectSpec("crayon_box", "box", str(tmp_path / "box.urdf"), T)
    return SceneSpec(
        name="t",
        embodiment=robot.name,
        objects=[obj],
        T_base_support=np.eye(4),
        cameras={},
        reference_camera="c",
        reference_step=0,
        joint_positions=np.asarray(robot.home_q),
    )


def _scene_of(objects):
    return SceneSpec(
        "s", "franka_panda", objects, np.eye(4), {}, "c", 0, np.zeros(7)
    )  # fmt: skip


@pytest.fixture(name="robot", params=sorted(ROBOTS))
def fixture_robot(request):
    """Each robot's spec."""
    return _robot(request.param)


def test_gripper_geometry_from_the_model(robot):
    """Each gripper closes along the expected TCP axis; its tips reach past the TCP by
    its known amount."""
    geometry = gripper_geometry(KinematicRobot(robot, 1.0).model)
    axis, tip = GRIPPERS[robot.name]
    assert abs(geometry.closing_axis @ axis) == pytest.approx(1.0, abs=1e-3)
    assert geometry.tip_depth == pytest.approx(tip, abs=1e-3)


def test_grasp_closes_across_the_thin_side(tmp_path, robot):
    """The fingers close along the box's 3 cm side, from above, with the tips clear of
    the table."""
    scene = _scene(tmp_path, robot, 30.0)
    geometry = gripper_geometry(KinematicRobot(robot, 1.0).model)
    plan = plan_top_down_grasp(scene, "crayon_box", geometry, robot.max_opening)
    assert plan.width == pytest.approx(BOX[0], abs=3e-3)
    closing = plan.T_base_tcp[:3, :3] @ geometry.closing_axis
    thin = Rotation.from_euler("z", 30.0, degrees=True).apply([1.0, 0.0, 0.0])
    assert abs(closing @ thin) > 0.99
    assert np.allclose(plan.T_base_tcp[:3, 2], [0.0, 0.0, -1.0])
    assert plan.T_base_tcp[2, 3] == pytest.approx(BOX[2] - 0.03, abs=2e-3)
    assert plan.T_base_tcp[2, 3] - geometry.tip_depth > 0.0  # the tips clear the table
    with pytest.raises(ValueError, match="too wide"):
        plan_top_down_grasp(scene, "crayon_box", geometry, 0.03)


def test_pick_up_reaches_the_grasp(tmp_path, robot):
    """On a perfect robot the program passes through the grasp in small joint steps,
    closes and lifts."""
    scene = _scene(tmp_path, robot, -40.0)
    bot = KinematicRobot(robot, stop=0.6)
    result = pick_up(bot, scene, "crayon")
    assert result["holding"] and result["gripper_level_after_lift"] == 0.6
    assert result["grasp_ik_error_m"] < 1e-3
    q = np.array(bot.log.q)
    assert q.shape[1] == robot.dof
    assert np.max(np.abs(np.diff(q, axis=0))) < 0.05  # no jumps between steps
    grasp = np.array(result["grasp"]["T_base_tcp"])
    tcp_z = []
    for qi in q:
        bot.model.set(qi)
        tcp_z.append(bot.model.tcp_pose()[2, 3])
    assert min(tcp_z) < grasp[2, 3] + 1e-3
    lifted = bot.tcp_pose()
    assert lifted[2, 3] == pytest.approx(grasp[2, 3] + 0.15, abs=1e-3)
    assert np.allclose(invert(lifted)[:3, :3] @ grasp[:3, :3], np.eye(3), atol=1e-2)
    levels = np.array(bot.log.gripper)
    assert levels[0] == 0.0 and levels[-1] == 1.0


def test_find_object_and_lift_score():
    """Unique substrings resolve; the target must rise 5 cm."""
    box = ObjectSpec("crayon_box", "box", "", np.eye(4))
    assert find_object(_scene_of([box]), "crayon").name == "crayon_box"
    before = {"a": np.zeros(3), "b": np.zeros(3)}
    after = {"a": np.array([0.0, 0.0, 0.06]), "b": np.array([0.01, 0.0, 0.0])}
    score = score_lift(before, after, "a")
    assert score["success"] and score["object_shift_m"]["b"] == pytest.approx(0.01)
    assert not score_lift(before, after, "b")["success"]


def test_match_target_takes_the_nearest_match(tmp_path):
    """Of two scene objects matched to the target, the nearer is the target; none
    matched is an error."""
    mesh = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    mesh.export(tmp_path / "cube.obj")
    (tmp_path / "cube.urdf").write_text(
        '<robot name="c"><link name="base"><visual><geometry>'
        '<mesh filename="cube.obj"/></geometry></visual></link></robot>'
    )

    def obj(name, x):
        return ObjectSpec(
            name,
            name,
            str(tmp_path / "cube.urdf"),
            make_transform(np.eye(3), [x, 0, 0]),
        )

    truth = {
        "target": "block",
        "support": {"T_base_support": np.eye(4), "height": 0.0, "size": [1, 1]},
        "objects": {
            n: {"T_base_obj": np.eye(4), "center": [x, 0, 0], "size": [0.02] * 3}
            for n, x in (("block", 0.5), ("box", 0.0))
        },
    }
    capture = Capture(
        "c", "mujoco", "franka_panda", "", {}, [], (0, 1), tmp_path,
        metadata={"ground_truth": truth},
    )  # fmt: skip
    scene = _scene_of([obj("far", 0.42), obj("near", 0.49), obj("other", 0.01)])
    assert match_target(scene, capture) == "near"
    scene.objects = [obj("other", 0.01)]
    with pytest.raises(KeyError, match="block"):
        match_target(scene, capture)
