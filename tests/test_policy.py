"""Tests for testbed/policy.py and testbed/pick.py: the motion primitives and the pick
program on a kinematic robot, for every robot (skipped without its assets), and
choosing the scene object a world's target is."""

import numpy as np
import pytest
from conftest import box_urdf, robot_or_skip, towel_mesh
from scipy.spatial.transform import Rotation

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.assets import ROOT_LINK, UrdfLink, write_object_urdf  # noqa: E402
from r2s2r.structs import Capture, ObjectSpec, SceneSpec  # noqa: E402
from r2s2r.testbed.pick import match_target  # noqa: E402
from r2s2r.testbed.policy import (  # noqa: E402
    RobotInterface,
    find_object,
    gripper_geometry,
    opening_at_gap,
    pick_up,
    plan_cloth_pinch,
    plan_top_down_grasp,
    score_lift,
)
from r2s2r.transforms import invert, make_transform  # noqa: E402

BOX = (0.03, 0.07, 0.12)  # thin along x in its own frame
# The closing axis in the TCP frame, how far the fingertips reach past the TCP, and
# the opening at which the fingers meet (the 2F-140's pads touch before it is shut).
GRIPPERS = {
    "fr3_robotiq": ((0, 1, 0), 0.019, 1.0),
    "ur5e_2f140": ((1, 0, 0), 0.036, 0.9),
}


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
    urdf = box_urdf(tmp_path, BOX)
    probe = KinematicRobot(robot, 1.0)
    probe.model.set(np.asarray(robot.home_q))
    x, y = probe.model.tcp_pose()[:2, 3] + 0.05
    T = make_transform(
        Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix(), [x, y, 0.0]
    )
    obj = ObjectSpec("crayon_box", "box", str(urdf), T)
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
        "s", "fr3_robotiq", objects, np.eye(4), {}, "c", 0, np.zeros(7)
    )  # fmt: skip


@pytest.fixture(name="robot", params=sorted(GRIPPERS))
def fixture_robot(request):
    """Each one-armed robot's spec."""
    return robot_or_skip(request.param)


def test_gripper_geometry_from_the_model(robot):
    """Each gripper closes along the expected TCP axis; its tips reach past the TCP by
    its known amount; its fingers meet where they do."""
    geometry = gripper_geometry(KinematicRobot(robot, 1.0).model)
    axis, tip, empty = GRIPPERS[robot.name]
    assert abs(geometry.closing_axis @ axis) == pytest.approx(1.0, abs=1e-3)
    assert geometry.tip_depth == pytest.approx(tip, abs=1e-3)
    assert geometry.empty_level == pytest.approx(empty, abs=0.01)


def test_grasp_closes_across_the_thin_side(tmp_path, robot):
    """The fingers close along the box's 3 cm side, from above, with the tips clear of
    the table."""
    scene = _scene(tmp_path, robot, 30.0)
    geometry = gripper_geometry(KinematicRobot(robot, 1.0).model)
    plan = plan_top_down_grasp(scene, "crayon_box", geometry, robot.arm.max_opening)
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
    closes and lifts, holding what stopped its fingers (not what did not)."""
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
    empty = GRIPPERS[robot.name][2]
    assert not pick_up(KinematicRobot(robot, stop=empty), scene, "crayon")["holding"]


def _towel_scene(tmp_path, robot, offset=(0.1, 0.0), T_base_support=np.eye(4)):
    """A 30 cm towel lying on the table ``offset`` (x, y) from where the robot's home
    pose points (the support at ``T_base_support``)."""
    towel_mesh(0.3, 15).export(tmp_path / "visual.obj")
    urdf = tmp_path / "towel.urdf"
    link = UrdfLink(ROOT_LINK, "visual.obj", 0.05, np.zeros(3), np.eye(3) * 1e-4, [])
    write_object_urdf(urdf, "towel", [link])
    probe = KinematicRobot(robot, 1.0)
    probe.model.set(np.asarray(robot.home_q))
    x, y = probe.model.tcp_pose()[:2, 3] + offset
    cloth = {"thickness": 0.002, "youngs_modulus": 5e5, "poissons_ratio": 0.3}
    T = T_base_support @ make_transform(np.eye(3), [x, y, 0.001])
    obj = ObjectSpec("towel", "towel", str(urdf), T, mass=0.05, cloth=cloth)
    return SceneSpec(
        "t", robot.name, [obj], T_base_support, {}, "c", 0, np.asarray(robot.home_q)
    )


def test_the_pads_close_to_a_gap(robot):
    """The narrower the gap between the pads, the more closed the gripper; no gap is
    where the fingers meet."""
    model = KinematicRobot(robot, 1.0).model
    openings = [opening_at_gap(model, gap) for gap in (0.05, 0.02, 0.004)]
    assert openings == sorted(openings) and openings[0] > 0.0
    assert opening_at_gap(model, 0.0) == pytest.approx(
        GRIPPERS[robot.name][2], abs=0.01
    )


def test_a_cloth_is_pinched_at_its_border_within_reach(tmp_path, robot):
    """A towel: pinched 4 cm in from its border, pointing down, closing across the line
    to its middle, the fingertips a millimetre above the table, where the arm reaches;
    the program closes to its thickness and leaves its rise to tell. A grasp of a
    body refuses it, and a pinch of a body."""
    scene = _towel_scene(tmp_path, robot)
    bot = KinematicRobot(robot, 1.0)
    geometry = gripper_geometry(bot.model)
    plan = plan_cloth_pinch(scene, "towel", geometry, bot.model)
    T = plan.T_base_tcp
    towel = scene.objects[0].T_base_obj[:2, 3]
    to_border = 0.15 - np.abs(T[:2, 3] - towel).max()  # inside the towel, 4 cm at most
    assert 0.0 < to_border < 0.04 + 1e-6
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0])
    closing = T[:3, :3] @ geometry.closing_axis
    inward = (towel - T[:2, 3]) / np.linalg.norm(towel - T[:2, 3])
    assert abs(closing[:2] @ inward) < 1e-6 and abs(closing[2]) < 1e-6
    assert T[2, 3] - geometry.tip_depth == pytest.approx(0.001, abs=1e-6)
    assert plan.width == pytest.approx(0.002)
    reached = bot.model.ik(T, np.asarray(robot.home_q))
    assert reached.pos_error < 1e-3 and reached.rot_error < 0.01
    result = pick_up(bot, scene, "towel")
    assert result["holding"] is None and result["grasp_ik_error_m"] < 1e-3
    pinch = opening_at_gap(bot.model, 0.002)
    assert bot.log.gripper[-1] == pytest.approx(pinch, abs=1e-3)
    with pytest.raises(ValueError, match="pinched .*, not grasped"):
        plan_top_down_grasp(scene, "towel", geometry, robot.arm.max_opening)
    with pytest.raises(ValueError, match="no cloth"):
        plan_cloth_pinch(_scene(tmp_path, robot, 0.0), "crayon_box", geometry)


def test_a_cloth_is_pinched_where_the_arm_reaches(tmp_path, robot):
    """Near the base, the towel's nearest border is out of reach (the arm cannot point
    straight down there): a farther point is pinched; a towel 3 m away is refused. On
    a tilted table the fingertips stop a millimetre above it."""
    bot = KinematicRobot(robot, 1.0)
    geometry = gripper_geometry(bot.model)
    home = bot.tcp_pose()[:2, 3]
    near = _towel_scene(tmp_path, robot, offset=-0.8 * home)  # at a fifth of the way
    plan = plan_cloth_pinch(near, "towel", geometry, bot.model)
    unchecked = plan_cloth_pinch(near, "towel", geometry)  # the nearest point
    T = plan.T_base_tcp
    assert np.linalg.norm(T[:2, 3]) > np.linalg.norm(unchecked.T_base_tcp[:2, 3])
    reached = bot.model.ik(T, np.asarray(robot.home_q))
    assert reached.pos_error < 1e-3 and reached.rot_error < 0.01
    far = _towel_scene(tmp_path, robot, offset=(3.0, 0.0))
    with pytest.raises(ValueError, match="within the arm's reach"):
        plan_cloth_pinch(far, "towel", geometry, bot.model)
    tilt = make_transform(
        Rotation.from_euler("y", 2, degrees=True).as_matrix(), [0] * 3
    )
    tilted = _towel_scene(tmp_path, robot, T_base_support=tilt)
    T = plan_cloth_pinch(tilted, "towel", geometry).T_base_tcp
    tips = T[:3, 3] - geometry.tip_depth * np.array([0.0, 0.0, 1.0])
    above = (np.linalg.inv(tilt) @ np.r_[tips, 1.0])[2]
    assert above == pytest.approx(0.001, abs=2e-4)  # the TCP points straight down


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
    cube = str(box_urdf(tmp_path, (0.02, 0.02, 0.02), "cube"))

    def obj(name, x):
        return ObjectSpec(name, name, cube, make_transform(np.eye(3), [x, 0, -0.01]))

    truth = {
        "target": "block",
        "support": {"T_base_support": np.eye(4), "height": 0.0, "size": [1, 1]},
        "objects": {
            n: {"T_base_obj": np.eye(4), "center": [x, 0, 0], "size": [0.02] * 3}
            for n, x in (("block", 0.5), ("box", 0.0))
        },
    }
    capture = Capture(
        "c", "mujoco", "fr3_robotiq", "", {}, [], (0, 1), tmp_path,
        metadata={"ground_truth": truth},
    )  # fmt: skip
    scene = _scene_of([obj("far", 0.42), obj("near", 0.49), obj("other", 0.01)])
    assert match_target(scene, capture) == "near"
    scene.objects = [obj("other", 0.01)]
    with pytest.raises(KeyError, match="block"):
        match_target(scene, capture)
