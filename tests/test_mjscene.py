"""Tests for sim/mjscene.py: a reconstructed scene in MuJoCo, its objects resting on
the support, an articulated object's joints moving as their dynamics say, the robot
held and driven."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import hinged_box, rgbd_capture, robot_or_skip

from r2s2r.sim.mjscene import Session
from r2s2r.structs import JointDynamics, SceneSpec
from r2s2r.tools.objects import assemble
from r2s2r.transforms import look_at, make_transform
from r2s2r.workspace import Workspace

pytest.importorskip("mujoco")

ON_TABLE = make_transform(np.eye(3), [0.5, 0.0, 0.0])
CAMERAS = {  # (a capture's, which a run needs)
    "c1": ("ext1", look_at((0.95, 0.35, 0.45), (0.5, 0.0, 0.04))),
    "c2": ("wrist", look_at((0.45, -0.4, 0.4), (0.5, 0.0, 0.04))),
}


def _box_scene(tmp_path):
    """The hinged box of :func:`conftest.hinged_box`, assembled, on the table: its lid
    turns from 0 (shut) to -1.9 rad (open), recorded at -0.8."""
    capture = rgbd_capture(tmp_path / "capture", CAMERAS)
    ws = Workspace.create(capture, tmp_path / "run", "agentic")
    hinged_box(tmp_path, ON_TABLE)
    assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "hull")
    return SceneSpec.load(tmp_path / "scene")


def _lid(scene, dynamics, angle, seconds=1.5):
    """The lid's angle ``seconds`` after it is let go at ``angle``."""
    box = replace(scene.objects[0], joint_dynamics={"hinge": dynamics})
    session = Session(replace(scene, objects=[box]))
    session.set_joints("box", {"hinge": angle})
    session.step(int(seconds / session.dt))
    return session.object_joints()["box"]["hinge"]


def test_an_object_rests_on_the_support_and_its_joints_are_as_recorded(tmp_path):
    """Settling moves the box by less than a millimetre; it starts with its lid where
    the scene has it."""
    scene = _box_scene(tmp_path)
    session = Session(scene)
    assert session.object_joints()["box"]["hinge"] == pytest.approx(-0.8)
    before = session.object_poses()["box"]
    session.step(int(1.0 / session.dt))
    after = session.object_poses()["box"]
    assert np.linalg.norm(after[:3, 3] - before[:3, 3]) < 1e-3


def test_a_lid_moves_as_its_dynamics_say(tmp_path):
    """Free, a lid let go short of upright falls shut; its dry friction holds it where
    it is let go; a spring pulls it back past where gravity would leave it."""
    scene = _box_scene(tmp_path)
    assert _lid(scene, JointDynamics(), -0.8) > -0.05  # shut
    assert _lid(scene, JointDynamics(friction=0.5), -0.8) < -0.7  # held
    # Let go past upright, gravity opens the lid fully; a spring toward shut closes it.
    assert _lid(scene, JointDynamics(), -1.75) < -1.85
    assert _lid(scene, JointDynamics(stiffness=0.5), -1.75) > -0.05


def test_the_robot_is_set_and_driven(tmp_path):
    """Set, the arm stands at the joints given; driven, its actuators take it to a
    target; the gripper closes."""
    robot = robot_or_skip("franka_panda")
    scene = _box_scene(tmp_path)
    session = Session(scene, robot)
    home = np.asarray(robot.home_q)
    session.set_robot(home, 0.0)
    assert session.arm_q() == pytest.approx(home)
    target = home + np.array([0.2, -0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
    session.command(target, 1.0)
    session.step(int(1.5 / session.dt))
    assert np.abs(session.arm_q() - target).max() < 0.02
    assert session.gripper_level() > 0.5


def test_the_gripper_closes_at_the_real_ones_speed(tmp_path):
    """Commanded shut from open, the gripper's opening closes over about the time the
    real one takes at its speed (max_opening / speed), not at once."""
    robot = robot_or_skip("fr3_robotiq")
    scene = replace(_box_scene(tmp_path), embodiment=robot.name)
    session = Session(scene, robot)
    session.set_robot(np.asarray(robot.home_q), 0.0)
    session.command(np.asarray(robot.home_q), 1.0)
    levels = []
    for _ in range(int(2.0 / session.dt)):
        session.step()
        levels.append(session.gripper_level())
    shut = (np.argmax(np.asarray(levels) > 0.95) + 1) * session.dt
    assert shut == pytest.approx(robot.max_opening / robot.gripper.speed, rel=0.2)
