"""Tests for sim/mjscene.py: a reconstructed scene in MuJoCo, its objects resting on
the support, an articulated object's joints moving as their dynamics say, the robot
held and driven."""

import json
from dataclasses import replace

import numpy as np
import pytest
from conftest import hinged_box, rgbd_capture, robot_or_skip, towel_mesh

from r2s2r.assets import urdf_visual_meshes
from r2s2r.sim.mjscene import CONDIM, Session
from r2s2r.structs import JointDynamics, ObjectSpec, SceneSpec
from r2s2r.tools.objects import assemble
from r2s2r.transforms import look_at, make_transform, transform_points
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
    robot = robot_or_skip("fr3_robotiq")
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


def test_every_contact_has_torsional_and_rolling_friction(tmp_path):
    """Every geom that collides has MuJoCo's torsional and rolling friction (condim
    6), the gripper's pads too, whose priority makes their contacts theirs: without
    it a block pinched between the pads turns about them as the arm carries it."""
    robot = robot_or_skip("fr3_robotiq")
    scene = replace(_box_scene(tmp_path), embodiment=robot.name)
    m = Session(scene, robot).model
    colliding = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    pads = [i for i in np.flatnonzero(colliding) if "pad" in m.geom(i).name]
    assert pads and set(m.geom_condim[colliding]) == {CONDIM}
    assert all(m.geom_friction[i][1] > 0 for i in pads)


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
    assert shut == pytest.approx(
        robot.arm.max_opening / robot.arm.gripper.speed, rel=0.2
    )


def test_a_drawn_objects_physics_is_set_and_the_state_kept(tmp_path):
    """Set to a draw, the box weighs what it says, its colliders grip with its friction
    and its lid moves as its joint now says; where everything is stays."""
    scene = _box_scene(tmp_path)
    box = replace(scene.objects[0], mass=0.4)
    session = Session(replace(scene, objects=[box]))
    session.set_joints("box", {"hinge": -0.8})
    poses = session.object_poses()
    drawn = replace(
        box,
        mass=0.8,
        friction=0.9,
        joint_dynamics={"hinge": JointDynamics(friction=0.5)},
    )
    session.set_physics(drawn)
    m = session.model
    ids = [b for b in range(m.nbody) if m.body(b).name in ("box", "box/lid")]
    assert m.body_mass[ids].sum() == pytest.approx(0.8)
    colliders = [
        g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and m.geom_group[g] == 3
    ]
    assert colliders and all(m.geom_friction[g][0] == 0.9 for g in colliders)
    np.testing.assert_allclose(session.object_poses()["box"], poses["box"])
    assert session.object_joints()["box"]["hinge"] == pytest.approx(-0.8)
    session.step(int(1.5 / session.dt))
    assert session.object_joints()["box"]["hinge"] < -0.7  # held by the drawn friction


def test_two_arms_are_set_and_driven_each_by_its_own(tmp_path):
    """RoboDojo's two arms: each follows its own targets and its gripper its own
    opening; the session gives one opening per arm."""
    robot = robot_or_skip("dual_x5")
    home = np.asarray(robot.home_q)
    scene = replace(_box_scene(tmp_path), embodiment=robot.name, joint_positions=home)
    session = Session(scene, robot)
    assert session.arm_q() == pytest.approx(home)
    assert session.gripper_level() == pytest.approx([0.0, 0.0])
    # The left arm reaching ahead, the right one up; both well above the table.
    target = (
        home + np.r_[0.3, 0.8, 0.9, -0.4, 0.2, -0.3, -0.3, 0.6, 0.8, 0.3, -0.2, 0.3]
    )
    session.command(target, [1.0, 0.0])
    session.step(int(1.5 / session.dt))
    assert np.abs(session.arm_q() - target).max() < 0.02
    left, right = session.gripper_level()
    assert left > 0.95 and right < 0.05


def _with_towel(tmp_path):
    """The hinged box's scene and a 30 cm towel through it, 5 cm up."""
    scene = _box_scene(tmp_path)
    towel_mesh(0.3).export(tmp_path / "towel.obj")
    objects = {
        "support": {"T_base_support": np.eye(4).tolist(), "extent": [1.0, 1.0]},
        "objects": [
            {
                "name": "towel",
                "mesh": "towel.obj",
                "T_base_obj": make_transform(np.eye(3), [0.5, 0.0, 0.05]).tolist(),
                "mass": 0.05,
                "cloth": {"thickness": 0.002, "youngs_modulus": 5e5},
            }
        ],
    }
    (tmp_path / "towel.json").write_text(json.dumps(objects))
    ws = Workspace.load(tmp_path / "run")
    assemble(ws, tmp_path / "towel.json", tmp_path / "towel_scene", "hull")
    (towel,) = SceneSpec.load(tmp_path / "towel_scene").objects
    assert isinstance(towel, ObjectSpec) and towel.cloth
    return replace(scene, objects=[*scene.objects, towel]), towel


def test_a_cloth_stays_where_it_lies_and_touches_nothing(tmp_path):
    """A towel draped through the box (its surface where the lid is): no free joint,
    held where the scene has it, at rest; the box settles as it would without it."""
    scene, towel = _with_towel(tmp_path)
    session = Session(scene)
    assert "towel/root" not in [
        session.model.joint(j).name for j in range(session.model.njnt)
    ]
    session.step(int(1.0 / session.dt))
    assert np.allclose(session.object_poses()["towel"], towel.T_base_obj)
    assert not session.object_velocities()["towel"].any()
    box = session.object_poses()["box"]
    assert np.linalg.norm(box[:3, 3] - ON_TABLE[:3, 3]) < 1e-3
    with pytest.raises(ValueError, match="a cloth"):
        session.place("towel", ON_TABLE)


@pytest.mark.gl
def test_a_cloth_is_drawn_where_it_moved(tmp_path):
    """Seen from above, the towel's surface 10 cm higher is 10 cm nearer, in a render
    made before it moved too."""
    scene, towel = _with_towel(tmp_path)
    session = Session(scene)
    K = np.array([[300.0, 0.0, 79.5], [0.0, 300.0, 59.5], [0.0, 0.0, 1.0]])
    above = look_at((0.5, 0.06, 0.6), (0.55, 0.06, 0.0))  # over the towel, off the box
    try:
        before = session.render(K, 160, 120, above)["depth"][60, 80]
        surface = urdf_visual_meshes(towel.asset_path)[0].mesh.vertices
        lifted = transform_points(towel.T_base_obj, surface) + [0.0, 0.0, 0.1]
        session.show_surface("towel", lifted)
        after = session.render(K, 160, 120, above)["depth"][60, 80]
    finally:
        session.close()
    assert before == pytest.approx(0.55, abs=0.01)
    assert after == pytest.approx(before - 0.1, abs=0.01)
