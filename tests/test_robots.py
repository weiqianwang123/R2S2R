"""Tests for the robot registry and robots/model.py (skipped per robot without its
assets), the masker and the viewer's robot on the UR5e."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import robot_or_skip

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.robots import (  # noqa: E402
    ROBOTS,
    droid_franka,
    fr3_robotiq,
    get_robot,
    ur5e_2f140,
)
from r2s2r.robots.model import RobotModel  # noqa: E402
from r2s2r.transforms import intrinsics_matrix, invert, look_at  # noqa: E402


@pytest.fixture(name="model", scope="module", params=sorted(ROBOTS))
def fixture_model(request):
    """Each robot's model."""
    return RobotModel(robot_or_skip(request.param))


def test_registry_names_the_known_robots():
    """Every robot is registered under its name; unknown names list the known ones."""
    assert set(ROBOTS) == {"franka_panda", "droid_franka", "fr3_robotiq", "ur5e_2f140"}
    assert all(get_robot(n).name == n for n in ROBOTS)
    with pytest.raises(ValueError, match="franka_panda"):
        get_robot("ur10")


def test_robot_compiles_with_its_arm_and_gripper(model):
    """The spec's joints, actuator and bodies exist; the home pose is within limits."""
    robot = model.robot
    assert model.arm_qadr.shape == (robot.dof,) == (len(robot.home_q),)
    dofs = {"franka_panda": 7, "droid_franka": 7, "fr3_robotiq": 7, "ur5e_2f140": 6}
    assert robot.dof == dofs[robot.name]
    model.model.actuator(robot.gripper.actuator)
    model.model.body(robot.tcp_body)
    assert np.all(model.q_min <= robot.home_q) and np.all(robot.home_q <= model.q_max)
    assert 0.05 < robot.max_opening < 0.15


def test_gripper_opens_and_closes(model):
    """Levels 0 and 1 put the driver at its ends, and the fingers apart and together."""
    gripper = model.robot.gripper
    driver = model.gripper.joints.index(gripper.driver)
    assert len(model.gripper.joints) > 1
    widths = []
    for level, end, tol in ((0.0, gripper.open, 0.01), (1.0, gripper.closed, 0.03)):
        assert abs(model.gripper.positions(level)[driver] - end) < tol
        model.set(np.asarray(model.robot.home_q), level)
        m, d = model.model, model.data
        tips = [
            invert(model.tcp_pose())[:3] @ np.r_[d.geom_xpos[g], 1.0]
            for g in range(m.ngeom)
            if any(k in m.body(m.geom_bodyid[g]).name for k in ("finger", "pad"))
        ]
        widths.append(np.ptp(np.array(tips)[:, :2], axis=0).max())
    assert widths[0] > widths[1] + 0.03


@pytest.mark.parametrize("name", ["droid_franka", "fr3_robotiq", "ur5e_2f140"])
def test_robotiq_tcp_is_between_the_closed_pads(name):
    """Closed, the Robotiq's pads meet at the TCP."""
    model = RobotModel(robot_or_skip(name))
    model.set(np.asarray(model.robot.home_q), 1.0)
    m, d = model.model, model.data
    pads = [
        g for g in range(m.ngeom) if (m.geom(g).name or "").endswith(("pad1", "pad2"))
    ]
    assert len(pads) == 4
    centre = np.mean([d.geom_xpos[g] for g in pads], axis=0)
    assert np.linalg.norm((invert(model.tcp_pose()) @ np.r_[centre, 1.0])[:3]) < 0.01


def test_ur5e_followers_follow_the_mjcf_equalities():
    """The 2F-140's followers are the MJCF's linear equalities of finger_joint."""
    model = RobotModel(robot_or_skip("ur5e_2f140"))
    closed = dict(zip(model.gripper.joints, model.gripper.positions(1.0)))
    q = ur5e_2f140.FINGER_CLOSED
    assert closed["finger_joint"] == pytest.approx(q)
    assert closed["right_inner_knuckle_joint"] == pytest.approx(-q)
    for joint in ("right_outer_knuckle_joint", "left_inner_finger_pad_joint"):
        assert closed[joint] == pytest.approx(q)
    # Isaac's table (measured there) has the same joints and closed values.
    isaac = ur5e_2f140.ISAAC_GRIPPER
    assert {n: c for n, (_, c) in isaac.items()} == pytest.approx(closed)
    assert all(o == 0.0 for o, _ in isaac.values())


def test_fk_ik_round_trip(model):
    """IK from the home pose finds poses that FK produced, for the TCP and a body."""
    robot = model.robot
    rng = np.random.default_rng(0)
    home = np.asarray(robot.home_q)
    for _ in range(5):
        q = home + rng.uniform(-0.3, 0.3, robot.dof)
        model.set(q, 0.0)
        T_tcp, T_body = model.tcp_pose(), model.pose(robot.tcp_body)
        result = model.ik(T_tcp, home)
        assert result.success and result.pos_error < 1e-4, result
        model.set(result.q)
        assert np.allclose(model.tcp_pose(), T_tcp, atol=1e-3)
        assert model.ik(T_body, home, frame=robot.tcp_body).success


def test_ik_stays_within_joint_limits_and_reports_failure(model):
    """A target out of reach fails, with the error left, and the joints in limits."""
    robot = model.robot
    model.set(np.asarray(robot.home_q))
    T = model.tcp_pose()
    T[:3, 3] += [3.0, 0.0, 0.0]
    result = model.ik(T, np.asarray(robot.home_q), max_iters=50)
    assert not result.success and result.pos_error > 1.0
    assert np.all(result.q >= model.q_min) and np.all(result.q <= model.q_max)


def test_ik_on_a_camera_frame_uses_the_opencv_convention():
    """The UR5e's wrist camera can be pointed at a target (OpenCV camera frame)."""
    model = RobotModel(robot_or_skip("ur5e_2f140"))
    home = np.asarray(model.robot.home_q)
    T_cam = model.pose("wrist", "camera")
    assert T_cam[2, 2] < -0.9  # at home the wrist camera looks down
    target = look_at(T_cam[:3, 3] + [0.05, 0.05, 0.0], [-0.5, 0.0, 0.0])
    result = model.ik(target, home, frame="wrist", kind="camera")
    assert result.success, result
    assert np.allclose(model.pose("wrist", "camera"), target, atol=1e-3)


@pytest.mark.parametrize("joint_name", ["", "obj_free"])
def test_simulated_gripper_leaves_a_worlds_other_joints_alone(joint_name):
    """In a world with a free object, the 2F-85's table covers the gripper only, and
    posing the robot does not move the object."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco

    robot = robot_or_skip("droid_franka")
    mjspec = robot.mjcf()
    body = mjspec.worldbody.add_body(name="obj", pos=[0.5, 0.0, 0.1])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02])
    body.add_freejoint().name = joint_name
    model = RobotModel(robot, mjspec)
    assert model.gripper.joints == [
        model.model.joint(j).name
        for j in range(model.model.njnt)
        if model.model.joint(j).name.startswith(droid_franka.PREFIX)
    ]
    adr = model.model.body("obj").jntadr[0]
    qadr = model.model.jnt_qposadr[adr]
    moved = np.array([0.3, -0.2, 0.4, 0.0, 1.0, 0.0, 0.0])
    model.data.qpos[qadr : qadr + 7] = moved
    model.set(np.asarray(robot.home_q), 1.0)
    assert np.array_equal(model.data.qpos[qadr : qadr + 7], moved)


def test_droid_robotiq_mount_matches_the_mjcf():
    """Isaac's re-mount constant is where the MuJoCo model has the Robotiq base."""
    model = RobotModel(robot_or_skip("droid_franka"))
    T = invert(model.pose("link7")) @ model.pose(f"{droid_franka.PREFIX}base")
    expected = np.eye(4)
    expected[2, 3] = droid_franka.ROBOTIQ_BASE_Z
    assert np.allclose(T, expected, atol=1e-4)


def test_fr3_and_droid_franka_put_the_tcp_in_the_same_place():
    """The FR3 has the Panda's kinematics and the Robotiq on the same coupling: the same
    joints give the same TCP pose, within both arms' limits."""
    fr3 = RobotModel(robot_or_skip("fr3_robotiq"))
    panda = RobotModel(robot_or_skip("droid_franka"))
    lower = np.maximum(fr3.q_min, panda.q_min)
    upper = np.minimum(fr3.q_max, panda.q_max)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(lower, upper)
        fr3.set(q, 0.5)
        panda.set(q, 0.5)
        T_fr3, T_panda = fr3.tcp_pose(), panda.tcp_pose()
        assert np.linalg.norm(T_fr3[:3, 3] - T_panda[:3, 3]) < 1e-3
        assert np.allclose(T_fr3[:3, :3], T_panda[:3, :3], atol=1e-3)


@pytest.mark.parametrize("name", ["droid_franka", "fr3_robotiq"])
def test_the_robotiq_squeezes_as_the_real_one_with_rubber_pads(name):
    """Closed on a fixed 40 mm block, the pads press with the real 2F-85's force as the
    lab drives it (droid_franka.GRIP_FORCE; their mean: the arm holds the gripper a
    little off a fixed block); the pads have its rubber's friction."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco
    from r2s2r.robots.model import prepare_robot

    robot = robot_or_skip(name)
    spec = robot.mjcf()
    prepare_robot(spec, robot)
    model = RobotModel(robot)
    model.set(np.asarray(robot.home_q), 0.0)
    pads = {
        side: model.data.geom_xpos[model.model.geom(f"gripper/{side}_pad1").id]
        for side in ("left", "right")
    }
    across = (pads["right"] - pads["left"]) / np.linalg.norm(
        pads["right"] - pads["left"]
    )
    R = np.linalg.qr(np.column_stack([across, np.eye(3)[:, :2]]))[0]
    R[:, 0] = across
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (R * np.sign(np.linalg.det(R))).ravel())
    spec.worldbody.add_geom(
        name="block",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        pos=(pads["left"] + pads["right"]) / 2,
        quat=quat,
    )
    m = spec.compile()
    d = mujoco.MjData(m)
    arm = [m.joint(j).qposadr[0] for j in robot.arm_joints]
    d.qpos[arm] = robot.home_q
    gripper = m.actuator(robot.gripper.actuator).id
    d.ctrl[[a for a in range(m.nu) if a != gripper]] = robot.home_q
    d.ctrl[gripper] = robot.gripper.ctrl_at(1.0)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    block, force = m.geom("block").id, np.zeros(6)
    squeeze = {}
    for side in ("left", "right"):
        ids = {m.geom(f"gripper/{side}_pad{k}").id for k in (1, 2)}
        assert all(m.geom_friction[g][0] == droid_franka.PAD_FRICTION for g in ids)
        squeeze[side] = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} & ids and block in (c.geom1, c.geom2):
                mujoco.mj_contactForce(m, d, i, force)
                squeeze[side] += force[0]
    assert np.mean(list(squeeze.values())) == pytest.approx(
        droid_franka.GRIP_FORCE, rel=0.1
    )


def test_fr3_joint_limits_match_the_mjcf():
    """The limits Isaac's Panda joints get are the FR3 MJCF's ranges."""
    model = RobotModel(robot_or_skip("fr3_robotiq"))
    limits = np.array(fr3_robotiq.JOINT_LIMITS)
    assert np.allclose(limits[:, 0], model.q_min) and np.allclose(
        limits[:, 1], model.q_max
    )


def test_ur5e_position_control_holds_the_arm():
    """The hook turns the torque motors into position servos from the MJCF defaults."""
    robot = robot_or_skip("ur5e_2f140")
    mjspec = robot.mjcf()
    robot.position_control(mjspec)
    m = mjspec.compile()
    for joint in robot.arm_joints:
        act = m.actuator(joint)
        kp = 500.0 if "wrist" in joint else 2000.0
        assert act.gainprm[0] == kp and act.biasprm[1] == -kp
        assert np.allclose(act.ctrlrange, m.joint(joint).range)


def test_viewer_robot_draws_primitive_geoms():
    """The UR5e's pads are boxes: the GLB has them; poses follow the gripper."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import geom_mesh
    from r2s2r.structs import Capture, RobotTrajectory
    from r2s2r.viewer.robot import robot_glb, robot_poses

    robot = robot_or_skip("ur5e_2f140")
    m = robot.mjcf().compile()
    pad = m.geom("gripper_left_inner_finger_pad_legacy_0")
    box = geom_mesh(m, pad.id)
    assert box is not None and np.allclose(sorted(box.extents), sorted(2 * pad.size))
    glb, bodies = robot_glb(robot)
    assert glb[:4] == b"glTF" and "left_inner_finger" in bodies
    capture = Capture(
        "c", "mujoco", robot.name, "", {}, [], (0, 2), Path("."),
        trajectory=RobotTrajectory(
            np.arange(2), np.arange(2) * 0.1, np.tile(robot.home_q, (2, 1)),
            np.array([0.0, 1.0]),
        ),
    )  # fmt: skip
    poses = robot_poses(robot, capture, bodies)
    finger = bodies.index("left_inner_finger")
    assert poses["poses"][0][finger] != poses["poses"][1][finger]


@pytest.mark.gl
def test_ur5e_mask_covers_the_robot_not_the_table():
    """Synthetic view of the UR5e over a table: the robot's pixels are cut, the table
    around it is kept."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.robots.mask import NO_ROBOT, RobotMasker

    robot = robot_or_skip("ur5e_2f140")
    masker = RobotMasker(robot, max_size=(320, 240))
    K = intrinsics_matrix(250.0, 250.0, 159.5, 119.5)
    T_base_cam = look_at(np.array([-1.2, 0.6, 0.8]), np.array([-0.3, 0.1, 0.2]))
    q = np.asarray(robot.home_q)
    robot_depth = masker.robot_depth(K, 320, 240, T_base_cam, q, 0.5)
    on_robot = robot_depth < NO_ROBOT
    # The table: the plane z = 0 seen from the camera (depth along the optical axis).
    rays = np.linalg.inv(K) @ np.stack(
        [*np.meshgrid(np.arange(320), np.arange(240)), np.ones((240, 320))]
    ).reshape(3, -1)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    table = (-t[2] / (R[2] @ rays)).reshape(240, 320)
    table = np.where(table > 0, table, 0.0)
    depth = np.where(on_robot, robot_depth, table)
    cut, mask = masker.mask_depth(depth.astype(np.float32), K, T_base_cam, q, 0.5)
    masker.close()
    assert 0.02 < on_robot.mean() < 0.5
    assert np.all(mask[on_robot]) and np.all(cut[on_robot] == 0)
    near_robot = cv2.dilate(on_robot.astype(np.uint8), np.ones((11, 11), np.uint8))
    far = (table > 0) & (near_robot == 0)
    assert far.sum() > 1000 and not mask[far].any()
