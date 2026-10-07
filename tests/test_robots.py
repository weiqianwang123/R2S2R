"""Tests for the robot registry and robots/model.py (skipped per robot without its
assets), RoboDojo's two arms, the masker and the viewer's robot on the UR5e."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import robot_or_skip

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.robots import (  # noqa: E402
    ROBOTS,
    dual_x5,
    fr3_robotiq,
    get_robot,
    ur5e_2f140,
)
from r2s2r.robots.model import RobotModel, in_subtree  # noqa: E402
from r2s2r.transforms import (  # noqa: E402
    intrinsics_matrix,
    invert,
    look_at,
    transform_points,
)


@pytest.fixture(name="model", scope="module", params=sorted(ROBOTS))
def fixture_model(request):
    """Each robot's model."""
    return RobotModel(robot_or_skip(request.param))


def test_registry_names_the_known_robots():
    """Every robot is registered under its name; unknown names list the known ones."""
    assert set(ROBOTS) == {"fr3_robotiq", "ur5e_2f140", "dual_x5"}
    assert all(get_robot(n).name == n for n in ROBOTS)
    with pytest.raises(ValueError, match="fr3_robotiq"):
        get_robot("ur10")


def test_robot_compiles_with_its_arms_and_grippers(model):
    """The spec's joints, actuators and bodies exist; the home pose is within limits."""
    robot = model.robot
    assert model.arm_qadr.shape == (robot.dof,) == (len(robot.home_q),)
    dofs = {"fr3_robotiq": [7], "ur5e_2f140": [6], "dual_x5": [6, 6]}
    assert [len(arm.joints) for arm in robot.arms] == dofs[robot.name]
    for arm in robot.arms:
        model.model.actuator(arm.gripper.actuator)
        model.model.body(arm.tcp_body)
        assert 0.05 < arm.max_opening < 0.15
    assert np.all(model.q_min <= robot.home_q) and np.all(robot.home_q <= model.q_max)


def test_gripper_opens_and_closes(model):
    """Levels 0 and 1 put each driver at its ends, and its fingers (the bodies its
    joints move) apart and together; the other gripper stays."""
    robot, m, d = model.robot, model.model, model.data
    for arm, poser in enumerate(model.grippers):
        gripper = robot.arms[arm].gripper
        driver = poser.joints.index(gripper.driver)
        assert len(poser.joints) > 1
        moved = [m.jnt_bodyid[m.joint(j).id] for j in poser.joints]
        fingers = [
            g
            for g in range(m.ngeom)
            if any(in_subtree(m, int(m.geom_bodyid[g]), int(b)) for b in moved)
        ]
        widths = []
        for level, end, tol in ((0.0, gripper.open, 0.01), (1.0, gripper.closed, 0.03)):
            assert abs(poser.positions(level)[driver] - end) < tol
            levels = np.zeros(len(robot.arms))
            levels[arm] = level
            model.set(np.asarray(robot.home_q), robot.gripper_position(levels))
            tips = transform_points(invert(model.tcp_pose(arm)), d.geom_xpos[fingers])
            widths.append(np.ptp(tips[:, :2], axis=0).max())
        assert widths[0] > widths[1] + 0.03


@pytest.mark.parametrize("name", ["fr3_robotiq", "ur5e_2f140"])
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
    closed = dict(zip(model.grippers[0].joints, model.grippers[0].positions(1.0)))
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
    """IK from the home pose finds poses that FK produced, for each arm's TCP and a
    body, moving that arm only."""
    robot = model.robot
    rng = np.random.default_rng(0)
    home = np.asarray(robot.home_q)
    for arm, spec in enumerate(robot.arms):
        others = np.ones(robot.dof, bool)
        others[robot.arm_slice(arm)] = False
        for _ in range(5):
            q = home + rng.uniform(-0.3, 0.3, robot.dof)
            model.set(q)
            T_tcp, T_body = model.tcp_pose(arm), model.pose(spec.tcp_body)
            result = model.ik(T_tcp, home, arm=arm)
            assert result.success and result.pos_error < 1e-4, result
            assert np.array_equal(result.q[others], home[others])
            model.set(result.q)
            assert np.allclose(model.tcp_pose(arm), T_tcp, atol=1e-3)
            assert model.ik(T_body, home, frame=spec.tcp_body, arm=arm).success


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

    robot = robot_or_skip("fr3_robotiq")
    mjspec = robot.mjcf()
    body = mjspec.worldbody.add_body(name="obj", pos=[0.5, 0.0, 0.1])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02])
    body.add_freejoint().name = joint_name
    model = RobotModel(robot, mjspec)
    assert model.grippers[0].joints == [
        model.model.joint(j).name
        for j in range(model.model.njnt)
        if model.model.joint(j).name.startswith(fr3_robotiq.PREFIX)
    ]
    adr = model.model.body("obj").jntadr[0]
    qadr = model.model.jnt_qposadr[adr]
    moved = np.array([0.3, -0.2, 0.4, 0.0, 1.0, 0.0, 0.0])
    model.data.qpos[qadr : qadr + 7] = moved
    model.set(np.asarray(robot.home_q), 1.0)
    assert np.array_equal(model.data.qpos[qadr : qadr + 7], moved)


def test_robotiq_mount_matches_the_mjcf():
    """Isaac's re-mount constant is where the MuJoCo model has the Robotiq base."""
    model = RobotModel(robot_or_skip("fr3_robotiq"))
    T = invert(model.pose("fr3_link7")) @ model.pose(f"{fr3_robotiq.PREFIX}base")
    expected = np.eye(4)
    expected[2, 3] = fr3_robotiq.ROBOTIQ_BASE_Z
    assert np.allclose(T, expected, atol=1e-4)


def test_the_robotiq_squeezes_as_the_real_one_with_rubber_pads():
    """Closed on a fixed 40 mm block, the pads press with the real 2F-85's force as the
    lab drives it (fr3_robotiq.GRIP_FORCE; their mean: the arm holds the gripper a
    little off a fixed block); the pads have its rubber's friction."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import mujoco
    from r2s2r.robots.model import prepare_robot

    robot = robot_or_skip("fr3_robotiq")
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
    gripper = m.actuator(robot.arm.gripper.actuator).id
    d.ctrl[[a for a in range(m.nu) if a != gripper]] = robot.home_q
    d.ctrl[gripper] = robot.arm.gripper.ctrl_at(1.0)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    block, force = m.geom("block").id, np.zeros(6)
    squeeze = {}
    for side in ("left", "right"):
        ids = {m.geom(f"gripper/{side}_pad{k}").id for k in (1, 2)}
        assert all(m.geom_friction[g][0] == fr3_robotiq.PAD_FRICTION for g in ids)
        squeeze[side] = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} & ids and block in (c.geom1, c.geom2):
                mujoco.mj_contactForce(m, d, i, force)
                squeeze[side] += force[0]
    assert np.mean(list(squeeze.values())) == pytest.approx(
        fr3_robotiq.GRIP_FORCE, rel=0.1
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


def test_robodojos_arms_stand_where_robodojo_has_them():
    """In RoboDojo's world frame the arms' bases are where its dual_x5 puts them,
    facing its +y; at home each gripper points ahead of its arm."""
    model = RobotModel(robot_or_skip("dual_x5"))
    for side, x in (("left", -0.3), ("right", 0.3)):
        T = dual_x5.T_ROBODOJO_BASE @ model.pose(f"{side}_base_link")
        assert np.allclose(T[:3, 3], [x, -0.45, 0.765])
        assert np.allclose(T[:3, :3], dual_x5.T_ROBODOJO_BASE[:3, :3])
        assert np.allclose(T[:3, 0], [0.0, 1.0, 0.0])
    for arm in range(2):
        assert np.allclose(model.tcp_pose(arm)[:3, 2], [1.0, 0.0, 0.0])


def test_robodojos_grippers_open_and_close_each_its_own():
    """A gripper position is one opening per arm: the left closed, the right open,
    each one's joint8 following its joint7; one number for both is refused."""
    robot = robot_or_skip("dual_x5")
    model = RobotModel(robot)
    model.set(np.asarray(robot.home_q), [1.0, 0.0])
    m, d = model.model, model.data
    for side, q in (("left", 0.0), ("right", dual_x5.FINGER_OPEN)):
        for finger in dual_x5.FINGERS:
            assert d.qpos[m.joint(f"{side}_{finger}").qposadr[0]] == pytest.approx(q)
    with pytest.raises(ValueError, match="2 grippers"):
        model.set(np.asarray(robot.home_q), 0.5)


def test_robodojos_tcp_is_between_the_pads():
    """Closed, each gripper's fingers lie either side of its TCP along its y, their
    tips 8 mm past it along the approach (z), the TCP within their width (x)."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import geom_mesh

    robot = robot_or_skip("dual_x5")
    model = RobotModel(robot)
    model.set(np.asarray(robot.home_q), [1.0, 1.0])
    m, d = model.model, model.data
    for arm, side in enumerate(("left", "right")):
        T_tcp_base = invert(model.tcp_pose(arm))
        ys = []
        for link in ("link7", "link8"):
            b = m.body(f"{side}_{link}").id
            T_base_body = np.eye(4)
            T_base_body[:3, :3], T_base_body[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
            points = np.vstack(
                [
                    transform_points(T_tcp_base @ T_base_body, geom_mesh(m, g).vertices)
                    for g in range(m.ngeom)
                    if m.geom_bodyid[g] == b and m.geom_contype[g]
                ]
            )
            assert points[:, 2].max() == pytest.approx(0.008, abs=1e-3)
            assert points[:, 0].min() < 0.0 < points[:, 0].max()
            ys.append(points[:, 1])
        near, far = sorted(ys, key=np.mean)
        assert near.max() < 0.001 and far.min() > -0.001
