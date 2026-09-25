"""Tests for policy/: motion primitives and the pick program on a kinematic robot."""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from r2s2r.policy.pick import find_object, pick_up, plan_top_down_grasp
from r2s2r.policy.robot import RobotInterface
from r2s2r.robots.franka import Q_READY
from r2s2r.structs import ObjectSpec, SceneSpec
from r2s2r.transforms import make_transform

BOX = (0.03, 0.07, 0.12)  # thin along x in its own frame


class KinematicRobot(RobotInterface):
    """Tracks every command exactly; the gripper stops at ``object_width``."""

    def __init__(self, object_width: float) -> None:
        super().__init__()
        self.q = Q_READY + 0.2
        self.width = 0.08
        self.object_width = object_width

    def joint_positions(self):
        return self.q.copy()

    def gripper_width(self):
        return self.width

    def _hold(self, q, gripper_closed):
        self.q = q.copy()
        self.width = self.object_width if gripper_closed else 0.08


def _scene(tmp_path, yaw_deg):
    mesh = trimesh.creation.box(extents=BOX)
    mesh.apply_translation([0, 0, BOX[2] / 2])
    mesh.export(tmp_path / "box.obj")
    (tmp_path / "box.urdf").write_text(
        '<robot name="box"><link name="base"><visual><geometry>'
        '<mesh filename="box.obj"/></geometry></visual></link></robot>'
    )
    T = make_transform(
        Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix(), [0.5, -0.1, 0.0]
    )
    obj = ObjectSpec("crayon_box", "box", str(tmp_path / "box.urdf"), T)
    return SceneSpec(
        name="t",
        embodiment="franka_panda",
        objects=[obj],
        T_base_support=np.eye(4),
        cameras={},
        reference_camera="c",
        reference_step=0,
        joint_positions=Q_READY,
    )


def test_find_object(tmp_path):
    """Exact names and unique substrings resolve."""
    scene = _scene(tmp_path, 0.0)
    assert find_object(scene, "crayon").name == "crayon_box"


def test_grasp_closes_across_the_thin_side(tmp_path):
    """The fingers close along the box's 3 cm side, below its top."""
    scene = _scene(tmp_path, 30.0)
    plan = plan_top_down_grasp(scene, "crayon_box")
    assert np.isclose(plan.width, BOX[0], atol=3e-3)
    closing = plan.T_base_tcp[:3, 1]
    thin_axis = Rotation.from_euler("z", 30.0, degrees=True).apply([1.0, 0.0, 0.0])
    assert abs(closing @ thin_axis) > 0.99
    assert np.allclose(plan.T_base_tcp[:3, 3], [0.5, -0.1, BOX[2] - 0.03], atol=3e-3)
    assert np.allclose(plan.T_base_tcp[:3, 2], [0.0, 0.0, -1.0])


def test_pick_up_reaches_the_grasp(tmp_path):
    """On a perfect robot the program passes through the grasp and lifts."""
    scene = _scene(tmp_path, -40.0)
    robot = KinematicRobot(object_width=BOX[0])
    result = pick_up(robot, scene, "crayon")
    assert result["holding"]
    assert result["grasp_ik_error_m"] < 1e-3
    q = np.array(robot.log.q)
    # Continuous joint commands: no jumps between control steps.
    assert np.max(np.abs(np.diff(q, axis=0))) < 0.05
    grasp = np.array(result["grasp"]["T_base_tcp"])
    tcp_z = [robot.kin.fk(qi)[2, 3] for qi in q]
    assert min(tcp_z) < grasp[2, 3] + 1e-3
    assert np.isclose(robot.tcp_pose()[2, 3], grasp[2, 3] + 0.15, atol=1e-3)
    closed = np.array(robot.log.gripper_closed)
    assert not closed[0] and closed[-1]
