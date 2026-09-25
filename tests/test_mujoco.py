"""Tests for real/mujoco/ (skipped without MuJoCo or its assets)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.real.mujoco import world as mw  # noqa: E402
from r2s2r.real.mujoco.capture import record_capture  # noqa: E402
from r2s2r.reconstruct.refine import view_points  # noqa: E402
from r2s2r.robots.franka import FRANKA_HAND_MAX_WIDTH, PandaKinematics  # noqa: E402
from r2s2r.robots.mask import NO_ROBOT, RobotMasker  # noqa: E402
from r2s2r.structs import DepthView  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (Path(mw.MujocoWorldConfig().menagerie_dir) / "franka_emika_panda").exists()
    or not Path(mw.MujocoWorldConfig().gso_dir).exists(),
    reason="MuJoCo assets not fetched (scripts/setup/fetch_mujoco_assets.sh)",
)


@pytest.fixture(name="world", scope="module")
def fixture_world():
    """The default world, settled."""
    world = mw.MujocoWorld()
    world.reset()
    yield world
    world.close()


def test_kinematics_match_the_simulator(world):
    """The shared numpy FK is the simulator's TCP."""
    T_fk = PandaKinematics().fk(world.arm_q())
    assert np.allclose(T_fk, world.tcp_pose(), atol=1e-4)


def test_depth_backprojects_onto_the_table(world):
    """K, T_base_cam and planar depth agree: table pixels land on z = 0."""
    spec = world.camera_spec("ext1")
    _, depth = world.render("ext1")
    view = DepthView("ext1", 0, depth.astype(float), spec.K, spec.T_base_cam)
    z = view_points(view, max_depth=3.0)[:, 2]
    near = z[np.abs(z) < 0.02]
    assert len(near) > 0.3 * len(z)
    assert abs(np.median(near)) < 0.002


def test_objects_rest_where_placed(world):
    """Objects settle upright at their configured positions."""
    for obj in world.cfg.objects:
        T = world.object_pose(obj.name)
        assert np.allclose(T[:2, 3], obj.xy, atol=5e-3)
        assert T[2, 2] > 0.999


def _robot_pixels(world, name):
    """The world's own segmentation of the robot in camera ``name``."""
    spec = world.camera_spec(name)
    r = world._renderer(spec.width, spec.height)  # pylint: disable=protected-access
    r.enable_segmentation_rendering()
    r.update_scene(world.data, camera=name)
    seg = r.render()
    r.disable_segmentation_rendering()
    geom = np.where(seg[..., 1] == int(mw.mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
    body = world.model.geom_bodyid[np.maximum(geom, 0)]
    root = world.model.body_rootid[body]
    return (geom >= 0) & (root == world.model.body("link0").id)


def test_robot_masker_matches_the_simulator(world):
    """Rendering the robot from calibration + joints reproduces what cameras see."""
    masker = RobotMasker("franka_panda", world.cfg.menagerie_dir)
    for name in world.camera_names():
        spec = world.camera_spec(name)
        truth = _robot_pixels(world, name)
        pred = (
            masker.robot_depth(
                spec.K,
                spec.width,
                spec.height,
                world.camera_pose(name),
                world.arm_q(),
                1 - world.finger_width() / FRANKA_HAND_MAX_WIDTH,
            )
            < NO_ROBOT
        )
        union = (truth | pred).sum()
        assert union == 0 or (truth & pred).sum() / union > 0.97, name
    masker.close()


def test_capture_records_the_requested_cameras(tmp_path):
    """A wrist-only capture has one moving camera with per-frame poses."""
    capture = record_capture(tmp_path / "cap", cameras=["wrist"], every=40)
    (cam,) = capture.cameras.values()
    assert cam.role == "wrist" and not cam.is_static
    poses = np.stack([f.T_base_cam for f in capture.frames])
    assert len(capture.frames) > 3 and np.ptp(poses[:, :3, 3], axis=0).max() > 0.1
    assert all(f.depth_image for f in capture.frames)
