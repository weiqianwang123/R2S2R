"""Tests for real/mujoco_world.py (skipped without MuJoCo or its assets)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.real import mujoco_world as mw  # noqa: E402
from r2s2r.refine import DepthView, view_points  # noqa: E402
from r2s2r.robots.franka import PandaKinematics  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (Path(mw.MujocoWorldConfig().menagerie_dir) / "franka_emika_panda").exists()
    or not Path(mw.MujocoWorldConfig().gso_dir).exists(),
    reason="MuJoCo assets not fetched (scripts/fetch_mujoco_assets.sh)",
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
