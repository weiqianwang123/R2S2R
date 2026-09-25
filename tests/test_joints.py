"""Tests for reconstruct/joints.py on a synthetic drawer (needs GL rendering)."""

import pytest

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from synthetic_drawer import (  # noqa: E402
    CAMERAS,
    TRUTH,
    cabinet,
    drawer_cabinet,
    look_at,
    scene_of,
    video,
)

from r2s2r.assets import urdf_movable_joints  # noqa: E402
from r2s2r.reconstruct import joints  # noqa: E402

pytestmark = pytest.mark.gl

CONFIG = joints.JointFitConfig(view_scale=1.0)


def test_fit_follows_the_drawer_and_widens_its_limits(tmp_path):
    """Opened to 12 cm and closed again: the trajectory is recovered, the limits cover
    the travel, and the check passes."""
    scene = scene_of(drawer_cabinet(tmp_path))
    steps = video(scene, TRUTH, CAMERAS)
    away = look_at((0.3, 0.0, 0.5), target=(0.0, 0.0, 1.0))  # sees only sky
    steps[6] = video(scene, {6: 0.0}, [away])[6]

    fitted, report = joints.fit_joints(scene, steps, rest_steps=[0, 1], config=CONFIG)
    entry = report["cab"]["joints"]["slide"]
    fit = {int(t): q for t, q in entry["trajectory"].items()}
    assert fit[6] is None and entry["observed_steps"] == 6
    for t, q in TRUTH.items():
        assert fit[t] == pytest.approx(q, abs=5e-3), t
    assert entry["moved"] and report["cab"]["check"]["consistent"]
    assert entry["observed_range"] == pytest.approx([0.05, 0.12], abs=5e-3)
    assert entry["limits"][0] == 0.0
    assert entry["limits"][1] == pytest.approx(0.12, abs=5e-3)
    assert fitted.objects[0].asset_path.endswith("cab_fit.urdf")
    assert urdf_movable_joints(fitted.objects[0].asset_path)[0]["upper"] == (
        pytest.approx(0.12, abs=5e-3)
    )


def test_wrong_axis_is_flagged_and_limits_kept(tmp_path):
    """The drawer slides out but the model slides it sideways: the open steps stay
    unexplained, and the asset keeps its limits."""
    steps = video(scene_of(drawer_cabinet(tmp_path)), TRUTH, CAMERAS)
    model = scene_of(drawer_cabinet(tmp_path, axis="0 1 0", name="sideways"))
    fitted, report = joints.fit_joints(model, steps, rest_steps=[0, 1], config=CONFIG)
    check = report["cab"]["check"]
    assert not check["consistent"] and {3, 4} <= set(check["unexplained_steps"])
    assert fitted.objects[0].asset_path == model.objects[0].asset_path


def test_still_drawer_is_not_moved(tmp_path):
    """A video where nothing moves leaves the asset alone."""
    scene = scene_of(drawer_cabinet(tmp_path))
    steps = video(scene, {0: 0.0, 1: 0.0}, CAMERAS[:1])
    fitted, report = joints.fit_joints(scene, steps, config=CONFIG)
    assert not report["cab"]["joints"]["slide"]["moved"]
    assert report["cab"]["check"]["consistent"]
    assert fitted.objects[0].asset_path == scene.objects[0].asset_path


def test_reachable_stops_where_the_part_enters_its_parent(tmp_path):
    """A panel in front of a box cannot be pushed into it, only pulled out."""
    urdf = cabinet(
        tmp_path,
        carcass=[((0.2, 0.2, 0.2), (0.0, 0.0, 0.1))],
        drawer=[((0.01, 0.16, 0.12), (0.107, 0.0, 0.1))],  # 2 mm in front
    )
    obj = scene_of(urdf).objects[0]
    joint = urdf_movable_joints(urdf)[0]
    lo, hi = joints._reachable(  # pylint: disable=protected-access
        obj, joint, (-0.2, 0.2), max_penetration=0.1, step=0.005
    )
    assert lo == 0.0 and hi > 0.19
