"""Tests for reconstruct/existence.py on synthetic objects (needs GL rendering)."""

import numpy as np
import pytest
import trimesh

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.reconstruct.existence import drop_unseen  # noqa: E402
from r2s2r.reconstruct.render import SceneRenderer  # noqa: E402
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec  # noqa: E402
from r2s2r.transforms import intrinsics_matrix, make_transform  # noqa: E402

pytestmark = pytest.mark.gl

K = intrinsics_matrix(300.0, 300.0, 159.5, 119.5)


def _box(tmp_path, name, extents):
    """A URDF of a box standing on its origin."""
    box = trimesh.creation.box(extents=extents)
    box.apply_translation([0.0, 0.0, extents[2] / 2])
    box.export(tmp_path / f"{name}.obj")
    urdf = tmp_path / f"{name}.urdf"
    urdf.write_text(
        f'<robot name="{name}"><link name="base"><visual><geometry>'
        f'<mesh filename="{name}.obj"/></geometry></visual></link></robot>'
    )
    return ObjectSpec(name, name, str(urdf), np.eye(4))


def _at(obj, xyz):
    return ObjectSpec(
        obj.name, obj.name, obj.asset_path, make_transform(np.eye(3), xyz)
    )


def _scene(objects):
    return SceneSpec(
        name="t",
        embodiment="franka_panda",
        objects=objects,
        T_base_support=np.eye(4),
        cameras={},
        reference_camera="c",
        reference_step=0,
        joint_positions=np.zeros(7),
    )


def _look_at(eye, target=(0.5, 0.0, 0.03)):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return make_transform(np.column_stack([right, np.cross(fwd, right), fwd]), eye)


def _views(scene):
    """Depth of ``scene`` from three cameras around the table."""
    renderer = SceneRenderer(scene, (320, 240))
    for k, obj in enumerate(scene.objects):
        renderer.pose(k, obj.T_base_obj)
    views = []
    for eye in ((0.1, 0.35, 0.45), (0.5, -0.5, 0.4), (0.95, 0.25, 0.45)):
        T = _look_at(eye)
        out = renderer.render(DepthView("cam", 0, np.zeros((240, 320)), K, T), 0)
        views.append(DepthView("cam", 0, out["depth"].astype(float), K, T))
    renderer.close()
    return views


def test_a_ghost_is_dropped_and_what_rested_on_it_settles(tmp_path):
    """A slab that is not there goes, the box standing on it drops to the table, and a
    real box that shares its neighbour's point cluster stays."""
    tall = _box(tmp_path, "tall", (0.06, 0.06, 0.09))
    left = _box(tmp_path, "left", (0.05, 0.05, 0.06))
    beside = _box(tmp_path, "beside", (0.05, 0.05, 0.04))  # touches ``left``
    ghost = _box(tmp_path, "ghost", (0.22, 0.18, 0.03))
    truth = _scene(
        [
            _at(tall, (0.5, -0.1, 0.0)),
            _at(left, (0.5, 0.12, 0.0)),
            _at(beside, (0.5, 0.17, 0.0)),
        ]
    )
    views = _views(truth)
    built = _scene(
        [
            _at(tall, (0.5, -0.1, 0.03)),  # on the ghost
            _at(left, (0.5, 0.12, 0.0)),
            _at(beside, (0.5, 0.17, 0.0)),
            _at(ghost, (0.55, -0.12, 0.0)),
        ]
    )
    kept, report = drop_unseen(built, views)
    assert [o.name for o in kept.objects] == ["tall", "left", "beside"]
    assert report["objects"]["ghost"]["dropped"]
    assert report["objects"]["ghost"]["seen_through"] > 0.5
    beside_entry = report["objects"]["beside"]
    assert not beside_entry["matched"] and not beside_entry["dropped"]
    assert beside_entry["seen_through"] < 0.1
    assert report["settled"] == {"tall": pytest.approx(0.03, abs=0.003)}
    assert abs(kept.objects[0].T_base_obj[2, 3]) < 0.003
    assert kept.provenance["existence"] == report


def test_nothing_changes_without_ghosts(tmp_path):
    """A scene the depth confirms comes back as it was."""
    box = _box(tmp_path, "box", (0.06, 0.06, 0.08))
    scene = _scene([_at(box, (0.5, 0.0, 0.0))])
    kept, report = drop_unseen(scene, _views(scene))
    assert kept is scene and not report["settled"]
    assert report["objects"]["box"] == {"matched": True, "dropped": False}


def test_a_real_object_with_a_wrong_shape_is_kept(tmp_path):
    """A box reconstructed as a thin tower matches no cluster and is mostly seen
    through, but the box's own points stand under it: it is kept, and flagged."""
    box = _box(tmp_path, "box", (0.06, 0.06, 0.09))
    tower = _box(tmp_path, "tower", (0.04, 0.04, 0.35))
    views = _views(_scene([_at(box, (0.5, 0.0, 0.0))]))
    kept, report = drop_unseen(_scene([_at(tower, (0.5, 0.0, 0.0))]), views)
    entry = report["objects"]["tower"]
    assert not entry["matched"] and entry["seen_through"] > 0.5
    assert not entry["dropped"] and entry["unexplained_points_under"] > 0
    assert [o.name for o in kept.objects] == ["tower"]
