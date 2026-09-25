"""Tests for reconstruct/refine.py and assets.py on synthetic geometry."""

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from r2s2r.assets import bake_mesh_scales, urdf_visual_points
from r2s2r.reconstruct.refine import (
    fit_support_outline,
    refine_scene,
    register_footprint,
)
from r2s2r.structs import CameraSpec, DepthView, ObjectSpec, SceneSpec
from r2s2r.transforms import intrinsics_matrix, make_transform

BOX = (0.12, 0.06, 0.05)  # object size (m)
TABLE = (0.40, 0.30)  # support size (m)
TABLE_YAW = np.deg2rad(20.0)


def _write_box_urdf(tmp_path, scale=(1.0, 1.0, 1.0)):
    mesh = trimesh.creation.box(extents=np.array(BOX) / np.array(scale))
    mesh.apply_translation([0, 0, BOX[2] / 2 / scale[2]])  # origin at the bottom
    mesh.export(tmp_path / "box.obj")
    s = " ".join(str(v) for v in scale)
    (tmp_path / "box.urdf").write_text(
        '<robot name="box"><link name="base"><visual><geometry>'
        f'<mesh filename="box.obj" scale="{s}"/></geometry></visual>'
        "<collision><geometry>"
        f'<mesh filename="box.obj" scale="{s}"/></geometry></collision>'
        "</link></robot>"
    )
    return tmp_path / "box.urdf"


def _yaw(T_or_yaw, xyz=(0.0, 0.0, 0.0)):
    return make_transform(Rotation.from_euler("z", T_or_yaw).as_matrix(), xyz)


def _overhead_depth(T_base_obj, K, size=(160, 120), cam_z=1.0):
    """Depth of a downward camera over a rotated table centred at (0.5, 0).

    The table is at z = 0 with a box on it.
    """
    w, h = size
    v, u = np.mgrid[0:h, 0:w]
    rx, ry = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]

    # Camera at (0.5, 0, cam_z) looking down: cam x = base x, cam y = -base y.
    def hit(depth):
        return np.stack([0.5 + rx * depth, -ry * depth], -1)

    depth = np.full((h, w), cam_z + 0.7)  # floor 0.7 m below the table
    table_local = (hit(cam_z) - [0.5, 0.0]) @ _yaw(-TABLE_YAW)[:2, :2].T
    on_table = np.all(np.abs(table_local) < np.array(TABLE) / 2, -1)
    depth[on_table] = cam_z
    top = cam_z - BOX[2]
    inv = np.linalg.inv(T_base_obj)
    box_local = hit(top) @ inv[:2, :2].T + inv[:2, 3]
    depth[np.all(np.abs(box_local) < np.array(BOX[:2]) / 2, -1)] = top
    T_base_cam = make_transform(np.diag([1.0, -1.0, -1.0]), [0.5, 0.0, cam_z])
    return depth, T_base_cam


def test_fit_support_outline():
    """The rectangle of the on-plane region is recovered, not the floor."""
    rng = np.random.default_rng(0)
    local = rng.uniform(-0.5, 0.5, (20000, 2)) * np.array(TABLE)
    xy = local @ _yaw(TABLE_YAW)[:2, :2].T + [0.1, -0.05]
    floor = np.c_[rng.uniform(-1, 1, (5000, 2)), np.full(5000, -0.7)]
    pts = np.vstack([np.c_[xy, np.zeros(len(xy))], floor])
    centre, size, yaw = fit_support_outline(pts, np.array([0.1, -0.05]), 0.01, 0.01)
    assert np.allclose(centre, [0.1, -0.05], atol=0.01)
    assert np.allclose(sorted(size), sorted(TABLE), atol=0.02)
    # minAreaRect's yaw is defined modulo 90 degrees.
    assert abs((np.rad2deg(yaw) - 20.0 + 45) % 90 - 45) < 2.0


def test_register_footprint_recovers_shift_and_yaw():
    """A shifted, turned box seen from its top and two sides is put back."""
    box = trimesh.creation.box(extents=BOX)
    model, faces = trimesh.sample.sample_surface(box, 4000, seed=1)
    truth = _yaw(np.deg2rad(8.0), [0.03, -0.02, 0.0])
    # Only the top and the +x / +y sides face the (virtual) cameras.
    normals = box.face_normals[faces]
    seen = (normals[:, 2] > 0.5) | (normals[:, 0] > 0.5) | (normals[:, 1] > 0.5)
    observed = (model @ truth[:3, :3].T + truth[:3, 3])[seen]
    T, before, after = register_footprint(model, observed)
    assert np.allclose(T[:2, 3], truth[:2, 3], atol=0.004)
    assert abs(np.arctan2(T[1, 0], T[0, 0]) - np.deg2rad(8.0)) < np.deg2rad(1.0)
    assert after > 0.8 > before


def test_bake_mesh_scales(tmp_path):
    """Scaled meshes are baked to their true size and lose the attribute."""
    urdf = _write_box_urdf(tmp_path, scale=(0.5, 2.0, 1.0))
    baked = bake_mesh_scales(urdf)
    assert baked.name == "box_r2s2r.urdf"
    assert "scale" not in baked.read_text(encoding="utf-8")
    pts = urdf_visual_points(baked, 2000)
    assert np.allclose(np.ptp(pts, 0), BOX, atol=0.002)
    assert np.allclose(np.ptp(urdf_visual_points(urdf, 2000), 0), BOX, atol=0.002)


def test_refine_scene_moves_misplaced_object(tmp_path):
    """A box placed 4 cm and 12 degrees off is put back where the depth sees it."""
    urdf = _write_box_urdf(tmp_path)
    truth = _yaw(np.deg2rad(30.0), [0.52, 0.03, 0.0])
    K = intrinsics_matrix(120.0, 120.0, 80.0, 60.0)
    depth, T_base_cam = _overhead_depth(truth, K)
    wrong = _yaw(np.deg2rad(42.0), [0.55, 0.0, 0.0])
    cam = CameraSpec("0", "ext1", 160, 120, K, T_base_cam=T_base_cam)
    scene = SceneSpec(
        name="synthetic",
        embodiment="droid_franka",
        objects=[ObjectSpec("box", "box", str(urdf), wrong)],
        T_base_support=make_transform(np.eye(3), [0.45, 0.0, 0.0]),
        cameras={"0": cam},
        reference_camera="0",
        reference_step=0,
        joint_positions=np.zeros(7),
    )
    views = [DepthView("0", 0, depth, K, T_base_cam)]
    refined, report = refine_scene(scene, views)
    T = refined.objects[0].T_base_obj
    assert report["objects"]["box"]["matched"]
    assert np.linalg.norm(T[:2, 3] - truth[:2, 3]) < 0.01
    yaw = np.rad2deg(np.arctan2(T[1, 0], T[0, 0]))
    assert abs((yaw - 30.0 + 90) % 180 - 90) < 3.0  # a box is 180-degree symmetric
    assert abs(T[2, 3]) < 0.003  # resting on the support
    assert refined.support_extent == pytest.approx(TABLE, abs=0.03) or (
        refined.support_extent == pytest.approx(TABLE[::-1], abs=0.03)
    )
