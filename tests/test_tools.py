"""Tests for r2s2r.tools: points from depth, the plane fit, the pattern search,
assembly, fitting a mesh to several views."""

import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
import trimesh
from conftest import RGBD_K, RGBD_SIZE, box_urdf, rgbd_capture
from scipy.spatial.transform import Rotation

from r2s2r.pipeline.workspace import Workspace
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec
from r2s2r.tools.geometry import fit_plane, parse_masks, pattern_search, view_points
from r2s2r.tools.objects import assemble
from r2s2r.transforms import look_at, make_transform

BOX = (0.06, 0.04, 0.08)  # the true object
TARGET = (0.5, 0.05, 0.04)  # where the cameras look
CAMERAS = {
    "c1": ("ext1", look_at((0.95, 0.35, 0.45), TARGET)),
    "c2": ("wrist", look_at((0.45, -0.4, 0.4), TARGET)),
}
TRUE_POSE = make_transform(
    Rotation.from_euler("z", 30, degrees=True).as_matrix(), [0.5, 0.05, 0.0]
)


def test_view_points_keep_the_masked_strided_pixels():
    """Points land in the base frame, in the order of their pixels (for colours);
    the mask is shrunk first, and only every stride-th pixel is kept."""
    T = make_transform(np.eye(3), [0.0, 0.0, 1.0])
    depth = np.full((6, 8), 0.5)
    depth[0, 0] = 0.0  # no depth
    depth[5, 7] = 3.0  # too far
    view = DepthView("c", 0, depth, RGBD_K, T, np.arange(144).reshape(6, 8, 3))
    pts, pixels = view_points(view)
    assert pixels.sum() == len(pts) == 46
    assert np.allclose(pts[:, 2], 1.5)
    assert view.image is not None
    assert view.image[pixels][0].tolist() == [3, 4, 5]  # pixel (0, 1)
    mask = np.zeros((6, 8), bool)
    mask[1:5, 1:7] = True
    assert view_points(view, mask)[1].sum() == 24
    assert view_points(view, mask, erode_px=1)[1].sum() == 8
    assert view_points(view, mask, stride=2)[1].sum() == 6


def test_fit_plane_ignores_outliers():
    """A tilted plane among clutter."""
    rng = np.random.default_rng(1)
    n = np.array([0.1, 0.0, 1.0]) / np.linalg.norm([0.1, 0.0, 1.0])
    xy = rng.uniform(-0.5, 0.5, (3000, 2))
    plane = np.c_[xy, (0.02 - n[0] * xy[:, 0]) / n[2]]
    clutter = rng.uniform(-0.5, 0.5, (800, 3))
    normal, d, inliers = fit_plane(np.concatenate([plane, clutter]))
    assert abs(abs(normal @ n) - 1) < 1e-4
    assert abs(abs(d) - 0.02) < 1e-3
    assert inliers[:3000].mean() > 0.99


def test_pattern_search_climbs_to_the_maximum():
    """A quadratic's maximum; fixed coordinates stay put."""
    target = np.array([0.3, -0.2])

    def f(x):
        return -float(np.sum((x - target) ** 2))

    x = pattern_search(f, np.zeros(2), np.array([0.1, 0.1]), [True, True], levels=8)
    assert np.allclose(x, target, atol=0.01)
    y = pattern_search(f, np.zeros(2), np.array([0.1, 0.1]), [True, False])
    assert y[1] == 0.0
    assert parse_masks(["ext1@0=a.png"]) == {"ext1@0": "a.png"}


def test_assemble_writes_a_simulation_ready_scene(tmp_path):
    """A scaled mesh becomes a scene of a sim-ready URDF, with the object's own
    collision parts when it has them; scale in the pose is refused."""
    capture = rgbd_capture(tmp_path / "capture", CAMERAS)
    ws = Workspace.create(capture, tmp_path / "run", "agentic")
    mesh = trimesh.creation.box(extents=(0.12, 0.08, 0.16))  # twice the true size
    mesh.export(tmp_path / "mesh.obj")
    T = TRUE_POSE @ make_transform(np.eye(3), [0, 0, 0.04])  # the box's centre
    objects = {
        "support": {"T_base_support": np.eye(4).tolist(), "extent": [1.0, 1.0]},
        "objects": [
            {
                "name": "Box 1",
                "mesh": "mesh.obj",
                "scale": 0.5,
                "T_base_obj": T.tolist(),
                "mass": 0.2,
                "friction": 0.7,
            }
        ],
    }
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    report = assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "hull")
    scene = SceneSpec.load(tmp_path / "scene")
    (obj,) = scene.objects
    assert obj.name == "box_1" and obj.mass == 0.2 and obj.friction == 0.7
    assert np.allclose(obj.T_base_obj, T)
    assert scene.support_extent == (1.0, 1.0)
    assert scene.reference_camera == "c1"  # the static camera
    assert report["objects"]["box_1"]["size_m"] == pytest.approx(BOX, abs=1e-6)
    assert report["objects"]["box_1"]["lowest_point_above_support_m"] == pytest.approx(
        0.0, abs=1e-6
    )
    root = ET.parse(obj.asset_path).getroot()
    assert float(root.find("link/inertial/mass").attrib["value"]) == pytest.approx(0.2)
    names = [c.attrib.get("name") for c in root.iter("collision")]
    assert "hull_0" in names and "r2s2r_resting_base" in names

    assert scene.provenance["method"] == "agentic"
    assert scene.provenance["collision"] == "hull"

    # Collision parts of its own: kept (scaled like the mesh), none made.
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(tmp_path / "part.obj")
    objects["objects"][0]["collision"] = ["part.obj", "part.obj"]
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    report = assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "coacd")
    assert report["objects"]["box_1"]["hulls"] == 2
    assert SceneSpec.load(tmp_path / "scene").provenance["collision"] == "given"
    hull = trimesh.load(tmp_path / "scene/objects/box_1/collision/hull_1.obj")
    assert np.allclose(hull.extents, 0.05)
    del objects["objects"][0]["collision"]

    objects["objects"][0]["T_base_obj"] = (2 * np.eye(4)).tolist()
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    with pytest.raises(ValueError, match="rigid"):
        assemble(ws, tmp_path / "objects.json", tmp_path / "scene2", "hull")


@pytest.mark.gl
def test_fit_recovers_scale_yaw_and_position(tmp_path):
    """The mesh is twice the object's size and not turned; two views see the object."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import SceneRenderer
    from r2s2r.tools.geometry import fit

    truth = SceneSpec(
        "t",
        "franka_panda",
        [ObjectSpec("box", "box", str(box_urdf(tmp_path, BOX)), TRUE_POSE)],
        np.eye(4),
        {},
        "c1",
        0,
        np.zeros(7),
    )
    renderer = SceneRenderer(truth, RGBD_SIZE)
    renderer.pose(0, TRUE_POSE)
    images, depths, masks = {}, {}, {}
    for serial, (_, T) in CAMERAS.items():
        blank = np.zeros(RGBD_SIZE[::-1])
        out = renderer.render(DepthView(serial, 0, blank, RGBD_K, T), 0)
        images[serial], depths[serial] = out["rgb"], out["depth"].astype(float)
        masks[serial] = out["mask"]
    renderer.close()
    capture = rgbd_capture(tmp_path / "capture", CAMERAS, 5, images, depths)
    ws = Workspace.create(capture, tmp_path / "run", "agentic")
    mask_paths = {}
    for serial, (role, _) in CAMERAS.items():
        path = tmp_path / f"mask_{role}.png"
        cv2.imwrite(str(path), masks[serial].astype(np.uint8) * 255)
        mask_paths[f"{role}@2"] = str(path)
    mesh = trimesh.creation.box(extents=(0.12, 0.08, 0.16))
    mesh.export(tmp_path / "mesh.obj")
    support = {"T_base_support": np.eye(4).tolist(), "extent": [1.0, 1.0]}

    result = fit(
        ws, tmp_path / "mesh.obj", support, mask_paths, tmp_path / "fit", up="z"
    )
    ws.close()
    assert result["scale"] == pytest.approx(0.5, abs=0.025)
    assert result["mean_iou"] > 0.9
    T = np.asarray(result["T_base_obj"])
    centre = T[:3, 3]  # the mesh file's origin is the box centre
    assert np.linalg.norm(centre[:2] - TRUE_POSE[:2, 3]) < 0.005
    assert centre[2] == pytest.approx(0.04, abs=0.005)  # resting on the support
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    assert abs((yaw - 30 + 90) % 180 - 90) < 4  # a box looks the same turned 180
    assert (tmp_path / "fit" / "ext1@2.png").exists()
