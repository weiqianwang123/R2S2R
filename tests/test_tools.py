"""Tests for r2s2r.tools: points from depth, the plane fit, the pattern search,
assembly, fitting a mesh to several views."""

import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
import trimesh
from conftest import (
    RGBD_K,
    RGBD_SIZE,
    box_urdf,
    hinged_box,
    rgbd_capture,
    towel_mesh,
)
from scipy.spatial.transform import Rotation

from r2s2r import cli
from r2s2r.assets import joint_transform, object_points, urdf_visual_meshes
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec
from r2s2r.tools import check as check_module
from r2s2r.tools import cli as tools_cli
from r2s2r.tools.check import check
from r2s2r.tools.geometry import fit_plane, parse_masks, pattern_search, view_points
from r2s2r.tools.objects import (
    CLOTH_AREAL_DENSITY,
    CLOTH_POISSONS_RATIO,
    articulation_problems,
    assemble,
    cloth_problems,
)
from r2s2r.transforms import invert, look_at, make_transform, transform_points
from r2s2r.workspace import Workspace

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


def _hinged_box(tmp_path):
    """A workspace, and the objects file of :func:`conftest.hinged_box` posed at
    ``TRUE_POSE``: the workspace, the objects, and the body, the lid as recorded, and
    the lid shut."""
    capture = rgbd_capture(tmp_path / "capture", CAMERAS)
    ws = Workspace.create(capture, tmp_path / "run", "agentic")
    objects, meshes = hinged_box(tmp_path, TRUE_POSE)
    return ws, objects, meshes


def test_assemble_an_articulated_object_keeps_its_parts_where_recorded(tmp_path):
    """The hinged box becomes a two-link URDF: at the recorded joint position every
    part is where it was, at zero the lid lies shut; broken joints are refused."""
    ws, objects, (body, lid, shut) = _hinged_box(tmp_path)
    report = assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "hull")
    (obj,) = SceneSpec.load(tmp_path / "scene").objects
    assert obj.joints == {"hinge": pytest.approx(-0.8)}
    assert report["objects"]["box"]["joints"]["hinge"]["limits"] == [-1.9, 0.0]
    root = ET.parse(obj.asset_path).getroot()
    assert [link.attrib["name"] for link in root.iter("link")] == ["base", "lid"]
    masses = [float(m.attrib["value"]) for m in root.iter("mass")]
    assert sum(masses) == pytest.approx(0.4) and masses[1] < masses[0]
    as_recorded = urdf_visual_meshes(obj.asset_path, obj.joints)[1].mesh  # the lid
    assert np.allclose(np.sort(as_recorded.vertices, 0), np.sort(lid.vertices, 0))
    at_zero = urdf_visual_meshes(obj.asset_path)[1].mesh
    assert np.allclose(np.sort(at_zero.vertices, 0), np.sort(shut.vertices, 0))
    # Placed in the base frame as recorded: the body and the raised lid.
    seen = trimesh.util.concatenate([body, lid])
    local = transform_points(invert(TRUE_POSE), object_points(obj, 2000))
    assert np.allclose(local.min(0), seen.bounds[0], atol=2e-3)
    assert np.allclose(local.max(0), seen.bounds[1], atol=2e-3)

    joint = objects["objects"][0]["joints"][0]
    for broken, problem in (
        ({"position": 0.5}, "outside its limits"),
        ({"parent": "lid"}, "tree"),
        ({"type": "ball"}, "type"),
        ({"axis": [0, 0, 0]}, "axis not 0"),
    ):
        objects["objects"][0]["joints"] = [{**joint, **broken}]
        assert any(problem in p for p in articulation_problems(objects["objects"][0]))


def test_nested_and_prismatic_joints_are_where_recorded(tmp_path):
    """A lid turned about a slanted hinge carrying a knob that slides along the lid's
    normal, the child joint listed first: at the recorded positions and at zero every
    link is where it should be."""
    capture = rgbd_capture(tmp_path / "capture", CAMERAS)
    ws = Workspace.create(capture, tmp_path / "run", "agentic")
    body = trimesh.creation.box(extents=(0.2, 0.1, 0.06))
    body.apply_translation([0, 0, 0.03])
    lid0 = trimesh.creation.box(extents=(0.2, 0.1, 0.01))
    lid0.apply_translation([0, 0, 0.065])
    knob0 = trimesh.creation.box(extents=(0.02, 0.02, 0.01))
    knob0.apply_translation([0.05, -0.03, 0.075])
    hinge = np.array([0.0, 0.05, 0.06])
    axis = np.array([1.0, 0.2, 0.0]) / np.linalg.norm([1.0, 0.2, 0.0])
    turn = (
        make_transform(np.eye(3), hinge)
        @ joint_transform("revolute", axis, -0.7)
        @ make_transform(np.eye(3), -hinge)
    )
    slide = turn[:3, :3] @ [0.0, 0.0, 1.0]
    lid, knob = lid0.copy(), knob0.copy()
    lid.apply_transform(turn)
    knob.apply_transform(turn)
    knob.apply_translation(slide * 0.015)
    for name, mesh in (("body", body), ("lid", lid), ("knob", knob)):
        mesh.export(tmp_path / f"{name}.obj")
    objects = {
        "support": {"T_base_support": np.eye(4).tolist(), "extent": [1.0, 1.0]},
        "objects": [
            {
                "name": "box",
                "mesh": "body.obj",
                "T_base_obj": TRUE_POSE.tolist(),
                "parts": [
                    {"name": "knob", "mesh": "knob.obj"},
                    {"name": "lid", "mesh": "lid.obj"},
                ],
                "joints": [
                    {
                        "name": "slide",
                        "type": "prismatic",
                        "parent": "lid",
                        "child": "knob",
                        "origin": transform_points(turn, [[0.05, -0.03, 0.07]])[
                            0
                        ].tolist(),
                        "axis": slide.tolist(),
                        "limits": [0.0, 0.03],
                        "position": 0.015,
                    },
                    {
                        "name": "hinge",
                        "type": "revolute",
                        "parent": "base",
                        "child": "lid",
                        "origin": hinge.tolist(),
                        "axis": axis.tolist(),
                        "limits": [-1.9, 0.0],
                        "position": -0.7,
                    },
                ],
            }
        ],
    }
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "none")
    (obj,) = SceneSpec.load(tmp_path / "scene").objects
    for joints, want in (
        (obj.joints, {"base": body, "lid": lid, "knob": knob}),
        (None, {"base": body, "lid": lid0, "knob": knob0}),
    ):
        for visual in urdf_visual_meshes(obj.asset_path, joints):
            assert np.allclose(
                np.sort(visual.mesh.vertices, 0),
                np.sort(want[visual.link].vertices, 0),
                atol=1e-6,
            ), visual.link


def test_articulation_problems_name_the_bad_names(tmp_path):
    """Part and joint names: missing, not lower case, taken, or twice."""
    _, objects, _ = _hinged_box(tmp_path)
    box = objects["objects"][0]
    joint = box["joints"][0]
    for change, problem in (
        ({"joints": [{**joint, "name": None}]}, "joint names must be lower case"),
        ({"parts": [{"name": "Lid", "mesh": "lid.obj"}]}, "part names must be lower"),
        ({"parts": [{"name": "visual", "mesh": "lid.obj"}]}, "no part may be called"),
        ({"joints": [joint, joint]}, "joint names must be unique"),
    ):
        assert any(problem in p for p in articulation_problems({**box, **change}))


def test_assemble_a_cloth_keeps_its_surface_and_material(tmp_path):
    """A towel: its surface as it lies (no collision parts, no resting base), its
    material, its mass (else an areal density); a bad material is refused."""
    capture = rgbd_capture(tmp_path / "capture", CAMERAS)
    ws = Workspace.create(capture, tmp_path / "run", "agentic")
    towel_mesh(0.3).export(tmp_path / "towel.obj")
    towel = {
        "name": "towel",
        "mesh": "towel.obj",
        "T_base_obj": make_transform(np.eye(3), [0.5, 0.0, 0.001]).tolist(),
        "friction": 0.8,
        "cloth": {"thickness": 0.002, "youngs_modulus": 5e5},
    }
    objects = {
        "support": {"T_base_support": np.eye(4).tolist(), "extent": [1.0, 1.0]},
        "objects": [towel],
    }
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    report = assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "coacd")
    (obj,) = SceneSpec.load(tmp_path / "scene").objects
    assert obj.cloth == {
        "thickness": 0.002,
        "youngs_modulus": 5e5,
        "poissons_ratio": CLOTH_POISSONS_RATIO,
    }
    assert obj.mass == pytest.approx(CLOTH_AREAL_DENSITY * 0.09)
    assert report["objects"]["towel"]["area_m2"] == pytest.approx(0.09)
    root = ET.parse(obj.asset_path).getroot()
    assert not list(root.iter("collision"))
    visuals = urdf_visual_meshes(obj.asset_path)
    assert len(visuals) == 1 and np.allclose(visuals[0].mesh.extents, [0.3, 0.3, 0])

    for change, problem in (
        ({"cloth": {"thickness": 0.0, "youngs_modulus": 5e5}}, "thickness"),
        ({"cloth": {"thickness": 0.002}}, "needs thickness and youngs_modulus"),
        ({"cloth": {**towel["cloth"], "poissons_ratio": 0.7}}, "poissons_ratio"),
        ({"parts": [{"name": "lid", "mesh": "towel.obj"}]}, "a cloth has no parts"),
    ):
        assert any(problem in p for p in cloth_problems({**towel, **change}))


def test_check_cli_reads_joint_positions(monkeypatch):
    """``r2s2r tool check --joint OBJECT:JOINT=VALUE ...`` per object and joint."""
    seen = []
    monkeypatch.setattr(tools_cli, "_ws", lambda args: None)
    monkeypatch.setattr(check_module, "check", lambda *args: seen.append(args) or {})
    argv = ["tool", "check", "scene", "--frames", "ext1@0", "--out", "o", "--joint"]
    cli.main(argv + ["box:hinge=1.2", "box:slide=0.01", "door:pin=-0.5"])
    assert seen[0][-1] == {"box": {"hinge": 1.2, "slide": 0.01}, "door": {"pin": -0.5}}
    with pytest.raises(SystemExit, match="OBJECT:JOINT=VALUE"):
        cli.main(argv + ["box=1.2"])


@pytest.mark.gl
def test_check_renders_an_articulated_object_with_its_joints_moved(tmp_path):
    """``check`` with a joint moved renders the part elsewhere; unknown joints are
    refused."""
    ws, _, _ = _hinged_box(tmp_path)
    assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "none")
    fid = ws.frame_id(ws.capture.frames[0])
    recorded = check(ws, tmp_path / "scene", [fid], tmp_path / "check_open")
    shut = check(
        ws, tmp_path / "scene", [fid], tmp_path / "check_shut", {"box": {"hinge": 0}}
    )
    assert shut["joints"] == {"box": {"hinge": 0}}
    images = [cv2.imread(r["frames"][0]["image"]) for r in (recorded, shut)]
    assert not np.array_equal(*images)
    with pytest.raises(ValueError, match="no joints"):
        check(ws, tmp_path / "scene", [fid], tmp_path / "c", {"box": {"lock": 0}})


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
        out = renderer.render(DepthView(serial, 0, blank, RGBD_K, T))
        images[serial], depths[serial] = out["rgb"], out["depth"].astype(float)
        masks[serial] = out["object"] == 0
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
