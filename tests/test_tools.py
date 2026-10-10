"""Tests for r2s2r.tools: points from depth, the plane fit, the pattern search,
assembly, fitting a mesh to several views."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

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
    robot_or_skip,
    towel_mesh,
)
from scipy.spatial.transform import Rotation

from r2s2r import cli
from r2s2r.assets import joint_transform, object_points, urdf_visual_meshes
from r2s2r.structs import DepthView, JointDynamics, ObjectSpec, SceneSpec
from r2s2r.tools import check as check_module
from r2s2r.tools import cli as tools_cli
from r2s2r.tools import envjobs, geometry
from r2s2r.tools import segment as segment_module
from r2s2r.tools.check import check
from r2s2r.tools.geometry import (
    UP_ROTATIONS,
    _initial_pose,
    fit_plane,
    parse_masks,
    pattern_search,
    view_points,
)
from r2s2r.tools.objects import (
    CLOTH_AREAL_DENSITY,
    CLOTH_POISSONS_RATIO,
    articulation_problems,
    assemble,
    cloth_problems,
    physics_problems,
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

    # Names that are one file name: refused, not one written over the other.
    objects["objects"][0]["T_base_obj"] = T.tolist()
    objects["objects"].append({**objects["objects"][0], "name": "box_1"})
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    with pytest.raises(ValueError, match=r"unique as file names: \['Box 1', 'box_1'\]"):
        assemble(ws, tmp_path / "objects.json", tmp_path / "scene3", "hull")


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


def test_a_joints_dynamics_reach_the_scene(tmp_path):
    """A joint's damping, friction and spring go into the scene and the report; a
    joint that gives none is free; negative values, and a spring's rest outside the
    limits, are refused."""
    ws, objects, _ = _hinged_box(tmp_path)
    joint = objects["objects"][0]["joints"][0]
    sprung = {"damping": 0.02, "friction": 0.05, "stiffness": 0.3, "rest": -0.1}
    objects["objects"][0]["joints"] = [{**joint, **sprung}]
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    report = assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "none")
    (obj,) = SceneSpec.load(tmp_path / "scene").objects
    assert obj.joint_dynamics == {"hinge": JointDynamics(**sprung)}
    assert report["objects"]["box"]["joints"]["hinge"]["stiffness"] == 0.3
    objects["objects"][0]["joints"] = [joint]
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    assemble(ws, tmp_path / "objects.json", tmp_path / "free", "none")
    (free,) = SceneSpec.load(tmp_path / "free").objects
    assert free.dynamics("hinge") == JointDynamics()
    for broken, problem in (
        ({"damping": -0.1}, "damping must not be negative"),
        ({"friction": "stiff"}, "must be numbers"),
        ({"stiffness": 0.3, "rest": 0.5}, "rest 0.5 is outside its limits"),
    ):
        objects["objects"][0]["joints"] = [{**joint, **broken}]
        assert any(problem in p for p in articulation_problems(objects["objects"][0]))


def test_ranges_reach_the_scene_and_bad_ones_are_refused(tmp_path):
    """An object's and its joints' ranges go into the scene and the report; a range
    that misses its estimate, goes below 0, puts a rest outside the limits or has no
    estimate is refused."""
    ws, objects, _ = _hinged_box(tmp_path)
    box = objects["objects"][0]
    joint = box["joints"][0]
    box.update(mass=0.3, mass_range=[0.2, 0.45], friction=0.6)
    box["joints"] = [{**joint, "damping": 0.01, "damping_range": [0.002, 0.05]}]
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    report = assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "none")
    (obj,) = SceneSpec.load(tmp_path / "scene").objects
    assert obj.ranges == {"mass": (0.2, 0.45), "hinge.damping": (0.002, 0.05)}
    assert report["objects"]["box"]["ranges"]["mass"] == [0.2, 0.45]
    lower, upper = joint["limits"]
    for broken, problem in (
        ({"mass_range": [0.35, 0.45]}, "must hold the estimate 0.3"),
        ({"friction_range": "wide"}, "must be two numbers"),
        ({"mass": None}, "mass_range needs a mass too"),
    ):
        assert any(problem in p for p in physics_problems({**box, **broken}))
    for broken, problem in (
        ({"friction_range": [-0.1, 0.1]}, "must lie within [0.0, inf]"),
        ({"rest_range": [lower - 1.0, upper]}, "must lie within"),
    ):
        entry = {**box, "joints": [{**joint, **broken}]}
        assert any(problem in p for p in articulation_problems(entry))


def test_assemble_a_cloth_keeps_its_surface_and_material(tmp_path):
    """A towel: its surface as it lies (no collision parts, no resting base), its
    material, its mass (else an areal density); a bad material, and a surface in
    pieces, are refused."""
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
    assert obj.ranges is None
    assert report["objects"]["towel"]["area_m2"] == pytest.approx(0.09)
    assert report["objects"]["towel"]["lowest_point_above_support_m"] == 0.001
    root = ET.parse(obj.asset_path).getroot()
    assert not list(root.iter("collision"))
    visuals = urdf_visual_meshes(obj.asset_path)
    assert len(visuals) == 1 and np.allclose(visuals[0].mesh.extents, [0.3, 0.3, 0])

    for change, problem in (
        ({"cloth": {"thickness": 0.0, "youngs_modulus": 5e5}}, "thickness"),
        ({"cloth": {"thickness": 0.002}}, "needs thickness and youngs_modulus"),
        ({"cloth": {**towel["cloth"], "poissons_ratio": 0.7}}, "poissons_ratio"),
        ({"parts": [{"name": "lid", "mesh": "towel.obj"}]}, "a cloth has no parts"),
        ({"collision": ["hull.obj"]}, "a cloth has no parts"),
        ({"cloth": 0.002}, "cloth must be an object"),
        ({"cloth": {"thickness": 0.002, "youngs_modulus": 0}}, "must be positive"),
        ({"friction_range": [0.6, 1.0]}, "a cloth has no ranges"),
    ):
        assert any(problem in p for p in cloth_problems({**towel, **change}))
    halves = [towel_mesh(0.14, 5).apply_translation([x, 0, 0]) for x in (-0.08, 0.08)]
    trimesh.util.concatenate(halves).export(tmp_path / "towel.obj")
    with pytest.raises(ValueError, match="one piece"):
        assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "coacd")


def test_check_cli_reads_joint_positions(monkeypatch):
    """``r2s2r tool check --joint OBJECT:JOINT=VALUE ...`` per object and joint."""
    seen = []
    monkeypatch.setattr(tools_cli, "_ws", lambda args: None)
    monkeypatch.setattr(check_module, "check", lambda *args: seen.append(args) or {})
    argv = ["tool", "check", "scene", "--frames", "ext1@0", "--out", "o", "--joint"]
    cli.main(argv + ["box:hinge=1.2", "box:slide=0.01", "door:pin=-0.5"])
    assert seen[0][-1] == {"box": {"hinge": 1.2, "slide": 0.01}, "door": {"pin": -0.5}}
    cli.main(argv[:-1] + ["--joint", "box:hinge=1", "--joint", "door:pin=2"])
    assert seen[1][-1] == {"box": {"hinge": 1.0}, "door": {"pin": 2.0}}
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
        "fr3_robotiq",
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
    # A given yaw is where the search starts, not where it stays.
    started = fit(
        ws, tmp_path / "mesh.obj", support, mask_paths, tmp_path / "f2", "z", 0.5, 22.0
    )
    ws.close()
    assert abs((started["yaw_deg"] - 30 + 90) % 180 - 90) < 4
    assert result["scale"] == pytest.approx(0.5, abs=0.025)
    assert result["mean_iou"] > 0.9
    T = np.asarray(result["T_base_obj"])
    centre = T[:3, 3]  # the mesh file's origin is the box centre
    assert np.linalg.norm(centre[:2] - TRUE_POSE[:2, 3]) < 0.005
    assert centre[2] == pytest.approx(0.04, abs=0.005)  # resting on the support
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    assert abs((yaw - 30 + 90) % 180 - 90) < 4  # a box looks the same turned 180
    assert (tmp_path / "fit" / "ext1@2.png").exists()


def test_print_json_keeps_number_lists_on_one_line(capsys):
    """Lists of numbers on one line; a long one ending in something else is quick."""
    tools_cli.print_json(
        {"a": [1, -2.5, 3e-05], "b": ["x", 1], "c": [1.0] * 40 + [None]}
    )
    out = capsys.readouterr().out
    assert '"a": [1, -2.5, 3e-05]' in out and '"x",' in out
    assert json.loads(out)["c"][-1] is None


def test_fit_cli_knows_the_up_axes(monkeypatch):
    """``--up`` offers what fit turns, and refuses the rest."""
    seen = []
    monkeypatch.setattr(tools_cli, "_ws", lambda args: None)
    monkeypatch.setattr(geometry, "fit", lambda *a, **kw: seen.append(kw["up"]) or {})
    argv = ["tool", "fit", "m.obj", "--support", "s", "--mask", "f=m", "--out", "o"]
    for up in UP_ROTATIONS:
        cli.main(argv + [f"--up={up}"])
    assert sorted(seen) == sorted(UP_ROTATIONS)
    with pytest.raises(SystemExit):
        cli.main(argv + ["--up=-z"])


def test_points_and_support_want_frames_or_masks(tmp_path):
    """Neither frames nor masks: said so."""
    ws = Workspace.create(
        rgbd_capture(tmp_path / "c", CAMERAS), tmp_path / "r", "agentic"
    )
    with pytest.raises(ValueError, match="--frames or --mask"):
        geometry.points(ws, [], {}, tmp_path / "p.ply")
    with pytest.raises(ValueError, match="--frames or --mask"):
        geometry.support(ws, {}, tmp_path / "s.json")


def test_initial_pose_of_a_lifted_object():
    """An object not resting on the support: scale from its own height, starting at
    its bottom; a given scale puts its top on the observed top."""
    rng = np.random.default_rng(0)
    observed = rng.uniform([-0.03, -0.02, 0.10], [0.03, 0.02, 0.18], (2000, 3))
    model = rng.uniform([-0.06, -0.04, 0.0], [0.06, 0.04, 0.16], (2000, 3))
    params, start = _initial_pose(observed, model, 0.16, None, 0.0, rest=False)
    assert np.exp(params[0]) == pytest.approx(0.5, abs=0.03)
    assert params[4] == pytest.approx(0.10, abs=0.005)
    assert start["observed_bottom_m"] == pytest.approx(0.10, abs=0.005)
    params, _ = _initial_pose(observed, model, 0.16, 0.5, 0.0, rest=False)
    assert params[4] == pytest.approx(0.10, abs=0.005)
    params, start = _initial_pose(observed, model, 0.16, None, 0.0, rest=True)
    assert params[4] == 0.0 and "observed_bottom_m" not in start


def _fake_sam3(calls):
    """A SAM3 job writing one mask per request (its frame's left half)."""

    def run(script, job, workdir, env):  # pylint: disable=unused-argument
        calls.append(job)
        results = []
        for req in job["requests"]:
            out = Path(req["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros(RGBD_SIZE[::-1], np.uint8)
            mask[:, : RGBD_SIZE[0] // 2] = 255
            cv2.imwrite(f"{out}_0.png", mask)
            inst = {"mask": f"{out}_0.png", "score": 0.9, "box": [0, 0, 10, 10]}
            results.append({"request": req, "instances": [inst]})
        return {"results": results}

    return run


def test_segment_keeps_every_text_apart(tmp_path, monkeypatch):
    """Each text in its own directory with its own masks.json; one --name is for one
    text; text and point prompts do not mix."""
    robot_or_skip("fr3_robotiq")  # each mask's share on the robot, rendered
    calls = []
    monkeypatch.setattr(segment_module, "run_env_job", _fake_sam3(calls))
    ws = Workspace.create(
        rgbd_capture(tmp_path / "c", CAMERAS), tmp_path / "r", "agentic"
    )
    out = tmp_path / "masks"
    summary = segment_module.segment(ws, ["ext1@0", "ext1@1"], out, ["red mug", "box"])
    assert sorted(summary) == ["box", "red_mug"]
    assert summary["red_mug"]["prompt"] == "red mug"
    inst = summary["box"]["frames"]["ext1@1"]["instances"][0]
    assert inst["mask"] == "box/ext1@1_0.png" and inst["on_robot"] >= 0
    saved = json.loads((out / "red_mug" / "masks.json").read_text())
    assert sorted(saved["frames"]) == ["ext1@0", "ext1@1"]
    summary = segment_module.segment(ws, ["ext1@0"], out, ["mug"], name="cup")
    assert list(summary) == ["cup"] and summary["cup"]["prompt"] == "mug"
    with pytest.raises(ValueError, match="one text"):
        segment_module.segment(ws, ["ext1@0"], out, ["mug", "cup"], name="cup")
    with pytest.raises(ValueError, match="share"):
        segment_module.segment(ws, ["ext1@0"], out, ["red mug", "red-mug"])
    with pytest.raises(ValueError, match="not both"):
        segment_module.segment(ws, ["ext1@0"], out, ["mug"], box=(0, 0, 5, 5))
    assert len(calls) == 2


def test_a_failing_env_job_raises_its_log(tmp_path, monkeypatch):
    """A job that fails: its exit code and its log's tail are raised."""
    mamba = tmp_path / "mamba"
    mamba.write_text("#!/bin/sh\necho 'CUDA out of memory'\nexit 3\n")
    mamba.chmod(0o755)
    monkeypatch.setattr(envjobs, "mamba_exe", lambda: str(mamba))
    with pytest.raises(RuntimeError, match="exit 3(.|\n)*CUDA out of memory"):
        envjobs.run_env_job("sam3_job.py", {"a": 1}, tmp_path, "simfoundry")
    assert json.loads(next((tmp_path / "logs").glob("*[0-9].json")).read_text()) == {
        "a": 1
    }
