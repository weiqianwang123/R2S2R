"""Tests for reconstruct/orientation.py on synthetic objects (needs GL rendering)."""

import json

import numpy as np
import pytest
import trimesh

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.reconstruct import orientation  # noqa: E402
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec  # noqa: E402
from r2s2r.transforms import intrinsics_matrix, make_transform  # noqa: E402

K = intrinsics_matrix(300.0, 300.0, 159.5, 119.5)


def _object(tmp_path, name, parts):
    """A URDF whose single visual mesh is the union of boxes (extents, centre)."""
    meshes = []
    for extents, centre in parts:
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(centre)
        meshes.append(box)
    trimesh.util.concatenate(meshes).export(tmp_path / f"{name}.obj")
    urdf = tmp_path / f"{name}.urdf"
    urdf.write_text(
        f'<robot name="{name}"><link name="base"><visual><geometry>'
        f'<mesh filename="{name}.obj"/></geometry></visual></link></robot>'
    )
    return urdf


def _yaw(degrees, xy=(0.5, 0.0)):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return make_transform([[c, -s, 0], [s, c, 0], [0, 0, 1]], [xy[0], xy[1], 0.0])


def _look_at(eye, target=(0.5, 0.0, 0.04)):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return make_transform(np.column_stack([right, np.cross(fwd, right), fwd]), eye)


def _scene(urdf, T):
    obj = ObjectSpec("thing", "thing", str(urdf), T)
    return SceneSpec(
        name="t",
        embodiment="franka_panda",
        objects=[obj],
        T_base_support=np.eye(4),
        cameras={},
        reference_camera="c",
        reference_step=0,
        joint_positions=np.zeros(7),
    )


def _views(scene):
    """Depth and colour of ``scene`` from three cameras around the object."""
    renderer = orientation._SceneRenderer(  # pylint: disable=protected-access
        scene, (320, 240)
    )
    renderer.pose(0, scene.objects[0].T_base_obj)
    views = []
    for eye in ((0.1, 0.3, 0.4), (0.5, -0.45, 0.35), (0.95, 0.2, 0.4)):
        T = _look_at(eye)
        stub = DepthView("cam", 0, np.zeros((240, 320)), K, T)
        out = renderer.render(stub, 0)
        views.append(DepthView("cam", 0, out["depth"].astype(float), K, T, out["rgb"]))
    renderer.close()
    return views


@pytest.fixture(name="gl", autouse=True)
def fixture_gl():
    """Skip where MuJoCo cannot open an offscreen GL context (e.g. CI)."""
    mujoco = orientation.mujoco
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    try:
        mujoco.Renderer(model, 8, 8).close()
    except Exception as exc:  # pylint: disable=broad-except
        pytest.skip(f"no offscreen rendering: {exc}")


def test_geometry_turns_an_asymmetric_object_back(tmp_path):
    """An L-shaped block left turned by 90 degrees is turned back without a VLM."""
    urdf = _object(
        tmp_path,
        "ell",
        [
            ((0.16, 0.06, 0.08), (0.0, 0.0, 0.04)),
            ((0.06, 0.12, 0.08), (0.05, 0.08, 0.04)),
        ],
    )
    truth = _yaw(20.0)
    views = _views(_scene(urdf, truth))
    off = orientation.turned(_scene(urdf, truth).objects[0], np.eye(4), 90.0)
    fixed, report = orientation.check_orientations(_scene(urdf, off), views, vlm=None)
    entry = report["thing"]
    assert entry["decided_by"] == "geometry" and entry["turn_deg"] == 270.0
    assert np.allclose(fixed.objects[0].T_base_obj, truth, atol=1e-6)


def test_vlm_breaks_a_symmetric_tie(tmp_path):
    """A box turned by 180 degrees looks the same in depth; the VLM decides."""
    urdf = _object(tmp_path, "box", [((0.16, 0.08, 0.08), (0.0, 0.0, 0.04))])
    truth = _yaw(20.0)
    views = _views(_scene(urdf, truth))
    off = orientation.turned(_scene(urdf, truth).objects[0], np.eye(4), 180.0)
    asked = []

    def vlm(prompt, images):
        asked.append((prompt, list(images)))
        return json.dumps({"image": 3, "confident": True})  # the second candidate

    fixed, report = orientation.check_orientations(_scene(urdf, off), views, vlm=vlm)
    entry = report["thing"]
    assert entry["candidates"] == [0.0, 180.0] and entry["decided_by"] == "vlm"
    assert len(asked) == 1 and len(asked[0][1]) == 3  # photo + two candidates
    assert np.allclose(fixed.objects[0].T_base_obj, truth, atol=1e-6)

    unsure = orientation.check_orientations(
        _scene(urdf, off), views, vlm=lambda p, i: '{"image": 3, "confident": false}'
    )[0]
    assert np.allclose(unsure.objects[0].T_base_obj, off)  # the backend's pose stays
