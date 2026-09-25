"""Tests for assets.py: sim-ready URDFs and the URDF readers."""

import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from r2s2r.assets import (
    RESTING_BASE,
    make_sim_ready,
    urdf_movable_joints,
    urdf_visual_points,
)


def _write_urdf(tmp_path, name, body):
    urdf = tmp_path / f"{name}.urdf"
    urdf.write_text(f'<robot name="{name}">{body}</robot>')
    return urdf


def _mesh_link(link, mesh_file, scale=None, collision=True):
    s = f' scale="{scale}"' if scale else ""
    geometry = f'<geometry><mesh filename="{mesh_file}"{s}/></geometry>'
    col = f"<collision>{geometry}</collision>" if collision else ""
    return f'<link name="{link}"><visual>{geometry}</visual>{col}</link>'


def _tapered_block(tmp_path):
    """4 x 4 cm block whose bottom 5 mm narrows to a 1 x 1 cm foot."""
    pts = [
        [x * s, y * s, z]
        for s, z in ((0.005, 0.0), (0.02, 0.005), (0.02, 0.1))
        for x in (-1, 1)
        for y in (-1, 1)
    ]
    trimesh.convex.convex_hull(np.array(pts)).export(tmp_path / "block.obj")
    return _write_urdf(tmp_path, "block", _mesh_link("base", "block.obj"))


def _collision_names(urdf):
    return [c.attrib.get("name") for c in ET.parse(urdf).getroot().iter("collision")]


def test_mesh_scales_are_baked(tmp_path):
    """Scaled meshes are written at their true size and lose the attribute."""
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(tmp_path / "unit.obj")
    urdf = _write_urdf(tmp_path, "box", _mesh_link("base", "unit.obj", "0.5 2 1"))
    lifted = np.eye(4)
    lifted[2, 3] = 0.5  # off the support: no base, only the baking
    out = make_sim_ready(urdf, lifted, articulated=False)
    assert out.name == "box_r2s2r.urdf"
    assert "scale" not in out.read_text(encoding="utf-8")
    assert np.allclose(
        np.ptp(urdf_visual_points(out, 2000), 0), (0.05, 0.2, 0.1), atol=2e-3
    )
    assert np.allclose(
        np.ptp(urdf_visual_points(urdf, 2000), 0), (0.05, 0.2, 0.1), atol=2e-3
    )


def test_resting_base_fills_the_footprint(tmp_path):
    """Resting objects get a flat base as wide as the body just above the bottom."""
    out = make_sim_ready(_tapered_block(tmp_path), np.eye(4), articulated=False)
    assert _collision_names(out).count(RESTING_BASE) == 1
    base = trimesh.load(tmp_path / "block_r2s2r_base.obj", force="mesh")
    assert np.allclose(base.bounds[:, 2], [0.0, 0.005], atol=1e-6)
    assert np.allclose(base.bounds[1, :2] - base.bounds[0, :2], 0.04, atol=1e-3)
    # Rebuilding from the output keeps a single base.
    again = make_sim_ready(out, np.eye(4), articulated=False)
    assert _collision_names(again).count(RESTING_BASE) == 1


def test_no_base_off_the_support(tmp_path):
    """An object 5 cm above the support is not resting on it."""
    lifted = np.eye(4)
    lifted[2, 3] = 0.05
    out = make_sim_ready(_tapered_block(tmp_path), lifted, articulated=False)
    assert RESTING_BASE not in _collision_names(out)


def test_movable_joints_in_root_frame(tmp_path):
    """Axes and pivots are composed along the chain into the root link frame."""
    urdf = _write_urdf(
        tmp_path,
        "c",
        '<link name="body"/><link name="mid"/><link name="drawer"/><link name="door"/>'
        '<joint name="fix" type="fixed"><parent link="body"/><child link="mid"/>'
        '<origin xyz="0 0 0.1" rpy="0 0 1.5707963"/></joint>'
        '<joint name="slide" type="prismatic">'
        '<parent link="mid"/><child link="drawer"/>'
        '<origin xyz="0.2 0 0"/><axis xyz="1 0 0"/>'
        '<limit lower="0" upper="0.15"/></joint>'
        '<joint name="hinge" type="revolute"><parent link="body"/><child link="door"/>'
        '<axis xyz="0 0 2"/><limit lower="-1.5" upper="0"/></joint>',
    )
    joints = {j["name"]: j for j in urdf_movable_joints(urdf)}
    assert set(joints) == {"slide", "hinge"}
    slide = joints["slide"]
    assert slide["type"] == "prismatic" and (slide["lower"], slide["upper"]) == (
        0.0,
        0.15,
    )
    assert np.allclose(slide["axis"], [0, 1, 0], atol=1e-6)  # turned by the fixed joint
    assert np.allclose(slide["origin"], [0, 0.2, 0.1], atol=1e-6)
    assert np.allclose(joints["hinge"]["axis"], [0, 0, 1])
