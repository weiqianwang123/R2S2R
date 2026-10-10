"""Tests for assets.py: sim-ready URDFs, the URDF readers and bottom offsets."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from r2s2r.assets import (
    RESTING_BASE,
    base_color_texture,
    bottom_offset,
    export_visual,
    make_sim_ready,
    urdf_visual_meshes,
)
from r2s2r.tools.geometry import load_mesh


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


def test_a_scaled_mesh_is_refused(tmp_path):
    """A ``<mesh scale>`` other than 1 is refused (Isaac's importer mishandles it); a
    scale of 1 is as good as none."""
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(tmp_path / "unit.obj")
    lifted = np.eye(4)
    lifted[2, 3] = 0.5  # off the support: no base
    urdf = _write_urdf(tmp_path, "box", _mesh_link("base", "unit.obj", "0.5 2 1"))
    with pytest.raises(ValueError, match="scaled"):
        make_sim_ready(urdf, lifted)
    urdf = _write_urdf(tmp_path, "one", _mesh_link("base", "unit.obj", "1 1 1"))
    assert make_sim_ready(urdf, lifted).name == "one_r2s2r.urdf"


def test_resting_base_fills_the_footprint(tmp_path):
    """Resting objects get a flat base as wide as the body just above the bottom."""
    out = make_sim_ready(_tapered_block(tmp_path), np.eye(4))
    assert _collision_names(out).count(RESTING_BASE) == 1
    base = trimesh.load(tmp_path / "block_r2s2r_base.obj", force="mesh")
    assert np.allclose(base.bounds[:, 2], [0.0, 0.005], atol=1e-6)
    assert np.allclose(base.bounds[1, :2] - base.bounds[0, :2], 0.04, atol=1e-3)
    # Rebuilding from the output keeps a single base.
    again = make_sim_ready(out, np.eye(4))
    assert _collision_names(again).count(RESTING_BASE) == 1


def test_no_base_off_the_support(tmp_path):
    """An object 5 cm above the support is not resting on it."""
    lifted = np.eye(4)
    lifted[2, 3] = 0.05
    out = make_sim_ready(_tapered_block(tmp_path), lifted)
    assert RESTING_BASE not in _collision_names(out)


def test_links_are_placed_by_their_joints(tmp_path):
    """A second link's mesh sits where the joint's origin puts it, in the root frame."""
    trimesh.creation.box(extents=(0.02, 0.02, 0.02)).export(tmp_path / "cube.obj")
    urdf = _write_urdf(
        tmp_path,
        "two",
        _mesh_link("body", "cube.obj")
        + _mesh_link("lid", "cube.obj")
        + '<joint name="fix" type="fixed"><parent link="body"/><child link="lid"/>'
        '<origin xyz="0 0 0.1" rpy="0 0 1.5707963"/></joint>',
    )
    meshes = urdf_visual_meshes(urdf)
    assert len(meshes) == 2
    assert np.allclose(meshes[0].mesh.centroid, 0.0, atol=1e-6)
    assert np.allclose(meshes[1].mesh.centroid, [0.0, 0.0, 0.1], atol=1e-6)


def test_bottom_offset_is_physcoders_convention():
    """An upright block's bottom centre, below its origin, identity rotation (like
    physcoder's block: z -0.021639); an object lying on its side turns z up."""
    block = trimesh.creation.box(extents=[0.0412, 0.041, 0.043278]).vertices
    pos, quat = bottom_offset(block, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(pos, [0.0, 0.0, -0.021639], atol=1e-6)
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])
    # A y-up mesh (up is the object's +y), off-centre in x.
    mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3]).vertices + [0.05, 0.0, 0.0]
    pos, quat = bottom_offset(mesh, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(pos, [0.05, -0.1, 0.0], atol=1e-9)
    R = Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
    assert np.allclose(R[:, 2], [0.0, 1.0, 0.0]) and quat[0] > 0


def test_meshes_of_a_directory_keep_their_own_textures(tmp_path):
    """Two textured links (a glTF's PBR materials) written into one directory each
    keep their texture, named after their mesh."""
    image = pytest.importorskip("PIL.Image")
    for stem, colour in (("visual", (200, 30, 30)), ("lid", (30, 30, 200))):
        mesh = trimesh.creation.box()
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.random.default_rng(0).random((len(mesh.vertices), 2)),
            image=image.new("RGB", (4, 4), colour),
        )
        mesh.export(tmp_path / f"{stem}.glb")
    for stem in ("visual", "lid"):
        export_visual(load_mesh(tmp_path / f"{stem}.glb"), tmp_path / f"{stem}.obj")
    for stem, colour in (("visual", (200, 30, 30)), ("lid", (30, 30, 200))):
        texture = base_color_texture(tmp_path / f"{stem}.obj")
        assert texture is not None and texture.name == f"{stem}.png"
        assert image.open(texture).convert("RGB").getpixel((0, 0)) == colour
