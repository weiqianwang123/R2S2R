"""Tests for sim/isaac.py: where PhysX runs a scene, which way gravity points, and
the support's colour."""

import cv2
import numpy as np
import pytest

from r2s2r.sim.isaac import gravity, physics_device, support_color
from r2s2r.structs import (
    CameraSpec,
    Capture,
    FrameRecord,
    ObjectSpec,
    SceneSpec,
    write_depth,
)
from r2s2r.transforms import make_transform


def _scene(objects, T_base_support=np.eye(4)):
    return SceneSpec(
        name="t",
        embodiment="franka_panda",
        objects=objects,
        T_base_support=T_base_support,
        cameras={},
        reference_camera="c",
        reference_step=0,
        joint_positions=np.zeros(7),
    )


def _object(name, **kwargs):
    return ObjectSpec(name, name, f"{name}.urdf", np.eye(4), **kwargs)


def test_a_scene_runs_on_the_cpu_unless_it_has_a_cloth():
    """Rigid and articulated objects on the CPU; a cloth needs the GPU."""
    box = _object("box", joints={"lid_hinge": 0.0})
    towel = _object(
        "towel",
        cloth={"thickness": 0.002, "youngs_modulus": 5e5, "poissons_ratio": 0.3},
    )
    assert physics_device(_scene([_object("cup"), box]), "cuda:0") == "cpu"
    assert physics_device(_scene([box, towel]), "cuda:0") == "cuda:0"


def test_gravity_is_along_the_supports_normal():
    """A support leaning 0.4 degrees: gravity leans with it, 9.81 m/s^2 still."""
    a = np.radians(0.4)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    g = np.array(gravity(_scene([], make_transform(R, [0.4, 0.0, -0.02]))))
    assert np.linalg.norm(g) == pytest.approx(9.81)
    assert g / 9.81 == pytest.approx(-R[:, 2])


def test_the_support_takes_the_colour_of_the_pixels_on_its_plane(tmp_path):
    """A camera 0.5 m above the table, looking down: the table's pixels give its
    colour; a box standing on it, above the plane, and the pixels beyond the
    table's outline do not."""
    depth = np.full((48, 64), 0.5)
    depth[:36] = 0.45  # a box's top, 5 cm up, over most of the view
    rgb = np.full((48, 64, 3), (200, 160, 90), np.uint8)
    rgb[:36] = (0, 0, 255)
    rgb[:, 60:] = (255, 255, 255)  # beyond x = 0.14 m
    write_depth(tmp_path / "d.png", depth)
    cv2.imwrite(str(tmp_path / "c.png"), rgb[..., ::-1])
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    down = make_transform(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 0.5])
    frame = FrameRecord(
        0, "w", "c.png", None, down, np.zeros(7), 0.0, depth_image="d.png"
    )
    camera = CameraSpec("w", "wrist", 64, 48, K)
    capture = Capture(
        "c", "test", "fr3_robotiq", "", {"w": camera}, [frame], (0, 1), root=tmp_path
    )
    colour = support_color(capture, np.eye(4), (0.26, 1.0), stride=1)
    assert np.array(colour) * 255 == pytest.approx([200, 160, 90], abs=0.5)


def test_a_scene_keeps_its_support_colour(tmp_path):
    """Saved and loaded with the scene; a scene without one has none."""
    scene = SceneSpec("t", "franka_panda", [], np.eye(4), {}, "c", 0, np.zeros(7))
    scene.save(tmp_path / "a")
    assert SceneSpec.load(tmp_path / "a").support_color is None
    coloured = SceneSpec(
        "t",
        "franka_panda",
        [],
        np.eye(4),
        {},
        "c",
        0,
        np.zeros(7),
        support_color=(0.78, 0.64, 0.36),
    )
    coloured.save(tmp_path / "b")
    assert SceneSpec.load(tmp_path / "b").support_color == (0.78, 0.64, 0.36)
