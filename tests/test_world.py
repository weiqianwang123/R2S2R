"""Tests for sim/world.py, what every simulator does the same: which way gravity
points, the support's colour, a settling's report, a replay's steps."""

import cv2
import numpy as np
import pytest

from r2s2r.sim.world import (
    gravity,
    replay_steps,
    settled,
    start_row,
    support_color,
)
from r2s2r.structs import (
    CameraSpec,
    Capture,
    FrameRecord,
    ObjectSpec,
    RobotTrajectory,
    SceneSpec,
    write_depth,
)
from r2s2r.transforms import make_transform


def _scene(objects, T_base_support=np.eye(4)):
    return SceneSpec(
        name="t",
        embodiment="fr3_robotiq",
        objects=objects,
        T_base_support=T_base_support,
        cameras={},
        reference_camera="c",
        reference_step=0,
        joint_positions=np.zeros(7),
    )


def _object(name, **kwargs):
    return ObjectSpec(name, name, f"{name}.urdf", np.eye(4), **kwargs)


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

    rgb[36:, :45] = (30, 30, 30)  # a dark cloth lying flat on most of the table seen
    cv2.imwrite(str(tmp_path / "c.png"), rgb[..., ::-1])
    cloth = np.array([[-0.2, -0.2, 0], [0.068, -0.2, 0], [0.068, -0.055, 0]])
    cloth = np.vstack([cloth, [[-0.2, -0.055, 0]]])
    line = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])  # no area
    covered = support_color(capture, np.eye(4), (0.26, 1.0), 1, [cloth, line])
    assert np.array(covered) * 255 == pytest.approx([200, 160, 90], abs=0.5)
    seen = support_color(capture, np.eye(4), (0.26, 1.0), stride=1)
    assert np.array(seen) * 255 != pytest.approx([200, 160, 90], abs=0.5)


def test_a_scene_keeps_its_support_colour(tmp_path):
    """Saved and loaded with the scene; a scene without one has none."""
    scene = SceneSpec("t", "fr3_robotiq", [], np.eye(4), {}, "c", 0, np.zeros(7))
    scene.save(tmp_path / "a")
    assert SceneSpec.load(tmp_path / "a").support_color is None
    coloured = SceneSpec(
        "t",
        "fr3_robotiq",
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


def test_a_settled_scene_has_its_objects_where_they_came_to_rest(tmp_path):
    """The objects at their poses and joints after settling; how far each moved,
    turned and dropped, and each joint; the support colour kept when the capture
    shows none."""
    box = _object("box", joints={"hinge": -0.8})
    scene = _scene([box, _object("cup")])
    moved = make_transform(np.eye(3), [0.003, 0.0, -0.004])
    capture = Capture("c", "test", "fr3_robotiq", "", {}, [], (0, 1), root=tmp_path)
    settled_scene, report = settled(
        scene,
        capture,
        2.0,
        ({"box": np.eye(4), "cup": np.eye(4)}, {"box": moved, "cup": np.eye(4)}),
        ({"box": {"hinge": -0.8}}, {"box": {"hinge": -0.75}}),
    )
    box_after, cup_after = settled_scene.objects
    assert np.allclose(box_after.T_base_obj, moved) and box_after.joints == {
        "hinge": -0.75
    }
    assert cup_after.joints is None
    assert report["objects"]["box"] == {
        "moved_m": 0.005,
        "turned_deg": 0.0,
        "dropped_m": 0.004,
        "joints_moved": {"hinge": 0.05},
    }
    assert settled_scene.provenance["settle"] is report
    assert "too few pixels" in report["support_color"]


def test_a_replay_needs_frames_and_the_robot_at_the_static_start(tmp_path):
    """The static period's steps with a frame, every n-th; none is an error, not an
    empty replay. Settling starts from the trajectory's row at the static period's
    start, which must be there."""
    frames = [
        FrameRecord(s, "w", "c.png", None, np.eye(4), np.zeros(7), 0.0)
        for s in (0, 2, 4, 9)
    ]
    camera = CameraSpec("w", "wrist", 64, 48, np.eye(3))
    traj = RobotTrajectory(
        np.arange(2, 10), np.arange(8) * 0.1, np.zeros((8, 7)), np.zeros(8)
    )
    capture = Capture(
        "c", "test", "fr3_robotiq", "", {"w": camera}, frames, (0, 6), tmp_path
    )
    assert replay_steps(capture, {"w": camera}, 1) == [0, 2, 4]
    assert replay_steps(capture, {"w": camera}, 2) == [0, 4]
    with pytest.raises(ValueError, match="no frame of cameras"):
        replay_steps(capture, {}, 1)
    with pytest.raises(ValueError, match="no robot trajectory"):
        start_row(capture)
    capture.trajectory = traj
    with pytest.raises(ValueError, match="no row for step 0"):
        start_row(capture)
    capture.static_steps = (3, 6)
    assert start_row(capture) == 1
