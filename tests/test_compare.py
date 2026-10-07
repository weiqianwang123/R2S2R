"""Tests for sim/compare.py: panels and numbers from a replay log (needs GL
rendering)."""

import json

import cv2
import numpy as np
import pytest
from conftest import box_urdf

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from r2s2r.mjrender import SceneRenderer  # noqa: E402
from r2s2r.sim.compare import (  # noqa: E402
    SHEET_ROWS,
    THUMB_HEIGHT,
    compare_replay,
)
from r2s2r.structs import (  # noqa: E402
    CameraSpec,
    Capture,
    DepthView,
    FrameRecord,
    ObjectSpec,
    SceneSpec,
    write_depth,
)
from r2s2r.transforms import (  # noqa: E402
    intrinsics_matrix,
    look_at,
    make_transform,
)

pytestmark = pytest.mark.gl

K = intrinsics_matrix(300.0, 300.0, 159.5, 119.5)
TARGET = (0.5, 0.0, 0.04)  # where the cameras look


def _scene(tmp_path, xy):
    urdf = box_urdf(tmp_path, (0.06, 0.06, 0.08))
    obj = ObjectSpec("box", "box", str(urdf), make_transform(np.eye(3), [*xy, 0.0]))
    return SceneSpec("t", "fr3_robotiq", [obj], np.eye(4), {}, "c", 0, np.zeros(7))


def _render(scene, T):
    renderer = SceneRenderer(scene, (320, 240))
    renderer.pose(0, scene.objects[0].T_base_obj)
    out = renderer.render(DepthView("c", 0, np.zeros((240, 320)), K, T))
    renderer.close()
    return out["rgb"], out["depth"].astype(float)


def test_compare_replay_scores_the_depth_and_outlines_the_objects(tmp_path):
    """The real frame is the true scene; the replay puts the box 3 cm off.

    Where the replay matches the capture, the residual is zero; the moved box shows up.
    """
    T = look_at((0.9, 0.3, 0.45), TARGET)
    truth = _scene(tmp_path, (0.5, 0.0))
    real_rgb, real_depth = _render(truth, T)
    root = tmp_path / "capture"
    (root / "rgb").mkdir(parents=True)
    cv2.imwrite(str(root / "rgb/0.png"), cv2.cvtColor(real_rgb, cv2.COLOR_RGB2BGR))
    write_depth(root / "rgb/0_depth.png", real_depth)
    cam = CameraSpec("c", "ext1", 320, 240, K, T_base_cam=T)
    frame = FrameRecord(
        0, "c", "rgb/0.png", None, T, np.zeros(7), 0.0, "rgb/0_depth.png"
    )
    capture = Capture("t", "test", "fr3_robotiq", "", {"c": cam}, [frame], (0, 1), root)

    replay_dir = tmp_path / "replay"
    (replay_dir / "frames").mkdir(parents=True)
    for name, xy in (("same", (0.5, 0.0)), ("moved", (0.53, 0.0))):
        scene = _scene(tmp_path, xy)
        sim_rgb, sim_depth = _render(scene, T)
        cv2.imwrite(
            str(replay_dir / f"frames/{name}.png"),
            cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR),
        )
        write_depth(replay_dir / f"frames/{name}_depth.png", sim_depth)
        log = {
            "frames": [
                {
                    "step": 0,
                    "camera": "c",
                    "role": "ext1",
                    "sim_rgb": f"frames/{name}.png",
                    "sim_depth": f"frames/{name}_depth.png",
                }
            ],
            "objects": {"box": [[0, *xy, 0.0, 1.0, 0.0, 0.0, 0.0]]},
            "arm_error_rad": [[0, 0.0]],
        }
        summary = compare_replay(
            log, scene, capture, tmp_path / f"compare_{name}", replay_dir
        )
        assert len(summary["frames"]) == 1
        row = summary["frames"][0]
        if name == "same":
            assert row["depth_residual_m"] == 0.0
            assert row["object_depth_residual_m"]["box"] < 0.002
        else:
            assert row["object_depth_residual_m"]["box"] > 0.01
        panel = cv2.imread(str(tmp_path / f"compare_{name}" / row["image"]))
        assert panel.shape == (240, 4 * 320, 3)  # real+outline | sim | blend | residual
        saved = json.loads((tmp_path / f"compare_{name}" / "compare.json").read_text())
        assert saved["motion_m"] == {"box": [[0, 0.0]]}


def test_a_long_replay_s_contact_sheet_shows_evenly_spaced_frames(tmp_path):
    """Every frame gets its panel; the camera's contact sheet shows SHEET_ROWS of
    them."""
    T = look_at((0.9, 0.3, 0.45), TARGET)
    scene = _scene(tmp_path, (0.5, 0.0))
    rgb, _ = _render(scene, T)
    root, replay_dir = tmp_path / "capture", tmp_path / "replay"
    root.mkdir()
    replay_dir.mkdir()
    cv2.imwrite(str(root / "0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(replay_dir / "0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    n = SHEET_ROWS + 6
    frames = [FrameRecord(s, "c", "0.png", None, T, np.zeros(7), 0.0) for s in range(n)]
    cam = CameraSpec("c", "wrist", 320, 240, K, is_static=False)
    capture = Capture("t", "test", "fr3_robotiq", "", {"c": cam}, frames, (0, n), root)
    log = {
        "frames": [
            {"step": s, "camera": "c", "role": "wrist", "sim_rgb": "0.png"}
            for s in range(n)
        ],
        "objects": {"box": [[s, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for s in range(n)]},
        "arm_error_rad": [],
    }
    summary = compare_replay(log, scene, capture, tmp_path / "compare", replay_dir)
    assert len(summary["frames"]) == n
    sheet = cv2.imread(str(tmp_path / "compare" / "sheet_wrist.png"))
    assert sheet.shape[0] == SHEET_ROWS * THUMB_HEIGHT
