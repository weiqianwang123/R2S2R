"""Shared fixtures and helpers: a tiny synthetic DROID episode and its calibration
files, a small RGB-D capture, a box URDF, robots skipped without their MJCF; the ``gl``
marker for tests that render with MuJoCo."""

from __future__ import annotations

import json
import subprocess
import sys
from functools import cache
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import trimesh

from r2s2r.robots import franka_panda, get_robot, ur5e_2f140
from r2s2r.robots.spec import MENAGERIE_DIR, RobotSpec
from r2s2r.structs import (
    CameraSpec,
    Capture,
    FrameRecord,
    RobotTrajectory,
    write_depth,
)
from r2s2r.transforms import intrinsics_matrix

UUID = "LAB+abc12345+2026-01-01-00h-00m-00s"
SERIALS = {"ext1": "111", "ext2": "222", "wrist": "333"}
SIZE = {"ext1": (32, 24), "ext2": (32, 24), "wrist": (16, 12)}  # per-eye (w, h)
NUM_STEPS = 12
CLOSE_STEP = 8  # gripper starts closing here
LATENCY_MS = 41
RGBD_K = intrinsics_matrix(300.0, 300.0, 159.5, 119.5)
RGBD_SIZE = (320, 240)
HOME_Q = np.array(franka_panda.HOME_Q)
ROBOT_ASSETS = {
    "franka_panda": MENAGERIE_DIR / "franka_emika_panda",
    "droid_franka": MENAGERIE_DIR / "robotiq_2f85",
    "ur5e_2f140": ur5e_2f140.MJCF_PATH,
}


@cache
def offscreen_gl() -> bool:
    """Whether MuJoCo can render here.

    Probed in a child process: without a GL driver (e.g. CI), creating a renderer
    aborts the process instead of raising.
    """
    probe = (
        "import os; os.environ.setdefault('MUJOCO_GL', 'egl'); import mujoco; "
        "mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 8, 8).close()"
    )
    run = subprocess.run(
        [sys.executable, "-c", probe], check=False, capture_output=True
    )
    return run.returncode == 0


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``gl`` marker."""
    config.addinivalue_line(
        "markers", "gl: renders with MuJoCo; skipped without offscreen GL"
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``gl`` tests where MuJoCo cannot render."""
    gl = [item for item in items if item.get_closest_marker("gl")]
    if gl and not offscreen_gl():
        for item in gl:
            item.add_marker(pytest.mark.skip(reason="no offscreen GL rendering"))


def robot_or_skip(name: str) -> RobotSpec:
    """Robot ``name``'s spec; the test is skipped where its MJCF is not."""
    if not Path(ROBOT_ASSETS[name]).exists():
        pytest.skip(f"{name}'s MJCF is not here ({ROBOT_ASSETS[name]})")
    return get_robot(name)


def box_urdf(root: Path, extents: tuple[float, ...], name: str = "box") -> Path:
    """``<name>.urdf`` in ``root``: a box of ``extents`` (m) standing on its origin."""
    box = trimesh.creation.box(extents=extents)
    box.apply_translation([0, 0, extents[2] / 2])
    box.export(root / f"{name}.obj")
    (root / f"{name}.urdf").write_text(
        f'<robot name="{name}"><link name="base"><visual><geometry>'
        f'<mesh filename="{name}.obj"/></geometry></visual></link></robot>'
    )
    return root / f"{name}.urdf"


def _pose6d(x: float) -> list[float]:
    return [x, 0.1, 0.5, np.pi, 0.0, 0.0]


@pytest.fixture(name="droid_episode")
def fixture_droid_episode(tmp_path: Path) -> tuple[Path, Path]:
    """Write an episode dir and a KarlP/droid-style calibration dir."""
    episode = tmp_path / "episode"
    mp4_dir = episode / "recordings" / "MP4"
    mp4_dir.mkdir(parents=True)
    metadata = {"uuid": UUID, "current_task": "task from metadata"}
    for role, serial in SERIALS.items():
        metadata[f"{role}_cam_serial"] = serial
    (episode / f"metadata_{UUID}.json").write_text(json.dumps(metadata))

    step_ms = np.arange(NUM_STEPS, dtype=np.int64) * 66 + 1_000_000
    gripper = np.zeros(NUM_STEPS)
    gripper[CLOSE_STEP:] = 0.5
    with h5py.File(episode / "trajectory.h5", "w") as f:
        f["observation/robot_state/joint_positions"] = np.tile(
            np.arange(7.0), (NUM_STEPS, 1)
        )
        f["observation/robot_state/gripper_position"] = gripper
        for i, serial in enumerate(SERIALS.values()):
            left = np.tile(_pose6d(0.1 * i), (NUM_STEPS, 1))
            right = left.copy()
            right[:, 0] += 0.12
            f[f"observation/camera_extrinsics/{serial}_left"] = left
            f[f"observation/camera_extrinsics/{serial}_right"] = right
            f[f"observation/timestamp/cameras/{serial}_estimated_capture"] = step_ms

    for role, serial in SERIALS.items():
        w, h = SIZE[role]
        writer = cv2.VideoWriter(
            str(mp4_dir / f"{serial}-stereo.mp4"),
            cv2.VideoWriter.fourcc(*"mp4v"),
            15,
            (2 * w, h),
        )
        for _ in range(NUM_STEPS):
            frame = np.zeros((h, 2 * w, 3), np.uint8)
            frame[:, :w] = (0, 0, 255)  # left eye red (BGR)
            frame[:, w:] = (255, 0, 0)  # right eye blue
            writer.write(frame)
        writer.release()
        (mp4_dir / f"{serial}_timestamps.json").write_text(
            json.dumps((step_ms + LATENCY_MS).tolist())
        )

    calib = tmp_path / "calib"
    calib.mkdir()
    intrinsics = {
        UUID: {
            serial: {
                "cameraMatrix": [20.0, SIZE[role][0] / 2, 20.0, SIZE[role][1] / 2],
                "width": SIZE[role][0],
                "height": SIZE[role][1],
            }
            for role, serial in SERIALS.items()
        }
    }
    (calib / "intrinsics.json").write_text(json.dumps(intrinsics))
    superset = {UUID: {SERIALS["ext1"]: _pose6d(1.0), "relative_path": "x"}}
    (calib / "cam2base_extrinsic_superset.json").write_text(json.dumps(superset))
    lang = {UUID: {"language_instruction1": "put the block in the bowl"}}
    (calib / "droid_language_annotations.json").write_text(json.dumps(lang))
    return episode, calib


def rgbd_capture(
    root: Path,
    cameras: dict[str, tuple[str, np.ndarray]],
    steps: int = 5,
    images: dict[str, np.ndarray] | None = None,
    depths: dict[str, np.ndarray] | None = None,
) -> Capture:
    """A Panda's RGB-D capture at ``root``: ``cameras`` ``{serial: (role, T_base_cam)}``
    (a ``wrist`` moves with the hand), ``steps`` frames each, all of the static period;
    grey images at 0.8 m unless ``images`` / ``depths`` ``{serial: array}`` are given.
    Its metadata holds a secret besides how its depth was made."""
    specs, frames = {}, []
    w, h = RGBD_SIZE
    for serial, (role, T) in cameras.items():
        static = role != "wrist"
        specs[serial] = CameraSpec(
            serial,
            role,
            w,
            h,
            RGBD_K,
            is_static=static,
            T_base_cam=T if static else None,
        )
        (root / serial).mkdir(parents=True)
        for step in range(steps):
            rgb = np.full((h, w, 3), 40 * step, np.uint8)
            depth = np.full((h, w), 0.8)
            if images is not None and depths is not None:
                rgb, depth = images[serial], depths[serial]
            cv2.imwrite(str(root / f"{serial}/{step}.png"), rgb[..., ::-1])
            write_depth(root / f"{serial}/{step}_d.png", depth)
            frames.append(
                FrameRecord(
                    step,
                    serial,
                    f"{serial}/{step}.png",
                    None,
                    T,
                    HOME_Q,
                    0.0,
                    f"{serial}/{step}_d.png",
                )
            )
    trajectory = RobotTrajectory(
        np.arange(steps),
        np.arange(steps, dtype=np.float64) * 0.1,
        np.tile(HOME_Q, (steps, 1)),
        np.zeros(steps),
    )
    capture = Capture(
        "t",
        "test",
        "franka_panda",
        "pick",
        specs,
        frames,
        (0, steps),
        root,
        metadata={"truth": "hidden", "depth": "measured"},
        trajectory=trajectory,
    )
    capture.save()
    return capture
