"""Shared fixtures: a tiny synthetic DROID episode and its calibration files."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

UUID = "LAB+abc12345+2026-01-01-00h-00m-00s"
SERIALS = {"ext1": "111", "ext2": "222", "wrist": "333"}
SIZE = {"ext1": (32, 24), "ext2": (32, 24), "wrist": (16, 12)}  # per-eye (w, h)
NUM_STEPS = 12
CLOSE_STEP = 8  # gripper starts closing here
LATENCY_MS = 41


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
