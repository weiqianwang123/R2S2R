"""Load a raw DROID episode into a :class:`~r2s2r.structs.Capture`.

The episode directory is laid out as in ``gs://gresearch/robotics/droid_raw/1.0.1``::

    metadata_<uuid>.json
    trajectory.h5
    recordings/MP4/<serial>-stereo.mp4      # left | right, side by side
    recordings/MP4/<serial>_timestamps.json # capture time (ms) of every video frame

DROID keeps camera intrinsics only inside its SVO files, and its original
extrinsics are noisy, so calibration comes from the improved release at
https://huggingface.co/KarlP/droid. ``calib_dir`` must hold its ``intrinsics.json``;
``cam2base_extrinsic_superset.json`` / ``cam2base_extrinsics.json`` (left camera
to base) and ``droid_language_annotations.json`` are used when present, and the
per-step extrinsics in ``trajectory.h5`` are the fallback.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray

from r2s2r.structs import CameraSpec, Capture, FrameRecord
from r2s2r.transforms import intrinsics_matrix, pose6d_to_matrix

logger = logging.getLogger(__name__)

ROLES = ("ext1", "ext2", "wrist")
IMPROVED_EXTRINSICS_FILES = (
    "cam2base_extrinsic_superset.json",
    "cam2base_extrinsics.json",
)


@dataclass
class DroidCalibration:
    """Improved calibration entries for one episode."""

    intrinsics: dict[str, dict[str, Any]]  # serial -> {cameraMatrix, width, height}
    cam2base: dict[str, list[float]]  # serial -> 6D left-camera pose
    instruction: str | None

    @classmethod
    def from_dir(cls, calib_dir: str | Path, uuid: str) -> DroidCalibration:
        """Read the entries for ``uuid`` from the KarlP/droid JSON files."""
        calib_dir = Path(calib_dir)
        intr_path = calib_dir / "intrinsics.json"
        if not intr_path.exists():
            raise FileNotFoundError(
                f"{intr_path} is required: DROID raw data keeps intrinsics only in SVO "
                "files. Download it from https://huggingface.co/KarlP/droid."
            )
        intrinsics = _load_entry(intr_path, uuid)
        if intrinsics is None:
            raise KeyError(f"no intrinsics for episode {uuid} in {intr_path}")

        cam2base: dict[str, list[float]] = {}
        for fname in IMPROVED_EXTRINSICS_FILES:
            entry = _load_entry(calib_dir / fname, uuid)
            if entry is None:
                continue
            for key, value in entry.items():
                if isinstance(value, list) and len(value) == 6:
                    cam2base.setdefault(key, value)

        lang = _load_entry(calib_dir / "droid_language_annotations.json", uuid)
        instruction = lang.get("language_instruction1") if lang else None
        return cls(intrinsics=intrinsics, cam2base=cam2base, instruction=instruction)


def _load_entry(path: Path, uuid: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp).get(uuid)


def align_steps_to_video(
    step_times_ms: NDArray[np.int64], video_times_ms: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Map every trajectory step to the index of its video frame.

    Video timestamps trail the step capture estimates by a roughly constant latency, so
    the median offset is removed before nearest-neighbour matching.
    """
    n = min(len(step_times_ms), len(video_times_ms))
    offset = float(np.median(video_times_ms[:n] - step_times_ms[:n]))
    shifted = step_times_ms.astype(np.float64) + offset
    idx = np.searchsorted(video_times_ms, shifted)
    idx = np.clip(idx, 1, len(video_times_ms) - 1)
    left = video_times_ms[idx - 1]
    right = video_times_ms[idx]
    idx -= (shifted - left < right - shifted).astype(np.int64)
    return idx.astype(np.int64)


def read_video_frames(
    path: Path, indices: Iterable[int]
) -> dict[int, NDArray[np.uint8]]:
    """Decode the requested frames (BGR) of a video."""
    wanted = set(int(i) for i in indices)
    frames: dict[int, NDArray[np.uint8]] = {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    i = 0
    while wanted - frames.keys():
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = np.asarray(frame, dtype=np.uint8)
        i += 1
    cap.release()
    missing = wanted - frames.keys()
    if missing:
        raise IndexError(f"{path} has {i} frames, missing {sorted(missing)}")
    return frames


def static_step_range(
    gripper_position: NDArray[np.float64], threshold: float
) -> tuple[int, int]:
    """Steps before the gripper first closes, when objects cannot have moved.

    Pushing without grasping is not detected; this is a heuristic.
    """
    closed = np.flatnonzero(np.asarray(gripper_position) > threshold)
    end = int(closed[0]) if len(closed) else len(gripper_position)
    return 0, max(end, 1)


def _camera_serials(metadata: dict[str, Any]) -> dict[str, str]:
    return {role: str(metadata[f"{role}_cam_serial"]) for role in ROLES}


def load_droid_episode(
    episode_dir: str | Path,
    out_dir: str | Path,
    calib_dir: str | Path,
    roles: Iterable[str] = ROLES,
    stride: int = 5,
    gripper_threshold: float = 0.05,
) -> Capture:
    """Extract calibrated stereo frames of the static part of an episode.

    Frames are written to ``out_dir/frames/<serial>/<step>_{left,right}.png`` and
    the capture to ``out_dir/capture.json``.
    """
    episode_dir, out_dir = Path(episode_dir), Path(out_dir)
    metadata_files = sorted(episode_dir.glob("metadata_*.json"))
    if len(metadata_files) != 1:
        raise FileNotFoundError(f"expected one metadata_*.json in {episode_dir}")
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    uuid = metadata["uuid"]
    serials = _camera_serials(metadata)
    calib = DroidCalibration.from_dir(calib_dir, uuid)

    with h5py.File(episode_dir / "trajectory.h5", "r") as traj:
        joints = traj["observation/robot_state/joint_positions"][:]
        gripper = traj["observation/robot_state/gripper_position"][:]
        extrinsics = traj["observation/camera_extrinsics"]
        step_times = {
            s: traj[f"observation/timestamp/cameras/{s}_estimated_capture"][:]
            for s in serials.values()
        }
        left_poses = {s: extrinsics[f"{s}_left"][:] for s in serials.values()}
        right_poses = {s: extrinsics[f"{s}_right"][:] for s in serials.values()}

    static = static_step_range(gripper, gripper_threshold)
    steps = list(range(static[0], static[1], stride))
    cameras: dict[str, CameraSpec] = {}
    frames: list[FrameRecord] = []
    calib_source: dict[str, str] = {}

    for role in roles:
        serial = serials[role]
        is_static = role != "wrist"
        video = episode_dir / "recordings" / "MP4" / f"{serial}-stereo.mp4"
        cap = cv2.VideoCapture(str(video))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        intr = calib.intrinsics[serial]
        fx, cx, fy, cy = intr["cameraMatrix"]
        K = intrinsics_matrix(fx, fy, cx, cy)
        if (intr["width"], intr["height"]) != (video_w, video_h):
            logger.warning(
                "camera %s: intrinsics are for %sx%s but video is %sx%s; rescaling K",
                serial,
                intr["width"],
                intr["height"],
                video_w,
                video_h,
            )
            K[0] *= video_w / intr["width"]
            K[1] *= video_h / intr["height"]

        if is_static and serial in calib.cam2base:
            T_static = pose6d_to_matrix(calib.cam2base[serial])
            calib_source[serial] = "improved"
        else:
            T_static = pose6d_to_matrix(left_poses[serial][static[0]])
            calib_source[serial] = "trajectory.h5"
        baseline = float(
            np.linalg.norm(left_poses[serial][0][:3] - right_poses[serial][0][:3])
        )
        cameras[serial] = CameraSpec(
            serial=serial,
            role=role,
            width=video_w,
            height=video_h,
            K=K,
            stereo_baseline=baseline,
            is_static=is_static,
            T_base_cam=T_static if is_static else None,
        )

        timestamps_path = video.with_name(f"{serial}_timestamps.json")
        if timestamps_path.exists():
            video_times = np.asarray(json.loads(timestamps_path.read_text()))
            step_to_frame = align_steps_to_video(step_times[serial], video_times)
        else:
            step_to_frame = np.arange(len(gripper))
        decoded = read_video_frames(video, {int(step_to_frame[s]) for s in steps})

        frame_dir = out_dir / "frames" / serial
        frame_dir.mkdir(parents=True, exist_ok=True)
        for step in steps:
            stereo = decoded[int(step_to_frame[step])]
            left_rel = f"frames/{serial}/{step:04d}_left.png"
            right_rel = f"frames/{serial}/{step:04d}_right.png"
            cv2.imwrite(str(out_dir / left_rel), stereo[:, :video_w])
            cv2.imwrite(str(out_dir / right_rel), stereo[:, video_w:])
            frames.append(
                FrameRecord(
                    step=step,
                    camera=serial,
                    left_image=left_rel,
                    right_image=right_rel,
                    T_base_cam=(
                        T_static
                        if is_static
                        else pose6d_to_matrix(left_poses[serial][step])
                    ),
                    joint_positions=joints[step].astype(np.float64),
                    gripper_position=float(gripper[step]),
                )
            )

    capture = Capture(
        name=uuid.replace("+", "_"),
        source="droid",
        embodiment="droid_franka",
        instruction=calib.instruction or metadata.get("current_task", ""),
        cameras=cameras,
        frames=frames,
        static_steps=static,
        root=out_dir,
        metadata={
            "droid_uuid": uuid,
            "episode_dir": str(episode_dir.resolve()),
            "calibration_source": calib_source,
            "num_steps": int(len(gripper)),
        },
    )
    capture.save()
    return capture
