"""Metric depth stored with a capture (RGB-D cameras, MuJoCo)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.structs import DEPTH_PNG_SCALE, Capture, DepthView, FrameRecord

if TYPE_CHECKING:  # MuJoCo is imported only when a masker is made
    from r2s2r.robots.mask import RobotMasker


def read_depth(path: str | Path) -> NDArray[np.float32]:
    """Metric depth from a uint16 PNG (see :data:`~r2s2r.structs.DEPTH_PNG_SCALE`)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"cannot read {path}")
    return raw.astype(np.float32) * DEPTH_PNG_SCALE


def write_depth(path: str | Path, depth: NDArray) -> None:
    """Metric depth as a uint16 PNG, the inverse of :func:`read_depth`."""
    units = np.clip(np.round(np.asarray(depth) / DEPTH_PNG_SCALE), 0, 65535)
    cv2.imwrite(str(path), units.astype(np.uint16))


def save_depth_steps(path: str | Path, steps: dict[int, list[DepthView]]) -> None:
    """Depth views grouped by step (colour dropped), compressed into one ``.npz``."""
    views = [v for t in sorted(steps) for v in steps[t]]
    arrays: dict[str, Any] = {
        "steps": np.array(sorted(steps), dtype=np.int64),
        "step": np.array([v.step for v in views], dtype=np.int64),
        "camera": np.array([v.camera for v in views]),
        "K": np.array([v.K for v in views]).reshape((-1, 3, 3)),
        "T": np.array([v.T_base_cam for v in views]).reshape((-1, 4, 4)),
    }
    for i, v in enumerate(views):
        units = np.round(v.depth / DEPTH_PNG_SCALE).clip(0, 65535)
        arrays[f"depth_{i}"] = units.astype(np.uint16)
    np.savez_compressed(path, **arrays)


def load_depth_steps(path: str | Path) -> dict[int, list[DepthView]]:
    """Inverse of :func:`save_depth_steps`."""
    with np.load(path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    out: dict[int, list[DepthView]] = {int(t): [] for t in arrays["steps"].tolist()}
    for i, (step, camera) in enumerate(zip(arrays["step"].tolist(), arrays["camera"])):
        depth = arrays[f"depth_{i}"].astype(np.float64) * DEPTH_PNG_SCALE
        K, T = arrays["K"][i], arrays["T"][i]
        out[int(step)].append(DepthView(str(camera), int(step), depth, K, T))
    return out


def capture_depth_views(
    capture: Capture,
    cameras: Iterable[str] | None = None,
    max_views: int | None = 12,
    steps: set[int] | None = None,
    masker: RobotMasker | None = None,
) -> list[DepthView]:
    """Measured depth from any mix of cameras over the static period.

    Views are spread over cameras and time like SimFoundry's candidates
    (:meth:`~r2s2r.structs.Capture.select_frames`). A
    :class:`~r2s2r.robots.mask.RobotMasker` removes the robot from each.
    """

    def keep(frame: FrameRecord) -> bool:
        return frame.depth_image is not None and (steps is None or frame.step in steps)

    return [
        _depth_view(capture, frame, masker)
        for frame in capture.select_frames(cameras, max_views, keep)
    ]


def capture_depth_steps(
    capture: Capture,
    cameras: Iterable[str] | None = None,
    max_steps: int | None = 48,
    masker: RobotMasker | None = None,
) -> dict[int, list[DepthView]]:
    """Depth views over the whole video, grouped by step: every camera's frame at up to
    ``max_steps`` evenly spaced steps (interaction included), the robot removed by
    ``masker``."""
    serials = set(capture.resolve_cameras(cameras))
    frames = [f for f in capture.frames if f.camera in serials and f.depth_image]
    all_steps = sorted({f.step for f in frames})
    if max_steps is not None and len(all_steps) > max_steps:
        idx = np.linspace(0, len(all_steps) - 1, max_steps).round().astype(int)
        all_steps = [all_steps[i] for i in sorted(set(idx))]
    keep = set(all_steps)
    out: dict[int, list[DepthView]] = {t: [] for t in all_steps}
    for frame in frames:
        if frame.step in keep:
            out[frame.step].append(_depth_view(capture, frame, masker))
    return out


def _depth_view(
    capture: Capture, frame: FrameRecord, masker: RobotMasker | None
) -> DepthView:
    assert frame.depth_image is not None
    cam = capture.cameras[frame.camera]
    depth = read_depth(capture.root / frame.depth_image)
    if masker is not None:
        depth, _ = masker.mask_depth(
            depth,
            cam.K,
            frame.T_base_cam,
            frame.joint_positions,
            frame.gripper_position,
        )
    bgr = cv2.imread(str(capture.root / frame.left_image))
    return DepthView(
        frame.camera,
        frame.step,
        depth.astype(float),
        cam.K,
        frame.T_base_cam,
        (
            None
            if bgr is None
            else np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.uint8)
        ),
    )
