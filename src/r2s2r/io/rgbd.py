"""Metric depth stored with a capture (RGB-D cameras, MuJoCo)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

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
