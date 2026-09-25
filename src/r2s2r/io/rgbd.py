"""Metric depth stored with a capture (RGB-D cameras, MuJoCo)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.structs import DEPTH_PNG_SCALE, Capture, DepthView


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
    capture: Capture, step: int, roles: list[str] | None = None
) -> list[DepthView]:
    """Depth views of the static cameras (or ``roles``) at one step."""
    views = []
    for cam in capture.cameras.values():
        if roles is not None and cam.role not in roles:
            continue
        if roles is None and not cam.is_static:
            continue
        frame = next((f for f in capture.frames_of(cam.serial) if f.step == step), None)
        if frame is None or frame.depth_image is None:
            continue
        views.append(
            DepthView(
                camera=cam.serial,
                step=step,
                depth=read_depth(capture.root / frame.depth_image).astype(float),
                K=cam.K,
                T_base_cam=frame.T_base_cam,
            )
        )
    return views
