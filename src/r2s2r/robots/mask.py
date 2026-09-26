"""Remove the robot from calibrated depth images.

A robot standing on the table is geometry above the support plane, so reconstruction
treats it as an object: SimFoundry's frame selection counts an arm that leaves the image
as a clipped object, and refinement clusters it. The masker renders the robot's own
model (:mod:`r2s2r.robots.mujoco_models`) at the recorded joint state from the
calibrated camera, with the exact pinhole intrinsics, and invalidates the depth wherever
the robot is the visible surface. Objects in front of the robot keep their depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.mjrender import CameraRenderer, add_camera, mujoco
from r2s2r.robots.mujoco_models import (
    ARM_JOINTS,
    EMBODIMENTS,
    MENAGERIE_DIR,
    GripperPoser,
    robot_spec,
)

NO_ROBOT = 1e6  # robot depth where the robot is not in view


@dataclass
class MaskConfig:
    """How generously to cut."""

    # Grow the rendered silhouette by this share of the image diagonal (~9 px at
    # 1280x720) to absorb calibration and model error.
    dilate_frac: float = 0.006
    # Observed surfaces this much closer than the robot are in front of it: kept.
    margin: float = 0.03


class RobotMasker:
    """Renders one embodiment from arbitrary calibrated cameras."""

    def __init__(
        self,
        embodiment: str,
        menagerie_dir: str | Path = MENAGERIE_DIR,
        config: MaskConfig | None = None,
        max_size: tuple[int, int] = (2560, 1600),
    ) -> None:
        self.config = config or MaskConfig()
        spec = robot_spec(embodiment, menagerie_dir)
        add_camera(spec, max_size)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.gripper = GripperPoser(self.model, embodiment)
        self.arm_qadr = [self.model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.renderer = CameraRenderer(self.model, self.data, max_size)

    def robot_depth(
        self,
        K: NDArray,
        width: int,
        height: int,
        T_base_cam: NDArray,
        joint_positions: NDArray,
        gripper_position: float,
    ) -> NDArray[np.float32]:
        """Planar depth of the robot surface, :data:`NO_ROBOT` elsewhere."""
        self.data.qpos[self.arm_qadr] = joint_positions
        self.gripper.set(self.data, gripper_position)
        out = self.renderer.render(K, width, height, T_base_cam)
        return np.where(out["geom"] >= 0, out["depth"], NO_ROBOT).astype(np.float32)

    def mask_depth(
        self,
        depth: NDArray,
        K: NDArray,
        T_base_cam: NDArray,
        joint_positions: NDArray,
        gripper_position: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """``depth`` with robot pixels set to 0, and the mask of those pixels."""
        h, w = depth.shape
        robot = self.robot_depth(K, w, h, T_base_cam, joint_positions, gripper_position)
        radius = max(1, int(round(self.config.dilate_frac * np.hypot(w, h))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        # Nearest robot depth around each pixel (erosion = local minimum).
        near = cv2.erode(robot, kernel)
        covered = near < NO_ROBOT
        mask = covered & ((depth <= 0) | (depth > near - self.config.margin))
        out = depth.astype(np.float32, copy=True)
        out[mask] = 0.0
        return out, mask

    def close(self) -> None:
        """Free the GL contexts."""
        self.renderer.close()


def robot_masker(embodiment: str) -> RobotMasker:
    """A masker for ``embodiment``, whose model must be known."""
    if embodiment not in EMBODIMENTS:
        raise ValueError(
            f"no robot model for {embodiment!r} (known: {sorted(EMBODIMENTS)}); add it "
            "to r2s2r.robots.mujoco_models, or turn robot masking off"
        )
    return RobotMasker(embodiment)
