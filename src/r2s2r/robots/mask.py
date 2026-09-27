"""Remove the robot from calibrated depth images.

A robot standing on the table is geometry above the support plane, so reconstruction
treats it as an object: SimFoundry's frame selection counts an arm that leaves the image
as a clipped object, and refinement clusters it. The masker renders the robot's own
model (:class:`~r2s2r.robots.model.RobotModel`) at the recorded joint state from the
calibrated camera, with the exact pinhole intrinsics, and invalidates the depth wherever
the robot is the visible surface. Objects in front of the robot keep their depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.mjrender import MAX_SIZE, CameraRenderer, add_camera
from r2s2r.robots.model import RobotModel
from r2s2r.robots.spec import RobotSpec

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
    """Renders one robot from arbitrary calibrated cameras."""

    def __init__(
        self,
        robot: RobotSpec,
        config: MaskConfig | None = None,
        max_size: tuple[int, int] = MAX_SIZE,
    ) -> None:
        self.config = config or MaskConfig()
        mjspec = robot.mjcf()
        add_camera(mjspec, max_size)
        self.robot = RobotModel(robot, mjspec)
        self.renderer = CameraRenderer(self.robot.model, self.robot.data, max_size)

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
        self.robot.set(joint_positions, gripper_position)
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
