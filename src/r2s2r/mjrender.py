"""MuJoCo rendering from calibrated pinhole cameras.

The robot masker and the orientation check render MuJoCo models exactly as a calibrated
camera saw the scene: OpenCV extrinsics and the full pinhole intrinsics, principal point
included.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order

# OpenCV camera axes -> MuJoCo camera axes (x right, y up, looking along -z).
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])
CAMERA_NAME = "r2s2r_camera"
# Clip planes in metres. MuJoCo scales its own by the model's extent, which puts the
# near plane centimetres out in a large scene and cuts off what a wrist camera sees.
NEAR, FAR = 0.005, 50.0


def quat_wxyz(R: NDArray) -> list[float]:
    """MuJoCo quaternion of a rotation matrix."""
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def set_clip_planes(model: Any, near: float = NEAR, far: float = FAR) -> None:
    """Fix a compiled model's clip planes in metres."""
    model.vis.map.znear = near / model.stat.extent
    model.vis.map.zfar = far / model.stat.extent


def add_camera(spec: Any, max_size: tuple[int, int]) -> None:
    """Give a spec the camera :class:`CameraRenderer` moves, and a render buffer."""
    spec.visual.global_.offwidth, spec.visual.global_.offheight = max_size
    spec.worldbody.add_camera(name=CAMERA_NAME)


class CameraRenderer:
    """Renders a compiled model (with :func:`add_camera`) from calibrated cameras."""

    def __init__(self, model: Any, data: Any, max_size: tuple[int, int]) -> None:
        self.model, self.data = model, data
        self.max_size = max_size
        set_clip_planes(model)
        self.cam = model.camera(CAMERA_NAME).id
        self._renderers: dict[tuple[int, int], Any] = {}

    def place(self, K: NDArray, width: int, height: int, T_base_cam: NDArray) -> None:
        """Point the camera; MuJoCo's principal offsets are from the image centre, x
        mirrored."""
        if width > self.max_size[0] or height > self.max_size[1]:
            raise ValueError(
                f"{width}x{height} exceeds the render buffer {self.max_size}"
            )
        m = self.model
        m.cam_pos[self.cam] = T_base_cam[:3, 3]
        m.cam_quat[self.cam] = quat_wxyz(T_base_cam[:3, :3] @ CV_TO_MJ)
        m.cam_sensorsize[self.cam] = [width, height]
        m.cam_resolution[self.cam] = [width, height]
        m.cam_intrinsic[self.cam] = [
            K[0, 0],
            K[1, 1],
            (width - 1) / 2.0 - K[0, 2],
            K[1, 2] - (height - 1) / 2.0,
        ]

    def render(
        self, K: NDArray, width: int, height: int, T_base_cam: NDArray
    ) -> dict[str, NDArray]:
        """``rgb`` (H, W, 3), planar ``depth`` (H, W) and ``geom`` ids (-1: none)."""
        self.place(K, width, height, T_base_cam)
        mujoco.mj_forward(self.model, self.data)
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height, width)
        r = self._renderers[key]
        r.update_scene(self.data, camera=self.cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self.cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=self.cam)
        seg = r.render()
        r.disable_segmentation_rendering()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        return {"rgb": rgb, "depth": depth, "geom": geom}

    def close(self) -> None:
        """Free the GL contexts."""
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
