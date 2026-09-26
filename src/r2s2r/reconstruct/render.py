"""Render a reconstructed scene into calibrated views (the orientation and existence
checks).

Every object is a mocap body, so objects can be moved between renders; the support is a
large plane. Renders use the views' exact intrinsics (:mod:`r2s2r.mjrender`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from r2s2r.assets import VisualMesh, urdf_visual_meshes
from r2s2r.mjrender import CameraRenderer, add_camera, mujoco, quat_wxyz
from r2s2r.structs import DepthView, SceneSpec

FAR = (100.0, 100.0, -100.0)  # where hidden objects go


class SceneRenderer:
    """The support and every object (textured) as mocap bodies."""

    def __init__(self, scene: SceneSpec, max_size: tuple[int, int]) -> None:
        spec = mujoco.MjSpec()
        add_camera(spec, max_size)
        spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        support = spec.worldbody.add_body(
            name="support",
            pos=scene.T_base_support[:3, 3].tolist(),
            quat=quat_wxyz(scene.T_base_support[:3, :3]),
        )
        support.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[3.0, 3.0, 0.001],
            pos=[0, 0, -0.001],
            rgba=[0.5, 0.5, 0.5, 1.0],
        )
        for i, obj in enumerate(scene.objects):
            body = spec.worldbody.add_body(name=f"obj{i}", mocap=True)
            for j, visual in enumerate(urdf_visual_meshes(obj.asset_path)):
                _add_mesh(spec, body, f"obj{i}/{j}", visual)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.camera = CameraRenderer(self.model, self.data, max_size)
        self.bodies = [self.model.body(f"obj{i}").id for i in range(len(scene.objects))]

    def pose(self, index: int, T_base_obj: NDArray | None) -> None:
        """Place object ``index`` (None: out of view)."""
        mocap = self.model.body_mocapid[self.bodies[index]]
        if T_base_obj is None:
            self.data.mocap_pos[mocap] = FAR
            return
        self.data.mocap_pos[mocap] = T_base_obj[:3, 3]
        self.data.mocap_quat[mocap] = quat_wxyz(T_base_obj[:3, :3])

    def render(self, view: DepthView, index: int) -> dict[str, NDArray]:
        """The view's ``rgb`` / ``depth`` / ``geom``; ``mask`` marks the pixels object
        ``index`` covers."""
        h, w = view.depth.shape
        out = self.camera.render(view.K, w, h, view.T_base_cam)
        body = self.model.geom_bodyid[np.maximum(out["geom"], 0)]
        out["mask"] = (out["geom"] >= 0) & (body == self.bodies[index])
        return out

    def close(self) -> None:
        """Free the GL contexts."""
        self.camera.close()


def _add_mesh(spec: Any, body: Any, name: str, visual: VisualMesh) -> None:
    mesh = visual.mesh
    uv = getattr(mesh.visual, "uv", None)
    kwargs: dict[str, Any] = {}
    if visual.texture is not None and uv is not None and len(uv) == len(mesh.vertices):
        # OBJ images start at the bottom-left; MuJoCo's at the top-left.
        kwargs["usertexcoord"] = np.c_[uv[:, 0], 1.0 - uv[:, 1]].reshape(-1).tolist()
        spec.add_texture(
            name=f"{name}_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(visual.texture),
        )
        material = spec.add_material(name=f"{name}_mat")
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"
    spec.add_mesh(
        name=name,
        uservert=np.asarray(mesh.vertices).reshape(-1).tolist(),
        userface=np.asarray(mesh.faces).reshape(-1).tolist(),
        **kwargs,
    )
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        material=f"{name}_mat" if kwargs else "",
        rgba=[1, 1, 1, 1] if kwargs else [0.75, 0.75, 0.75, 1],
    )
