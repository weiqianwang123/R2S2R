"""MuJoCo rendering from calibrated pinhole cameras, and MuJoCo geoms as meshes.

The robot masker, the tools' checks and fits, and the fixed method's refinement render
MuJoCo models exactly as a calibrated camera saw the scene: OpenCV extrinsics and the
full pinhole intrinsics, principal point included. :class:`SceneRenderer` renders a
reconstructed scene; every object is a mocap body, so objects can be moved between
renders, and the support is a large plane. :func:`geom_mesh` turns any geom (meshes and
primitives alike) into a coloured triangle mesh, for whatever draws a model elsewhere.
"""

from __future__ import annotations

import ctypes.util
import os
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from r2s2r.assets import VisualMesh, urdf_visual_meshes
from r2s2r.structs import DepthView, SceneSpec
from r2s2r.transforms import pos_quat_to_matrix, rotation_to_quat

# Headless rendering through EGL, where there is an EGL library. Without one (e.g. CI),
# MuJoCo imports with its default backend, and only rendering fails.
if "MUJOCO_GL" not in os.environ and ctypes.util.find_library("EGL"):
    os.environ["MUJOCO_GL"] = "egl"
import mujoco  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order

# OpenCV camera axes -> MuJoCo camera axes (x right, y up, looking along -z).
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])
CAMERA_NAME = "r2s2r_camera"
# Clip planes in metres. MuJoCo scales its own by the model's extent, which puts the
# near plane centimetres out in a large scene and cuts off what a wrist camera sees.
NEAR, FAR = 0.005, 50.0
HIDDEN = (100.0, 100.0, -100.0)  # where SceneRenderer puts objects out of view
MAX_SIZE = (2560, 1600)  # render buffer (width, height) for any camera a run has


def add_camera(spec: Any, max_size: tuple[int, int]) -> None:
    """Give a spec the camera :class:`CameraRenderer` moves, and a render buffer."""
    spec.visual.global_.offwidth, spec.visual.global_.offheight = max_size
    spec.worldbody.add_camera(name=CAMERA_NAME)


class CameraRenderer:
    """Renders a compiled model (with :func:`add_camera`) from calibrated cameras."""

    def __init__(self, model: Any, data: Any, max_size: tuple[int, int]) -> None:
        self.model, self.data = model, data
        self.max_size = max_size
        model.vis.map.znear = NEAR / model.stat.extent
        model.vis.map.zfar = FAR / model.stat.extent
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
        m.cam_quat[self.cam] = rotation_to_quat(T_base_cam[:3, :3] @ CV_TO_MJ)
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


def add_mesh(spec: Any, body: Any, name: str, visual: VisualMesh) -> None:
    """Add ``visual`` to ``body`` of an ``MjSpec`` as a mesh geom, textured if it has a
    texture and UVs."""
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


class SceneRenderer:
    """A scene's support and every object (textured, its joints as the scene has them)
    as mocap bodies."""

    def __init__(self, scene: SceneSpec, max_size: tuple[int, int]) -> None:
        spec = mujoco.MjSpec()
        add_camera(spec, max_size)
        spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        support = spec.worldbody.add_body(
            name="support",
            pos=scene.T_base_support[:3, 3].tolist(),
            quat=rotation_to_quat(scene.T_base_support[:3, :3]).tolist(),
        )
        support.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[3.0, 3.0, 0.001],
            pos=[0, 0, -0.001],
            rgba=[0.5, 0.5, 0.5, 1.0],
        )
        for i, obj in enumerate(scene.objects):
            body = spec.worldbody.add_body(name=f"obj{i}", mocap=True)
            for j, visual in enumerate(urdf_visual_meshes(obj.asset_path, obj.joints)):
                add_mesh(spec, body, f"obj{i}/{j}", visual)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.camera = CameraRenderer(self.model, self.data, max_size)
        self.bodies = [self.model.body(f"obj{i}").id for i in range(len(scene.objects))]
        self._object_of = np.full(self.model.nbody, -1)  # body id -> object index
        self._object_of[self.bodies] = np.arange(len(self.bodies))

    def pose(self, index: int, T_base_obj: NDArray | None) -> None:
        """Place object ``index`` (None: out of view)."""
        mocap = self.model.body_mocapid[self.bodies[index]]
        if T_base_obj is None:
            self.data.mocap_pos[mocap] = HIDDEN
            return
        self.data.mocap_pos[mocap] = T_base_obj[:3, 3]
        self.data.mocap_quat[mocap] = rotation_to_quat(T_base_obj[:3, :3])

    def render(self, view: DepthView) -> dict[str, NDArray]:
        """The view's ``rgb`` / ``depth`` / ``geom``, and ``object``: the index of the
        object each pixel shows (-1: none), so ``out["object"] == i`` is object ``i``'s
        mask."""
        h, w = view.depth.shape
        out = self.camera.render(view.K, w, h, view.T_base_cam)
        body = self.model.geom_bodyid[np.maximum(out["geom"], 0)]
        out["object"] = np.where(out["geom"] >= 0, self._object_of[body], -1)
        return out

    def close(self) -> None:
        """Free the GL contexts."""
        self.camera.close()


def geom_mesh(model: Any, g: int) -> trimesh.Trimesh | None:
    """Geom ``g`` of a compiled model as a triangle mesh in its body's frame, coloured
    by its material (else its rgba); None for planes, height fields and the like."""
    kind, size = model.geom_type[g], model.geom_size[g]
    geom = mujoco.mjtGeom
    if kind == geom.mjGEOM_MESH:
        i = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[i], model.mesh_vertnum[i]
        f0, nf = model.mesh_faceadr[i], model.mesh_facenum[i]
        mesh = trimesh.Trimesh(
            np.asarray(model.mesh_vert[v0 : v0 + nv], float),
            np.asarray(model.mesh_face[f0 : f0 + nf], int),
            process=False,
        )
    elif kind == geom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * size)
    elif kind in (geom.mjGEOM_SPHERE, geom.mjGEOM_ELLIPSOID):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        radii = size if kind == geom.mjGEOM_ELLIPSOID else np.full(3, size[0])
        mesh.apply_scale(radii)
    elif kind == geom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(height=2 * size[1], radius=size[0])
    elif kind == geom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1])
    else:
        return None
    mesh.apply_transform(pos_quat_to_matrix(model.geom_pos[g], model.geom_quat[g]))
    mat = model.geom_matid[g]
    rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
    color = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(  # type: ignore[no-untyped-call]
        mesh, vertex_colors=np.tile(color, (len(mesh.vertices), 1))
    )
    return mesh
