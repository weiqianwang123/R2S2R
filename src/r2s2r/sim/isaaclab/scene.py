"""Build an Isaac Lab scene from a :class:`~r2s2r.structs.SceneSpec`, and run it
(:class:`Session`: what the replay, the settling and the pick test share).

Isaac Lab modules can only be imported once the Omniverse app is running, so import this
module after ``isaaclab.app.AppLauncher`` has started.

The environment origin is the robot base frame, so every SceneSpec pose is used as-is:
the robot (its :class:`~r2s2r.robots.spec.RobotSpec`'s Isaac config) sits at the origin,
objects and the support surface at their ``T_base_*``, and each camera at its calibrated
pose with the real intrinsics (OpenCV convention == Isaac Lab's "ros" camera
convention). There is one environment, at the world origin: some robots' USDs pin their
root there.

Objects are spawned from single-file USDs (:func:`object_usd`): the assembled URDF
converted, its rigid body on the default prim (an articulated object's links under it,
the first the articulation root), and a physics material with the object's friction
bound to its colliders. physcoder loads the same files, with the ``metadata.yaml``
:func:`write_metadata` puts beside them. An articulated object's joints are passive,
damped; held where the scene has them when the objects are kinematic.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import isaaclab.sim as sim_utils
import numpy as np
import torch
import yaml
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from r2s2r.assets import bottom_offset
from r2s2r.robots.spec import RobotSpec
from r2s2r.structs import SUPPORT_THICKNESS, CameraSpec, ObjectSpec, SceneSpec
from r2s2r.transforms import (
    make_transform,
    matrix_to_pos_quat,
    pos_quat_to_matrix,
    transform_points,
)

SUPPORT_EXTENT = (0.6, 0.6)  # for scenes whose support outline is unknown
METADATA_FILENAME = "metadata.yaml"
PHYSICS_DT = 0.005  # seconds per physics step
# An articulated object's joints: damped (N m s/rad, N s/m), else free; held stiffly
# (N m/rad, N/m) where the scene has them when the objects are kinematic.
JOINT_DAMPING = 1.0
JOINT_HOLD_STIFFNESS = 1e4


def _pose(T: NDArray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    pos, quat = matrix_to_pos_quat(T)
    return tuple(float(v) for v in pos), tuple(float(v) for v in quat)


# ------------------------------------------------------------------------- robot
def robot_cfg(robot: RobotSpec, joint_positions: NDArray) -> ArticulationCfg:
    """The robot at the origin, its arm at ``joint_positions``."""
    cfg = robot.isaac_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    joint_pos = dict(cfg.init_state.joint_pos)
    joint_pos.update(
        {n: float(q) for n, q in zip(robot.isaac_arm_joints, joint_positions)}
    )
    cfg.init_state = cfg.init_state.replace(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=joint_pos
    )
    return cfg


def gripper_joints(
    articulation: Articulation, robot: RobotSpec
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """The gripper's articulation joints, and their open and closed positions: the
    spec's table, else the driver alone between its soft limits."""
    gripper = robot.gripper
    if gripper.isaac_joints is None:
        ids, _ = articulation.find_joints([gripper.isaac_driver])
        limits = articulation.data.soft_joint_pos_limits[0, ids]
        return ids, limits[:, 0], limits[:, 1]
    names = list(gripper.isaac_joints)
    ids, _ = articulation.find_joints(names, preserve_order=True)
    ends = articulation.data.joint_pos.new_tensor(
        [gripper.isaac_joints[n] for n in names]
    )
    return ids, ends[:, 0], ends[:, 1]


def make_scene(cfg: InteractiveSceneCfg, robot: RobotSpec) -> InteractiveScene:
    """The interactive scene, the robot fixed up by its spec (call before
    ``sim.reset()``)."""
    scene = InteractiveScene(cfg)
    if scene.num_envs != 1 or bool(scene.env_origins.abs().max() > 0):
        raise ValueError("the scene must have one environment, at the origin")
    if robot.isaac_post_spawn is not None:
        robot.isaac_post_spawn(f"{scene.env_prim_paths[0]}/Robot")
    return scene


# ----------------------------------------------------------------------- objects
def object_usd(obj: ObjectSpec, out_dir: str | Path) -> Path:
    """Write ``obj`` as one USD file, ``out_dir/<name>.usd``, and return its path.

    Isaac's URDF importer writes its layered output under ``out_dir/usd/``; this
    flattens it into a single file with the rigid body (and its mass) on the default
    prim, the visuals and colliders underneath (an articulated object keeps its links,
    and its joints, under the default prim), and a ``PhysicsMaterial`` with the
    object's friction bound to every collider. Texture paths stay relative, so the
    directory can move.
    """
    out_dir = Path(out_dir).resolve()
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=obj.asset_path,
            usd_dir=str(out_dir / "usd"),
            usd_file_name=f"{obj.name}.usd",
            # The converter's cache looks at the URDF's text, not at the meshes it
            # names, which a redone stage 4 may have changed.
            force_usd_conversion=True,
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=None,
            # The collision meshes are convex parts already (CoACD, SimFoundry).
            collider_type="convex_hull",
        )
    )
    stage = Usd.Stage.Open(converter.usd_path)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        # The importer instances visuals and colliders; flatten them in place.
        while instances := [p for p in stage.Traverse() if p.IsInstance()]:
            for prim in instances:
                prim.SetInstanceable(False)
    flat = Usd.Stage.Open(stage.Flatten())
    root = flat.GetDefaultPrim()
    links = [p for p in root.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if obj.joints:
        # Articulated: the links stay as the importer made them, the first the root.
        if not links[0].HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError(f"{obj.asset_path}'s first link is no articulation root")
    elif len(links) != 1:
        raise ValueError(f"{obj.asset_path} is not a single rigid link")
    else:
        _move_rigid_body(links[0], root)
    for prim in flat.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and os.path.isabs(value.path):
                attr.Set(Sdf.AssetPath(f"./{os.path.relpath(value.path, out_dir)}"))
    if obj.friction is not None:
        material = UsdShade.Material.Define(
            flat, root.GetPath().AppendChild("PhysicsMaterial")
        )
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(float(obj.friction))
        physics.CreateDynamicFrictionAttr(float(obj.friction))
        physics.CreateRestitutionAttr(0.0)
        for prim in _colliders(root):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.strongerThanDescendants, "physics"
            )
    path = out_dir / f"{obj.name}.usd"
    flat.GetRootLayer().Export(str(path))
    return path


def _move_rigid_body(link: Usd.Prim, root: Usd.Prim) -> None:
    """Move the link's physics schemas and properties to ``root`` (the link sits at
    root's origin), and drop Isaac's robot bookkeeping."""
    xform = UsdGeom.Xformable(link).GetLocalTransformation()
    if not np.allclose(np.array(xform), np.eye(4), atol=1e-6):
        raise ValueError(f"{link.GetPath()} is not at its object's origin")
    for prim in (root, link):
        for schema in prim.GetAppliedSchemas():
            if schema.startswith("Isaac"):
                prim.RemoveAppliedSchema(schema)
        for prop in prim.GetProperties():
            if prop.GetName().startswith("isaac:"):
                prim.RemoveProperty(prop.GetName())
    for schema in link.GetAppliedSchemas():
        if schema.startswith(("Physics", "Physx")):
            root.AddAppliedSchema(schema)
            link.RemoveAppliedSchema(schema)
    for attr in link.GetAttributes():
        name = attr.GetName()
        if name.startswith(("physics:", "physx")) and attr.HasAuthoredValue():
            root.CreateAttribute(name, attr.GetTypeName()).Set(attr.Get())
            link.RemoveProperty(name)


def _colliders(root: Usd.Prim) -> Iterable[Usd.Prim]:
    return (p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI))


def write_metadata(
    usd_path: str | Path, T_base_obj: NDArray, T_base_support: NDArray
) -> Path:
    """``metadata.yaml`` beside an object's USD, in physcoder's convention.

    ``bottom_offset`` (and ``assembled_offset``, the same here) is the bottom centre of
    the colliders as the object rests at ``T_base_obj`` (:func:`~r2s2r.assets.
    bottom_offset`, up being the support's normal); an articulated object's links at
    its USD's joint zero.
    """
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    points = []
    for collider in _colliders(root):
        for prim in Usd.PrimRange(collider):
            if prim.IsA(UsdGeom.Mesh):
                pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
                M = np.array(cache.ComputeRelativeTransform(prim, root)[0]).T
                points.append(transform_points(M, pts))
    if not points:
        raise ValueError(f"{usd_path} has no collision meshes")
    pos, quat = bottom_offset(
        np.concatenate(points), T_base_obj[:3, :3].T @ T_base_support[:3, 2]
    )

    def offset() -> dict[str, list[float]]:
        return {
            "pos": [round(float(v), 6) + 0.0 for v in pos],
            "quat": [round(float(v), 6) + 0.0 for v in quat],
        }

    path = Path(usd_path).with_name(METADATA_FILENAME)
    path.write_text(
        yaml.safe_dump(
            {"assembled_offset": offset(), "bottom_offset": offset()},
            default_flow_style=None,
        ),
        encoding="utf-8",
    )
    return path


def with_object_usds(scene: SceneSpec, out_dir: str | Path) -> SceneSpec:
    """``scene`` with every object that has no USD yet converted into
    ``out_dir/objects/<name>/``."""
    objects = [
        (
            obj
            if obj.usd is not None
            else replace(
                obj, usd=str(object_usd(obj, Path(out_dir) / "objects" / obj.name))
            )
        )
        for obj in scene.objects
    ]
    return replace(scene, objects=objects)


def object_cfg(
    obj: ObjectSpec, index: int, kinematic: bool
) -> RigidObjectCfg | ArticulationCfg:
    """An object from its USD (:func:`object_usd`); ``kinematic`` holds it where it
    is placed (for comparing geometry, not physics), an articulated one with its
    joints as the scene has them."""
    if obj.usd is None:
        raise ValueError(f"{obj.name} has no USD yet (with_object_usds)")
    pos, rot = _pose(obj.T_base_obj)
    prim_path = f"{{ENV_REGEX_NS}}/Object_{index}"
    if obj.joints:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=obj.usd,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=kinematic
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos, rot=rot, joint_pos=dict(obj.joints)
            ),
            actuators={
                "joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=JOINT_HOLD_STIFFNESS if kinematic else 0.0,
                    damping=JOINT_DAMPING,
                )
            },
        )
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=obj.usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


# ----------------------------------------------------------------- support, cameras
def support_cfg(scene: SceneSpec) -> AssetBaseCfg:
    """A static slab whose top face is the reconstructed support plane."""
    extent = scene.support_extent or SUPPORT_EXTENT
    below = make_transform(np.eye(3), [0.0, 0.0, -SUPPORT_THICKNESS / 2])
    pos, rot = _pose(scene.T_base_support @ below)
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=(extent[0], extent[1], SUPPORT_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.45, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def centered_render_size(cam: CameraSpec) -> tuple[int, int, int, int]:
    """Render size with the principal point at its centre, and the crop back.

    Omniverse cameras ignore aperture offsets (the principal point is always the image
    centre), so render a larger image centred on (cx, cy) and crop the real image's
    window out of it. Returns (width, height, crop_x, crop_y).
    """
    cx, cy = float(cam.K[0, 2]), float(cam.K[1, 2])
    half_w = int(np.ceil(max(cx, cam.width - cx)))
    half_h = int(np.ceil(max(cy, cam.height - cy)))
    return 2 * half_w, 2 * half_h, int(round(half_w - cx)), int(round(half_h - cy))


def crop_window(cam: CameraSpec) -> tuple[slice, slice]:
    """Where the real image lies in the camera's render (rows, columns)."""
    _, _, x0, y0 = centered_render_size(cam)
    return slice(y0, y0 + cam.height), slice(x0, x0 + cam.width)


def camera_key(cam: CameraSpec) -> str:
    """The scene entity of a camera."""
    return f"camera_{cam.role}"


def object_key(index: int) -> str:
    """The scene entity of the scene's object ``index``."""
    return f"object_{index}"


def camera_cfg(cam: CameraSpec, T_base_cam: NDArray, near: float) -> CameraCfg:
    """A camera with the real intrinsics at ``T_base_cam``, seeing nothing nearer than
    ``near`` (m).

    The render is larger than the real image (see :func:`centered_render_size`); crop it
    with the offsets that function returns.
    """
    pos, rot = _pose(T_base_cam)
    width, height, crop_x, crop_y = centered_render_size(cam)
    K = cam.K.copy()
    K[0, 2] += crop_x
    K[1, 2] += crop_y
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Camera_{cam.role}",
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=K.reshape(-1).tolist(),
            width=width,
            height=height,
            clipping_range=(near, 20.0),
            # Without a focal length Isaac Lab picks a 1 mm aperture and a
            # sub-millimetre focal length, which renders with the wrong field of
            # view; 24 mm gives physically sized camera parameters.
            focal_length=24.0,
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
    )


def build_scene_cfg(
    scene: SceneSpec,
    robot: RobotSpec,
    kinematic_objects: bool,
    cameras: Iterable[tuple[CameraSpec, NDArray]] = (),
) -> InteractiveSceneCfg:
    """The full interactive scene: ``robot`` (the scene's embodiment; pass the same
    spec to :func:`make_scene`), support, objects (with their USDs), lights, and
    ``cameras`` (each at its pose)."""
    if robot.name != scene.embodiment:
        raise ValueError(
            f"scene {scene.name} is for {scene.embodiment}, not {robot.name}"
        )
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.0)
    # InteractiveScene reads entities from the cfg instance's attributes.
    setattr(cfg, "robot", robot_cfg(robot, scene.joint_positions))
    setattr(cfg, "support", support_cfg(scene))
    for i, obj in enumerate(scene.objects):
        setattr(cfg, object_key(i), object_cfg(obj, i, kinematic_objects))
    # A dim dome keeps the background dark so renders overlay cleanly on real frames;
    # a distant light gives the geometry readable shading.
    setattr(
        cfg,
        "dome_light",
        AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(0.2, 0.2, 0.2)),
        ),
    )
    setattr(
        cfg,
        "sun_light",
        AssetBaseCfg(
            prim_path="/World/SunLight",
            spawn=sim_utils.DistantLightCfg(intensity=2000.0, angle=5.0),
            init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.0, 0.3827, 0.0)),
        ),
    )
    for cam, T_base_cam in cameras:
        setattr(
            cfg,
            camera_key(cam),
            camera_cfg(cam, T_base_cam, robot.isaac_camera_near),
        )
    return cfg


# ------------------------------------------------------------------------ session
class Session:
    """A scene running in Isaac Lab (:func:`build_scene_cfg`): the simulation, the
    interactive scene, the robot's arm and gripper joints, the objects by name.

    ``robot_spec`` is the scene's embodiment; ``kinematic_objects`` holds the objects
    where they are placed; ``cameras`` as for :func:`build_scene_cfg`.
    """

    def __init__(
        self,
        spec: SceneSpec,
        robot_spec: RobotSpec,
        kinematic_objects: bool,
        device: str,
        cameras: Iterable[tuple[CameraSpec, NDArray]] = (),
    ) -> None:
        self.robot_spec = robot_spec
        self.sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT, device=device))
        self.scene = make_scene(
            build_scene_cfg(spec, robot_spec, kinematic_objects, cameras), robot_spec
        )
        self.sim.reset()
        self.dt = self.sim.get_physics_dt()
        self.articulation: Articulation = self.scene["robot"]
        self.arm_ids, _ = self.articulation.find_joints(
            list(robot_spec.isaac_arm_joints), preserve_order=True
        )
        self.grip_ids, self.grip_open, self.grip_closed = gripper_joints(
            self.articulation, robot_spec
        )
        self.names = {object_key(i): obj.name for i, obj in enumerate(spec.objects)}
        for obj in self._objects().values():
            if isinstance(obj, Articulation):  # the joints as the scene has them
                pos = obj.data.default_joint_pos
                obj.write_joint_state_to_sim(pos, torch.zeros_like(pos))
                obj.set_joint_position_target(pos)  # held there when kinematic

    def gripper_targets(self, level: float) -> torch.Tensor:
        """The gripper joints' positions at opening ``level`` (0 open, 1 closed)."""
        return self.grip_open + level * (self.grip_closed - self.grip_open)

    def step_physics(self, n: int, render: bool = False) -> None:
        """``n`` physics steps; ``render`` renders on the last."""
        for i in range(n):
            self.scene.write_data_to_sim()
            self.sim.step(render=render and i == n - 1)
            self.scene.update(self.dt)

    def _objects(self) -> dict[str, RigidObject | Articulation]:
        """The scene's objects (rigid or articulated) by name."""
        entities = {**self.scene.rigid_objects, **self.scene.articulations}
        return {name: entities[key] for key, name in self.names.items()}

    def object_poses(self) -> dict[str, NDArray[np.float64]]:
        """Every object's pose in the robot base frame (the environment is at the
        origin)."""
        out = {}
        for name, obj in self._objects().items():
            pos = obj.data.root_pos_w[0].cpu().numpy()
            quat = obj.data.root_quat_w[0].cpu().numpy()
            out[name] = pos_quat_to_matrix(pos, quat)
        return out

    def object_joints(self) -> dict[str, dict[str, float]]:
        """Every articulated object's joint positions, by name, within their limits
        (PhysX lets a joint pass a limit a little, which Isaac Lab then refuses as a
        starting position)."""
        out = {}
        for name, obj in self._objects().items():
            if isinstance(obj, Articulation):
                limits = obj.data.joint_pos_limits[0]
                q = obj.data.joint_pos[0].clamp(limits[:, 0], limits[:, 1])
                out[name] = {
                    joint: float(v) for joint, v in zip(obj.joint_names, q.tolist())
                }
        return out
