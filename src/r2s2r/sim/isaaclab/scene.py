"""Build an Isaac Lab scene from a :class:`~r2s2r.structs.SceneSpec`.

Isaac Lab modules can only be imported once the Omniverse app is running, so import this
module after ``isaaclab.app.AppLauncher`` has started.

The environment origin is the robot base frame, so every SceneSpec pose is used as-is:
the robot sits at the origin, objects and the support surface at their ``T_base_*``, and
each static camera at its calibrated ``T_base_cam`` with the real intrinsics (OpenCV
convention == Isaac Lab's "ros" camera convention).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab_assets.robots.franka import (
    FRANKA_PANDA_HIGH_PD_CFG,
    FRANKA_ROBOTIQ_GRIPPER_CFG,
)

from r2s2r.structs import CameraSpec, ObjectSpec, SceneSpec
from r2s2r.transforms import make_transform, matrix_to_pos_quat

PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
SUPPORT_THICKNESS = 0.02


def _pose(T: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    pos, quat = matrix_to_pos_quat(T)
    return tuple(float(v) for v in pos), tuple(float(v) for v in quat)


# Embodiment -> Isaac Lab robot. Both hold joint position targets stiffly with
# gravity compensated, like Franka's own controller.
ROBOT_CFGS = {
    "droid_franka": FRANKA_ROBOTIQ_GRIPPER_CFG,  # Panda + Robotiq 2F-85
    "franka_panda": FRANKA_PANDA_HIGH_PD_CFG,  # Panda + Franka Hand
}


def robot_cfg(scene: SceneSpec) -> ArticulationCfg:
    """The scene's robot at the origin, in the recorded pose."""
    if scene.embodiment not in ROBOT_CFGS:
        raise NotImplementedError(f"no Isaac Lab config for {scene.embodiment!r}")
    cfg = ROBOT_CFGS[scene.embodiment].replace(prim_path="{ENV_REGEX_NS}/Robot")
    joint_pos = dict(cfg.init_state.joint_pos)
    joint_pos.update(
        {name: float(q) for name, q in zip(PANDA_JOINTS, scene.joint_positions)}
    )
    cfg.init_state = cfg.init_state.replace(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=joint_pos
    )
    return cfg


def support_cfg(scene: SceneSpec, extent: tuple[float, float]) -> AssetBaseCfg:
    """A static slab whose top face is the reconstructed support plane."""
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


def articulated_object_cfg(obj: ObjectSpec, index: int) -> ArticulationCfg:
    """A reconstructed articulated object (drawer, door, lid): its base is fixed, as for
    furniture, and its joints are passive with a little damping."""
    pos, rot = _pose(obj.T_base_obj)
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Object_{index}",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=obj.asset_path,
            fix_base=True,
            merge_fixed_joints=True,
            collider_type="convex_hull",
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot),
        actuators={
            "passive": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=0.0, damping=2.0
            )
        },
    )


def object_cfg(obj: ObjectSpec, index: int) -> RigidObjectCfg | ArticulationCfg:
    """A reconstructed object, spawned from the backend's URDF."""
    if obj.articulated:
        return articulated_object_cfg(obj, index)
    pos, rot = _pose(obj.T_base_obj)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Object_{index}",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=obj.asset_path,
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=None,
            # SimFoundry's collision meshes are already convex (CoACD) parts.
            collider_type="convex_hull",
            # The importer adds an articulation root, which RigidObject rejects.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=(
                sim_utils.MassPropertiesCfg(mass=obj.mass) if obj.mass else None
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
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


def camera_cfg(cam: CameraSpec, T_base_cam: np.ndarray) -> CameraCfg:
    """A camera with the real intrinsics at the calibrated pose.

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
            clipping_range=(0.02, 20.0),
            # Without a focal length Isaac Lab picks a 1 mm aperture and a
            # sub-millimetre focal length, which renders with the wrong field of
            # view; 24 mm gives physically sized camera parameters.
            focal_length=24.0,
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
    )


def build_scene_cfg(
    scene: SceneSpec,
    num_envs: int = 1,
    env_spacing: float = 5.0,
    support_extent: tuple[float, float] = (0.6, 0.6),  # when the scene has none
    with_cameras: bool = True,
    camera_roles: list[str] | None = None,  # None: every static camera
) -> InteractiveSceneCfg:
    """The full interactive scene: robot, support, objects, lights, cameras."""
    cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    # InteractiveScene reads entities from the cfg instance's attributes.
    setattr(cfg, "robot", robot_cfg(scene))
    setattr(cfg, "support", support_cfg(scene, scene.support_extent or support_extent))
    for i, obj in enumerate(scene.objects):
        setattr(cfg, f"object_{i}", object_cfg(obj, i))
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
    if with_cameras:
        for cam in scene.cameras.values():
            if camera_roles is not None and cam.role not in camera_roles:
                continue
            if cam.is_static and cam.T_base_cam is not None:
                setattr(cfg, f"camera_{cam.role}", camera_cfg(cam, cam.T_base_cam))
    return cfg
