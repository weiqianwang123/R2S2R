"""The MuJoCo world: a Franka Panda (mujoco_menagerie) on a table with scanned household
objects (Google Scanned Objects, MuJoCo port), watched by calibrated RGB-D cameras, and
its robot behind :class:`~r2s2r.policy.robot.RobotInterface`.

The world frame is the robot base frame. Assets are fetched by
``scripts/setup/fetch_mujoco_assets.sh`` into ``~/.cache/r2s2r``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh
from numpy.typing import NDArray

from r2s2r.mjrender import CV_TO_MJ, mujoco, quat_wxyz, set_clip_planes
from r2s2r.policy.robot import RobotInterface
from r2s2r.robots.franka import FRANKA_HAND_TCP, Q_READY, PandaKinematics
from r2s2r.robots.mujoco_models import CACHE_DIR, MENAGERIE_DIR, robot_spec
from r2s2r.structs import CameraSpec, Capture
from r2s2r.transforms import intrinsics_matrix, make_transform

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
PANDA_BODIES = ["link0", *[f"link{i}" for i in range(1, 8)], "hand"]


@dataclass
class ObjectConfig:
    """A ground-truth object: a GSO model placed upright on the table."""

    name: str
    model: str  # directory name under ``gso_dir``
    xy: tuple[float, float]
    yaw_deg: float
    mass: float


@dataclass
class CameraConfig:
    """A static pinhole camera looking at ``lookat`` (RealSense-like defaults)."""

    role: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    width: int = 1280
    height: int = 720
    focal: float = 910.0  # pixels, square pixels, centred principal point


def default_objects() -> list[ObjectConfig]:
    """A graspable crayon box between a mug and a figurine."""
    return [
        ObjectConfig(
            "crayon_box", "Crayola_Crayons_24_count", (0.52, -0.02), 20.0, 0.12
        ),
        ObjectConfig(
            "blue_mug", "Cole_Hardware_Mug_Classic_Blue", (0.60, 0.22), -30.0, 0.35
        ),
        ObjectConfig(
            "android_figure", "Android_Figure_Orange", (0.48, -0.25), 60.0, 0.10
        ),
    ]


@dataclass
class CabinetConfig:
    """A procedural cabinet with one drawer (a prismatic joint), fixed to the table.

    Sizes are outer dimensions: depth along the drawer's travel, width, height.
    ``yaw_deg`` 180 puts the drawer front toward the robot.
    """

    name: str
    xy: tuple[float, float]
    yaw_deg: float = 180.0
    size: tuple[float, float, float] = (0.22, 0.26, 0.18)
    travel: float = 0.14  # how far the drawer opens


def default_cameras() -> list[CameraConfig]:
    """Two external views, front-left and right, like DROID's ext1/ext2."""
    return [
        CameraConfig("ext1", (1.05, 0.42, 0.52), (0.52, 0.0, 0.03)),
        CameraConfig("ext2", (0.40, -0.72, 0.58), (0.54, 0.0, 0.03)),
    ]


@dataclass
class MujocoWorldConfig:
    """Everything needed to rebuild the same world at capture and deploy time."""

    objects: list[ObjectConfig] = field(default_factory=default_objects)
    cabinets: list[CabinetConfig] = field(default_factory=list)
    cameras: list[CameraConfig] = field(default_factory=default_cameras)
    wrist_camera: bool = True
    wrist_size: tuple[int, int] = (1280, 720)
    wrist_focal: float = 800.0  # pixels
    # The capture motion points the wrist camera at this point from a few directions.
    scan_target: tuple[float, float, float] = (0.55, 0.0, 0.03)
    scan_distance: float = 0.5
    # After the scan, the capture opens every drawer by this much and closes it
    # again, like a teleoperated warmup demonstration (0: no interaction).
    demo_pull: float = 0.0
    target: str = "crayon_box"  # ground-truth object a pick task should lift
    table_center: tuple[float, float] = (0.35, 0.0)
    table_size: tuple[float, float] = (1.2, 1.2)
    table_height: float = 0.75  # the table top is the base plane, z = 0
    q_start: tuple[float, ...] = tuple(float(v) for v in Q_READY)
    timestep: float = 0.002
    gripper_kp: float = 1000.0  # ~7 N per finger on a 3 cm object
    menagerie_dir: str = str(MENAGERIE_DIR)
    gso_dir: str = str(CACHE_DIR / "gso" / "models")

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MujocoWorldConfig:
        """Inverse of :meth:`as_dict`."""
        d = dict(d)
        d["objects"] = [ObjectConfig(**o) for o in d["objects"]]
        d["cameras"] = [CameraConfig(**c) for c in d["cameras"]]
        d["cabinets"] = [CabinetConfig(**c) for c in d.get("cabinets", [])]
        return cls(**d)

    @classmethod
    def preset(cls, name: str) -> MujocoWorldConfig:
        """``pick``: crayon box, mug, figurine.

        ``cabinet``: the mug is replaced by a drawer cabinet, to exercise articulated
        objects.
        """
        if name == "pick":
            return cls()
        if name == "cabinet":
            objects = [o for o in default_objects() if o.name != "blue_mug"]
            return cls(
                objects=objects,
                cabinets=[CabinetConfig("drawer_cabinet", (0.66, 0.27))],
                scan_target=(0.6, 0.1, 0.06),
                scan_distance=0.62,
                demo_pull=0.12,
            )
        raise KeyError(f"unknown world preset {name!r} (pick, cabinet)")


def look_at(pos: NDArray, target: NDArray) -> NDArray[np.float64]:
    """``T_base_cam`` of an OpenCV camera (x right, y down, z forward)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return make_transform(np.column_stack([right, down, fwd]), pos)


def camera_intrinsics(cam: CameraConfig) -> NDArray[np.float64]:
    """MuJoCo renders a symmetric frustum: the principal point is the image centre."""
    return intrinsics_matrix(
        cam.focal, cam.focal, (cam.width - 1) / 2.0, (cam.height - 1) / 2.0
    )


# ---------------------------------------------------------------------- building
def _add_gso_object(spec: Any, obj: ObjectConfig, gso_dir: Path) -> None:
    """Add a GSO model as a free body, with namespaced assets and absolute paths."""
    model_dir = gso_dir / obj.model
    xml_root = ET.parse(model_dir / "model.xml").getroot()
    tex_el = xml_root.find("asset/texture")
    assert tex_el is not None, f"{obj.model} has no texture"
    tex_file = tex_el.attrib["file"]
    spec.add_texture(
        name=f"{obj.name}_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(model_dir / tex_file),
    )
    mat = spec.add_material(name=f"{obj.name}_mat", specular=0.3, shininess=0.3)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{obj.name}_tex"

    collision_files = [
        m.attrib["file"]
        for m in xml_root.iter("mesh")
        if m.attrib["name"].startswith("model_collision")
    ]
    # Density that gives the configured mass over the convex collision parts.
    volume = 0.0
    for f in collision_files:
        part = trimesh.load(model_dir / f, force="mesh")
        assert isinstance(part, trimesh.Trimesh)
        volume += part.convex_hull.volume
    density = obj.mass / volume

    yaw = np.deg2rad(obj.yaw_deg)
    body = spec.worldbody.add_body(
        name=obj.name,
        pos=[obj.xy[0], obj.xy[1], 0.002],
        quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
    )
    body.add_freejoint(name=f"{obj.name}_free")
    spec.add_mesh(name=f"{obj.name}_visual", file=str(model_dir / "model.obj"))
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{obj.name}_visual",
        material=f"{obj.name}_mat",
        contype=0,
        conaffinity=0,
        group=2,
        density=0.0,
    )
    for i, f in enumerate(collision_files):
        spec.add_mesh(name=f"{obj.name}_col{i}", file=str(model_dir / f))
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{obj.name}_col{i}",
            group=3,
            density=density,
            condim=4,
        )


def _add_cabinet_materials(spec: Any) -> None:
    spec.add_texture(
        name="cabinet_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        mark=mujoco.mjtMark.mjMARK_RANDOM,
        random=0.05,
        rgb1=[0.42, 0.27, 0.16],
        markrgb=[0.34, 0.21, 0.12],
        width=512,
        height=512,
    )
    body_mat = spec.add_material(name="cabinet_mat", specular=0.15)
    body_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cabinet_tex"
    spec.add_material(
        name="drawer_front_mat", rgba=[0.86, 0.84, 0.78, 1.0], specular=0.2
    )
    spec.add_material(name="handle_mat", rgba=[0.12, 0.12, 0.13, 1.0], specular=0.6)


def _add_cabinet(spec: Any, cab: CabinetConfig, wall: float = 0.012) -> None:
    """A fixed carcass open at the front (+x in its frame) and a sliding drawer."""
    dx, dy, dz = (v / 2 for v in cab.size)
    yaw = np.deg2rad(cab.yaw_deg)
    body = spec.worldbody.add_body(
        name=cab.name,
        pos=[cab.xy[0], cab.xy[1], 0.0],
        quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
    )
    box = mujoco.mjtGeom.mjGEOM_BOX
    panels = {  # name: (half sizes, centre)
        "bottom": ((dx, dy, wall / 2), (0, 0, wall / 2)),
        "top": ((dx, dy, wall / 2), (0, 0, 2 * dz - wall / 2)),
        "left": ((dx, wall / 2, dz), (0, dy - wall / 2, dz)),
        "right": ((dx, wall / 2, dz), (0, -dy + wall / 2, dz)),
        "back": ((wall / 2, dy, dz), (-dx + wall / 2, 0, dz)),
    }
    for name, (size, pos) in panels.items():
        body.add_geom(
            name=f"{cab.name}_{name}",
            type=box,
            size=list(size),
            pos=list(pos),
            material="cabinet_mat",
        )
    # The drawer fills the opening; closed, its front is flush with the carcass.
    inner_x, inner_y = dx - wall, dy - wall
    inner_z = dz - wall - 0.004
    drawer = body.add_body(name=f"{cab.name}_drawer", pos=[wall / 2, 0.0, dz])
    # The drawer runs on rails: it slides along its joint without rubbing the carcass
    # (MuJoCo's parent filter does not apply to a body welded to the world).
    spec.add_exclude(bodyname1=cab.name, bodyname2=f"{cab.name}_drawer")
    drawer.add_joint(
        name=f"{cab.name}_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=[0.0, cab.travel],
        damping=5.0,
        frictionloss=0.5,
    )
    t = 0.008
    for name, size, pos, mat in [
        (
            "front",
            (t, dy - 0.004, dz - 0.004),
            (inner_x - t / 2 + wall / 2, 0, 0),
            "drawer_front_mat",
        ),
        (
            "floor",
            (inner_x - t, inner_y - 0.002, t / 2),
            (-t, 0, -inner_z + t / 2),
            "cabinet_mat",
        ),
        (
            "side_l",
            (inner_x - t, t / 2, inner_z * 0.7),
            (-t, inner_y - t, -inner_z * 0.3),
            "cabinet_mat",
        ),
        (
            "side_r",
            (inner_x - t, t / 2, inner_z * 0.7),
            (-t, -inner_y + t, -inner_z * 0.3),
            "cabinet_mat",
        ),
        (
            "rear",
            (t / 2, inner_y - 0.002, inner_z * 0.7),
            (-inner_x + t, 0, -inner_z * 0.3),
            "cabinet_mat",
        ),
    ]:
        drawer.add_geom(
            name=f"{cab.name}_drawer_{name}",
            type=box,
            size=list(size),
            pos=list(pos),
            material=mat,
            mass=0.05,
        )
    drawer.add_geom(
        name=f"{cab.name}_handle",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.008, 0.045, 0],
        pos=[inner_x + wall / 2 + 0.022, 0, 0.02],
        quat=[np.cos(np.pi / 4), np.cos(np.pi / 4), 0.0, 0.0],
        material="handle_mat",
        mass=0.02,
    )
    for side in (1, -1):
        drawer.add_geom(
            name=f"{cab.name}_handle_post{side}",
            type=box,
            size=[0.011, 0.005, 0.005],
            pos=[inner_x + wall / 2 + 0.011, side * 0.035, 0.02],
            material="handle_mat",
            mass=0.005,
        )


def _add_camera(spec: Any, cam: CameraConfig) -> None:
    T = look_at(np.array(cam.pos), np.array(cam.lookat))
    fovy = np.rad2deg(2 * np.arctan(cam.height / 2.0 / cam.focal))
    spec.worldbody.add_camera(
        name=cam.role,
        pos=list(cam.pos),
        quat=quat_wxyz(T[:3, :3] @ CV_TO_MJ),
        fovy=float(fovy),
    )


def build_spec(cfg: MujocoWorldConfig) -> Any:
    """The world as an ``mujoco.MjSpec`` (compile it with ``spec.compile()``)."""
    spec = robot_spec("franka_panda", cfg.menagerie_dir)
    spec.modelname = "r2s2r_world"
    spec.option.timestep = cfg.timestep
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    width = max([c.width for c in cfg.cameras] + [cfg.wrist_size[0]])
    height = max([c.height for c in cfg.cameras] + [cfg.wrist_size[1]])
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.15, 0.15, 0.15]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]

    # Franka's controller compensates gravity internally.
    for name in PANDA_BODIES + ["left_finger", "right_finger"]:
        spec.body(name).gravcomp = 1.0
    grip = spec.actuator("actuator8")
    grip.gainprm[0] = 0.04 * cfg.gripper_kp / 255.0
    grip.biasprm[1] = -cfg.gripper_kp
    grip.biasprm[2] = -0.05 * cfg.gripper_kp
    hand = spec.body("hand")
    # Group 4 is never rendered: the site must not show up in camera images.
    hand.add_site(
        name="tcp", pos=[0.0, 0.0, FRANKA_HAND_TCP], size=[0.005, 0, 0], group=4
    )
    if cfg.wrist_camera:
        # Beside the hand, looking along its approach axis; the fingertips are in view.
        hand.add_camera(
            name="wrist",
            pos=[0.06, 0.0, 0.02],
            quat=quat_wxyz(CV_TO_MJ),
            fovy=float(
                np.rad2deg(2 * np.arctan(cfg.wrist_size[1] / 2 / cfg.wrist_focal))
            ),
        )

    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.75, 0.78, 0.82],
        rgb2=[0.35, 0.37, 0.4],
        width=512,
        height=512,
    )
    spec.add_texture(
        name="floor_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.42, 0.42, 0.44],
        rgb2=[0.36, 0.36, 0.38],
        width=512,
        height=512,
    )
    floor_mat = spec.add_material(name="floor_mat", texrepeat=[6, 6])
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    spec.add_texture(
        name="table_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        mark=mujoco.mjtMark.mjMARK_RANDOM,
        random=0.04,
        rgb1=[0.62, 0.5, 0.38],
        markrgb=[0.52, 0.41, 0.3],
        width=1024,
        height=1024,
    )
    table_mat = spec.add_material(name="table_mat", texrepeat=[2, 2], specular=0.1)
    table_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "table_tex"

    world = spec.worldbody
    world.add_light(
        name="key",
        pos=[0.3, 0.3, 2.0],
        dir=[0.15, -0.15, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.35, 0.35, 0.35],
        castshadow=True,
    )
    world.add_light(
        name="fill",
        pos=[1.5, -1.0, 1.5],
        dir=[-0.6, 0.4, -0.6],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.15, 0.15, 0.15],
        castshadow=False,
    )
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[4, 4, 0.05],
        pos=[0, 0, -cfg.table_height],
        material="floor_mat",
    )
    tx, ty = cfg.table_center
    sx, sy = cfg.table_size[0] / 2, cfg.table_size[1] / 2
    table = world.add_body(name="table", pos=[tx, ty, 0.0])
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[sx, sy, 0.02],
        pos=[0, 0, -0.02],
        material="table_mat",
    )
    for i, (lx, ly) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        table.add_geom(
            name=f"table_leg{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.03, 0.03, (cfg.table_height - 0.04) / 2],
            pos=[lx * (sx - 0.05), ly * (sy - 0.05), -(cfg.table_height + 0.04) / 2],
            rgba=[0.3, 0.25, 0.2, 1.0],
        )
    for obj in cfg.objects:
        _add_gso_object(spec, obj, Path(cfg.gso_dir))
    if cfg.cabinets:
        _add_cabinet_materials(spec)
    for cabinet in cfg.cabinets:
        _add_cabinet(spec, cabinet)
    for cam in cfg.cameras:
        _add_camera(spec, cam)
    return spec


# ------------------------------------------------------------------------ world
class MujocoWorld:
    """A compiled world plus the handles the capture and deploy code need."""

    def __init__(self, cfg: MujocoWorldConfig | None = None) -> None:
        self.cfg = cfg or MujocoWorldConfig()
        self.spec = build_spec(self.cfg)
        self.model = self.spec.compile()
        set_clip_planes(self.model)
        self.data = mujoco.MjData(self.model)
        self._arm_qadr = [self.model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self._finger_qadr = [self.model.joint(j).qposadr[0] for j in FINGER_JOINTS]
        self._renderers: dict[tuple[int, int], Any] = {}

    # ----------------------------------------------------------------- state
    def reset(self, settle: float = 1.0) -> None:
        """Robot at ``q_start`` with the gripper open, objects placed, then settle."""
        mujoco.mj_resetData(self.model, self.data)
        q = np.asarray(self.cfg.q_start)
        self.data.qpos[self._arm_qadr] = q
        self.data.qpos[self._finger_qadr] = 0.04
        self.data.ctrl[:7] = q
        self.data.ctrl[7] = 255.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(settle / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def arm_q(self) -> NDArray[np.float64]:
        """Measured arm joints."""
        return self.data.qpos[self._arm_qadr].copy()

    def finger_width(self) -> float:
        """Distance between the fingers."""
        return float(self.data.qpos[self._finger_qadr].sum())

    def joint_position(self, name: str) -> float:
        """Current position of a (1-dof) joint, e.g. a drawer's slide."""
        return float(self.data.qpos[self.model.joint(name).qposadr[0]])

    def object_pose(self, name: str) -> NDArray[np.float64]:
        """Ground-truth ``T_base_obj`` (GSO origin: bottom centre)."""
        body = self.data.body(name)
        return make_transform(body.xmat.reshape(3, 3), body.xpos)

    def tcp_pose(self) -> NDArray[np.float64]:
        """Measured ``T_base_tcp`` from the simulator's own kinematics."""
        site = self.data.site("tcp")
        return make_transform(site.xmat.reshape(3, 3), site.xpos)

    # ------------------------------------------------------------- cameras
    def camera_names(self) -> list[str]:
        """Static cameras first, then the wrist camera if present."""
        names = [c.role for c in self.cfg.cameras]
        return names + (["wrist"] if self.cfg.wrist_camera else [])

    def camera_spec(self, name: str) -> CameraSpec:
        """Calibration as a real rig would report it."""
        static = {c.role: c for c in self.cfg.cameras}
        if name in static:
            cam = static[name]
            return CameraSpec(
                serial=f"mj_{name}",
                role=name,
                width=cam.width,
                height=cam.height,
                K=camera_intrinsics(cam),
                is_static=True,
                T_base_cam=self.camera_pose(name),
            )
        width, height = self.cfg.wrist_size
        focal = self.cfg.wrist_focal
        K = intrinsics_matrix(focal, focal, (width - 1) / 2.0, (height - 1) / 2.0)
        return CameraSpec(
            serial=f"mj_{name}",
            role=name,
            width=width,
            height=height,
            K=K,
            is_static=False,
        )

    def camera_pose(self, name: str) -> NDArray[np.float64]:
        """Current OpenCV ``T_base_cam``."""
        cam = self.data.camera(name)
        return make_transform(cam.xmat.reshape(3, 3) @ CV_TO_MJ, cam.xpos)

    def _renderer(self, width: int, height: int) -> Any:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height, width)
        return self._renderers[key]

    def render(self, name: str) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """RGB (H, W, 3) and planar depth in meters (H, W)."""
        spec = self.camera_spec(name)
        r = self._renderer(spec.width, spec.height)
        r.disable_depth_rendering()
        r.update_scene(self.data, camera=name)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=name)
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        return rgb, depth

    def close(self) -> None:
        """Free the GL contexts."""
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


class MujocoRobot(RobotInterface):
    """The world's Panda behind the policy interface."""

    control_dt = 0.02

    def __init__(
        self,
        world: MujocoWorld,
        on_step: Callable[[MujocoRobot], None] | None = None,
    ) -> None:
        super().__init__(PandaKinematics())
        self.world = world
        self.on_step = on_step
        self.n_substeps = int(round(self.control_dt / world.model.opt.timestep))
        self.steps = 0

    def joint_positions(self) -> NDArray[np.float64]:
        return self.world.arm_q()

    def gripper_width(self) -> float:
        return self.world.finger_width()

    def _hold(self, q: NDArray[np.float64], gripper_closed: bool) -> None:
        data = self.world.data
        data.ctrl[:7] = q
        data.ctrl[7] = 0.0 if gripper_closed else 255.0
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.world.model, data)
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self)


def world_from_capture(capture: Capture) -> MujocoWorld:
    """Rebuild the world a MuJoCo capture was recorded in."""
    return MujocoWorld(MujocoWorldConfig.from_dict(capture.metadata["world_config"]))
