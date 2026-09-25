"""MuJoCo standing in for the real world.

The world is a Franka Panda (mujoco_menagerie) on a table with scanned household
objects (Google Scanned Objects, MuJoCo port), watched by calibrated RGB-D cameras.
It plays both ends of the loop:

* **real -> sim**: :func:`record_capture` writes a :class:`~r2s2r.structs.Capture`
  (RGB + metric depth, intrinsics, extrinsics, joint states) exactly like a real
  recording, and nothing else; ground truth goes into ``capture.metadata`` for
  evaluation only;
* **sim -> real**: :class:`MujocoRobot` is a :class:`RobotInterface`, so a program
  policy validated in Isaac Lab runs here unchanged.

The world frame is the robot base frame. Assets are fetched by
``scripts/fetch_mujoco_assets.sh`` into ``~/.cache/r2s2r``.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.policy.robot import RobotInterface
from r2s2r.robots.franka import FRANKA_HAND_TCP, Q_READY, PandaKinematics
from r2s2r.structs import DEPTH_PNG_SCALE, CameraSpec, Capture, FrameRecord
from r2s2r.transforms import intrinsics_matrix, make_transform

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order

CACHE_DIR = Path(os.environ.get("R2S2R_CACHE", Path.home() / ".cache" / "r2s2r"))

MAX_DEPTH = 10.0  # meters; farther pixels (the sky) are stored as invalid (0)
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
    cameras: list[CameraConfig] = field(default_factory=default_cameras)
    wrist_camera: bool = True
    target: str = "crayon_box"  # ground-truth object a pick task should lift
    table_center: tuple[float, float] = (0.35, 0.0)
    table_size: tuple[float, float] = (1.2, 1.2)
    table_height: float = 0.75  # the table top is the base plane, z = 0
    q_start: tuple[float, ...] = tuple(float(v) for v in Q_READY)
    timestep: float = 0.002
    gripper_kp: float = 1000.0  # ~7 N per finger on a 3 cm object
    menagerie_dir: str = str(CACHE_DIR / "mujoco_menagerie")
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
        return cls(**d)


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


# OpenCV camera axes -> MuJoCo camera axes (x right, y up, looking along -z).
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def _quat_wxyz(R: NDArray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


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


def _add_camera(spec: Any, cam: CameraConfig) -> None:
    T = look_at(np.array(cam.pos), np.array(cam.lookat))
    fovy = np.rad2deg(2 * np.arctan(cam.height / 2.0 / cam.focal))
    spec.worldbody.add_camera(
        name=cam.role,
        pos=list(cam.pos),
        quat=_quat_wxyz(T[:3, :3] @ CV_TO_MJ),
        fovy=float(fovy),
    )


def build_spec(cfg: MujocoWorldConfig) -> Any:
    """The world as an ``mujoco.MjSpec`` (compile it with ``spec.compile()``)."""
    spec = mujoco.MjSpec.from_file(
        str(Path(cfg.menagerie_dir) / "franka_emika_panda" / "panda.xml")
    )
    spec.modelname = "r2s2r_world"
    spec.option.timestep = cfg.timestep
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    width = max([c.width for c in cfg.cameras] + [640])
    height = max([c.height for c in cfg.cameras] + [480])
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
    hand.add_site(name="tcp", pos=[0.0, 0.0, FRANKA_HAND_TCP], size=[0.005, 0, 0])
    if cfg.wrist_camera:
        # Behind the fingers, looking along the hand's approach axis.
        hand.add_camera(
            name="wrist",
            pos=[0.06, 0.0, 0.02],
            quat=_quat_wxyz(
                Rotation.from_euler("y", -15, degrees=True).as_matrix() @ CV_TO_MJ
            ),
            fovy=70.0,
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
        cam_model = self.model.camera(name)
        height, width = 480, 640
        focal = height / 2.0 / np.tan(np.deg2rad(cam_model.fovy[0]) / 2.0)
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


# ---------------------------------------------------------------------- capture
def _capture_motion(robot: MujocoRobot) -> None:
    """Sweep the hand over the table so the wrist camera sees the scene."""
    T0 = robot.tcp_pose()
    for dx, dy, dz in [(0.12, 0.18, -0.1), (0.12, -0.18, -0.1), (0.0, 0.0, 0.0)]:
        T = T0.copy()
        T[:3, 3] += [dx, dy, dz]
        robot.move_tcp(T, speed=0.12, settle=0.2)


def record_capture(
    out_dir: str | Path,
    cfg: MujocoWorldConfig | None = None,
    name: str = "mujoco_pick",
    instruction: str = "pick up the crayon box",
    every: int = 5,
) -> Capture:
    """Run the capture motion and save RGB-D frames every ``every`` control steps."""
    out_dir = Path(out_dir)
    world = MujocoWorld(cfg)
    world.reset()
    cameras = {n: world.camera_spec(n) for n in world.camera_names()}
    frames: list[FrameRecord] = []

    def grab(robot: MujocoRobot) -> None:
        if robot.steps % every:
            return
        q, width = world.arm_q(), world.finger_width()
        for cam_name in world.camera_names():
            spec = cameras[cam_name]
            rgb, depth = world.render(cam_name)
            rel = Path("frames") / spec.serial
            (out_dir / rel).mkdir(parents=True, exist_ok=True)
            rgb_rel = rel / f"{robot.steps:04d}_rgb.png"
            depth_rel = rel / f"{robot.steps:04d}_depth.png"
            cv2.imwrite(str(out_dir / rgb_rel), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            depth[depth > MAX_DEPTH] = 0.0  # like a real sensor: 0 = no return
            depth_mm = np.clip(np.round(depth / DEPTH_PNG_SCALE), 0, 65535)
            cv2.imwrite(str(out_dir / depth_rel), depth_mm.astype(np.uint16))
            frames.append(
                FrameRecord(
                    step=robot.steps,
                    camera=spec.serial,
                    left_image=str(rgb_rel),
                    right_image=None,
                    depth_image=str(depth_rel),
                    T_base_cam=world.camera_pose(cam_name),
                    joint_positions=q,
                    gripper_position=1.0 - width / 0.08,
                )
            )

    robot = MujocoRobot(world, on_step=grab)
    grab(robot)
    _capture_motion(robot)
    capture = Capture(
        name=name,
        source="mujoco",
        embodiment="franka_panda",
        instruction=instruction,
        cameras={c.serial: c for c in cameras.values()},
        frames=frames,
        static_steps=(0, robot.steps + 1),
        root=out_dir,
        metadata={
            "world_config": world.cfg.as_dict(),
            "depth_png_scale": DEPTH_PNG_SCALE,
            # For evaluating reconstructions only; never read by the pipeline.
            "ground_truth_T_base_obj": {
                o.name: world.object_pose(o.name).tolist() for o in world.cfg.objects
            },
        },
    )
    capture.save()
    world.close()
    return capture


def world_from_capture(capture: Capture) -> MujocoWorld:
    """Rebuild the world a MuJoCo capture was recorded in."""
    return MujocoWorld(MujocoWorldConfig.from_dict(capture.metadata["world_config"]))


def save_json(path: str | Path, payload: Any) -> None:
    """Small helper for result files."""
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
