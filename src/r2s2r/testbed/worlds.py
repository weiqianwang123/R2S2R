"""MuJoCo worlds standing in for the real world: a robot from the registry, a scene,
calibrated cameras, and the ground truth a reconstruction is scored against.

A :class:`MujocoWorld` is one compiled ``MjSpec`` posed through the robot's
:class:`~r2s2r.robots.model.RobotModel` (so the recorder's IK and the rendered robot
are the same model) and simulated with the robot's arm on position control and its
gravity compensated, as a real arm's controller does. The builders:

- :func:`panda_table`: a Franka Panda with the Franka Hand on a wooden table with three
  Google Scanned Objects, an exterior camera ``ext1`` and a wrist camera on the hand;
- :func:`physcoder_box_block`: physcoder's UR5e + Robotiq 2F-140 scene
  (``box_block.xml``, used by path) with a seeded layout of its box and block, its
  MJCF ``wrist`` camera, and ``ext1`` at physcoder's real front camera pose.

A capture records the world's name and parameters, so :func:`world_from_capture`
rebuilds the same world for the pick test. The world frame is the robot base frame.
Assets: ``scripts/setup/fetch_mujoco_assets.sh`` (menagerie, GSO) and physcoder's.
"""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import trimesh
from numpy.typing import NDArray

from r2s2r.mjrender import CV_TO_MJ, CameraRenderer, add_camera, geom_mesh, mujoco
from r2s2r.paths import CACHE_DIR, PHYSCODER_ASSETS
from r2s2r.robots import get_robot
from r2s2r.robots.model import RobotModel, in_subtree
from r2s2r.robots.spec import MENAGERIE_DIR, RobotSpec
from r2s2r.structs import CameraSpec, Capture
from r2s2r.transforms import (
    intrinsics_matrix,
    look_at,
    make_transform,
    rotation_to_quat,
    transform_points,
)

GSO_DIR = CACHE_DIR / "gso" / "models"
BOX_BLOCK_XML = (
    PHYSCODER_ASSETS / "mujoco" / "objects" / "single_ur_scene" / "box_block.xml"
)
BLOCK_LAYOUTS = ("corner", "beside")
GT_SETTLE_SECONDS = 1.0  # physics before ground truth is read: objects come to rest
MOUNT_GAP = 0.05  # robot-scene pairs this near at the home pose are the arm's mount
SUPPORT_TOLERANCE = 0.003  # m: the support's visible top is this near its collision top


@dataclass(frozen=True)
class WorldObject:
    """A ground-truth object (the body of that name), and what it rests on (an object
    or the support's body)."""

    name: str
    rests_on: str


class MujocoWorld:
    """A compiled world: the robot (arm on position control), the scene, its cameras
    (``cameras``: roles of the MJCF cameras recorded, static ones first) and ground
    truth.

    ``robot`` is the robot's spec and ``kinematics`` poses it in the world's model and
    data. ``name`` and ``params`` rebuild the world (:func:`build_world`); ``layout``
    is what the builder placed where, ``target`` the object a pick test should lift,
    and ``support`` the box geom whose top face the objects stand on.
    """

    def __init__(
        self,
        name: str,
        params: dict[str, Any],
        robot: RobotSpec,
        mjspec: Any,
        *,
        objects: list[WorldObject],
        support: str,
        cameras: tuple[str, ...],
        target: str,
        instruction: str,
        layout: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        self.name, self.params = name, params
        self.objects, self.support, self.cameras = objects, support, cameras
        self.target, self.instruction = target, instruction
        self.layout, self.source = layout, source
        self.robot = robot
        robot.position_control(mjspec)
        # A real arm's controller compensates gravity; the objects feel it. (Set before
        # compiling: MuJoCo counts the compensated bodies then.)
        base = mjspec.joint(robot.arm_joints[0]).parent
        while base.parent.name != mjspec.worldbody.name:
            base = base.parent
        for body in (base, *base.find_all(mujoco.mjtObj.mjOBJ_BODY)):
            body.gravcomp = 1.0
        sizes = np.array([mjspec.camera(c).resolution for c in cameras])
        max_size = (int(sizes[:, 0].max()), int(sizes[:, 1].max()))
        add_camera(mjspec, max_size)
        self.kinematics = RobotModel(robot, mjspec)
        self.model, self.data = self.kinematics.model, self.kinematics.data
        self.renderer = CameraRenderer(self.model, self.data, max_size)
        m = self.model
        first = m.jnt_bodyid[m.joint(robot.arm_joints[0]).id]
        robot_bodies = np.flatnonzero(m.body_rootid == m.body_rootid[first])
        by_joint = {
            int(m.actuator_trnid[a, 0]): a
            for a in range(m.nu)
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
        }
        self.arm_actuators = np.array(
            [by_joint[m.joint(j).id] for j in robot.arm_joints]
        )
        self.gripper_actuator = m.actuator(robot.gripper.actuator).id
        self.driver_qadr = m.joint(robot.gripper.driver).qposadr[0]
        # Clearance is between what the arm moves and everything off the robot, but
        # for what is this close at the home pose already: the arm's mount (the
        # shoulder turning above its plate).
        colliding = (m.geom_contype | m.geom_conaffinity) > 0
        moving = [b for b in robot_bodies if in_subtree(m, int(b), int(first))]
        self.robot_geoms = np.flatnonzero(colliding & np.isin(m.geom_bodyid, moving))
        self.scene_geoms = np.flatnonzero(
            colliding & ~np.isin(m.geom_bodyid, robot_bodies)
        )
        self._mount = {pair for pair, _ in self._near_pairs(MOUNT_GAP)}

    # ----------------------------------------------------------------- state
    def reset(self) -> None:
        """The robot at its home pose, gripper open and held there, the objects where
        the builder put them; then :data:`GT_SETTLE_SECONDS` of physics."""
        mujoco.mj_resetData(self.model, self.data)
        home = np.asarray(self.robot.home_q)
        self.kinematics.set(home, 0.0)
        self.hold(home, 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.step(GT_SETTLE_SECONDS)

    def hold(self, q: NDArray, level: float) -> None:
        """Arm joint targets and gripper opening (0 open, 1 closed) for the
        actuators."""
        self.data.ctrl[self.arm_actuators] = q
        self.data.ctrl[self.gripper_actuator] = self.robot.gripper.ctrl_at(level)

    def step(self, seconds: float) -> None:
        """Simulate ``seconds``."""
        for _ in range(int(round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def arm_q(self) -> NDArray[np.float64]:
        """Arm joints."""
        return self.data.qpos[self.kinematics.arm_qadr].copy()

    def gripper_level(self) -> float:
        """The gripper's opening from its driver joint, 0 open to 1 closed."""
        return self.robot.gripper.level_of(self.data.qpos[self.driver_qadr])

    def body_pose(self, name: str) -> NDArray[np.float64]:
        """``T_base_body``."""
        return self.kinematics.pose(name)

    def object_positions(self) -> dict[str, NDArray[np.float64]]:
        """Every ground-truth object's position."""
        return {o.name: self.body_pose(o.name)[:3, 3].copy() for o in self.objects}

    def clearance(self, limit: float) -> float:
        """The moving robot's distance to the scene, its mount aside, as posed
        (``limit`` when farther)."""
        return min(
            (dist for pair, dist in self._near_pairs(limit) if pair not in self._mount),
            default=limit,
        )

    def _near_pairs(self, limit: float) -> list[tuple[tuple[int, int], float]]:
        """(robot geom, scene geom) pairs nearer than ``limit``, and how near."""
        m, d = self.model, self.data
        a, b = self.robot_geoms, self.scene_geoms
        gap = np.linalg.norm(d.geom_xpos[a][:, None] - d.geom_xpos[b][None], axis=2)
        gap -= m.geom_rbound[a][:, None] + m.geom_rbound[b][None]
        gap[:, m.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE] = -np.inf
        out = []
        for i, j in zip(*np.nonzero(gap < limit)):
            pair = (int(a[i]), int(b[j]))
            dist = float(mujoco.mj_geomDistance(m, d, *pair, limit, None))
            if dist < limit:
                out.append((pair, dist))
        return out

    # ------------------------------------------------------------- cameras
    def camera_spec(self, role: str) -> CameraSpec:
        """A camera's calibration as a real rig reports it; static cameras carry their
        pose."""
        m = self.model
        c = m.camera(role).id
        width, height = (int(v) for v in m.cam_resolution[c])
        if m.cam_sensorsize[c].any():
            fx, fy = m.cam_intrinsic[c][:2] / m.cam_sensorsize[c] * (width, height)
        else:
            fx = fy = height / 2.0 / np.tan(np.deg2rad(m.cam_fovy[c]) / 2.0)
        static = m.cam_bodyid[c] == 0
        return CameraSpec(
            serial=f"mj_{role}",
            role=role,
            width=width,
            height=height,
            K=intrinsics_matrix(fx, fy, (width - 1) / 2.0, (height - 1) / 2.0),
            is_static=bool(static),
            T_base_cam=self.camera_pose(role) if static else None,
        )

    def camera_pose(self, role: str) -> NDArray[np.float64]:
        """A camera's current OpenCV ``T_base_cam``."""
        return self.kinematics.pose(role, "camera")

    def render(self, role: str) -> dict[str, NDArray]:
        """``rgb`` (H, W, 3), planar ``depth`` in metres and ``body`` ids (-1: none)
        from a camera, as it is posed."""
        cam = self.camera_spec(role)
        out = self.renderer.render(cam.K, cam.width, cam.height, self.camera_pose(role))
        geom = out.pop("geom")
        out["body"] = np.where(geom >= 0, self.model.geom_bodyid[geom], -1)
        return out

    def close(self) -> None:
        """Free the GL contexts."""
        self.renderer.close()

    # --------------------------------------------------------- ground truth
    def ground_truth(self) -> dict[str, Any]:
        """The scene as it is now, self-contained (scoring needs nothing else): per
        object its pose, visual box, collision boxes, mass, friction, whether it is
        static and what it rests on; the support; the layout; the assets' source."""
        return {
            "world": self.name,
            "target": self.target,
            "support": self._support_truth(),
            "objects": {o.name: self._object_truth(o) for o in self.objects},
            "layout": self.layout,
            "source": self.source,
        }

    def geoms(self, body: str) -> tuple[list[int], list[int]]:
        """A body's visual geoms (the groups cameras render) and collision geoms."""
        m = self.model
        own = np.flatnonzero(m.geom_bodyid == m.body(body).id)
        visual = [int(g) for g in own if m.geom_group[g] < 3 and m.geom_rgba[g, 3] > 0]
        colliding = [int(g) for g in own if m.geom_contype[g] or m.geom_conaffinity[g]]
        return visual, colliding

    def _object_truth(self, obj: WorldObject) -> dict[str, Any]:
        m = self.model
        body = m.body(obj.name).id
        T = self.body_pose(obj.name)
        visual, colliding = self.geoms(obj.name)
        points = np.concatenate([_vertices(m, g) for g in visual])
        points = transform_points(T, points)
        lo, hi = points.min(axis=0), points.max(axis=0)
        boxes = []
        for g in colliding:
            v = _vertices(m, g)
            c_lo, c_hi = v.min(axis=0), v.max(axis=0)
            boxes.append(
                {
                    "T_base_box": T @ make_transform(np.eye(3), (c_lo + c_hi) / 2),
                    "size": c_hi - c_lo,
                }
            )
        static = bool(m.body_jntnum[body] == 0)
        return {
            "T_base_obj": T,
            "center": (lo + hi) / 2,
            "size": hi - lo,
            "collision": boxes,
            "mass": None if static else float(m.body_subtreemass[body]),
            "friction": float(max(m.geom_friction[g, 0] for g in colliding)),
            "static": static,
            "rests_on": obj.rests_on,
        }

    def _support_truth(self) -> dict[str, Any]:
        """The support's collision top (its height, where the objects stand), centred
        and sized by what the cameras see there: its body's upward visual faces within
        :data:`SUPPORT_TOLERANCE` of that top. (physcoder's table box is turned 90
        degrees from the table it draws.)"""
        m, d = self.model, self.data
        g = m.geom(self.support).id
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"support geom {self.support} is not a box")
        R = d.geom_xmat[g].reshape(3, 3)
        if R[2, 2] < 0.999:
            raise ValueError(f"support geom {self.support} is not level")
        top = d.geom_xpos[g] + R[:, 2] * m.geom_size[g][2]
        body = m.body(m.geom_bodyid[g]).name
        T_base_body = self.body_pose(body)
        seen = [np.empty((0, 3))]
        for v in self.geoms(body)[0]:
            mesh = geom_mesh(m, v)
            if mesh is None:
                continue
            faces = transform_points(T_base_body, mesh.vertices)[mesh.faces]
            up = mesh.face_normals @ T_base_body[:3, :3].T
            level = np.all(np.abs(faces[..., 2] - top[2]) < SUPPORT_TOLERANCE, axis=1)
            seen.append(faces[(up[:, 2] > 0.9) & level].reshape(-1, 3))
        local = (np.concatenate(seen) - top) @ R  # in the support's frame
        if not len(local):
            raise ValueError(f"nothing of {body} is seen at its collision top")
        lo, hi = local[:, :2].min(axis=0), local[:, :2].max(axis=0)
        return {
            "T_base_support": make_transform(R, top + R[:, :2] @ ((lo + hi) / 2)),
            "height": float(top[2]),
            "size": hi - lo,
            "body": body,
        }


def _vertices(model: Any, g: int) -> NDArray[np.float64]:
    mesh = geom_mesh(model, g)
    if mesh is None:
        raise ValueError(f"geom {model.geom(g).name} has no mesh form")
    return np.asarray(mesh.vertices, float)


# ------------------------------------------------------------------ panda_table
@dataclass(frozen=True)
class GsoObject:
    """A Google Scanned Object placed upright on the table."""

    name: str
    model: str  # directory under GSO_DIR
    xy: tuple[float, float]
    yaw_deg: float
    mass: float


PANDA_OBJECTS = (
    GsoObject("crayon_box", "Crayola_Crayons_24_count", (0.52, -0.02), 20.0, 0.12),
    GsoObject("blue_mug", "Cole_Hardware_Mug_Classic_Blue", (0.60, 0.22), -30.0, 0.35),
    GsoObject("android_figure", "Android_Figure_Orange", (0.48, -0.25), 60.0, 0.10),
)
TABLE_CENTER, TABLE_SIZE, TABLE_HEIGHT = (0.35, 0.0), (1.2, 1.2), 0.75
GRIPPER_KP = 1000.0  # the Franka Hand's position gain: ~7 N per finger on 3 cm


def panda_table() -> MujocoWorld:
    """A graspable crayon box between a mug and a figurine on a wooden table (its top
    is z = 0), an exterior camera front-left of it and a camera on the hand."""
    robot = get_robot("franka_panda")
    spec = robot.mjcf()
    spec.modelname = "panda_table"
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.15, 0.15, 0.15]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    grip = spec.actuator(robot.gripper.actuator)
    grip.gainprm[0] = 0.04 * GRIPPER_KP / 255.0
    grip.biasprm[1] = -GRIPPER_KP
    grip.biasprm[2] = -0.05 * GRIPPER_KP
    # Beside the hand, looking along its approach axis; the fingertips are in view.
    spec.body("hand").add_camera(
        name="wrist",
        pos=[0.06, 0.0, 0.02],
        quat=rotation_to_quat(CV_TO_MJ).tolist(),
        **_intrinsics(1280, 720, 800.0),
    )
    T = look_at([1.05, 0.42, 0.52], [0.52, 0.0, 0.03])
    spec.worldbody.add_camera(
        name="ext1",
        pos=T[:3, 3].tolist(),
        quat=rotation_to_quat(T[:3, :3] @ CV_TO_MJ).tolist(),
        **_intrinsics(1280, 720, 910.0),
    )
    _add_table(spec)
    for obj in PANDA_OBJECTS:
        _add_gso_object(spec, obj)
    return MujocoWorld(
        "panda_table",
        {},
        robot,
        spec,
        objects=[WorldObject(o.name, "table") for o in PANDA_OBJECTS],
        support="table_top",
        cameras=("ext1", "wrist"),
        target="crayon_box",
        instruction="pick up the crayon box beside the blue mug and the orange "
        "android figure",
        layout={"objects": [asdict(o) for o in PANDA_OBJECTS]},
        source={"menagerie": str(MENAGERIE_DIR), "gso": str(GSO_DIR)},
    )


def _intrinsics(width: int, height: int, focal: float) -> dict[str, Any]:
    """An MJCF camera's resolution and focal length in pixels (principal point at the
    image centre)."""
    return {
        "resolution": [width, height],
        "sensor_size": [width, height],
        "focal_pixel": [focal, focal],
    }


def _add_table(spec: Any) -> None:
    """Sky, a checkered floor, lights, and the table (``table_top`` at z = 0)."""
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
        pos=[0, 0, -TABLE_HEIGHT],
        material="floor_mat",
    )
    sx, sy = TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2
    table = world.add_body(name="table", pos=[*TABLE_CENTER, 0.0])
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
            size=[0.03, 0.03, (TABLE_HEIGHT - 0.04) / 2],
            pos=[lx * (sx - 0.05), ly * (sy - 0.05), -(TABLE_HEIGHT + 0.04) / 2],
            rgba=[0.3, 0.25, 0.2, 1.0],
        )


def _add_gso_object(spec: Any, obj: GsoObject) -> None:
    """A GSO model as a free body 2 mm above the table: its textured visual mesh and
    convex collision parts, with the density that gives ``obj.mass``."""
    model_dir = GSO_DIR / obj.model
    xml_root = ET.parse(model_dir / "model.xml").getroot()
    tex_el = xml_root.find("asset/texture")
    if tex_el is None:
        raise ValueError(f"{obj.model} has no texture")
    spec.add_texture(
        name=f"{obj.name}_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(model_dir / tex_el.attrib["file"]),
    )
    mat = spec.add_material(name=f"{obj.name}_mat", specular=0.3, shininess=0.3)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{obj.name}_tex"
    parts = [
        m.attrib["file"]
        for m in xml_root.iter("mesh")
        if m.attrib["name"].startswith("model_collision")
    ]
    volume = 0.0
    for f in parts:
        part = trimesh.load(model_dir / f, force="mesh")
        assert isinstance(part, trimesh.Trimesh)
        volume += part.convex_hull.volume
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
    for i, f in enumerate(parts):
        spec.add_mesh(name=f"{obj.name}_col{i}", file=str(model_dir / f))
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{obj.name}_col{i}",
            group=3,
            density=obj.mass / volume,
            condim=4,
        )


# ---------------------------------------------------------- physcoder_box_block
# physcoder's ranges (mujoco_env/single_ur/box_block.py): the box's origin above the
# table top (z = -0.013), its xy and yaw; the block's footprint half-sizes, the inner
# half-width of the box, and its body origin above what it stands on.
TABLE_TOP = -0.013
BOX_Z = TABLE_TOP + 0.069016
BOX_X, BOX_Y, BOX_YAW = (-0.6, -0.5), (-0.1, 0.1), (11 * np.pi / 12, 13 * np.pi / 12)
BLOCK_HALF = (0.0206, 0.0205)
BOX_INNER, BOX_OUTER = 0.11, 0.16
BLOCK_ORIGIN = 0.021639  # above its collision box's bottom
BOX_FLOOR = -0.029016  # the box floor's top, in the box frame
# `beside`: the block's gap to the box's outer wall; the open 2F-140's fingers reach
# 7 cm from its centre.
BESIDE_GAP = (0.08, 0.12)
# physcoder's real front camera (robot/single_ur/real/utils/ur_real_kinematics.py), an
# OpenGL camera pose (looking along -z, y up) as MuJoCo's cameras are; a RealSense at
# 640x480 with a stand-in focal length.
FRONT_CAMERA = np.array(
    [
        [-0.00345512, 0.78655414, -0.61751165, -0.8806788014],
        [-0.99979578, -0.01501287, -0.01352853, 0.0343685208],
        [-0.01991154, 0.61733880, 0.78644538, 0.4224737591],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
FRONT_SIZE, FRONT_FOCAL = (640, 480), 605.0


def physcoder_box_block(seed: int = 0, block: str = "corner") -> MujocoWorld:
    """physcoder's box and block on its table (top z = -0.013): the box at a seeded
    place and yaw in physcoder's ranges, the block in one of its corners 1-4 mm from
    the walls (``corner``, physcoder's layout) or on the table beside it (``beside``,
    where the 2F-140 can pick it)."""
    if block not in BLOCK_LAYOUTS:
        raise ValueError(f"block must be one of {BLOCK_LAYOUTS}, not {block!r}")
    if not BOX_BLOCK_XML.exists():
        raise FileNotFoundError(
            f"{BOX_BLOCK_XML} not found: fetch physcoder's assets, or point "
            "PHYSCODER_ASSETS_DIR or PHYSCODER_ROOT at them"
        )
    robot = get_robot("ur5e_2f140")
    spec = mujoco.MjSpec.from_file(str(BOX_BLOCK_XML))
    layout = box_block_layout(np.random.default_rng(seed), block)
    box, blk = spec.body("box"), spec.body("block")
    box.pos = [*layout["box_xy"], BOX_Z]
    box.quat = [np.cos(layout["box_yaw"] / 2), 0.0, 0.0, np.sin(layout["box_yaw"] / 2)]
    blk.pos = layout["block_pos"]
    yaw = layout["box_yaw"] + layout["block_yaw"]
    blk.quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    spec.worldbody.add_camera(
        name="ext1",
        pos=FRONT_CAMERA[:3, 3].tolist(),
        quat=rotation_to_quat(FRONT_CAMERA[:3, :3]).tolist(),
        **_intrinsics(*FRONT_SIZE, FRONT_FOCAL),
    )
    where = "in a corner of" if block == "corner" else "beside"
    return MujocoWorld(
        "physcoder_box_block",
        {"seed": seed, "block": block},
        robot,
        spec,
        objects=[
            WorldObject("box", "table"),
            WorldObject("block", "box" if block == "corner" else "table"),
        ],
        support="table_mesh_0",
        cameras=("ext1", "wrist"),
        target="block",
        instruction=f"pick up the green block {where} the grey box",
        layout=layout,
        source=physcoder_source(),
    )


def box_block_layout(rng: np.random.Generator, block: str) -> dict[str, Any]:
    """Our seeded sampler over physcoder's ranges: the box's xy and yaw, the block's
    position and its yaw relative to the box."""
    box_xy = np.array([rng.uniform(*BOX_X), rng.uniform(*BOX_Y)])
    box_yaw = rng.uniform(*BOX_YAW)
    block_yaw = rng.uniform(-np.pi, np.pi)
    c, s = abs(np.cos(block_yaw)), abs(np.sin(block_yaw))
    half = np.array(
        [BLOCK_HALF[0] * c + BLOCK_HALF[1] * s, BLOCK_HALF[0] * s + BLOCK_HALF[1] * c]
    )
    if block == "corner":
        wall_gap = rng.uniform(0.001, 0.004, 2)
        local = (BOX_INNER - half - wall_gap) * rng.choice([-1.0, 1.0], 2)
        z = BOX_Z + BOX_FLOOR + BLOCK_ORIGIN + rng.uniform(0.001, 0.003)
    else:
        side = rng.choice([-1.0, 1.0])
        gap = rng.uniform(*BESIDE_GAP)
        local = np.array([rng.uniform(-0.06, 0.06), side * (BOX_OUTER + gap + half[1])])
        z = TABLE_TOP + BLOCK_ORIGIN + rng.uniform(0.001, 0.003)
    cy, sy = np.cos(box_yaw), np.sin(box_yaw)
    xy = box_xy + np.array([[cy, -sy], [sy, cy]]) @ local
    return {
        "block": block,
        "box_xy": box_xy.tolist(),
        "box_yaw": float(box_yaw),
        "block_in_box": local.tolist(),
        "block_yaw": float(block_yaw),
        "block_pos": [float(xy[0]), float(xy[1]), float(z)],
    }


def physcoder_source() -> dict[str, Any]:
    """Which physcoder assets the world was built from: the checkout's commit and its
    asset manifest's ``asset_set_sha256`` (None where unknown)."""
    root = PHYSCODER_ASSETS.parent
    manifest = root / "config" / "assets.json"
    sha = None
    if manifest.exists():
        sha = json.loads(manifest.read_text(encoding="utf-8")).get("asset_set_sha256")
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "assets": str(PHYSCODER_ASSETS),
        "physcoder_commit": git.stdout.strip() if git.returncode == 0 else None,
        "asset_set_sha256": sha,
    }


# -------------------------------------------------------------------- registry
WORLDS: dict[str, Callable[..., MujocoWorld]] = {
    "panda_table": panda_table,
    "physcoder_box_block": physcoder_box_block,
}


def build_world(name: str, **params: Any) -> MujocoWorld:
    """World ``name`` built with ``params`` (its builder's keyword arguments)."""
    if name not in WORLDS:
        raise ValueError(f"unknown world {name!r} (known: {', '.join(WORLDS)})")
    try:
        return WORLDS[name](**params)
    except TypeError as exc:
        raise ValueError(f"world {name} takes no {sorted(params)}: {exc}") from exc


def world_from_capture(capture: Capture) -> MujocoWorld:
    """The world a testbed capture was recorded in, rebuilt and settled; its layout
    must be the recorded one."""
    meta = capture.metadata.get("world")
    if meta is None:
        raise ValueError(f"capture {capture.name} was not recorded in a MuJoCo world")
    world = build_world(meta["name"], **meta["params"])
    recorded = capture.metadata["ground_truth"]["layout"]
    if not _same(world.layout, recorded):
        raise ValueError(
            f"world {meta['name']} {meta['params']} no longer has capture "
            f"{capture.name}'s layout: {world.layout} vs {recorded}"
        )
    world.reset()
    return world


def _same(a: Any, b: Any) -> bool:
    """JSON-like values equal, numbers to 1e-9."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(a, b, rtol=0.0, atol=1e-9))
    return bool(a == b)
