"""Build a MuJoCo scene from a :class:`~r2s2r.structs.SceneSpec`, and run it
(:class:`Session`: what MuJoCo's settling and replay share, :mod:`r2s2r.sim.mujoco`, and
what anyone simulating a reconstructed scene in MuJoCo starts from).

The world frame is the robot base frame, so every SceneSpec pose is used as-is, as in
Isaac Lab (:mod:`r2s2r.sim.isaaclab.scene`): the robot (its
:class:`~r2s2r.robots.spec.RobotSpec`'s MJCF) at the origin, its arm on position
control with gravity compensated; the support drawn as a slab whose top face is its
plane, and colliding as that plane (:func:`_add_support`);
every object from its URDF at its ``T_base_obj``, a free body (an articulated one with
its links under it, joined by its joints with their dynamics,
:class:`~r2s2r.structs.JointDynamics`), its collision parts convex colliders with its
friction, its visual meshes textured and not colliding. Gravity is along the support's
normal (:func:`~r2s2r.sim.world.gravity`).

MuJoCo's URDF import cannot take the assembled URDFs (it drops the ``collision/``
directory from mesh paths), so the bodies are built from the URDF here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from r2s2r.assets import VisualMesh, base_color_texture, urdf_origin, urdf_vector
from r2s2r.mjrender import MAX_SIZE, CameraRenderer, add_camera, add_mesh, mujoco
from r2s2r.robots.model import GripperPoser, prepare_robot
from r2s2r.robots.spec import RobotSpec
from r2s2r.sim.world import gravity, support_extent
from r2s2r.structs import SUPPORT_THICKNESS, ObjectSpec, SceneSpec
from r2s2r.transforms import make_transform, matrix_to_pos_quat, pos_quat_to_matrix

TIMESTEP = 0.002  # s, the longest (MuJoCo's default): a pinch between the pads holds
# Sliding friction when the scene gives an object none (PhysX's default material's);
# MuJoCo's own torsional and rolling friction.
DEFAULT_FRICTION = 0.5
SPIN_FRICTION, ROLL_FRICTION = 0.005, 0.0001
VISUAL_GROUP, COLLISION_GROUP = 2, 3  # rendered by default, and not
# A joint's dry friction as a stiff constraint: MuJoCo's default lets a lid its friction
# should hold creep shut under its weight, the same at any friction.
FRICTION_SOLREF = (2 * TIMESTEP, 1.0)
FRICTION_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
SUPPORT_RGBA = (0.2, 0.45, 0.9, 1.0)  # when the scene has no support colour
JOINT_KINDS = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
}


# --------------------------------------------------------------------- the URDF
@dataclass
class _Link:
    """A URDF link: its inertia (mass, centre of mass, 3x3 inertia about it, in the
    link frame) and its meshes (file, scale, pose in the link frame)."""

    mass: float
    com: NDArray[np.float64]
    inertia: NDArray[np.float64]
    visuals: list[tuple[Path, NDArray[np.float64], NDArray[np.float64]]] = field(
        default_factory=list
    )
    collisions: list[tuple[Path, NDArray[np.float64], NDArray[np.float64]]] = field(
        default_factory=list
    )


@dataclass
class _Joint:
    """A URDF joint: its child link's frame in its parent's at zero, the axis in the
    child's frame, the limits."""

    name: str
    kind: str
    parent: str
    child: str
    origin: NDArray[np.float64]
    axis: NDArray[np.float64]
    limits: tuple[float, float]


def _read_urdf(path: Path) -> tuple[str, dict[str, _Link], list[_Joint]]:
    """An object's URDF: its root link, its links, its joints parents first."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            raise ValueError(f"{path}: link {link.attrib['name']} has no inertial")
        frame = urdf_origin(inertial)
        I_el = inertial.find("inertia")
        assert I_el is not None
        I = _inertia_matrix(I_el.attrib)
        R = frame[:3, :3]
        meshes = {}
        for kind in ("visual", "collision"):
            meshes[kind] = [
                (
                    path.parent / mesh.attrib["filename"],
                    urdf_vector(mesh, "scale", "1 1 1"),
                    urdf_origin(element),
                )
                for element in link.iter(kind)
                if (mesh := element.find("geometry/mesh")) is not None
            ]
        links[link.attrib["name"]] = _Link(
            mass=float(urdf_vector(inertial.find("mass"), "value", "0")[0]),
            com=frame[:3, 3],
            inertia=R @ I @ R.T,
            visuals=meshes["visual"],
            collisions=meshes["collision"],
        )
    joints = []
    for joint in root.iter("joint"):
        kind = joint.attrib.get("type", "fixed")
        if kind not in JOINT_KINDS:
            raise ValueError(f"{path}: joint {joint.attrib['name']} is {kind}")
        limit = joint.find("limit")
        joints.append(
            _Joint(
                name=joint.attrib["name"],
                kind=kind,
                parent=_named(joint, "parent"),
                child=_named(joint, "child"),
                origin=urdf_origin(joint),
                axis=urdf_vector(joint.find("axis"), "xyz", "1 0 0"),
                limits=(
                    float(urdf_vector(limit, "lower", "0")[0]),
                    float(urdf_vector(limit, "upper", "0")[0]),
                ),
            )
        )
    children = {j.child for j in joints}
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(f"{path}: {len(roots)} root links, {roots}")
    order: list[_Joint] = []
    placed = {roots[0]}
    while len(order) < len(joints):  # parents first
        ready = [j for j in joints if j.parent in placed and j.child not in placed]
        if not ready:
            raise ValueError(f"{path}: its joints do not form a tree")
        order += ready
        placed |= {j.child for j in ready}
    return roots[0], links, order


def _inertia_matrix(a: dict[str, str]) -> NDArray[np.float64]:
    xx, yy, zz = float(a["ixx"]), float(a["iyy"]), float(a["izz"])
    xy, xz, yz = float(a["ixy"]), float(a["ixz"]), float(a["iyz"])
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _named(joint: ET.Element, tag: str) -> str:
    el = joint.find(tag)
    if el is None:
        raise ValueError(f"joint {joint.attrib.get('name')} has no <{tag}>")
    return el.attrib["link"]


# ------------------------------------------------------------------ the scene
def scene_spec(
    scene: SceneSpec, robot: RobotSpec | None = None, timestep: float = TIMESTEP
) -> Any:
    """``scene`` as a ``mujoco.MjSpec`` (see the module doc), with ``robot`` when
    given (the scene's embodiment), stepping ``timestep`` (s, at most
    :data:`TIMESTEP`): its support, its objects (bodies named as the objects; an
    articulated object's links ``<object>/<link>``, its joints ``<object>/<joint>``)
    and lights; the objects at their poses, their joints at zero (:class:`Session`
    puts them where the scene has them)."""
    if timestep > TIMESTEP:
        raise ValueError(f"a {timestep} s step is too long (at most {TIMESTEP} s)")
    if robot is not None and robot.name != scene.embodiment:
        raise ValueError(
            f"scene {scene.name} is for {scene.embodiment}, not {robot.name}"
        )
    spec = mujoco.MjSpec() if robot is None else robot.mjcf()
    if robot is not None:
        prepare_robot(spec, robot)
    spec.compiler.degree = False  # every angle here is in radians
    spec.option.timestep = timestep
    spec.option.gravity = gravity(scene)
    # Several contact points between convex meshes, not one: on one, a book's flat
    # bottom balanced on a point and sank 2 cm into the table.
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.worldbody.add_light(
        name="sun",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        dir=[-0.5, 0.0, -1.0],
        diffuse=[0.5, 0.5, 0.5],
    )
    _add_support(spec, scene)
    for obj in scene.objects:
        _add_object(spec, obj)
    return spec


def _add_support(spec: Any, scene: SceneSpec) -> None:
    """The support: a slab of its extent and colour to see, and its plane to stand on.
    The plane, not the slab, collides: MuJoCo's contacts between a box and a convex
    mesh put a flat-bottomed object on a line across its bottom face, where it
    balanced, tipped and fell through the table. So, unlike Isaac Lab's slab, the
    support reaches past its outline: what is pushed off its edge stays at its
    height."""
    extent = support_extent(scene)
    below = make_transform(np.eye(3), [0.0, 0.0, -SUPPORT_THICKNESS / 2])
    pos, quat = matrix_to_pos_quat(scene.T_base_support @ below)
    body = spec.worldbody.add_body(name="support", pos=pos, quat=quat)
    rgba = SUPPORT_RGBA if scene.support_color is None else (*scene.support_color, 1.0)
    body.add_geom(
        name="support_slab",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[extent[0] / 2, extent[1] / 2, SUPPORT_THICKNESS / 2],
        rgba=list(rgba),
        contype=0,
        conaffinity=0,
        group=VISUAL_GROUP,
    )
    body.add_geom(
        name="support",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, SUPPORT_THICKNESS / 2],
        size=[extent[0] / 2, extent[1] / 2, 0.01],
        friction=[DEFAULT_FRICTION, SPIN_FRICTION, ROLL_FRICTION],
        group=COLLISION_GROUP,
    )


def _add_object(spec: Any, obj: ObjectSpec) -> None:
    """An object's bodies: its root link free at ``T_base_obj``, its other links
    under it at their joints' zero."""
    root, links, joints = _read_urdf(Path(obj.asset_path))
    friction = DEFAULT_FRICTION if obj.friction is None else float(obj.friction)
    pos, quat = matrix_to_pos_quat(np.asarray(obj.T_base_obj, float))
    bodies = {root: spec.worldbody.add_body(name=obj.name, pos=pos, quat=quat)}
    bodies[root].add_freejoint(name=f"{obj.name}/root")
    for joint in joints:
        pos, quat = matrix_to_pos_quat(joint.origin)
        body = bodies[joint.parent].add_body(
            name=f"{obj.name}/{joint.child}", pos=pos, quat=quat
        )
        dynamics = obj.dynamics(joint.name)
        body.add_joint(
            name=f"{obj.name}/{joint.name}",
            type=JOINT_KINDS[joint.kind],
            axis=joint.axis / np.linalg.norm(joint.axis),
            range=list(joint.limits),
            limited=True,
            damping=dynamics.damping,
            frictionloss=dynamics.friction,
            solref_friction=np.array(FRICTION_SOLREF),
            solimp_friction=np.array(FRICTION_SOLIMP),
            stiffness=dynamics.stiffness,
            springref=dynamics.rest,
        )
        bodies[joint.child] = body
    for name, link in links.items():
        _add_link(spec, bodies[name], f"{obj.name}/{name}", link, friction)


def _add_link(spec: Any, body: Any, label: str, link: _Link, friction: float) -> None:
    body.explicitinertial = True
    body.mass = link.mass
    body.ipos = link.com
    I = link.inertia
    body.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    for k, (path, scale, T) in enumerate(link.visuals):
        mesh = trimesh.load(path, force="mesh")
        assert isinstance(mesh, trimesh.Trimesh), f"{path} is not a single mesh"
        mesh.apply_transform(np.diag([*scale, 1.0]))
        mesh.apply_transform(T)
        add_mesh(
            spec,
            body,
            f"{label}/visual{k}",
            VisualMesh(mesh, base_color_texture(path)),
            contype=0,
            conaffinity=0,
            group=VISUAL_GROUP,
        )
    for k, (path, scale, T) in enumerate(link.collisions):
        name = f"{label}/collision{k}"
        spec.add_mesh(name=name, file=str(path.resolve()), scale=list(scale))
        pos, quat = matrix_to_pos_quat(T)
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=name,
            pos=pos,
            quat=quat,
            group=COLLISION_GROUP,
            friction=[friction, SPIN_FRICTION, ROLL_FRICTION],
            rgba=[0.5, 0.5, 0.5, 1.0],
        )


# --------------------------------------------------------------------- session
def _address(model: Any, joint: str) -> tuple[int, int]:
    """A joint's qpos and dof addresses."""
    j = model.joint(joint)
    return int(j.qposadr[0]), int(j.dofadr[0])


def _range(model: Any, joint: str) -> tuple[float, float]:
    lo, hi = model.joint(joint).range
    return float(lo), float(hi)


class Session:
    """A scene running in MuJoCo (:func:`scene_spec`): the model and data, the objects
    by name (their free joints and joints), the robot's arm and gripper when given,
    and a renderer of calibrated cameras (renders up to ``max_size``); physics steps
    ``timestep`` (:func:`scene_spec`).

    Every object starts where the scene has it, an articulated one with its joints as
    the scene has them; the robot at the scene's recorded arm joints, gripper open.
    """

    def __init__(
        self,
        scene: SceneSpec,
        robot: RobotSpec | None = None,
        max_size: tuple[int, int] = MAX_SIZE,
        timestep: float = TIMESTEP,
    ) -> None:
        spec = scene_spec(scene, robot, timestep)
        add_camera(spec, max_size)
        self.scene, self.robot = scene, robot
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        m = self.model
        self.objects = {obj.name: obj for obj in scene.objects}
        # Each object's free joint's (qpos, dof) addresses, and each articulated
        # object's joints' (qpos, dof, range).
        self._root = {name: _address(m, f"{name}/root") for name in self.objects}
        self._joints = {
            name: {
                joint: (*_address(m, f"{name}/{joint}"), _range(m, f"{name}/{joint}"))
                for joint in obj.joints
            }
            for name, obj in self.objects.items()
            if obj.joints
        }
        if robot is not None:
            self.arm_qadr = np.array(
                [m.joint(j).qposadr[0] for j in robot.arm_joints], int
            )
            self.arm_dofadr = np.array(
                [m.joint(j).dofadr[0] for j in robot.arm_joints], int
            )
            by_joint = {
                int(m.actuator_trnid[a, 0]): a
                for a in range(m.nu)
                if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            }
            self.arm_actuators = np.array(
                [by_joint[m.joint(j).id] for j in robot.arm_joints], int
            )
            self.gripper_actuator = m.actuator(robot.gripper.actuator).id
            self.gripper = GripperPoser(m, robot)
            self.gripper_qadr = np.array(
                [m.joint(j).qposadr[0] for j in self.gripper.joints], int
            )
            self.gripper_dofadr = np.array(
                [m.joint(j).dofadr[0] for j in self.gripper.joints], int
            )
            self.set_robot(scene.joint_positions, 0.0)
        for name, obj in self.objects.items():
            self.place(name, obj.T_base_obj)
            if obj.joints:
                self.set_joints(name, obj.joints)
        self.camera = CameraRenderer(m, self.data, max_size)

    # ------------------------------------------------------------------ robot
    def set_robot(self, q: NDArray, level: float) -> None:
        """The robot at arm joints ``q``, gripper opening ``level`` (0 open, 1
        closed), at rest, its actuators holding it there."""
        assert self.robot is not None, "the session has no robot"
        d = self.data
        d.qpos[self.arm_qadr] = q
        d.qvel[self.arm_dofadr] = 0.0
        d.qpos[self.gripper_qadr] = self.gripper.positions(level)
        d.qvel[self.gripper_dofadr] = 0.0
        self.command(q, level)
        mujoco.mj_forward(self.model, d)

    def command(self, q: NDArray, level: float) -> None:
        """The arm's actuators' targets ``q`` (within its joints' ranges) and the
        gripper's opening ``level``."""
        assert self.robot is not None, "the session has no robot"
        lo, hi = self.model.actuator_ctrlrange[self.arm_actuators].T
        self.data.ctrl[self.arm_actuators] = np.clip(q, lo, hi)
        self.data.ctrl[self.gripper_actuator] = self.robot.gripper.ctrl_at(level)

    def arm_q(self) -> NDArray[np.float64]:
        """The arm's joints."""
        return self.data.qpos[self.arm_qadr].copy()

    def gripper_level(self) -> float:
        """The gripper's opening, 0 open to 1 closed, from its driver joint."""
        assert self.robot is not None, "the session has no robot"
        q = self.data.qpos[self.model.joint(self.robot.gripper.driver).qposadr[0]]
        return self.robot.gripper.level_of(float(q))

    # ---------------------------------------------------------------- physics
    def step(self, n: int = 1, hold_robot: bool = False) -> None:
        """``n`` physics steps; ``hold_robot`` keeps the robot exactly where it is,
        as if kinematic (what the objects meet when they settle)."""
        held = None
        if hold_robot and self.robot is not None:
            held = (
                self.data.qpos[self.arm_qadr].copy(),
                self.data.qpos[self.gripper_qadr].copy(),
            )
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if held is not None:
                self.data.qpos[self.arm_qadr], self.data.qpos[self.gripper_qadr] = held
                self.data.qvel[self.arm_dofadr] = 0.0
                self.data.qvel[self.gripper_dofadr] = 0.0
        if held is not None:
            mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- objects
    def place(
        self, name: str, T_base_obj: NDArray, velocity: NDArray | None = None
    ) -> None:
        """Object ``name`` at ``T_base_obj`` moving at ``velocity`` (linear, angular;
        the base frame; at rest if not given). An articulated one's joints stay as
        they are."""
        qadr, dadr = self._root[name]
        pos, quat = matrix_to_pos_quat(np.asarray(T_base_obj, float))
        self.data.qpos[qadr : qadr + 7] = [*pos, *quat]
        v = np.zeros(6) if velocity is None else np.asarray(velocity, float)
        # A free joint's velocity: linear in the world frame, angular in the body's.
        R = pos_quat_to_matrix(pos, quat)[:3, :3]
        self.data.qvel[dadr : dadr + 6] = [*v[:3], *(R.T @ v[3:])]
        mujoco.mj_forward(self.model, self.data)

    def set_joints(
        self,
        name: str,
        positions: dict[str, float],
        velocities: dict[str, float] | None = None,
    ) -> None:
        """Articulated object ``name``'s joints at ``positions`` (as they are, for
        those it leaves out), moving at ``velocities`` (at rest, for those it leaves
        out)."""
        joints = self._joints[name]
        unknown = set(positions) - set(joints)
        if unknown:
            raise ValueError(f"{name} has no joint {', '.join(sorted(unknown))}")
        for joint, (jq, jd, _) in joints.items():
            if joint in positions:
                self.data.qpos[jq] = positions[joint]
            self.data.qvel[jd] = (velocities or {}).get(joint, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def object_poses(self) -> dict[str, NDArray[np.float64]]:
        """Every object's ``T_base_obj``."""
        out = {}
        for name, (qadr, _) in self._root.items():
            q = self.data.qpos[qadr : qadr + 7]
            out[name] = pos_quat_to_matrix(q[:3], q[3:])
        return out

    def object_velocities(self) -> dict[str, NDArray[np.float64]]:
        """Every object's velocity (linear, angular; the base frame)."""
        out = {}
        poses = self.object_poses()
        for name, (_, dadr) in self._root.items():
            v = self.data.qvel[dadr : dadr + 6]
            out[name] = np.r_[v[:3], poses[name][:3, :3] @ v[3:]]
        return out

    def object_joints(self, within_limits: bool = True) -> dict[str, dict[str, float]]:
        """Every articulated object's joint positions: ``within_limits``, or as far
        past them as the limits' soft constraints let them be."""
        return {
            name: {
                joint: float(
                    np.clip(self.data.qpos[jq], lo, hi)
                    if within_limits
                    else self.data.qpos[jq]
                )
                for joint, (jq, _, (lo, hi)) in joints.items()
            }
            for name, joints in self._joints.items()
        }

    def object_joint_velocities(self) -> dict[str, dict[str, float]]:
        """Every articulated object's joint velocities."""
        return {
            name: {joint: float(self.data.qvel[jd]) for joint, (_, jd, _) in j.items()}
            for name, j in self._joints.items()
        }

    # -------------------------------------------------------------- rendering
    def render(
        self, K: NDArray, width: int, height: int, T_base_cam: NDArray
    ) -> dict[str, NDArray]:
        """``rgb`` (H, W, 3), planar ``depth`` (H, W, m) and ``geom`` ids of a
        calibrated pinhole camera (OpenCV axes) at ``T_base_cam``."""
        return self.camera.render(K, width, height, T_base_cam)

    def close(self) -> None:
        """Free the GL contexts."""
        self.camera.close()
