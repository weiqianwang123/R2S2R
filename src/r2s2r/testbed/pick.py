"""The pick test in MuJoCo: the pick program on the world a capture was recorded in,
acting on a reconstructed scene, scored by how far the world's true target rose.

The program only sees the scene; the world's ground truth names the target
(:func:`match_target`: naming what a task refers to is the policy writer's job) and
scores the lift. :func:`oracle_scene` turns the ground truth itself into a scene, to
test the policy (in MuJoCo and in Isaac Lab) with a perfect reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.mjrender import geom_mesh
from r2s2r.structs import Capture, ObjectSpec, SceneSpec
from r2s2r.testbed.evaluate import ground_truth, scene_errors
from r2s2r.testbed.policy import (
    RobotInterface,
    VideoRecorder,
    pick_up,
    save_rollout,
    score_lift,
)
from r2s2r.testbed.worlds import MujocoWorld, world_from_capture
from r2s2r.transforms import quat_wxyz_to_xyzw

# A static ground-truth object (no joint in the world) needs a mass in a scene.
STATIC_MASS = 2.0


class MujocoRobot(RobotInterface):
    """The world's robot behind the policy interface."""

    control_dt = 0.02

    def __init__(
        self,
        world: MujocoWorld,
        on_step: Callable[[MujocoRobot], None] | None = None,
    ) -> None:
        super().__init__(world.spec)
        self.world = world
        self.on_step = on_step
        self.steps = 0

    def joint_positions(self) -> NDArray[np.float64]:
        return self.world.arm_q()

    def gripper_level(self) -> float:
        return self.world.gripper_level()

    def _hold(self, q: NDArray[np.float64], level: float) -> None:
        self.world.hold(q, level)
        self.world.step(self.control_dt)
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self)


def match_target(scene: SceneSpec, capture: Capture) -> str:
    """The scene object that is the world's pick target: of those matched to it, the
    nearest."""
    truth = ground_truth(capture)
    errors = scene_errors(scene, truth)
    matches = [n for n, e in errors.items() if e["ground_truth"] == truth["target"]]
    if not matches:
        raise KeyError(
            f"no object of {scene.name} matches the target {truth['target']}"
        )
    return min(matches, key=lambda n: errors[n]["center_error_m"])


def oracle_scene(world: MujocoWorld, capture: Capture, out_dir: str | Path) -> Path:
    """The capture's ground-truth objects as a scene in ``out_dir``: each a URDF of
    its body's visual and collision geoms in ``world`` (as meshes) and its inertia, at
    its recorded pose, on the recorded support. Returns the scene directory."""
    out_dir = Path(out_dir)
    truth = ground_truth(capture)
    objects = []
    for obj in world.objects:
        spec = truth["objects"][obj.name]
        urdf = _object_urdf(world, obj.body, out_dir / "objects" / obj.name)
        objects.append(
            ObjectSpec(
                name=obj.name,
                category=obj.name,
                asset_path=str(urdf),
                T_base_obj=spec["T_base_obj"],
                mass=STATIC_MASS if spec["static"] else spec["mass"],
                friction=spec["friction"],
            )
        )
    support = truth["support"]
    scene = SceneSpec(
        name=f"{capture.name}_oracle",
        embodiment=capture.embodiment,
        objects=objects,
        T_base_support=support["T_base_support"],
        cameras=capture.cameras,
        reference_camera=next(iter(capture.cameras)),
        reference_step=0,
        joint_positions=np.asarray(world.spec.home_q, float),
        provenance={"method": "oracle"},
        support_extent=(float(support["size"][0]), float(support["size"][1])),
    )
    scene.save(out_dir)
    return out_dir


def _object_urdf(world: MujocoWorld, body: str, out_dir: Path) -> Path:
    """A body as ``<body>.urdf`` in ``out_dir``: one visual mesh, a collision mesh per
    colliding geom, the body's inertia (a static body: :data:`STATIC_MASS` spread over
    its visual box)."""
    m = world.model
    b = m.body(body).id
    visual, colliding = world.geoms(body)
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes = [geom_mesh(m, g) for g in visual]
    mesh = cast(
        trimesh.Trimesh, trimesh.util.concatenate([x for x in meshes if x is not None])
    )
    mesh.export(out_dir / "visual.obj")
    collisions = []
    for i, g in enumerate(colliding):
        part = geom_mesh(m, g)
        if part is None:
            continue
        part.export(out_dir / f"collision_{i}.obj")
        collisions.append(f"collision_{i}.obj")
    if m.body_jntnum[b]:
        mass, com, inertia = float(m.body_mass[b]), m.body_ipos[b], m.body_inertia[b]
        rpy = Rotation.from_quat(quat_wxyz_to_xyzw(m.body_iquat[b])).as_euler("xyz")
    else:
        lo, hi = mesh.bounds
        e2 = (hi - lo) ** 2
        mass, com, rpy = STATIC_MASS, (lo + hi) / 2, np.zeros(3)
        inertia = (
            STATIC_MASS / 12 * np.array([e2[1] + e2[2], e2[0] + e2[2], e2[0] + e2[1]])
        )
    ixx, iyy, izz = (float(v) for v in inertia)
    geometry = "".join(
        f'<collision><geometry><mesh filename="{c}"/></geometry></collision>'
        for c in collisions
    )
    path = out_dir / f"{body}.urdf"
    path.write_text(
        f'<robot name="{body}"><link name="base"><inertial>'
        f'<origin xyz="{_floats(com)}" rpy="{_floats(rpy)}"/><mass value="{mass}"/>'
        f'<inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>'
        '</inertial><visual><geometry><mesh filename="visual.obj"/></geometry>'
        f"</visual>{geometry}</link></robot>",
        encoding="utf-8",
    )
    return path


def _floats(values: Any) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def run_pick(
    capture: Capture,
    scene: SceneSpec,
    target: str,
    out_dir: str | Path,
    video_camera: str | None = "ext1",
    video_every: int = 2,
) -> dict[str, Any]:
    """Pick ``target`` (an object of ``scene``) in the world ``capture`` came from,
    and score by how far the world's own target rose."""
    out_dir = Path(out_dir)
    world = world_from_capture(capture)
    video = None
    if video_camera:
        fps = 1 / (MujocoRobot.control_dt * video_every)
        video = VideoRecorder(out_dir / f"mujoco_{video_camera}.mp4", fps)

    def record(robot: MujocoRobot) -> None:
        if video is not None and robot.steps % video_every == 0:
            video.add(world.render(video_camera or "")["rgb"])

    before = world.object_positions()
    robot = MujocoRobot(world, on_step=record)
    policy = pick_up(robot, scene, target)
    result = {
        "deployment": "mujoco",
        "scene": scene.name,
        "scene_method": scene.provenance.get("method"),
        "policy": policy,
        "ground_truth_target": world.target,
        **score_lift(before, world.object_positions(), world.target),
    }
    save_rollout(out_dir, result, robot.log)
    if video is not None:
        video.close()
    world.close()
    return result
