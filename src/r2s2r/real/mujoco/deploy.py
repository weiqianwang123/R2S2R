"""Deploy a program policy to the MuJoCo world and score it with ground truth."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from r2s2r.policy.pick import object_points, pick_up
from r2s2r.policy.scoring import save_rollout, score_lift
from r2s2r.real.mujoco.world import (
    MujocoRobot,
    world_config_of,
    world_from_capture,
)
from r2s2r.structs import Capture, ObjectSpec, SceneSpec
from r2s2r.transforms import make_transform
from r2s2r.video import VideoRecorder


def oracle_scene(capture: Capture, out_dir: str | Path) -> SceneSpec:
    """The scene with ground-truth meshes and poses, to test a policy on its own.

    Each GSO model gets a minimal URDF so it looks like any backend's output.
    """
    world = world_from_capture(capture)
    world.reset()
    asset_dir = Path(out_dir) / "oracle_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    for obj in world.cfg.objects:
        mesh = Path(world.cfg.gso_dir) / obj.model / "model.obj"
        urdf = asset_dir / f"{obj.name}.urdf"
        urdf.write_text(
            f'<robot name="{obj.name}"><link name="base">'
            f'<inertial><mass value="{obj.mass}"/>'
            '<inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/>'
            "</inertial>"
            f'<visual><geometry><mesh filename="{mesh}"/></geometry></visual>'
            f'<collision><geometry><mesh filename="{mesh}"/></geometry></collision>'
            "</link></robot>",
            encoding="utf-8",
        )
        objects.append(
            ObjectSpec(
                name=obj.name,
                category=obj.name,
                asset_path=str(urdf),
                T_base_obj=world.object_pose(obj.name),
                mass=obj.mass,
            )
        )
    tx, ty = world.cfg.table_center
    scene = SceneSpec(
        name=f"{capture.name}_oracle",
        embodiment=capture.embodiment,
        objects=objects,
        T_base_support=make_transform(np.eye(3), [tx, ty, 0.0]),
        cameras=capture.cameras,
        reference_camera=next(iter(capture.cameras)),
        reference_step=0,
        joint_positions=np.asarray(world.cfg.q_start),
        provenance={"backend": "oracle"},
        support_extent=(float(world.cfg.table_size[0]), float(world.cfg.table_size[1])),
    )
    world.close()
    return scene


def _box(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (lo + hi) / 2, hi - lo


def scene_errors(scene: SceneSpec, capture: Capture) -> dict[str, dict[str, Any]]:
    """Match reconstructed objects to ground truth by nearest centre and compare axis-
    aligned boxes in the base frame (centre offset, extents)."""
    with tempfile.TemporaryDirectory() as tmp:
        truth = oracle_scene(capture, tmp)
        gt = {o.name: _box(object_points(o)) for o in truth.objects}
    out: dict[str, dict[str, Any]] = {}
    for obj in scene.objects:
        center, size = _box(object_points(obj))
        dist = {n: float(np.linalg.norm(box[0] - center)) for n, box in gt.items()}
        name = min(dist, key=dist.__getitem__)
        offset = center - gt[name][0]
        out[obj.name] = {
            "ground_truth": name,
            "center_offset_m": offset.round(4).tolist(),
            "center_error_m": round(float(np.linalg.norm(offset)), 4),
            "size_m": size.round(4).tolist(),
            "ground_truth_size_m": gt[name][1].round(4).tolist(),
        }
    return out


def match_target(scene: SceneSpec, capture: Capture) -> str:
    """The reconstructed object that is the world's task target (``cfg.target``).

    Naming the object a task refers to is the policy writer's job (an agent reading the
    instruction and the scene's object list); in the MuJoCo harness the ground truth
    stands in for it, and the policy still acts on the reconstruction only.
    """
    target = world_config_of(capture).target
    matches = [
        n
        for n, e in scene_errors(scene, capture).items()
        if e["ground_truth"] == target
    ]
    if not matches:
        raise KeyError(f"no reconstructed object matches the target {target!r}")
    return matches[0]


def run_pick(
    capture: Capture,
    scene: SceneSpec,
    target: str,
    out_dir: str | Path,
    video_camera: str | None = "ext1",
    video_every: int = 2,
) -> dict[str, Any]:
    """Pick ``target`` (an object of ``scene``) in the world ``capture`` came from, and
    score it against the world's ground-truth target."""
    out_dir = Path(out_dir)
    world = world_from_capture(capture)
    world.reset()
    names = [o.name for o in world.cfg.objects]

    def positions() -> dict[str, np.ndarray]:
        return {n: world.object_pose(n)[:3, 3].copy() for n in names}

    cam = video_camera or ""
    video = None
    if cam:
        fps = 1 / (MujocoRobot.control_dt * video_every)
        video = VideoRecorder(out_dir / f"mujoco_{cam}.mp4", fps)

    def record(robot: MujocoRobot) -> None:
        if video is not None and robot.steps % video_every == 0:
            video.add(world.render(cam)[0])

    before = positions()
    robot = MujocoRobot(world, on_step=record)
    policy = pick_up(robot, scene, target)
    result = {
        "deployment": "mujoco",
        "scene": scene.name,
        "scene_backend": scene.provenance.get("backend"),
        "policy": policy,
        "ground_truth_target": world.cfg.target,
        **score_lift(before, positions(), world.cfg.target),
    }
    save_rollout(out_dir, result, robot.log)
    if video is not None:
        video.close()
    world.close()
    return result
