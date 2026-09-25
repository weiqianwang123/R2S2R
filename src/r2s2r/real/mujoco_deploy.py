"""Run a program policy on the MuJoCo "real" world and score it with ground truth."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from r2s2r.policy.pick import object_points, pick_up
from r2s2r.real.mujoco_world import MujocoRobot, MujocoWorld, world_from_capture
from r2s2r.structs import Capture, ObjectSpec, SceneSpec
from r2s2r.transforms import make_transform
from r2s2r.video import VideoRecorder

LIFT_SUCCESS_M = 0.05


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


def run_pick(
    capture: Capture,
    scene: SceneSpec,
    target: str,
    out_dir: str | Path,
    video_camera: str | None = "ext1",
    video_every: int = 2,
) -> dict[str, Any]:
    """Pick ``target`` (an object of ``scene``) in the world ``capture`` came from."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world: MujocoWorld = world_from_capture(capture)
    world.reset()
    names = [o.name for o in world.cfg.objects]
    before = {n: world.object_pose(n)[:3, 3].copy() for n in names}

    cam = video_camera or ""
    video = None
    if cam:
        fps = 1 / (MujocoRobot.control_dt * video_every)
        video = VideoRecorder(out_dir / f"mujoco_{cam}.mp4", fps)

    def record(robot: MujocoRobot) -> None:
        if video is not None and robot.steps % video_every == 0:
            video.add(world.render(cam)[0])

    robot = MujocoRobot(world, on_step=record)
    policy = pick_up(robot, scene, target)
    after = {n: world.object_pose(n)[:3, 3].copy() for n in names}
    lift = {n: float(after[n][2] - before[n][2]) for n in names}
    result = {
        "deployment": "mujoco",
        "scene": scene.name,
        "scene_backend": scene.provenance.get("backend"),
        "policy": policy,
        "ground_truth_target": world.cfg.target,
        "object_lift_m": lift,
        "object_shift_m": {
            n: float(np.linalg.norm(after[n] - before[n])) for n in names
        },
        "success": bool(lift[world.cfg.target] > LIFT_SUCCESS_M),
        "tcp_final": world.tcp_pose()[:3, 3].tolist(),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "commands.json").write_text(
        json.dumps(robot.log.as_dict()), encoding="utf-8"
    )
    if video is not None:
        video.close()
    world.close()
    return result
