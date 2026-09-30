"""Replay a capture's robot in a reconstructed scene, and settle a scene, in Isaac Lab.

Import after ``isaaclab.app.AppLauncher`` has started (``scripts/isaaclab/replay.py``,
``scripts/isaaclab/settle.py``).

:func:`replay` checks the reconstruction against every view: over the static period, the
robot is set to the recorded joints and gripper opening at each step the capture has
frames for, the objects are held where the scene places them, and each of the capture's
cameras renders RGB and depth, a static camera where it is calibrated and a moving
(wrist) camera at the frame's recorded pose.

:func:`settle` lets a scene's objects come to rest (an articulated object's joints
too), the robot held as it was at the start of the static period, and keeps each
object's USD (and physcoder's ``metadata.yaml``) with the settled scene.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from r2s2r.robots import get_robot
from r2s2r.sim.isaac import SETTLE_SECONDS
from r2s2r.sim.isaaclab.scene import (
    Session,
    camera_key,
    crop_window,
    object_usd,
    with_object_usds,
    write_metadata,
)
from r2s2r.structs import CameraSpec, Capture, SceneSpec, write_depth
from r2s2r.transforms import matrix_to_pos_quat

RENDER_PASSES = 2  # renders after moving a camera, so the image catches up


@dataclass
class ReplayConfig:
    """Knobs of :func:`replay`."""

    cameras: tuple[str, ...] | None = None  # roles or serials; None: every camera
    every: int = 1  # render every n-th step that has frames
    device: str = "cuda:0"


class _Recorded(Session):
    """A session with the capture's robot, set to its recorded states."""

    def __init__(
        self,
        spec: SceneSpec,
        capture: Capture,
        device: str,
        cameras: dict[str, CameraSpec],
        kinematic_objects: bool,
    ) -> None:
        traj = capture.trajectory
        if traj is None:
            raise ValueError(
                f"capture {capture.name} has no robot trajectory: record it again "
                "(its trajectory.npz)"
            )
        self.traj = traj
        poses = [
            (cam, next(f for f in capture.frames if f.camera == s).T_base_cam)
            for s, cam in cameras.items()
        ]
        robot = get_robot(capture.embodiment)
        super().__init__(spec, robot, kinematic_objects, device, poses)
        self.index = {int(s): i for i, s in enumerate(traj.steps.tolist())}

    def set_state(self, i: int) -> None:
        """The robot at trajectory row ``i``, and held there."""
        state = self.articulation.data.default_joint_pos.clone()
        state[0, self.arm_ids] = state.new_tensor(self.traj.joint_positions[i])
        level = float(np.clip(self.traj.gripper_position[i], 0.0, 1.0))
        state[0, self.grip_ids] = self.gripper_targets(level)
        self.articulation.write_joint_state_to_sim(state, torch.zeros_like(state))
        self.articulation.set_joint_position_target(state)


def replay(
    spec: SceneSpec,
    capture: Capture,
    out_dir: str | Path,
    config: ReplayConfig | None = None,
) -> dict[str, Any]:
    """Replay ``capture``'s robot in ``spec`` and render its cameras; the renders go to
    ``out_dir/frames``, and the returned log (also ``out_dir/replay.json`` via the
    script) says where, with the objects' poses at every rendered step. Objects without
    a USD are converted into ``out_dir/objects/``."""
    cfg = config or ReplayConfig()
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    cameras = {s: capture.cameras[s] for s in capture.resolve_cameras(cfg.cameras)}
    frame_steps = sorted(
        {
            f.step
            for f in capture.frames
            if f.camera in cameras and capture.in_static(f.step)
        }
    )
    render_steps = frame_steps[:: max(1, cfg.every)]
    session = _Recorded(
        with_object_usds(spec, out_dir), capture, cfg.device, cameras, True
    )
    log: dict[str, Any] = {
        "scene": spec.name,
        "capture": str(capture.root),
        "embodiment": capture.embodiment,
        "frames": [],
        "objects": {name: [] for name in session.names.values()},
        "arm_error_rad": [],
    }
    for step in render_steps:
        i = session.index[step]
        session.set_state(i)
        session.step_physics(1)
        measured = session.articulation.data.joint_pos[0, session.arm_ids].cpu().numpy()
        log["arm_error_rad"].append(
            [step, float(np.abs(measured - session.traj.joint_positions[i]).max())]
        )
        for name, T in session.object_poses().items():
            pos, quat = matrix_to_pos_quat(T)
            log["objects"][name].append([step, *pos.tolist(), *quat.tolist()])
        log["frames"] += _render(session, capture, cameras, step, out_dir)
    return log


@dataclass
class SettleConfig:
    """Knobs of :func:`settle`."""

    seconds: float = SETTLE_SECONDS
    device: str = "cuda:0"


def settle(
    spec: SceneSpec,
    capture: Capture,
    out_dir: str | Path,
    config: SettleConfig | None = None,
) -> tuple[SceneSpec, dict[str, Any]]:
    """Let the objects come to rest under gravity, the robot held at its state at the
    start of the static period.

    Each object's USD goes to ``out_dir/objects/<name>/<name>.usd`` with physcoder's
    ``metadata.yaml`` beside it (for the settled pose). Returns the scene with the
    objects where they came to rest (and their USDs), and how far each moved.
    """
    cfg = config or SettleConfig()
    out_dir = Path(out_dir).resolve()
    spec = replace(
        spec,
        objects=[
            replace(obj, usd=str(object_usd(obj, out_dir / "objects" / obj.name)))
            for obj in spec.objects
        ],
    )
    session = _Recorded(spec, capture, cfg.device, {}, False)
    start = capture.static_steps[0]
    session.set_state(session.index.get(start, 0))
    before, joints_before = session.object_poses(), session.object_joints()
    session.step_physics(int(round(cfg.seconds / session.dt)))
    after, joints_after = session.object_poses(), session.object_joints()
    report: dict[str, Any] = {
        "seconds": cfg.seconds,
        "robot_step": start,
        "objects": {},
    }
    objects = []
    for obj in spec.objects:
        T0, T1 = before[obj.name], after[obj.name]
        turn = Rotation.from_matrix(T0[:3, :3].T @ T1[:3, :3]).magnitude()
        report["objects"][obj.name] = {
            "moved_m": round(float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])), 4),
            "turned_deg": round(float(np.degrees(turn)), 2),
            "dropped_m": round(float(T0[2, 3] - T1[2, 3]), 4),
        }
        joints = joints_after.get(obj.name)
        if joints:  # how far each joint moved (rad or m)
            report["objects"][obj.name]["joints_moved"] = {
                j: round(q - joints_before[obj.name][j], 4) for j, q in joints.items()
            }
        assert obj.usd is not None
        write_metadata(obj.usd, T1, spec.T_base_support)
        objects.append(replace(obj, T_base_obj=T1, joints=joints or None))
    settled = replace(
        spec,
        objects=objects,
        provenance={**spec.provenance, "settle": report},
    )
    return settled, report


def _render(
    session: Session,
    capture: Capture,
    cameras: dict[str, CameraSpec],
    step: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Render every camera with a frame at ``step``; moving cameras go to the frame's
    recorded pose first."""
    scene = session.scene
    frames = {
        f.camera: f for f in capture.frames if f.step == step and f.camera in cameras
    }
    for serial, frame in frames.items():
        cam = cameras[serial]
        if not cam.is_static:
            pos, quat = matrix_to_pos_quat(frame.T_base_cam)
            like = scene.env_origins  # a tensor on the simulation's device
            scene[camera_key(cam)].set_world_poses(
                positions=like.new_tensor(pos)[None],
                orientations=like.new_tensor(quat)[None],
                convention="ros",
            )
    for _ in range(RENDER_PASSES):
        session.sim.render()
    scene.update(session.dt)
    out = []
    for serial, frame in frames.items():
        cam = cameras[serial]
        window = crop_window(cam)
        data = scene[camera_key(cam)].data.output
        rgb = data["rgb"][0, ..., :3].cpu().numpy()[window]
        depth = data["distance_to_image_plane"][0, ..., 0].cpu().numpy()[window]
        depth = np.where(np.isfinite(depth), depth, 0.0)
        stem = f"frames/{step:04d}_{cam.role}"
        cv2.imwrite(
            str(out_dir / f"{stem}_sim.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        write_depth(out_dir / f"{stem}_sim_depth.png", depth)
        out.append(
            {
                "step": step,
                "camera": serial,
                "role": cam.role,
                "sim_rgb": f"{stem}_sim.png",
                "sim_depth": f"{stem}_sim_depth.png",
                "real_rgb": str(capture.root / frame.left_image),
                "real_depth": (
                    None
                    if frame.depth_image is None
                    else str(capture.root / frame.depth_image)
                ),
            }
        )
    return out
