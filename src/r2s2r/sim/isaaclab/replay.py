"""Replay a capture's robot trajectory in a reconstructed scene, in Isaac Lab.

Import after ``isaaclab.app.AppLauncher`` has started (``scripts/isaaclab/replay.py``).

Two modes. ``geometry`` (the default) checks the reconstruction against every view: over
the static period only, the robot is set to the recorded joints and gripper at each
frame and the objects are held where the scene places them. ``physics`` replays the
whole recording: the arm follows the recorded joints (position targets on the same stiff
actuators the policies drive), the gripper the recorded opening, and the objects are
simulated, after coming to rest.

At every step the capture has frames for, each of its cameras renders RGB and depth: a
static camera where it is calibrated, a moving (wrist) camera at the frame's recorded
pose. The objects' poses are logged at those steps.

:func:`settle` lets a scene's objects come to rest, the robot held as it was at the
start of the static period.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.sim.isaac import SETTLE_SECONDS
from r2s2r.sim.isaaclab.scene import (
    PANDA_JOINTS,
    build_scene_cfg,
    camera_cfg,
    centered_render_size,
    make_scene,
)
from r2s2r.structs import CameraSpec, Capture, SceneSpec, write_depth
from r2s2r.transforms import matrix_to_pos_quat, pos_quat_to_matrix

# Joints that open and close each embodiment's gripper, and whether the joint's upper
# limit is the open end.
GRIPPER_JOINTS = {
    "franka_panda": (["panda_finger_joint1", "panda_finger_joint2"], True),
    "droid_franka": (["finger_joint"], False),  # Robotiq 2F-85's driven joint
}


@dataclass
class ReplayConfig:
    """Knobs of :func:`replay`."""

    mode: str = "geometry"  # or "physics"
    physics_dt: float = 0.005
    settle: float = 0.5  # seconds for the objects to come to rest before the replay
    cameras: tuple[str, ...] | None = None  # roles or serials; None: every camera
    every: int = 1  # render every n-th step that has frames
    render_passes: int = 2  # renders after moving a camera, so the image catches up
    device: str = "cuda:0"


class _Session:
    """A scene in Isaac Lab with the capture's robot, set to or driven along its
    recorded states."""

    def __init__(
        self,
        spec: SceneSpec,
        capture: Capture,
        physics_dt: float,
        device: str,
        cameras: dict[str, CameraSpec],
        kinematic_objects: bool,
    ) -> None:
        traj = capture.trajectory
        if traj is None:
            raise ValueError(
                f"capture {capture.name} has no robot trajectory: record it again "
                "(capture droid / mujoco-capture write trajectory.npz)"
            )
        self.traj = traj
        self.sim = SimulationContext(SimulationCfg(dt=physics_dt, device=device))
        scene_cfg = build_scene_cfg(
            spec, with_cameras=False, kinematic_objects=kinematic_objects
        )
        for cam in cameras.values():
            frame0 = next(f for f in capture.frames if f.camera == cam.serial)
            setattr(scene_cfg, _key(cam), camera_cfg(cam, frame0.T_base_cam))
        self.scene = make_scene(scene_cfg, capture.embodiment)
        self.sim.reset()
        self.dt = self.sim.get_physics_dt()
        self.robot = self.scene["robot"]
        self.arm_ids, _ = self.robot.find_joints(PANDA_JOINTS, preserve_order=True)
        grip_names, self.open_high = GRIPPER_JOINTS[capture.embodiment]
        self.grip_ids, _ = self.robot.find_joints(grip_names, preserve_order=True)
        self.limits = (
            self.robot.data.soft_joint_pos_limits[0, self.grip_ids].cpu().numpy()
        )
        self.names = {f"object_{i}": obj.name for i, obj in enumerate(spec.objects)}
        self.index = {int(s): i for i, s in enumerate(traj.steps.tolist())}

    def targets(self, i: int) -> torch.Tensor:
        """Joint targets of trajectory row ``i``."""
        target = self.robot.data.default_joint_pos.clone()
        target[0, self.arm_ids] = target.new_tensor(self.traj.joint_positions[i])
        closed = float(np.clip(self.traj.gripper_position[i], 0.0, 1.0))
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        grip = hi - closed * (hi - lo) if self.open_high else lo + closed * (hi - lo)
        target[0, self.grip_ids] = target.new_tensor(grip)
        return target

    def step_physics(self, n: int) -> None:
        """``n`` physics steps."""
        for _ in range(n):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.dt)

    def set_state(self, i: int) -> None:
        """The robot at trajectory row ``i``, and held there."""
        state = self.targets(i)
        self.robot.write_joint_state_to_sim(state, torch.zeros_like(state))
        self.robot.set_joint_position_target(state)

    def object_poses(self) -> dict[str, NDArray[np.float64]]:
        """Every object's pose in the robot base frame."""
        origin = self.scene.env_origins[0]
        out = {}
        for key, obj in self.scene.rigid_objects.items():
            pos = (obj.data.root_pos_w[0] - origin).cpu().numpy()
            quat = obj.data.root_quat_w[0].cpu().numpy()
            out[self.names[key]] = pos_quat_to_matrix(pos, quat)
        return out


def replay(
    spec: SceneSpec,
    capture: Capture,
    out_dir: str | Path,
    config: ReplayConfig | None = None,
) -> dict[str, Any]:
    """Replay ``capture``'s robot in ``spec`` and render its cameras; the renders go to
    ``out_dir/frames``, and the returned log (also ``out_dir/replay.json`` via the
    script) says where, with the objects' poses at every rendered step."""
    cfg = config or ReplayConfig()
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    if cfg.mode not in ("geometry", "physics"):
        raise ValueError(f"unknown replay mode {cfg.mode!r} (geometry, physics)")
    geometry = cfg.mode == "geometry"
    cameras = {s: capture.cameras[s] for s in capture.resolve_cameras(cfg.cameras)}
    frame_steps = sorted(
        {
            f.step
            for f in capture.frames
            if f.camera in cameras and (not geometry or capture.in_static(f.step))
        }
    )
    render_steps = set(frame_steps[:: max(1, cfg.every)])
    session = _Session(spec, capture, cfg.physics_dt, cfg.device, cameras, geometry)
    traj = session.traj
    log: dict[str, Any] = {
        "scene": spec.name,
        "capture": str(capture.root),
        "embodiment": capture.embodiment,
        "mode": cfg.mode,
        "frames": [],
        "objects": {name: [] for name in session.names.values()},
        "arm_error_rad": [],
    }

    def record(i: int, step: int) -> None:
        measured = session.robot.data.joint_pos[0, session.arm_ids].cpu().numpy()
        log["arm_error_rad"].append(
            [step, float(np.abs(measured - traj.joint_positions[i]).max())]
        )
        for name, T in session.object_poses().items():
            pos, quat = matrix_to_pos_quat(T)
            log["objects"][name].append([step, *pos.tolist(), *quat.tolist()])
        log["frames"] += _render(
            session.sim, session.scene, capture, cameras, step, out_dir, cfg
        )

    if geometry:
        # The robot at each frame's recorded state; the objects stay where placed.
        for step in sorted(render_steps):
            session.set_state(session.index[step])
            session.step_physics(1)
            record(session.index[step], step)
        return log

    # Start in the first recorded state, and let the objects come to rest.
    session.set_state(0)
    session.step_physics(int(round(cfg.settle / session.dt)))
    for i, step in enumerate(traj.steps.tolist()):
        session.robot.set_joint_position_target(session.targets(i))
        if i > 0:
            n = max(1, int(round((traj.times[i] - traj.times[i - 1]) / session.dt)))
            session.step_physics(n)
        if step in render_steps:
            record(i, step)
    return log


@dataclass
class SettleConfig:
    """Knobs of :func:`settle`."""

    seconds: float = SETTLE_SECONDS
    physics_dt: float = 0.005
    device: str = "cuda:0"


def settle(
    spec: SceneSpec, capture: Capture, config: SettleConfig | None = None
) -> tuple[SceneSpec, dict[str, Any]]:
    """Let the objects come to rest under gravity, the robot held at its state at the
    start of the static period; the scene with the objects where they came to rest, and
    how far each moved."""
    cfg = config or SettleConfig()
    session = _Session(spec, capture, cfg.physics_dt, cfg.device, {}, False)
    start = capture.static_steps[0]
    row = session.index.get(start, 0)
    session.set_state(row)
    before = session.object_poses()
    session.step_physics(int(round(cfg.seconds / session.dt)))
    after = session.object_poses()
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
        objects.append(replace(obj, T_base_obj=T1))
    settled = replace(
        spec,
        objects=objects,
        provenance={**spec.provenance, "settle": report},
    )
    return settled, report


def _key(cam: CameraSpec) -> str:
    return f"camera_{cam.role}"


def _render(
    sim: SimulationContext,
    scene: InteractiveScene,
    capture: Capture,
    cameras: dict[str, CameraSpec],
    step: int,
    out_dir: Path,
    cfg: ReplayConfig,
) -> list[dict[str, Any]]:
    """Render every camera with a frame at ``step``; moving cameras go to the frame's
    recorded pose first."""
    frames = {
        f.camera: f for f in capture.frames if f.step == step and f.camera in cameras
    }
    origin = scene.env_origins[0]
    for serial, frame in frames.items():
        cam = cameras[serial]
        if not cam.is_static:
            pos, quat = matrix_to_pos_quat(frame.T_base_cam)
            scene[_key(cam)].set_world_poses(
                positions=origin.new_tensor(pos)[None] + origin[None],
                orientations=origin.new_tensor(quat)[None],
                convention="ros",
            )
    for _ in range(cfg.render_passes):
        sim.render()
    scene.update(sim.get_physics_dt())
    out = []
    for serial, frame in frames.items():
        cam = cameras[serial]
        _, _, x0, y0 = centered_render_size(cam)
        window = (slice(y0, y0 + cam.height), slice(x0, x0 + cam.width))
        data = scene[_key(cam)].data.output
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
