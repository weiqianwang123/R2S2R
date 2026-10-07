"""Replay a capture's robot in a reconstructed scene, and settle a scene, in Isaac Lab.

Import after ``isaaclab.app.AppLauncher`` has started (``scripts/isaaclab/replay.py``,
``scripts/isaaclab/settle.py``). What every simulator does the same is in
:mod:`r2s2r.sim.world`.

:func:`replay` checks the reconstruction against every view: over the static period, the
robot is set to the recorded joints and gripper opening at each step the capture has
frames for, the objects are held where the scene places them, and each of the capture's
cameras renders RGB and depth, a static camera where it is calibrated and a moving
(wrist) camera at the frame's recorded pose.

:func:`settle` lets a scene's objects come to rest (an articulated object's joints
too), the robot held as it was at the start of the static period, and keeps each
object's USD (and physcoder's ``metadata.yaml``) with the settled scene, and the
support's colour from the capture's frames.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r2s2r.robots import get_robot
from r2s2r.sim.isaaclab.scene import (
    Session,
    camera_key,
    crop_window,
    object_usd,
    with_object_usds,
    write_metadata,
)
from r2s2r.sim.world import (
    SETTLE_SECONDS,
    frames_at,
    replay_steps,
    settled,
    write_render,
)
from r2s2r.structs import CameraSpec, Capture, SceneSpec
from r2s2r.transforms import matrix_to_pos_quat

RENDER_PASSES = 2  # renders after moving a camera, so the image catches up


@dataclass
class ReplayConfig:
    """Knobs of :func:`replay`."""

    cameras: tuple[str, ...] | None = None  # roles or serials; None: every camera
    every: int = 1  # render every n-th step that has frames


class _Recorded(Session):
    """A session with the capture's robot, set to its recorded states."""

    def __init__(
        self,
        spec: SceneSpec,
        capture: Capture,
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
        super().__init__(spec, robot, kinematic_objects, poses)
        self.index = {int(s): i for i, s in enumerate(traj.steps.tolist())}

    def set_state(self, i: int) -> None:
        """The robot at trajectory row ``i``, and held there."""
        state = self.articulation.data.default_joint_pos.clone()
        state[0, self.arm_ids] = state.new_tensor(self.traj.joint_positions[i])
        level = np.clip(self.traj.gripper_position[i], 0.0, 1.0)
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
    cameras = {s: capture.cameras[s] for s in capture.resolve_cameras(cfg.cameras)}
    session = _Recorded(with_object_usds(spec, out_dir), capture, cameras, True)
    log: dict[str, Any] = {
        "scene": spec.name,
        "capture": str(capture.root),
        "embodiment": capture.embodiment,
        "frames": [],
        "objects": {name: [] for name in session.names.values()},
        "arm_error_rad": [],
    }
    for step in replay_steps(capture, cameras, cfg.every):
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


def settle(
    spec: SceneSpec,
    capture: Capture,
    out_dir: str | Path,
    seconds: float = SETTLE_SECONDS,
) -> tuple[SceneSpec, dict[str, Any]]:
    """Let the objects come to rest under gravity for ``seconds``, the robot held at
    its state at the start of the static period (:func:`~r2s2r.sim.world.settled`).

    Each object's USD goes to ``out_dir/objects/<name>/<name>.usd`` with physcoder's
    ``metadata.yaml`` beside it (for the settled pose). Returns the settled scene, with
    the USDs, and how far each object moved.
    """
    out_dir = Path(out_dir).resolve()
    spec = replace(
        spec,
        objects=[
            replace(obj, usd=str(object_usd(obj, out_dir / "objects" / obj.name)))
            for obj in spec.objects
        ],
    )
    session = _Recorded(spec, capture, {}, False)
    session.set_state(session.index.get(capture.static_steps[0], 0))
    before, joints_before = session.object_poses(), session.object_joints()
    session.step_physics(int(round(seconds / session.dt)))
    after, joints_after = session.object_poses(), session.object_joints()
    scene, report = settled(
        spec, capture, seconds, (before, after), (joints_before, joints_after)
    )
    for obj in scene.objects:
        assert obj.usd is not None
        write_metadata(obj.usd, obj.T_base_obj, scene.T_base_support)
    return scene, report


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
    frames = frames_at(capture, cameras, step)
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
        out.append(write_render(capture, cam, frame, rgb, depth, out_dir))
    return out
