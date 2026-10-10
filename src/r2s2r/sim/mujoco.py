"""MuJoCo, run in this process: settling a scene and replaying a capture's recording
in it to compare every view, as :mod:`r2s2r.sim.isaac` does in Isaac Lab, taking the
same paths and leaving the same files (:mod:`r2s2r.sim.world`); the scene is built and
run by :mod:`r2s2r.sim.mjscene`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from r2s2r.robots import get_robot
from r2s2r.sim.compare import compare_replay
from r2s2r.sim.mjscene import Session
from r2s2r.sim.world import (
    SETTLE_SECONDS,
    frames_at,
    recorded_trajectory,
    replay_steps,
    replay_summary,
    settled,
    start_row,
    trajectory_rows,
    write_render,
)
from r2s2r.structs import Capture, SceneSpec
from r2s2r.transforms import matrix_to_pos_quat


def settle(
    scene_dir: str | Path,
    capture_dir: str | Path,
    out_dir: str | Path,
    seconds: float = SETTLE_SECONDS,
) -> dict[str, Any]:
    """The scene with its objects where they come to rest (``out_dir/scene.json``, the
    robot held as the capture has it at the start of the static period; the report in
    ``out_dir/settle.json``), and how far each moved. An object's Isaac USD, made for
    where it was, is dropped: Isaac converts it again."""
    out_dir = Path(out_dir).resolve()
    spec = SceneSpec.load(scene_dir)
    capture = Capture.load(capture_dir)
    session = held_session(spec, capture)
    try:
        before, joints_before = session.object_poses(), session.object_joints()
        session.step(int(round(seconds / session.dt)), hold_robot=True)
        after, joints_after = session.object_poses(), session.object_joints()
    finally:
        session.close()
    scene, report = settled(
        spec, capture, seconds, (before, after), (joints_before, joints_after)
    )
    scene = replace(scene, objects=[replace(o, usd=None) for o in scene.objects])
    scene.save(out_dir)
    (out_dir / "settle.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return {"scene": str(out_dir / "scene.json"), **report}


def held_session(spec: SceneSpec, capture: Capture) -> Session:
    """A session of ``spec`` with the capture's robot as it was at the start of the
    static period (what the objects come to rest against, held there by
    ``Session.step(hold_robot=True)``)."""
    traj, i = recorded_trajectory(capture), start_row(capture)
    session = Session(spec, get_robot(capture.embodiment))
    session.set_robot(traj.joint_positions[i], _level(traj.gripper_position[i]))
    return session


def replay(
    scene_dir: str | Path,
    capture_dir: str | Path,
    out_dir: str | Path,
    cameras: list[str] | None = None,
    every: int = 1,
) -> dict[str, Any]:
    """Replay the capture's static period in the scene (the robot at every recorded
    state, the objects where the scene puts them), render every camera, and compare
    with the real frames (``out_dir/compare/``, :func:`~r2s2r.sim.compare.
    compare_replay`; the renders under ``out_dir/frames/``, the log in
    ``out_dir/replay.json``). Returns the median depth residuals per camera and per
    object."""
    out_dir = Path(out_dir).resolve()
    spec = SceneSpec.load(scene_dir)
    capture = Capture.load(capture_dir)
    traj = recorded_trajectory(capture)
    rows = trajectory_rows(traj)
    chosen = {s: capture.cameras[s] for s in capture.resolve_cameras(cameras)}
    log: dict[str, Any] = {
        "scene": spec.name,
        "capture": str(capture.root),
        "embodiment": capture.embodiment,
        "frames": [],
        "objects": {obj.name: [] for obj in spec.objects},
        "arm_error_rad": [],
    }
    session = Session(spec, get_robot(capture.embodiment))
    try:
        for step in replay_steps(capture, chosen, every):
            i = rows[step]
            q = traj.joint_positions[i]
            session.set_robot(q, _level(traj.gripper_position[i]))
            log["arm_error_rad"].append(
                [step, float(np.abs(session.arm_q() - q).max())]
            )
            for name, T in session.object_poses().items():
                pos, quat = matrix_to_pos_quat(T)
                log["objects"][name].append([step, *pos.tolist(), *quat.tolist()])
            for serial, frame in frames_at(capture, chosen, step).items():
                cam = chosen[serial]
                out = session.render(cam.K, cam.width, cam.height, frame.T_base_cam)
                log["frames"].append(
                    write_render(capture, cam, frame, out["rgb"], out["depth"], out_dir)
                )
    finally:
        session.close()
    (out_dir / "replay.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    compare_replay(log, spec, capture, out_dir / "compare", out_dir)
    return replay_summary(out_dir)


def _level(gripper: ArrayLike) -> NDArray:
    """A recorded gripper position (one opening per arm) within 0 to 1."""
    return np.clip(gripper, 0.0, 1.0)
