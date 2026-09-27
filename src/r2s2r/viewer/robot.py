"""The capture's robot for the viewer: its visual meshes as one GLB (a node per body, in
the body's frame) and every body's pose at every recorded step.

It is the robot's MuJoCo model (:mod:`r2s2r.robots`), the one the pipeline cuts the
robot out of depth with, posed with the recorded joints and gripper opening.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from r2s2r.mjrender import geom_mesh
from r2s2r.robots import get_robot
from r2s2r.robots.model import RobotModel
from r2s2r.structs import Capture
from r2s2r.transforms import quat_wxyz_to_xyzw

VISUAL_GROUP = 2  # every robot spec's visual geoms


def robot_glb(embodiment: str) -> tuple[bytes, list[str]]:
    """The robot's visual meshes as a GLB, one node per body (``b<k>`` for the k-th of
    the bodies' names returned)."""
    model = get_robot(embodiment).mjcf().compile()
    parts: dict[int, list[trimesh.Trimesh]] = {}
    for g in range(model.ngeom):
        if model.geom_group[g] != VISUAL_GROUP:
            continue
        mesh = geom_mesh(model, g)
        if mesh is not None:
            parts.setdefault(int(model.geom_bodyid[g]), []).append(mesh)
    scene: Any = trimesh.Scene()
    names: list[str] = []
    for body, meshes in sorted(parts.items()):
        name = model.body(body).name or f"body{body}"
        merged = trimesh.util.concatenate(meshes)
        node = f"b{len(names)}"
        scene.add_geometry(merged, node_name=node, geom_name=node)
        names.append(name)
    return scene.export(file_type="glb"), names


def robot_poses(capture: Capture, bodies: list[str]) -> dict[str, Any]:
    """Every body's pose (x, y, z, qx, qy, qz, qw; base frame) at every step of the
    capture's trajectory (or of its frames, without one)."""
    robot = RobotModel(get_robot(capture.embodiment))
    ids = [robot.model.body(name).id for name in bodies]
    traj = capture.trajectory
    if traj is not None:
        steps = traj.steps.tolist()
        times = traj.times.tolist()
        joints, grips = traj.joint_positions, traj.gripper_position
    else:
        frames = sorted(
            {f.step: f for f in capture.frames}.values(), key=lambda f: f.step
        )
        steps = [f.step for f in frames]
        times = [float(s) for s in steps]
        joints = np.array([f.joint_positions for f in frames])
        grips = np.array([f.gripper_position for f in frames])
    poses = []
    for q, g in zip(joints, grips):
        robot.set(q, float(g))
        xpos, xquat = robot.data.xpos, robot.data.xquat
        poses.append(
            [
                [
                    *np.round(xpos[b], 5).tolist(),
                    *np.round(quat_wxyz_to_xyzw(xquat[b]), 5).tolist(),
                ]
                for b in ids
            ]
        )
    return {
        "bodies": bodies,
        "steps": steps,
        "times": [round(t, 4) for t in times],
        "gripper": np.round(np.asarray(grips, float), 3).tolist(),
        "poses": poses,
    }
