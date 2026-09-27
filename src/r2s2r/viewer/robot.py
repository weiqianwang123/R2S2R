"""The capture's robot for the viewer: its visual meshes as one GLB (a node per body, in
the body's frame) and every body's pose at every recorded step.

It is the MuJoCo model the pipeline uses to cut the robot out of depth (for DROID, the
Panda with the Robotiq 2F-85 on the flange), posed with the recorded joints and gripper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.mjrender import mujoco
from r2s2r.robots.mujoco_models import ARM_JOINTS, GripperPoser, robot_spec
from r2s2r.structs import Capture

VISUAL_GROUP = 2  # menagerie's visual geoms


def robot_glb(embodiment: str) -> tuple[bytes, list[str]]:
    """The robot's visual meshes as a GLB, one node per body (``b<k>`` for the k-th of
    the bodies' names returned)."""
    model = robot_spec(embodiment).compile()
    parts: dict[int, list[trimesh.Trimesh]] = {}
    for g in range(model.ngeom):
        if model.geom_group[g] != VISUAL_GROUP:
            continue
        mesh = _geom_mesh(model, g)
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
    model = robot_spec(capture.embodiment).compile()
    data = mujoco.MjData(model)
    gripper = GripperPoser(model, capture.embodiment)
    arm = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    ids = [model.body(name).id for name in bodies]
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
        data.qpos[:] = 0.0
        data.qpos[arm] = q
        gripper.set(data, float(g))
        mujoco.mj_kinematics(model, data)
        row = []
        for b in ids:
            w, x, y, z = data.xquat[b]
            row.append(
                [
                    *np.round(data.xpos[b], 5).tolist(),
                    *np.round([x, y, z, w], 5).tolist(),
                ]
            )
        poses.append(row)
    return {
        "bodies": bodies,
        "steps": steps,
        "times": [round(t, 4) for t in times],
        "gripper": np.round(np.asarray(grips, float), 3).tolist(),
        "poses": poses,
    }


def _geom_mesh(model: Any, g: int) -> trimesh.Trimesh | None:
    """A mesh geom in its body's frame, coloured by its material."""
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        return None
    i = model.geom_dataid[g]
    v0, nv = model.mesh_vertadr[i], model.mesh_vertnum[i]
    f0, nf = model.mesh_faceadr[i], model.mesh_facenum[i]
    vertices = np.asarray(model.mesh_vert[v0 : v0 + nv], float)
    faces = np.asarray(model.mesh_face[f0 : f0 + nf], int)
    T = _geom_transform(model, g)
    vertices = vertices @ T[:3, :3].T + T[:3, 3]
    mat = model.geom_matid[g]
    rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
    color = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(  # type: ignore[no-untyped-call]
        mesh, vertex_colors=np.tile(color, (nv, 1))
    )
    return mesh


def _geom_transform(model: Any, g: int) -> NDArray[np.float64]:
    T = np.eye(4)
    w, x, y, z = model.geom_quat[g]
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = model.geom_pos[g]
    return T
