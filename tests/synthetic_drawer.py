"""A synthetic cabinet with a sliding drawer, and depth "videos" of it, for the tests of
joint fitting and calibration (needs GL rendering)."""

import numpy as np
import trimesh

from r2s2r.reconstruct.render import SceneRenderer
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec
from r2s2r.transforms import intrinsics_matrix, make_transform

K = intrinsics_matrix(300.0, 300.0, 159.5, 119.5)
T_CABINET = make_transform(np.eye(3), [0.5, 0.0, 0.0])  # front face at x = 0.6


def boxes(path, parts):
    """Save the union of boxes (extents, centre) as one mesh."""
    meshes = []
    for extents, centre in parts:
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(centre)
        meshes.append(box)
    trimesh.util.concatenate(meshes).export(path)


def cabinet(tmp_path, carcass, drawer, axis="1 0 0", name="cab"):
    """A URDF: ``drawer`` boxes sliding along ``axis`` out of ``carcass`` boxes, with
    limits the backend underestimated."""
    boxes(tmp_path / "carcass.obj", carcass)
    boxes(tmp_path / "drawer.obj", drawer)
    urdf = tmp_path / f"{name}.urdf"
    urdf.write_text(
        f'<robot name="{name}">'
        '<link name="carcass"><visual><geometry><mesh filename="carcass.obj"/>'
        "</geometry></visual><collision><geometry>"
        '<mesh filename="carcass.obj"/></geometry></collision></link>'
        '<link name="drawer"><visual><geometry><mesh filename="drawer.obj"/>'
        "</geometry></visual><collision><geometry>"
        '<mesh filename="drawer.obj"/></geometry></collision></link>'
        '<joint name="slide" type="prismatic"><parent link="carcass"/>'
        f'<child link="drawer"/><axis xyz="{axis}"/>'
        '<limit lower="0" upper="0.02" effort="5" velocity="5"/></joint>'
        "</robot>"
    )
    return urdf


def drawer_cabinet(tmp_path, axis="1 0 0", name="cab"):
    """An open-front carcass (0.2 x 0.24 x 0.16 m) with a drawer box flush at q = 0."""
    carcass = [
        ((0.2, 0.24, 0.01), (0.0, 0.0, 0.005)),  # bottom
        ((0.2, 0.24, 0.01), (0.0, 0.0, 0.155)),  # top
        ((0.2, 0.01, 0.16), (0.0, 0.115, 0.08)),  # sides
        ((0.2, 0.01, 0.16), (0.0, -0.115, 0.08)),
        ((0.01, 0.24, 0.16), (-0.095, 0.0, 0.08)),  # back
    ]
    drawer = [((0.18, 0.2, 0.12), (0.01, 0.0, 0.075))]
    return cabinet(tmp_path, carcass, drawer, axis, name)


def scene_of(urdf):
    """A scene with the cabinet ``urdf`` on the support, its front facing +x."""
    return SceneSpec(
        name="t",
        embodiment="franka_panda",
        objects=[ObjectSpec("cab", "cabinet", str(urdf), T_CABINET, articulated=True)],
        T_base_support=np.eye(4),
        cameras={},
        reference_camera="c",
        reference_step=0,
        joint_positions=np.zeros(7),
    )


def look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float] = (0.6, 0.0, 0.08),
) -> np.ndarray:
    """A camera at ``eye`` looking at ``target`` (OpenCV axes)."""
    at, to = np.asarray(eye, float), np.asarray(target, float)
    fwd = (to - at) / np.linalg.norm(to - at)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return make_transform(np.column_stack([right, np.cross(fwd, right), fwd]), at)


def video(scene, trajectory, cameras):
    """Depth views of ``scene`` with the drawer at ``trajectory[step]``."""
    renderer = SceneRenderer(scene, (320, 240))
    steps = {}
    for step, q in trajectory.items():
        renderer.pose(0, T_CABINET, {"slide": q})
        steps[step] = []
        for T in cameras:
            stub = DepthView("cam", step, np.zeros((240, 320)), K, T)
            depth = renderer.render(stub, 0)["depth"].astype(float)
            depth[depth > 10.0] = 0.0  # the sky: no return
            steps[step].append(DepthView("cam", step, depth, K, T))
    renderer.close()
    return steps


TRUTH = {0: 0.0, 1: 0.0, 2: 0.05, 3: 0.12, 4: 0.12, 5: 0.0}  # open, then close
CAMERAS = [look_at((1.0, 0.3, 0.45)), look_at((0.95, -0.35, 0.3))]
