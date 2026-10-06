"""What every simulator does the same with a reconstructed scene, whichever runs it
(Isaac Lab, :mod:`r2s2r.sim.isaac`; MuJoCo, :mod:`r2s2r.sim.mujoco`): gravity along the
support's normal, the support's outline, how long the objects get to come to rest and
how far each moved, the support's colour from the capture, which of the capture's steps
a replay renders, each rendered frame written beside the real one, and a replay's
numbers.

Every pose is in the robot base frame, every simulator's world frame.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.structs import (
    CameraSpec,
    Capture,
    FrameRecord,
    SceneSpec,
    read_depth,
    read_rgb,
    write_depth,
)
from r2s2r.transforms import backproject, invert, transform_points

SETTLE_SECONDS = 2.0  # simulated time for the objects to come to rest
ON_SUPPORT = 0.005  # m: a point this near the support's plane lies on it
SUPPORT_EXTENT = (0.6, 0.6)  # m, for a scene whose support's outline is unknown


def gravity(scene: SceneSpec) -> tuple[float, float, float]:
    """Gravity in the robot base frame, along the support's normal: the table is
    level, while the base (or its estimate) leans a few tenths of a degree, enough for
    a round object to roll away."""
    normal = np.asarray(scene.T_base_support, float)[:3, 2]
    g = -9.81 * normal / np.linalg.norm(normal)
    return float(g[0]), float(g[1]), float(g[2])


def support_extent(scene: SceneSpec) -> tuple[float, float]:
    """The support's outline (m): the scene's, else :data:`SUPPORT_EXTENT`."""
    extent = scene.support_extent or SUPPORT_EXTENT
    return float(extent[0]), float(extent[1])


def support_color(
    capture: Capture,
    T_base_support: NDArray,
    extent: tuple[float, float] | None,
    stride: int = 4,
) -> tuple[float, float, float]:
    """The support's colour (sRGB, 0 to 1): the median, over the static period's
    frames with depth, of the pixels whose depth puts them on its plane (within
    :data:`ON_SUPPORT`) and inside its outline. What stands on it is above the
    plane and left out."""
    to_support = invert(np.asarray(T_base_support, float))
    found = []
    for frame in capture.frames:
        if frame.depth_image is None or not capture.in_static(frame.step):
            continue
        depth = read_depth(capture.root / frame.depth_image)
        image = read_rgb(capture.root / frame.left_image)
        if image.shape[:2] != depth.shape:
            image = np.asarray(
                cv2.resize(image, depth.shape[::-1], interpolation=cv2.INTER_AREA),
                np.uint8,
            )
        keep = np.zeros(depth.shape, bool)
        keep[::stride, ::stride] = True
        keep &= depth > 0
        pts = backproject(np.where(keep, depth, 0.0), capture.cameras[frame.camera].K)
        s = transform_points(to_support @ frame.T_base_cam, pts)
        on = np.abs(s[:, 2]) < ON_SUPPORT
        if extent is not None:
            on &= (np.abs(s[:, 0]) < extent[0] / 2) & (np.abs(s[:, 1]) < extent[1] / 2)
        found.append(image[keep][on])
    pixels = np.concatenate(found) if found else np.zeros((0, 3))
    if len(pixels) < 100:
        raise ValueError("too few pixels on the support for its colour")
    r, g, b = np.median(pixels, axis=0) / 255.0
    return float(r), float(g), float(b)


# ---------------------------------------------------------------------- settling
def settled(
    spec: SceneSpec,
    capture: Capture,
    seconds: float,
    poses: tuple[dict[str, NDArray], dict[str, NDArray]],
    joints: tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]],
) -> tuple[SceneSpec, dict[str, Any]]:
    """The scene with its objects where they came to rest (``poses`` and ``joints``
    before and after ``seconds`` of physics, the robot held as it was at the start of
    the capture's static period) and the support's colour as the capture's depth frames
    show it (:func:`support_color`); and how far each object moved."""
    before, after = poses
    joints_before, joints_after = joints
    report: dict[str, Any] = {
        "seconds": seconds,
        "robot_step": capture.static_steps[0],
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
        moved = joints_after.get(obj.name)
        if moved:  # how far each joint moved (rad or m)
            report["objects"][obj.name]["joints_moved"] = {
                j: round(q - joints_before[obj.name][j], 4) for j, q in moved.items()
            }
        objects.append(replace(obj, T_base_obj=T1, joints=moved or obj.joints))
    try:
        colour: tuple[float, float, float] | None = support_color(
            capture, spec.T_base_support, spec.support_extent
        )
    except ValueError as exc:  # no depth on the support; it stays as it was
        colour = spec.support_color
        report["support_color"] = str(exc)
    scene = replace(
        spec,
        objects=objects,
        provenance={**spec.provenance, "settle": report},
        support_color=colour,
    )
    return scene, report


def settle_summary(report: dict[str, Any]) -> str:
    """One line on how far each object moved and turned while settling."""
    return ", ".join(
        f"{name} moved {r['moved_m']:.3f} m, turned {r['turned_deg']:.1f} deg"
        for name, r in report["objects"].items()
    )


# ----------------------------------------------------------------------- replays
def replay_steps(
    capture: Capture, cameras: dict[str, CameraSpec], every: int
) -> list[int]:
    """The static period's steps that have a frame of one of ``cameras``, every
    ``every``-th of them."""
    steps = sorted(
        {
            f.step
            for f in capture.frames
            if f.camera in cameras and capture.in_static(f.step)
        }
    )
    return steps[:: max(1, every)]


def frames_at(
    capture: Capture, cameras: dict[str, CameraSpec], step: int
) -> dict[str, FrameRecord]:
    """The capture's frames at ``step`` of ``cameras``, by camera."""
    return {
        f.camera: f for f in capture.frames if f.step == step and f.camera in cameras
    }


def write_render(
    capture: Capture,
    cam: CameraSpec,
    frame: FrameRecord,
    rgb: NDArray[np.uint8],
    depth: NDArray,
    out_dir: Path,
) -> dict[str, Any]:
    """Write a camera's render of ``frame``'s step into ``out_dir/frames/``; the
    replay log's entry for it, beside the real frame's files."""
    stem = f"frames/{frame.step:04d}_{cam.role}"
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_sim.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    write_depth(
        out_dir / f"{stem}_sim_depth.png", np.where(np.isfinite(depth), depth, 0.0)
    )
    return {
        "step": frame.step,
        "camera": frame.camera,
        "role": cam.role,
        "sim_rgb": f"{stem}_sim.png",
        "sim_depth": f"{stem}_sim_depth.png",
        "real_rgb": str(capture.root / frame.left_image),
        "real_depth": (
            None if frame.depth_image is None else str(capture.root / frame.depth_image)
        ),
    }


def replay_summary(out_dir: str | Path) -> dict[str, Any]:
    """A replay's numbers from its comparison (``out_dir/compare/compare.json``,
    :func:`~r2s2r.sim.compare.compare_replay`): the median depth residual per camera
    and per object, the contact sheets."""
    out_dir = Path(out_dir)
    compare = json.loads((out_dir / "compare" / "compare.json").read_text())
    per_camera: dict[str, list[float]] = {}
    per_object: dict[str, dict[str, list[float]]] = {}
    for row in compare["frames"]:
        if row.get("depth_residual_m") is not None:
            per_camera.setdefault(row["camera"], []).append(row["depth_residual_m"])
        for name, value in row.get("object_depth_residual_m", {}).items():
            per_object.setdefault(name, {}).setdefault(row["camera"], []).append(value)
    return {
        "compare_dir": str(out_dir / "compare"),
        "sheets": sorted(str(p) for p in (out_dir / "compare").glob("sheet_*.png")),
        "frames": len(compare["frames"]),
        "median_depth_residual_m": {
            cam: round(float(np.median(v)), 4) for cam, v in per_camera.items()
        },
        "object_median_depth_residual_m": {
            name: {cam: round(float(np.median(v)), 4) for cam, v in cams.items()}
            for name, cams in per_object.items()
        },
    }
