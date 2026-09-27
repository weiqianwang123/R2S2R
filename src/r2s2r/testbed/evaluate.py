"""Score reconstructed scenes against a testbed capture's ground truth.

Only the capture's metadata is read (``ground_truth``, written by
:mod:`r2s2r.testbed.record`): no world is rebuilt, so scoring needs none of the world's
assets. Each reconstructed object is matched to the ground-truth object
whose centre is nearest, and their axis-aligned boxes in the base frame are compared
(centre offset, sizes); the support plane's height is compared where the objects stand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from r2s2r.assets import object_points
from r2s2r.structs import Capture, ObjectSpec, SceneSpec


def ground_truth(capture: Capture) -> dict[str, Any]:
    """The capture's ground truth, poses and boxes as arrays."""
    truth = capture.metadata.get("ground_truth")
    if truth is None:
        raise ValueError(
            f"capture {capture.name} has no ground truth: score against the original "
            "testbed capture (a run's copy drops it)"
        )
    objects = {
        name: {
            **obj,
            "T_base_obj": np.asarray(obj["T_base_obj"], float),
            "center": np.asarray(obj["center"], float),
            "size": np.asarray(obj["size"], float),
        }
        for name, obj in truth["objects"].items()
    }
    support = {
        **truth["support"],
        "T_base_support": np.asarray(truth["support"]["T_base_support"], float),
    }
    return {**truth, "objects": objects, "support": support}


def object_box(obj: ObjectSpec) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Centre and size of a scene object's axis-aligned box in the base frame."""
    points = object_points(obj)
    lo, hi = points.min(axis=0), points.max(axis=0)
    return (lo + hi) / 2, hi - lo


def scene_errors(scene: SceneSpec, truth: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per scene object: the ground-truth object it matches, the centre offset and
    its size against the truth's."""
    out: dict[str, dict[str, Any]] = {}
    names = list(truth["objects"])
    centres = np.array([truth["objects"][n]["center"] for n in names])
    for obj in scene.objects:
        centre, size = object_box(obj)
        name = names[int(np.argmin(np.linalg.norm(centres - centre, axis=1)))]
        gt = truth["objects"][name]
        offset = centre - gt["center"]
        out[obj.name] = {
            "ground_truth": name,
            "center_offset_m": offset.round(4).tolist(),
            "center_error_m": round(float(np.linalg.norm(offset)), 4),
            "size_m": size.round(4).tolist(),
            "ground_truth_size_m": gt["size"].round(4).tolist(),
        }
    return out


def support_error(scene: SceneSpec, truth: dict[str, Any]) -> dict[str, Any]:
    """The scene's support plane against the true support: its height where the
    objects stand (their centres' mean) and its tilt."""
    T = scene.T_base_support
    normal, origin = T[:3, 2], T[:3, 3]
    at = np.mean([o["center"][:2] for o in truth["objects"].values()], axis=0)
    height = origin[2] - normal[:2] @ (at - origin[:2]) / normal[2]
    true_normal = truth["support"]["T_base_support"][:3, 2]
    tilt = np.degrees(np.arccos(np.clip(abs(normal @ true_normal), -1.0, 1.0)))
    return {
        "height_error_m": round(float(height - truth["support"]["height"]), 4),
        "tilt_deg": round(float(tilt), 2),
        "extent_m": (
            None if scene.support_extent is None else list(scene.support_extent)
        ),
        "ground_truth_size_m": np.round(truth["support"]["size"], 3).tolist(),
    }


def evaluate(scene: SceneSpec, capture: Capture) -> dict[str, Any]:
    """Objects (and ground-truth objects no scene object matched) and support."""
    truth = ground_truth(capture)
    objects = scene_errors(scene, truth)
    matched = {e["ground_truth"] for e in objects.values()}
    return {
        "scene": scene.name,
        "objects": objects,
        "missed": sorted(set(truth["objects"]) - matched),
        "support": support_error(scene, truth),
    }


def summary(report: dict[str, Any]) -> list[str]:
    """Readable lines of an :func:`evaluate` report."""
    lines = []
    for name, e in report["objects"].items():
        lines.append(
            f"  {name} ~ {e['ground_truth']}: centre off by "
            f"{e['center_error_m'] * 100:.1f} cm {e['center_offset_m']}, size "
            f"{e['size_m']} (true {e['ground_truth_size_m']})"
        )
    if report["missed"]:
        lines.append(f"  missed: {', '.join(report['missed'])}")
    s = report["support"]
    lines.append(
        f"  support: height off by {s['height_error_m'] * 100:+.1f} cm, tilted "
        f"{s['tilt_deg']:.1f} deg, extent {s['extent_m']} "
        f"(true {s['ground_truth_size_m']})"
    )
    return lines
