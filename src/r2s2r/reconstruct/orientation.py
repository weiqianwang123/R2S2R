"""Check every object's turn about the support normal against all the views.

A single-frame pose fit can leave an object turned by a multiple of 90 degrees when
its shape is nearly symmetric under that turn: a box's sides look alike in depth, and
a generated texture may have lost what told them apart. For each object this module

1. renders each candidate turn (0, 90, 180, 270 degrees about the support normal
   through the object's footprint centre) into every calibrated depth view, with the
   other objects and the support in place, and scores it by the median depth residual
   over the pixels any candidate covers;
2. keeps the candidates the geometry cannot separate from the best one. Most objects
   end here, since their shape alone rules out every other turn;
3. asks a VLM to pick among the rest. It sees the photo and a render of each
   remaining candidate from the same camera, and looks for features that fix the
   orientation, such as handles, openings, printed text or labels. The backend's
   pose is kept unless the VLM is confident.

Nothing here depends on the kind of object.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from r2s2r.assets import urdf_visual_points
from r2s2r.reconstruct.render import SceneRenderer
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec
from r2s2r.transforms import make_transform

VLM = Callable[[str, list[Path]], str]

PROMPT = """Image 1 is a photo that shows a {name}.
Images 2 to {last} show a 3D model of that object, rendered from the camera that took
the photo, each time turned a different way about the vertical axis.
Which of images 2 to {last} shows the object turned the same way as in the photo?
Compare features that fix an object's orientation, such as handles, openings, doors,
spouts, printed text or labels, and on which side of the object they are.
Answer with JSON only: {{"image": <number>, "confident": <true or false>}}."""


@dataclass
class OrientationConfig:
    """Knobs of :func:`check_orientations`."""

    turns_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    residual_clip: float = 0.05  # metres; bounds each pixel's depth residual
    # Candidates with residual <= keep_ratio * best + keep_margin stay undecided.
    keep_ratio: float = 1.25
    keep_margin: float = 0.003
    crop_margin: float = 0.25  # of the object's box, around the VLM's image crops
    max_size: tuple[int, int] = (2560, 1600)


def turned(obj: ObjectSpec, T_base_support: NDArray, degrees: float) -> NDArray:
    """``T_base_obj`` turned about the support normal through the footprint centre."""
    pts = urdf_visual_points(obj.asset_path, 500)
    pts = pts @ obj.T_base_obj[:3, :3].T + obj.T_base_obj[:3, 3]
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
    normal = T_base_support[:3, 2]
    R = Rotation.from_rotvec(np.deg2rad(degrees) * normal).as_matrix()
    turn = make_transform(R, centre - R @ centre)
    return turn @ obj.T_base_obj


def check_orientations(
    scene: SceneSpec,
    views: list[DepthView],
    vlm: VLM | None = None,
    config: OrientationConfig | None = None,
    image_dir: str | Path | None = None,
) -> tuple[SceneSpec, dict[str, Any]]:
    """Turn objects whose views favour another orientation; see the module doc.

    ``image_dir`` keeps the images shown to the VLM (``<object>/*.png``).
    """
    cfg = config or OrientationConfig()
    renderer = SceneRenderer(scene, cfg.max_size)
    objects = list(scene.objects)
    report: dict[str, Any] = {}
    try:
        for i, obj in enumerate(scene.objects):
            for k, other in enumerate(objects):
                renderer.pose(k, other.T_base_obj)
            poses = {t: turned(obj, scene.T_base_support, t) for t in cfg.turns_deg}
            residual = _residuals(renderer, i, poses, views, cfg)
            best = min(residual.values())
            kept = [
                t
                for t in cfg.turns_deg
                if residual[t] <= cfg.keep_ratio * best + cfg.keep_margin
            ]
            entry: dict[str, Any] = {
                "residual_mm": {
                    str(int(t)): round(r * 1000, 1) for t, r in residual.items()
                },
                "candidates": kept,
            }
            if len(kept) == 1:
                choice, entry["decided_by"] = kept[0], "geometry"
            elif vlm is not None:
                keep = None if image_dir is None else Path(image_dir) / obj.name
                choice, answer = _ask_vlm(
                    renderer, i, obj, poses, kept, views, vlm, cfg, keep
                )
                entry["decided_by"], entry["vlm_answer"] = "vlm", answer
            else:
                choice = 0.0 if 0.0 in kept else kept[0]
                entry["decided_by"] = "undecided (no VLM)"
            entry["turn_deg"] = choice
            if choice:
                objects[i] = replace(obj, T_base_obj=poses[choice])
            report[obj.name] = entry
    finally:
        renderer.close()
    return replace(scene, objects=objects), report


def _residuals(
    renderer: SceneRenderer,
    index: int,
    poses: dict[float, NDArray],
    views: list[DepthView],
    cfg: OrientationConfig,
) -> dict[float, float]:
    """Median clipped depth residual per candidate, over the pixels any covers."""
    per_view = []
    for view in views:
        renders = {}
        for turn, T in poses.items():
            renderer.pose(index, T)
            renders[turn] = renderer.render(view, index)
        union = np.logical_or.reduce([r["mask"] for r in renders.values()])
        valid = union & (view.depth > 0)
        if valid.sum() < 50:
            continue
        per_view.append(
            {
                turn: np.minimum(
                    np.abs(view.depth[valid] - r["depth"][valid]), cfg.residual_clip
                )
                for turn, r in renders.items()
            }
        )
    renderer.pose(index, poses[0.0] if 0.0 in poses else next(iter(poses.values())))
    if not per_view:
        return {turn: 0.0 for turn in poses}
    return {
        turn: float(np.median(np.concatenate([v[turn] for v in per_view])))
        for turn in poses
    }


def _ask_vlm(
    renderer: SceneRenderer,
    index: int,
    obj: ObjectSpec,
    poses: dict[float, NDArray],
    kept: list[float],
    views: list[DepthView],
    vlm: VLM,
    cfg: OrientationConfig,
    keep_dir: Path | None,
) -> tuple[float, str]:
    """Show the VLM the photo and each kept candidate, from the view that sees the
    object best.

    Returns the chosen turn (the backend's when unsure) and the answer.
    """
    fallback = 0.0 if 0.0 in kept else kept[0]
    with_image = [v for v in views if v.image is not None]
    if not with_image:
        return fallback, "no image"
    for k in range(len(renderer.bodies)):  # the object alone, over nothing
        renderer.pose(k, None)
    renders = {}
    best_view, best_area = None, 0
    for view in with_image:
        renderer.pose(index, poses[kept[0]])
        area = int(renderer.render(view, index)["mask"].sum())
        if area > best_area:
            best_view, best_area = view, area
    if best_view is None:
        return fallback, "object not in view"
    for turn in kept:
        renderer.pose(index, poses[turn])
        renders[turn] = renderer.render(best_view, index)
    union = np.logical_or.reduce([r["mask"] for r in renders.values()])
    ys, xs = np.nonzero(union)
    pad_y = int(cfg.crop_margin * (ys.max() - ys.min() + 1))
    pad_x = int(cfg.crop_margin * (xs.max() - xs.min() + 1))
    h, w = union.shape
    box = (
        slice(max(0, ys.min() - pad_y), min(h, ys.max() + pad_y + 1)),
        slice(max(0, xs.min() - pad_x), min(w, xs.max() + pad_x + 1)),
    )
    with tempfile.TemporaryDirectory(prefix="r2s2r_orientation_") as tmp:
        out_dir = keep_dir or Path(tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [out_dir / "photo.png"]
        assert best_view.image is not None
        cv2.imwrite(
            str(paths[0]), cv2.cvtColor(best_view.image[box], cv2.COLOR_RGB2BGR)
        )
        for n, turn in enumerate(kept):
            rgb = np.where(renders[turn]["mask"][..., None], renders[turn]["rgb"], 255)
            paths.append(out_dir / f"candidate_{n + 1}_turn{int(turn)}.png")
            cv2.imwrite(
                str(paths[-1]),
                cv2.cvtColor(rgb[box].astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
        name = obj.category.replace("_", " ")
        answer = vlm(PROMPT.format(name=name, last=len(paths)), paths)
    parsed = _parse_answer(answer)
    if parsed is None or not parsed[1] or not 2 <= parsed[0] <= len(kept) + 1:
        return fallback, answer
    return kept[parsed[0] - 2], answer


def _parse_answer(answer: str) -> tuple[int, bool] | None:
    match = re.search(r"\{.*\}", answer, re.S)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
        return int(data["image"]), bool(data.get("confident", False))
    except (ValueError, KeyError, TypeError):
        return None
