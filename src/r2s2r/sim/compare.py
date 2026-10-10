"""Compare a replay with the recording it replays: images for a person or an agent to
look at, and numbers.

For every rendered frame of a replay (either simulator's;
:func:`~r2s2r.sim.world.write_render`): the real image, the sim render and a blend side
by side, with each object's outline where the replay has it drawn on the real image.
Where the real frame has depth, the depth residual too, over the whole image and per
object. Per camera, a contact sheet over time (at most
``SHEET_ROWS`` frames, evenly spaced); per object, how far it moved from where it
started.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.structs import Capture, DepthView, SceneSpec, read_depth, read_rgb
from r2s2r.transforms import pos_quat_to_matrix

RESIDUAL_CLIP = 0.05  # metres; the residual image saturates here
THUMB_HEIGHT = 180  # pixels, of a contact sheet's rows
SHEET_ROWS = 24  # a contact sheet's rows at most
# Object outlines (RGB), one colour per object in turn.
COLORS = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 200, 0),
    (200, 64, 255),
    (0, 220, 220),
    (255, 128, 0),
    (160, 160, 160),
]


def compare_replay(
    log: dict[str, Any],
    spec: SceneSpec,
    capture: Capture,
    out_dir: str | Path,
    replay_dir: str | Path,
) -> dict[str, Any]:
    """Write the comparison images under ``out_dir`` and return the numbers.

    The real images and depth come from ``capture``.
    """
    # MuJoCo renders the outlines; imported here so the module loads without it.
    # pylint: disable=import-outside-toplevel
    from r2s2r.mjrender import SceneRenderer

    out_dir, replay_dir = Path(out_dir), Path(replay_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    names = [obj.name for obj in spec.objects]
    poses = {
        name: {int(row[0]): pos_quat_to_matrix(row[1:4], row[4:8]) for row in rows}
        for name, rows in log["objects"].items()
    }
    size = (
        max(c.width for c in capture.cameras.values()),
        max(c.height for c in capture.cameras.values()),
    )
    renderer = SceneRenderer(spec, size)
    summary: dict[str, Any] = {"frames": [], "motion_m": {}}
    # Per camera, the frames its contact sheet shows.
    by_role: dict[str, list[int]] = {}
    for i, entry in enumerate(log["frames"]):
        by_role.setdefault(entry["role"], []).append(i)
    on_sheet: set[int] = set()
    for rows in by_role.values():
        picks = np.linspace(0, len(rows) - 1, min(SHEET_ROWS, len(rows))).round()
        on_sheet.update(rows[int(k)] for k in picks)
    sheets: dict[str, list[NDArray[np.uint8]]] = {}
    try:
        for i, entry in enumerate(log["frames"]):
            step, serial = int(entry["step"]), entry["camera"]
            frame = capture.frame(serial, step)
            cam = capture.cameras[serial]
            real = read_rgb(capture.root / frame.left_image)
            sim = read_rgb(replay_dir / entry["sim_rgb"])
            view = DepthView(
                serial, step, np.zeros((cam.height, cam.width)), cam.K, frame.T_base_cam
            )
            for k, name in enumerate(names):
                renderer.pose(k, poses[name].get(step))
            objects = renderer.render(view)["object"]
            masks = {name: objects == k for k, name in enumerate(names)}
            panel = comparison_panel(real, sim, masks)
            row: dict[str, Any] = {"step": step, "camera": entry["role"]}
            if frame.depth_image is not None:
                real_depth = read_depth(capture.root / frame.depth_image)
                sim_depth = read_depth(replay_dir / entry["sim_depth"])
                row.update(depth_residuals(real_depth, sim_depth, masks))
                panel = np.hstack([panel, residual_image(real_depth, sim_depth)])
            path = out_dir / "frames" / f"{step:04d}_{entry['role']}.png"
            cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            row["image"] = str(path.relative_to(out_dir))
            summary["frames"].append(row)
            if i in on_sheet:
                sheets.setdefault(entry["role"], []).append(thumbnail(panel, step))
    finally:
        renderer.close()
    for role, thumbs in sheets.items():
        cv2.imwrite(
            str(out_dir / f"sheet_{role}.png"),
            cv2.cvtColor(np.vstack(thumbs), cv2.COLOR_RGB2BGR),
        )
    for name, rows in log["objects"].items():
        start = np.asarray(rows[0][1:4])
        summary["motion_m"][name] = [
            [int(r[0]), round(float(np.linalg.norm(np.asarray(r[1:4]) - start)), 4)]
            for r in rows
        ]
    summary["arm_error_rad_max"] = max(
        (e for _, e in log["arm_error_rad"]), default=0.0
    )
    (out_dir / "compare.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    return summary


def comparison_panel(
    real: NDArray[np.uint8], sim: NDArray[np.uint8], masks: dict[str, NDArray[np.bool_]]
) -> NDArray[np.uint8]:
    """Real image with the objects' outlines | sim render | 50/50 blend."""
    outlined = real.copy()
    for i, (name, mask) in enumerate(masks.items()):
        if not mask.any():
            continue
        color = COLORS[i % len(COLORS)]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(outlined, contours, -1, color, 2)
        ys, xs = np.nonzero(mask)
        cv2.putText(
            outlined,
            name,
            (int(xs.min()), max(12, int(ys.min()) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    blend = ((real.astype(np.uint16) + sim.astype(np.uint16)) // 2).astype(np.uint8)
    return np.hstack([outlined, sim, blend])


def depth_residuals(
    real: NDArray, sim: NDArray, masks: dict[str, NDArray[np.bool_]]
) -> dict[str, Any]:
    """Median absolute depth residual (m) over the image and over each object."""
    valid = (real > 0) & (sim > 0)
    residual = np.abs(sim - real)
    per_object = {
        name: round(float(np.median(residual[mask & valid])), 4)
        for name, mask in masks.items()
        if (mask & valid).sum() >= 20
    }
    return {
        "depth_residual_m": (
            round(float(np.median(residual[valid])), 4) if valid.any() else None
        ),
        "object_depth_residual_m": per_object,
    }


def residual_image(real: NDArray, sim: NDArray) -> NDArray[np.uint8]:
    """|sim - real| depth, black at 0 and white at RESIDUAL_CLIP; blue where unknown."""
    valid = (real > 0) & (sim > 0)
    grey = (np.clip(np.abs(sim - real) / RESIDUAL_CLIP, 0, 1) * 255).astype(np.uint8)
    image = np.dstack([grey] * 3)
    image[~valid] = (0, 0, 96)
    return image


def thumbnail(panel: NDArray[np.uint8], step: int) -> NDArray[np.uint8]:
    """``panel`` scaled to :data:`THUMB_HEIGHT`, labelled with the step."""
    scale = THUMB_HEIGHT / panel.shape[0]
    thumb = cv2.resize(panel, (int(panel.shape[1] * scale), THUMB_HEIGHT))
    cv2.putText(
        thumb,
        f"step {step}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return np.asarray(thumb, np.uint8)
