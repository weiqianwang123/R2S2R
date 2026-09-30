"""A quick look at a scene from the recorded views: every object rendered (MuJoCo, no
robot) where the scene puts it, against the real frames and their robot-free depth.

The Isaac Lab replay (:mod:`r2s2r.sim.isaac`) is the full check; this one takes
seconds.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from r2s2r.mjrender import SceneRenderer
from r2s2r.sim.compare import (
    comparison_panel,
    depth_residuals,
    residual_image,
    thumbnail,
)
from r2s2r.structs import SCENE_FILENAME, SceneSpec
from r2s2r.tools.objects import assemble
from r2s2r.workspace import Workspace


def check(
    ws: Workspace,
    source: str | Path,
    frame_ids: list[str],
    out_dir: str | Path,
    joints: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Panels (real with the objects' outlines | render | blend | depth residual) per
    frame, a contact sheet, and the depth residual per object (``check.json``).

    ``source`` is a scene directory, or an objects file (assembled without collision
    first). ``joints`` ({object: {joint: position}}) renders articulated objects with
    those joints moved, to see where a part goes (the recording shows it where it was).
    """
    out_dir = Path(out_dir).resolve()
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    source = Path(source)
    if source.is_dir():
        scene = SceneSpec.load(source)
    else:
        assemble(ws, source, out_dir / "scene", collision="none")
        scene = SceneSpec.load(out_dir / "scene")
    if not scene.objects:
        raise ValueError("the scene has no objects to check")
    for name, moved in (joints or {}).items():
        index = next((i for i, o in enumerate(scene.objects) if o.name == name), None)
        if index is None or not scene.objects[index].joints:
            raise ValueError(f"no articulated object {name!r} in the scene")
        obj = scene.objects[index]
        unknown = set(moved) - set(obj.joints or {})
        if unknown:
            raise ValueError(f"{name} has no joints {sorted(unknown)}")
        scene.objects[index] = replace(obj, joints={**(obj.joints or {}), **moved})
    size = (
        max(c.width for c in ws.capture.cameras.values()),
        max(c.height for c in ws.capture.cameras.values()),
    )
    renderer = SceneRenderer(scene, size)
    for i, obj in enumerate(scene.objects):
        renderer.pose(i, obj.T_base_obj)
    rows, thumbs = [], []
    try:
        for fid in frame_ids:
            frame = ws.frame(fid)
            view = ws.depth_view(frame)
            assert view.image is not None
            out = renderer.render(view)
            masks = {
                obj.name: out["object"] == i for i, obj in enumerate(scene.objects)
            }
            panel = comparison_panel(view.image, out["rgb"], masks)
            numbers = depth_residuals(
                view.depth.astype(np.float32), out["depth"], masks
            )
            panel = np.hstack([panel, residual_image(view.depth, out["depth"])])
            path = out_dir / "frames" / f"{fid}.png"
            cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            rows.append({"frame": fid, "image": str(path), **numbers})
            thumbs.append(thumbnail(panel, frame.step))
    finally:
        renderer.close()
    if thumbs:
        # Cameras of different aspect ratios give thumbnails of different widths.
        width = max(t.shape[1] for t in thumbs)
        sheet = np.vstack(
            [np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0))) for t in thumbs]
        )
        cv2.imwrite(str(out_dir / "sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    summary = {
        "joints": joints or {},
        "scene": str(
            (source if source.is_dir() else out_dir / "scene") / SCENE_FILENAME
        ),
        "sheet": str(out_dir / "sheet.png"),
        "frames": rows,
    }
    (out_dir / "check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
