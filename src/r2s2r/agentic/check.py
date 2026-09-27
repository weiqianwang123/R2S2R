"""A quick look at a scene from the recorded views: every object rendered (MuJoCo, no
robot) where the scene puts it, against the real frames and their robot-free depth.

The Isaac Lab replay (:mod:`r2s2r.agentic.isaac`) is the full check; this one takes
seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from r2s2r.agentic.objects import assemble
from r2s2r.agentic.workspace import Workspace
from r2s2r.reconstruct.render import SceneRenderer
from r2s2r.sim.compare import (
    comparison_panel,
    depth_residuals,
    residual_image,
    thumbnail,
)
from r2s2r.structs import SCENE_FILENAME, SceneSpec


def check(
    ws: Workspace, source: str | Path, frame_ids: list[str], out_dir: str | Path
) -> dict[str, Any]:
    """Panels (real with the objects' outlines | render | blend | depth residual) per
    frame, a contact sheet, and the depth residual per object (``check.json``).

    ``source`` is a scene directory, or an objects file (assembled without collision
    first).
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
            renders = [renderer.render(view, i) for i in range(len(scene.objects))]
            masks = {obj.name: r["mask"] for obj, r in zip(scene.objects, renders)}
            sim = renders[0]["rgb"]
            panel = comparison_panel(view.image, sim, masks)
            numbers = depth_residuals(
                view.depth.astype(np.float32), renders[0]["depth"], masks
            )
            panel = np.hstack([panel, residual_image(view.depth, renders[0]["depth"])])
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
        "scene": str(
            (source if source.is_dir() else out_dir / "scene") / SCENE_FILENAME
        ),
        "sheet": str(out_dir / "sheet.png"),
        "frames": rows,
    }
    (out_dir / "check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
