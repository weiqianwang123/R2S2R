"""Tools on images: segmenting frames with SAM3, reading the masks, cropping an object
for generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.paths import ENV_SIMFOUNDRY
from r2s2r.sim.compare import COLORS
from r2s2r.tools.envjobs import run_env_job
from r2s2r.workspace import Workspace


def slug(text: str) -> str:
    """A file-name-safe version of ``text``."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "object"


def load_mask(path: str | Path, shape: tuple[int, int]) -> NDArray[np.bool_]:
    """A binary mask image, at ``shape`` (H, W)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"cannot read mask {path}")
    if raw.ndim == 3:
        raw = raw[..., -1] if raw.shape[2] == 4 else raw.max(axis=2)
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(raw > 127)


def segment(
    ws: Workspace,
    frame_ids: list[str],
    out_dir: str | Path,
    texts: list[str] | None = None,
    points: list[tuple[float, float, int]] | None = None,
    box: tuple[float, float, float, float] | None = None,
    name: str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Masks of what ``texts`` describe in every frame (every instance found), or of the
    one thing ``points`` / ``box`` pick out (one frame).

    Writes ``out_dir/<name>/<frame>_<k>.png`` (best score first), an overlay per frame
    numbering the instances, and ``masks.json``; returns what ``masks.json`` holds.
    Each text has its own ``name`` (by default from the text), so one ``name`` is for
    one text.
    """
    out_dir = Path(out_dir).resolve()
    requests: list[dict[str, Any]] = []
    if texts and (points or box):
        raise ValueError("give text prompts, or points / a box, not both")
    if texts and name and len(texts) > 1:
        raise ValueError("a name is for one text: segment each text on its own")
    if texts and len({slug(t) for t in texts}) != len(texts):
        raise ValueError(f"the texts {texts} would share an output directory")
    if texts:
        for text in texts:
            for fid in frame_ids:
                frame = ws.frame(fid)
                requests.append(
                    {
                        "image": str(ws.image_path(frame)),
                        "out": str(out_dir / slug(name or text) / fid),
                        "text": text,
                        "frame": fid,
                        "name": slug(name or text),
                    }
                )
    else:
        if points is None and box is None:
            raise ValueError("give text prompts, or points / a box")
        if len(frame_ids) != 1:
            raise ValueError("point and box prompts are for one frame at a time")
        fid = frame_ids[0]
        label = slug(name or "picked")
        requests.append(
            {
                "image": str(ws.image_path(ws.frame(fid))),
                "out": str(out_dir / label / fid),
                "points": [list(p) for p in points] if points else None,
                "box": list(box) if box else None,
                "frame": fid,
                "name": label,
            }
        )
    result = run_env_job(
        "sam3_job.py",
        {"threshold": threshold, "requests": requests},
        ws.root / "cache" / "jobs",
        ENV_SIMFOUNDRY,
    )
    summary: dict[str, Any] = {}
    for row in result["results"]:
        req = row["request"]
        frame = ws.frame(req["frame"])
        robot = ws.robot_mask(frame)
        instances = []
        for inst in row["instances"]:
            mask = load_mask(inst["mask"], (robot.shape[0], robot.shape[1]))
            instances.append(
                {
                    **inst,
                    "mask": str(Path(inst["mask"]).relative_to(out_dir)),
                    "on_robot": round(
                        float((mask & robot).sum() / max(1, mask.sum())), 3
                    ),
                }
            )
        overlay = out_dir / req["name"] / f"{req['frame']}_overlay.png"
        _write_overlay(ws.image(frame), row["instances"], overlay)
        entry = summary.setdefault(
            req["name"], {"prompt": req.get("text"), "frames": {}}
        )
        entry["frames"][req["frame"]] = {
            "overlay": str(overlay.relative_to(out_dir)),
            "instances": instances,
        }
    for name_, entry in summary.items():
        path = out_dir / name_ / "masks.json"
        old = json.loads(path.read_text()) if path.exists() else {"frames": {}}
        old["prompt"] = entry["prompt"]
        old["frames"].update(entry["frames"])
        path.write_text(json.dumps(old, indent=1), encoding="utf-8")
    return summary


def _write_overlay(
    image: NDArray[np.uint8], instances: list[dict[str, Any]], path: Path
) -> None:
    out = image.copy()
    tint = image.copy()
    for k, inst in enumerate(instances):
        mask = load_mask(inst["mask"], (image.shape[0], image.shape[1]))
        color = COLORS[k % len(COLORS)]
        tint[mask] = color
    out = np.asarray(cv2.addWeighted(out, 0.55, tint, 0.45, 0), np.uint8)
    for k, inst in enumerate(instances):
        mask = load_mask(inst["mask"], (image.shape[0], image.shape[1]))
        color = COLORS[k % len(COLORS)]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(out, contours, -1, color, 2)
        x0, y0 = inst["box"][:2]
        text = f"{k} ({inst['score']:.2f})"
        cv2.putText(
            out,
            text,
            (x0, max(14, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (x0, max(14, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def crop(
    ws: Workspace,
    frame_id: str,
    mask_path: str | Path,
    out_path: str | Path,
    pad: float = 0.15,
    size: int = 1024,
) -> dict[str, Any]:
    """The masked object cut out of the frame, square, centred, on a transparent
    background (RGBA PNG), for single-image generation."""
    frame = ws.frame(frame_id)
    image = ws.image(frame)
    mask = load_mask(mask_path, (image.shape[0], image.shape[1]))
    if not mask.any():
        raise ValueError(f"mask {mask_path} is empty")
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * pad) + 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    n = int(round(side))
    rgba = np.zeros((n, n, 4), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(image.shape[1], x0 + n), min(image.shape[0], y0 + n)
    rgba[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0, :3] = image[sy0:sy1, sx0:sx1]
    rgba[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0, 3] = (
        mask[sy0:sy1, sx0:sx1].astype(np.uint8) * 255
    )
    touches = bool(
        xs.min() == 0
        or ys.min() == 0
        or xs.max() == image.shape[1] - 1
        or ys.max() == image.shape[0] - 1
    )
    out = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_LANCZOS4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))
    return {
        "image": str(out_path),
        "object_pixels": int(mask.sum()),
        "source_side_px": n,
        "upscale": round(size / n, 2),
        "touches_image_border": touches,
        "on_robot": round(float((mask & ws.robot_mask(frame)).sum() / mask.sum()), 3),
    }
