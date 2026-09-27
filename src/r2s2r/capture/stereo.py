"""Metric depth for a capture's stereo frames, from FoundationStereo (SimFoundry's stage
2), so that stereo captures can be compared with renders in depth too.

The capture is copied to ``out_dir`` and every stereo frame in the chosen steps gets a
``depth_image`` at its left image's resolution. The robot is left in the depth: a
comparison renders it as well.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from r2s2r.pipeline.fixed.simfoundry import (
    FRAME_MAP_FILENAME,
    SimFoundryBackend,
    SimFoundryConfig,
)
from r2s2r.structs import Capture, write_depth


def add_stereo_depth(
    capture: Capture,
    out_dir: str | Path,
    config: SimFoundryConfig | None = None,
    static_only: bool = True,
) -> Capture:
    """``capture`` copied to ``out_dir`` with FoundationStereo depth on its stereo
    frames (those of the static period, unless ``static_only`` is off)."""
    out_dir = Path(out_dir)
    if out_dir.resolve() != capture.root.resolve():
        shutil.copytree(capture.root, out_dir, dirs_exist_ok=True)
    work = out_dir / "stereo_depth"
    cfg = replace(config or SimFoundryConfig(), max_frames=10_000, stages=("2",))
    frames = capture.frames
    source = (
        capture
        if static_only
        else replace(
            capture,
            static_steps=(min(f.step for f in frames), max(f.step for f in frames) + 1),
        )
    )
    backend = SimFoundryBackend(cfg)
    backend.prepare_inputs(source, work)
    backend.run(source, work, stages=("2",))
    fs_dir = backend.scene_dir(source, work) / "s2_fs"
    depth_of = {}
    frame_map = backend.scene_dir(source, work) / "s1_zed" / FRAME_MAP_FILENAME
    for m in json.loads(frame_map.read_text(encoding="utf-8")):
        if m["depth"] != "stereo":
            continue
        stem = fs_dir / f"image_{m['index']}"
        raw = Path(f"{stem}_depth_meter_raw.npy")  # before the robot was cut out
        depth_of[(m["camera"], int(m["step"]))] = np.load(
            raw if raw.exists() else f"{stem}_depth_meter.npy"
        )
    new_frames = []
    for frame in frames:
        depth = depth_of.get((frame.camera, frame.step))
        if depth is None:
            new_frames.append(frame)
            continue
        cam = capture.cameras[frame.camera]
        full = cv2.resize(
            depth.astype(np.float32),
            (cam.width, cam.height),
            interpolation=cv2.INTER_NEAREST,
        )
        rel = f"depth/{frame.camera}/{frame.step:04d}.png"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        write_depth(out_dir / rel, full)
        new_frames.append(replace(frame, depth_image=rel))
    shutil.rmtree(work)
    out = replace(
        capture,
        frames=new_frames,
        root=out_dir,
        metadata={**capture.metadata, "depth": "FoundationStereo (stage 2)"},
    )
    out.save()
    return out
