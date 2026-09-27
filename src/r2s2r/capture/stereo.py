"""Metric depth for a capture's stereo frames, from FoundationStereo.

A run works on depth; a stereo camera (DROID's ZED cameras) records image pairs. Every
static-period frame of a stereo camera that has no depth yet gets FoundationStereo's:
SimFoundry's stage-2 backend with its settings (half resolution), run as a job in
SimFoundry's environment (``scripts/tools/stereo_job.py``), scaled back to the left
image's resolution. The robot is left in the depth: a run cuts it out itself.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from r2s2r.paths import ENV_SIMFOUNDRY
from r2s2r.structs import Capture, FrameRecord, write_depth
from r2s2r.tools.envjobs import run_env_job

DEPTH_NOTE = "FoundationStereo, static frames of the stereo cameras"


def stereo_frames(capture: Capture) -> list[FrameRecord]:
    """The static-period frames of stereo cameras that have no depth yet."""
    return [
        f
        for f in capture.frames
        if f.depth_image is None
        and f.right_image is not None
        and capture.cameras[f.camera].stereo_baseline is not None
        and capture.in_static(f.step)
    ]


def add_stereo_depth(capture: Capture, jobs_dir: Path) -> Capture:
    """``capture`` (saved in place) with FoundationStereo depth on its
    :func:`stereo_frames`, as ``depth/<camera>/<step>.png``; the job's files go to
    ``jobs_dir``."""
    frames = stereo_frames(capture)
    if not frames:
        return capture
    work = jobs_dir.resolve() / "stereo_depth"
    pairs = []
    for f in frames:
        cam = capture.cameras[f.camera]
        assert f.right_image is not None and cam.stereo_baseline is not None
        pairs.append(
            {
                "left": str((capture.root / f.left_image).resolve()),
                "right": str((capture.root / f.right_image).resolve()),
                "K": cam.K.tolist(),
                "baseline": cam.stereo_baseline,
            }
        )
    result = run_env_job(
        "stereo_job.py", {"work": str(work), "pairs": pairs}, jobs_dir, ENV_SIMFOUNDRY
    )
    written = {}
    for f, path in zip(frames, result["depth"]):
        cam = capture.cameras[f.camera]
        full = cv2.resize(
            np.load(path).astype(np.float32),
            (cam.width, cam.height),
            interpolation=cv2.INTER_NEAREST,
        )
        rel = f"depth/{f.camera}/{f.step:04d}.png"
        (capture.root / rel).parent.mkdir(parents=True, exist_ok=True)
        write_depth(capture.root / rel, full)
        written[(f.camera, f.step)] = rel
    shutil.rmtree(work)
    out = replace(
        capture,
        frames=[
            replace(f, depth_image=written.get((f.camera, f.step), f.depth_image))
            for f in capture.frames
        ],
        metadata={**capture.metadata, "depth": DEPTH_NOTE},
    )
    out.save()
    return out
