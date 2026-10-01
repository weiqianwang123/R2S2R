"""Isaac Lab, run in its own process (the Omniverse app must start first): settling a
scene, replaying a capture's recording in it to compare every view, and the pick test.

All take paths (a scene directory, a capture directory), so anything can call them: the
runs' shared stage 5 and final replay, the agent's tools, and ``r2s2r pick``. Every
scene's simulation runs as this module sets it: where PhysX runs
(:func:`physics_device`), gravity (:func:`gravity`) and the rendering preset
(:data:`RENDERING_MODE`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from r2s2r.paths import REPO_ROOT
from r2s2r.structs import SceneSpec

ISAAC_SCRIPTS = REPO_ROOT / "scripts" / "isaaclab"
SETTLE_SECONDS = 2.0  # simulated time for the objects to come to rest
# Isaac Lab's rendering preset (the scripts' ``--rendering_mode`` default):
# "balanced" blends earlier frames into each image, so a cloth that moved leaves a
# ghost where it lay for many frames; "performance" renders each frame by itself.
# DLSS, which upscales from earlier frames too, is replaced after launch
# (:func:`r2s2r.sim.isaaclab.scene.render_frames_alone`).
RENDERING_MODE = "performance"


def physics_device(scene: SceneSpec, device: str) -> str:
    """Where PhysX runs ``scene``: on ``device`` if it has a cloth (only the GPU
    simulates one), else on the CPU. One scene of a few objects steps several times
    faster there, and GPU PhysX mis-solves contacts on an articulated object's links:
    what rests on a lid or a book's cover creeps up and tumbles off within seconds."""
    return device if any(obj.cloth for obj in scene.objects) else "cpu"


def gravity(scene: SceneSpec) -> tuple[float, float, float]:
    """Gravity in the robot base frame, along the support's normal: the table is
    level, while the base (or its estimate) leans a few tenths of a degree, enough for
    a round object to roll away."""
    normal = np.asarray(scene.T_base_support, float)[:3, 2]
    g = -9.81 * normal / np.linalg.norm(normal)
    return float(g[0]), float(g[1]), float(g[2])


def _run_isaac(script: str, args: list[str], log_path: Path) -> None:
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    cmd = [sys.executable, str(ISAAC_SCRIPTS / script), *args, "--headless"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(
            f"{script} failed (exit {proc.returncode}); log {log_path}:\n{tail}"
        )


def settle_summary(report: dict[str, Any]) -> str:
    """One line on how far each object moved while settling (a body also turned)."""
    return ", ".join(
        f"{name} moved {r['moved_m']:.3f} m"
        + (f", turned {r['turned_deg']:.1f} deg" if "turned_deg" in r else "")
        for name, r in report["objects"].items()
    )


def settle(
    scene_dir: str | Path,
    capture_dir: str | Path,
    out_dir: str | Path,
    seconds: float = SETTLE_SECONDS,
) -> dict[str, Any]:
    """The scene with its objects where they come to rest (``out_dir/scene.json``; the
    robot held as the capture has it at the start of the static period; each object's
    USD under ``out_dir/objects/``), and how far each moved."""
    out_dir = Path(out_dir).resolve()
    _run_isaac(
        "settle.py",
        [
            str(Path(scene_dir).resolve()),
            str(Path(capture_dir).resolve()),
            "--out",
            str(out_dir),
            "--seconds",
            str(seconds),
        ],
        out_dir / "settle.log",
    )
    report: dict[str, Any] = json.loads((out_dir / "settle.json").read_text())
    return {"scene": str(out_dir / "scene.json"), **report}


def replay(
    scene_dir: str | Path,
    capture_dir: str | Path,
    out_dir: str | Path,
    cameras: list[str] | None = None,
    every: int = 1,
) -> dict[str, Any]:
    """Replay the capture's static period in the scene (the robot at every recorded
    state, the objects held where the scene puts them), render every camera, and compare
    with the real frames.

    Returns the median depth residuals per camera and per object; the images are under
    ``out_dir/compare/``.
    """
    out_dir = Path(out_dir).resolve()
    args = [
        str(Path(scene_dir).resolve()),
        str(Path(capture_dir).resolve()),
        "--out",
        str(out_dir),
        "--every",
        str(every),
    ]
    if cameras:
        args += ["--cameras", *cameras]
    _run_isaac("replay.py", args, out_dir / "replay.log")
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


def pick(
    scene_dir: str | Path,
    target: str,
    out_dir: str | Path,
    video_camera: str | None = "ext1",
) -> dict[str, Any]:
    """Run the pick program on the scene for ``target`` (one of its objects) and
    score it (``out_dir/result.json``; the commands and a video from the static camera
    ``video_camera`` beside it)."""
    out_dir = Path(out_dir).resolve()
    _run_isaac(
        "pick.py",
        [
            str(Path(scene_dir).resolve()),
            "--target",
            target,
            "--out",
            str(out_dir),
            "--video-camera",
            video_camera or "",
        ],
        out_dir / "pick.log",
    )
    result: dict[str, Any] = json.loads((out_dir / "result.json").read_text())
    return result
