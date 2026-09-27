"""Tools that run Isaac Lab (in its own process, since the Omniverse app must start
first): settling a scene, and replaying the recording in it to compare every view."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from r2s2r.agentic.workspace import Workspace
from r2s2r.reconstruct.simfoundry import REPO_ROOT

ISAAC_SCRIPTS = REPO_ROOT / "scripts" / "isaaclab"


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


def settle(
    ws: Workspace, scene_dir: str | Path, out_dir: str | Path, seconds: float = 2.0
) -> dict[str, Any]:
    """The scene with its objects where they come to rest (``out_dir/scene.json``), and
    how far each moved."""
    out_dir = Path(out_dir).resolve()
    _run_isaac(
        "settle.py",
        [
            str(Path(scene_dir).resolve()),
            str(ws.capture.root),
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
    ws: Workspace,
    scene_dir: str | Path,
    out_dir: str | Path,
    cameras: list[str] | None = None,
    every: int = 1,
) -> dict[str, Any]:
    """Replay the recording's static period in the scene (the robot at every recorded
    state, the objects held where the scene puts them), render every camera, and compare
    with the real frames.

    Returns the median depth residuals per camera and per object; the images are under
    ``out_dir/compare/``.
    """
    out_dir = Path(out_dir).resolve()
    args = [
        str(Path(scene_dir).resolve()),
        str(ws.capture.root),
        "--out",
        str(out_dir),
        "--mode",
        "geometry",
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
