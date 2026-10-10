"""Isaac Lab, run in its own process (the Omniverse app must start first): settling a
scene, replaying a capture's recording in it to compare every view, and the pick test.

All take paths (a scene directory, a capture directory), so anything can call them: the
runs' shared stage 5 and final replay (:mod:`r2s2r.sim`, which picks the simulator),
the agent's tools, and ``r2s2r pick``. What every simulator does the same (gravity, the
settling time, the support's colour, a replay's numbers) is in
:mod:`r2s2r.sim.world`; physics runs on the CPU, the rendering preset is
:data:`RENDERING_MODE`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from r2s2r.paths import REPO_ROOT
from r2s2r.sim.world import SETTLE_SECONDS, replay_summary
from r2s2r.structs import SceneSpec

ISAAC_SCRIPTS = REPO_ROOT / "scripts" / "isaaclab"
# Isaac Lab's rendering preset (the scripts' ``--rendering_mode`` default):
# "balanced" blends earlier frames into each image, so what moved leaves a ghost
# where it was for many frames; "performance" renders each frame by itself.
# DLSS, which upscales from earlier frames too, is replaced by FXAA as the
# simulation starts (:func:`r2s2r.sim.isaaclab.scene.simulation_cfg`).
RENDERING_MODE = "performance"


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
    return replay_summary(out_dir)


def pick(
    scene_dir: str | Path,
    target: str,
    out_dir: str | Path,
    video_camera: str | None = "ext1",
) -> dict[str, Any]:
    """Run the pick program on the scene for ``target`` (one of its objects, not a
    cloth: it moves only in MuJoCo's scene pick) and score it
    (``out_dir/result.json``; the commands and a video from the static camera
    ``video_camera`` beside it)."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.testbed.policy import find_object

    obj = find_object(SceneSpec.load(scene_dir), target)
    if obj.cloth:
        raise ValueError(f"{obj.name} is a cloth: it moves only in MuJoCo's scene pick")
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
