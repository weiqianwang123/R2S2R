"""Running a method on a capture: the run's directory, every stage's status in
``run.json``, the stage every method shares (5: the scene settles in the run's
simulator, Isaac Lab or MuJoCo: :mod:`r2s2r.sim`) and the final replay of the scene
against the recording, in the same simulator.

A method does stages 2, 3 and 4, and 6 if it has one (:mod:`r2s2r.pipeline.stages` says
what each leaves); it is anything with

- ``name``, and ``stages``: the keys of the stages it does;
- ``run_stage(ws, key, stage_dir)``: do the stage; what to record about it;
- ``check(ws, key, stage_dir)``: problems with a stage's product beyond the shared
  checks (its own stricter rules; none for most stages).

A stage is skipped when it is done: its product passes both checks, and it and every
product before it are newer than the one before. ``run.json``, rewritten whole at every
change::

    {"method": "agentic", "sim": "isaac" | "mujoco", "capture": "inputs/capture",
     "stages": {"2": {"status": "running" | "done" | "failed", "started": epoch,
                      "seconds": s, "error": "..." (failed), ...what the method
                      recorded}},
     "final_replay": {"stage": "6", "status": ..., "started": epoch, "seconds": s,
                      ...the replay's numbers}}
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol

from r2s2r import sim
from r2s2r.pipeline.stages import (
    PRODUCTS,
    SETTLE,
    STAGE_DIRS,
    STAGES,
    VALIDATORS,
    fresh,
    newer_than_previous,
)
from r2s2r.structs import CAPTURE_FILENAME, Capture
from r2s2r.workspace import DEFAULT_SIM, RUN_FILENAME, Workspace

logger = logging.getLogger(__name__)


class Method(Protocol):
    """How a method does its stages (see the module doc)."""

    name: str
    stages: tuple[str, ...]

    def run_stage(self, ws: Workspace, key: str, stage_dir: Path) -> dict[str, Any]:
        """Do stage ``key`` in ``stage_dir``; what to record about it in
        ``run.json``."""

    def check(self, ws: Workspace, key: str, stage_dir: Path) -> list[str]:
        """Problems with stage ``key``'s product beyond the shared checks."""


class StageFailed(RuntimeError):
    """A stage ended without a valid product."""


def run_stages(method: Method) -> tuple[str, ...]:
    """The stages a run of ``method`` goes through: its own and the shared one."""
    return tuple(k for k in STAGES if k in method.stages or k == SETTLE)


def problems(ws: Workspace, method: Method, key: str) -> list[str]:
    """What is wrong with stage ``key``'s product: the shared checks, then the
    method's; when it passes both, whether it is older than the previous stage's."""
    d = ws.root / STAGE_DIRS[key]
    found = VALIDATORS[key](ws, d) + method.check(ws, key, d)
    if not found and not newer_than_previous(ws.root, key):
        previous = STAGES[STAGES.index(key) - 1]
        found.append(
            f"stage {previous}'s product changed after this one was written: write "
            f"{' and '.join(PRODUCTS[key])} again"
        )
    return found


def is_done(ws: Workspace, method: Method, key: str) -> bool:
    """Whether stage ``key``'s product is valid, and it and every product before it
    newer than the one before."""
    return fresh(ws.root, key) and not problems(ws, method, key)


def open_run(
    source: str | Path,
    out: str | Path | None,
    method: str,
    cameras: list[str] | None = None,
    simulator: str | None = None,
) -> Workspace:
    """The run to work in: ``source`` itself when it is a run; else a new run of
    ``method`` on the capture ``source`` at ``out`` (with only ``cameras``, if given;
    its scenes simulated in ``simulator``, Isaac Lab by default), or the run already
    there."""
    source = Path(source).resolve()
    if (source / RUN_FILENAME).exists():
        if out is not None and Path(out).resolve() != source:
            raise ValueError(f"{source} is a run: resume it without --out")
        root = source
    elif (source / CAPTURE_FILENAME).exists():
        if out is None:
            raise ValueError("a new run needs --out")
        root = Path(out).resolve()
    else:
        raise FileNotFoundError(f"{source} is neither a capture nor a run")
    if not (root / RUN_FILENAME).exists():
        return Workspace.create(
            Capture.load(source), root, method, cameras, simulator or DEFAULT_SIM
        )
    ws = Workspace.load(root)
    made_by = ws.read_run().get("method")
    if made_by != method:
        raise ValueError(f"{root} is a run of the {made_by} method, not {method}")
    if simulator is not None and simulator != ws.sim:
        raise ValueError(f"{root} simulates its scenes in {ws.sim}, not {simulator}")
    if cameras and set(ws.capture.resolve_cameras(cameras)) != set(ws.capture.cameras):
        roles = [c.role for c in ws.capture.cameras.values()]
        raise ValueError(f"{root} was made with the cameras {roles}, not {cameras}")
    return ws


def run(
    source: str | Path,
    out: str | Path | None,
    method: Method,
    stages: tuple[str, ...] | None = None,
    force: bool = False,
    cameras: list[str] | None = None,
    simulator: str | None = None,
) -> dict[str, Any]:
    """Run ``stages`` (default: all of the run's) of ``method``, in order, in the run
    :func:`open_run` gives; then, if the last is 5 or 6, replay the final scene. Done
    stages are skipped unless ``force``. Returns ``run.json``."""
    ws = open_run(source, out, method.name, cameras, simulator)
    own = run_stages(method)
    wanted = [k for k in STAGES if k in (stages or own)]
    if stages and set(stages) - set(own):
        raise ValueError(f"the {method.name} method has the stages {own}, not {stages}")
    info = ws.read_run()
    left_out = [k for k in STAGES if k not in own and k not in info["stages"]]
    if left_out:  # stages the method does not have
        info["stages"].update({k: {"status": "skipped"} for k in left_out})
        ws.write_run(info)
    try:
        for key in wanted:
            if not force and is_done(ws, method, key):
                logger.info("stage %s already done", key)
                entry = ws.read_run()["stages"].get(key, {})
                if entry.get("status") != "done":
                    entry.pop("error", None)
                    _record(ws, key, {**entry, "status": "done"})
                continue
            _run_stage(ws, method, key)
        if wanted and wanted[-1] in (SETTLE, "6"):
            final_replay(ws, method, force)
    finally:
        ws.close()
    return ws.read_run()


def final_replay(
    ws: Workspace, method: Method, force: bool = False
) -> dict[str, Any] | None:
    """Replay the final scene (stage 6's when done, else stage 5's) against the
    recording, into ``<stage dir>/final_replay``, unless that replay is newer than the
    scene; its numbers (also in ``run.json``), or None when there is nothing to do."""
    key = next(
        (
            k
            for k in ("6", SETTLE)
            if k in run_stages(method) and is_done(ws, method, k)
        ),
        None,
    )
    if key is None:
        return None
    scene = ws.root / STAGE_DIRS[key] / "scene"
    out = scene.parent / "final_replay"
    done = out / "compare" / "compare.json"
    if (
        not force
        and done.exists()
        and done.stat().st_mtime > (scene / "scene.json").stat().st_mtime
    ):
        return None
    logger.info("replaying the final scene -> %s", out)
    start = time.time()
    _record_replay(ws, {"stage": key, "status": "running", "started": round(start, 1)})
    try:
        summary = sim.replay(scene, ws.capture.root, out, ws.sim)
    except BaseException as exc:
        _record_replay(ws, {"stage": key, **_failed(start, exc)})
        raise
    summary["compare_dir"] = os.path.relpath(summary["compare_dir"], ws.root)
    summary["sheets"] = [os.path.relpath(p, ws.root) for p in summary["sheets"]]
    _record_replay(
        ws,
        {
            "stage": key,
            "status": "done",
            "started": round(start, 1),
            "seconds": round(time.time() - start),
            **summary,
        },
    )
    return summary


# -------------------------------------------------------------------------- stages
def _run_stage(ws: Workspace, method: Method, key: str) -> None:
    stage_dir = ws.root / STAGE_DIRS[key]
    stage_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    _record(ws, key, {"status": "running", "started": round(start, 1)})
    try:
        if key == SETTLE:
            entry = _settle(ws, stage_dir)
        else:
            entry = method.run_stage(ws, key, stage_dir)
        found = problems(ws, method, key)
        if found:
            raise StageFailed(f"stage {key}'s output is not valid: {found}")
    except BaseException as exc:
        _record(ws, key, _failed(start, exc))
        raise
    seconds = round(time.time() - start)
    _record(
        ws,
        key,
        {"status": "done", "started": round(start, 1), "seconds": seconds, **entry},
    )
    logger.info("stage %s done in %d s", key, seconds)


def _settle(ws: Workspace, stage_dir: Path) -> dict[str, Any]:
    """Stage 5: stage 4's scene settles in the run's simulator, the robot held (its
    cloths then in Newton)."""
    report = sim.settle(
        ws.root / STAGE_DIRS["4"] / "scene",
        ws.capture.root,
        stage_dir / "scene",
        ws.sim,
    )
    report["scene"] = "scene/scene.json"
    (stage_dir / "output.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    return {}


def _failed(start: float, exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "started": round(start, 1),
        "seconds": round(time.time() - start),
        "error": f"{type(exc).__name__}: {exc}"[:2000],
    }


def _record(ws: Workspace, key: str, entry: dict[str, Any]) -> None:
    run_ = ws.read_run()
    run_.setdefault("stages", {})[key] = entry
    ws.write_run(run_)


def _record_replay(ws: Workspace, entry: dict[str, Any]) -> None:
    run_ = ws.read_run()
    run_["final_replay"] = entry
    ws.write_run(run_)
