"""Drives the agent ("astra": Codex ``gpt-6-astra``) through the stages of an agentic
reconstruction (see :mod:`r2s2r.agentic`).

Each agent stage runs ``codex exec`` in its own directory, with the stage's brief and
the workspace's rules (``AGENTS.md``), unsandboxed (the tools need the GPU). The stage's
output is then validated; while it is missing or invalid, the same session is resumed
with what is wrong. The agent must not change the repository: its state (tracked changes
and untracked files, the SimFoundry submodule's too) is compared before and after every
stage. Stage 5 (settling under physics) needs no agent; stage 6 can be left out
(``--stages 2 3 4 5``). The final scene, stage 6's or else stage 5's, is then replayed
once more (``<stage dir>/final_replay``), so that its replay against the real frames is
there whatever the agent kept.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable

import numpy as np

from r2s2r.agentic import isaac
from r2s2r.agentic.geometry import load_support
from r2s2r.agentic.objects import load_objects
from r2s2r.agentic.workspace import WORKSPACE_FILENAME, Workspace
from r2s2r.reconstruct.simfoundry import DEFAULT_SIMFOUNDRY_DIR, REPO_ROOT
from r2s2r.structs import Capture, SceneSpec

logger = logging.getLogger(__name__)

STAGES = ("2", "3", "4", "5", "6")
STAGE_DIRS = {
    "2": "s2_frames",
    "3": "s3_objects",
    "4": "s4_scene",
    "5": "s5_settle",
    "6": "s6_refine",
}
MIN_FRAMES, MAX_FRAMES = 4, 8
# The ChatGPT desktop app's bundled CLI is kept current; a separately installed
# `codex` can be too old for the newest models.
DESKTOP_APP_CODEX = "/usr/lib/chatgpt/resources/codex"


def default_codex_bin() -> str:
    """The desktop app's Codex CLI if installed, else ``codex`` on the PATH."""
    if os.access(DESKTOP_APP_CODEX, os.X_OK):
        return DESKTOP_APP_CODEX
    return shutil.which("codex") or "codex"


@dataclass
class AgentConfig:
    """How to run the agent."""

    codex_bin: str = field(default_factory=default_codex_bin)
    model: str = "gpt-6-astra"
    reasoning: str = "xhigh"
    retries: int = 2  # resumptions while a stage's output is invalid
    timeout_s: float = 4 * 3600  # per codex call, a safety net only
    settle_seconds: float = 2.0


class RepositoryChanged(RuntimeError):
    """The agent changed files outside its workspace."""


def run(
    out_dir: str | Path,
    capture_dir: str | Path | None = None,
    stages: tuple[str, ...] = STAGES,
    config: AgentConfig | None = None,
    force: bool = False,
    cameras: list[str] | None = None,
) -> dict[str, Any]:
    """Run ``stages`` in the workspace ``out_dir`` (made from ``capture_dir``, with only
    ``cameras`` if given, when new).

    A stage whose output is already valid is skipped unless ``force``.
    """
    cfg = config or AgentConfig()
    out_dir = Path(out_dir).resolve()
    if (out_dir / WORKSPACE_FILENAME).exists():
        ws = Workspace.load(out_dir)
    else:
        if capture_dir is None:
            raise ValueError(f"{out_dir} is no workspace yet: give the capture")
        ws = Workspace.create(Capture.load(capture_dir), out_dir, cameras)
    (ws.root / "AGENTS.md").write_text(render_brief("AGENTS.md", ws), encoding="utf-8")
    log_path = ws.root / "run.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"stages": {}}
    try:
        for stage in stages:
            stage_dir = ws.root / STAGE_DIRS[stage]
            if not force and not VALIDATORS[stage](ws, stage_dir):
                logger.info("stage %s already done", stage)
                continue
            start = time.time()
            log.setdefault("started", {})[stage] = round(start, 1)
            log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
            if stage == "5":
                _settle_stage(ws, stage_dir, cfg)
                entry: dict[str, Any] = {}
            else:
                entry = _agent_stage(ws, stage, stage_dir, cfg)
            entry["seconds"] = round(time.time() - start)
            log["stages"][stage] = entry
            log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
            logger.info("stage %s done in %d s", stage, entry["seconds"])
        if stages[-1] in ("5", "6"):
            final = final_replay(ws, force)
            if final is not None:
                log["final_replay"] = final
                log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    finally:
        ws.close()
    return dict(log)


def final_replay(ws: Workspace, force: bool = False) -> dict[str, Any] | None:
    """Replay the final scene (stage 6's, or stage 5's when stage 6 was not run) against
    the recording, into ``<stage dir>/final_replay``, unless that replay is newer than
    the scene; its summary, or None when there is no final scene or nothing to redo."""
    for key in ("6", "5"):
        stage_dir = ws.root / STAGE_DIRS[key]
        if not VALIDATORS[key](ws, stage_dir):
            break
    else:
        return None
    scene = stage_dir / "scene" / "scene.json"
    out = stage_dir / "final_replay"
    done = out / "compare" / "compare.json"
    if not force and done.exists() and done.stat().st_mtime > scene.stat().st_mtime:
        return None
    logger.info("replaying the final scene -> %s", out)
    return isaac.replay(ws, scene.parent, out)


# ------------------------------------------------------------------------- stages
def _settle_stage(ws: Workspace, stage_dir: Path, cfg: AgentConfig) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    report = isaac.settle(
        ws, ws.root / STAGE_DIRS["4"] / "scene", stage_dir / "scene", cfg.settle_seconds
    )
    (stage_dir / "output.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )


def _agent_stage(
    ws: Workspace, stage: str, stage_dir: Path, cfg: AgentConfig
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "BRIEF.md").write_text(
        render_brief(f"stage{stage}.md", ws), encoding="utf-8"
    )
    before = repo_state()
    prompt = (
        f"Read {ws.root / 'AGENTS.md'} (the rules of this workspace, the recording, "
        f"the tools) and then {stage_dir / 'BRIEF.md'}, and carry out that stage. Work "
        f"in {stage_dir}."
    )
    session = _codex(prompt, stage_dir, cfg, None, "codex_0")
    problems = VALIDATORS[stage](ws, stage_dir)
    attempt = 0
    while problems and attempt < cfg.retries:
        attempt += 1
        logger.warning("stage %s output invalid: %s", stage, problems)
        _codex(
            "The stage's output is not complete yet:\n- "
            + "\n- ".join(problems)
            + "\nFix this and finish the stage.",
            stage_dir,
            cfg,
            session,
            f"codex_{attempt}",
        )
        problems = VALIDATORS[stage](ws, stage_dir)
    changed = diff_states(before, repo_state())
    if changed:
        raise RepositoryChanged(
            f"stage {stage} changed files outside its workspace: {changed}"
        )
    if problems:
        raise RuntimeError(f"stage {stage} output still invalid: {problems}")
    return {"session": session, "resumptions": attempt}


def _codex(
    prompt: str, cwd: Path, cfg: AgentConfig, session: str | None, name: str
) -> str:
    """One ``codex exec`` (or a resumption of ``session``); the session id."""
    common = [
        "-m",
        cfg.model,
        "-c",
        f'model_reasoning_effort="{cfg.reasoning}"',
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
        "-o",
        str(cwd / f"{name}_final.md"),
    ]
    if session is None:
        cmd = [cfg.codex_bin, "exec", *common, "-C", str(cwd), prompt]
    else:
        cmd = [cfg.codex_bin, "exec", "resume", *common, session, prompt]
    env = dict(os.environ)
    # The agent's `python` and `r2s2r` are this environment's.
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env["PATH"]])
    events = cwd / f"{name}.jsonl"
    logger.info("astra: %s (events: %s)", cwd.name, events)
    with open(events, "w", encoding="utf-8") as out:
        subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=cfg.timeout_s,
            check=False,
        )
    found = session
    for line in events.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            found = event.get("thread_id", found)
        if event.get("type") == "turn.failed":
            raise RuntimeError(
                f"codex turn failed ({events}): {event.get('error', {}).get('message')}"
            )
    if found is None:
        tail = "\n".join(events.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"codex started no session; {events}:\n{tail}")
    return found


def render_brief(name: str, ws: Workspace) -> str:
    """A brief with the workspace's facts filled in."""
    text = resources.files("r2s2r.agentic").joinpath("briefs", name).read_text()
    cap = ws.capture
    cameras = "\n".join(
        f"  - `{c.role}`: {c.width}x{c.height}, "
        + (
            "static, calibrated"
            if c.is_static
            else "on the wrist, moving with the hand"
        )
        + (", stereo" if c.stereo_baseline else "")
        for c in cap.cameras.values()
    )
    counts = ", ".join(
        f"{sum(1 for f in ws.reconstructable() if f.camera == s)} from `{c.role}`"
        for s, c in cap.cameras.items()
    )
    depth_note = (
        "computed from the stereo pairs by FoundationStereo; it is smooth at object "
        "edges and less reliable on thin, dark or shiny things."
        if "FoundationStereo" in str(cap.metadata.get("depth", ""))
        else "measured by the cameras."
    )
    values = {
        "ws": str(ws.root),
        "embodiment": cap.embodiment,
        "instruction": cap.instruction or "(none given)",
        "cameras": cameras,
        "static_start": str(cap.static_steps[0]),
        "static_last": str(cap.static_steps[1] - 1),
        "frame_counts": f"Frames to reconstruct from: {counts}.",
        "depth_note": depth_note,
    }
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


# ---------------------------------------------------------------------- validation
def _validate_frames(ws: Workspace, d: Path) -> list[str]:
    out = _json(d / "output.json")
    if isinstance(out, str):
        return [out]
    problems = []
    frames = out.get("frames", [])
    if not MIN_FRAMES <= len(frames) <= MAX_FRAMES:
        problems.append(
            f"choose {MIN_FRAMES} to {MAX_FRAMES} frames, not {len(frames)}"
        )
    for fid in frames:
        try:
            frame = ws.frame(fid)
        except (KeyError, ValueError) as exc:
            problems.append(str(exc))
            continue
        if not ws.is_static(frame) or frame.depth_image is None:
            problems.append(f"{fid} is not a static frame with depth")
    support = out.get("support", {}).get("file")
    if not support:
        problems.append("output.json names no support file")
    else:
        problems += _check_support(d / support)
    if not isinstance(out.get("objects"), list):
        problems.append("output.json lists no objects")
    return problems


def _check_support(path: Path) -> list[str]:
    try:
        T, extent = load_support(path)
    except (OSError, KeyError, ValueError) as exc:
        return [f"support {path}: {exc}"]
    if not _is_rigid(T):
        return [f"support {path}: T_base_support is not a rigid transform"]
    if extent is None or min(extent) <= 0:
        return [f"support {path}: no positive extent"]
    return []


def _validate_objects(_: Workspace, d: Path) -> list[str]:
    path = d / "objects.json"
    try:
        spec, base = load_objects(path)
    except (OSError, ValueError) as exc:
        return [f"{path}: {exc}"]
    problems = []
    objects = spec.get("objects", [])
    if not objects:
        problems.append("objects.json has no objects")
    names = [o.get("name") for o in objects]
    if len(set(names)) != len(names):
        problems.append(f"object names are not unique: {names}")
    for obj in objects:
        name = obj.get("name", "?")
        mesh = obj.get("mesh")
        if not mesh or not (base / mesh).exists():
            problems.append(f"{name}: mesh {mesh} not found")
        scale = np.asarray(obj.get("scale", 0.0), float)
        if scale.size not in (1, 3) or np.any(scale <= 0):
            problems.append(f"{name}: scale must be positive (one value or three)")
        if not _is_rigid(np.asarray(obj.get("T_base_obj", np.zeros((4, 4))), float)):
            problems.append(f"{name}: T_base_obj must be a rigid 4x4 transform")
    support = spec.get("support")
    if isinstance(support, str):
        problems += _check_support(base / support)
    elif not isinstance(support, dict) or "T_base_support" not in support:
        problems.append("objects.json has no support")
    return problems


def _validate_scene(sub: str, extra: tuple[str, ...] = ()) -> Callable[..., list[str]]:
    def validate(_: Workspace, d: Path) -> list[str]:
        problems = [f"{name} missing" for name in extra if not (d / name).exists()]
        try:
            scene = SceneSpec.load(d / sub)
        except (OSError, KeyError, ValueError) as exc:
            return problems + [f"{d / sub}: {exc}"]
        if not scene.objects:
            problems.append("the scene has no objects")
        for obj in scene.objects:
            if not Path(obj.asset_path).exists():
                problems.append(f"{obj.name}: asset {obj.asset_path} missing")
            if not obj.mass or obj.mass <= 0:
                problems.append(f"{obj.name}: no mass")
        return problems

    return validate


VALIDATORS: dict[str, Callable[[Workspace, Path], list[str]]] = {
    "2": _validate_frames,
    "3": _validate_objects,
    "4": _validate_scene("scene", ("output.json",)),
    "5": _validate_scene("scene", ("output.json",)),
    "6": _validate_scene("scene", ("report.md",)),
}


def _json(path: Path) -> Any:
    if not path.exists():
        return f"{path.name} not written yet"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"{path.name} is not valid JSON: {exc}"


def _is_rigid(T: np.ndarray) -> bool:
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    return bool(np.allclose(R @ R.T, np.eye(3), atol=1e-4) and np.linalg.det(R) > 0)


# ------------------------------------------------------------------ repository guard
def repo_state() -> dict[str, tuple[str, str]]:
    """A fingerprint of the repository's and the SimFoundry submodule's working trees:

    their status, and a digest of the tracked changes and the untracked files.
    """
    state = {}
    for repo in (REPO_ROOT, DEFAULT_SIMFOUNDRY_DIR):

        def git(*args: str, repo: Path = repo) -> str:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout

        status = git("status", "--porcelain=v1", "--untracked-files=all")
        digest = hashlib.sha256(git("diff", "HEAD", "--no-ext-diff").encode())
        for line in status.splitlines():
            if line.startswith("?? "):
                path = repo / line[3:]
                if path.is_file():
                    digest.update(path.read_bytes())
        state[str(repo)] = (status, digest.hexdigest())
    return state


def diff_states(
    before: dict[str, tuple[str, str]], after: dict[str, tuple[str, str]]
) -> list[str]:
    """Per repository whose state changed, what changed in its status."""
    changed = []
    for repo, (status, digest) in after.items():
        old_status, old_digest = before.get(repo, ("", ""))
        if (status, digest) != (old_status, old_digest):
            lines = set(status.splitlines()) ^ set(old_status.splitlines())
            changed.append(f"{repo}: {sorted(lines) or 'file contents changed'}")
    return changed
