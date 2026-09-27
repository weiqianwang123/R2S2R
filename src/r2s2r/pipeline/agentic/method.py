"""The agentic method: the agent ("astra": Codex ``gpt-6-astra``) does stages 2, 3, 4
and 6 (see :mod:`r2s2r.pipeline.agentic`); :mod:`r2s2r.pipeline.run` runs them.

Each stage runs ``codex exec`` in its own directory, with the stage's brief and the
run's rules (``AGENTS.md``), unsandboxed (the tools need the GPU). While the stage's
product is missing or invalid, the same session is resumed with what is wrong. The agent
must not change the repository: its state (tracked changes and untracked files, the
SimFoundry submodule's too) is compared before and after every stage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from r2s2r.paths import REPO_ROOT, SIMFOUNDRY_DIR, codex_bin
from r2s2r.pipeline.run import problems
from r2s2r.pipeline.stages import read_json
from r2s2r.pipeline.workspace import Workspace
from r2s2r.tools.geometry import load_support

logger = logging.getLogger(__name__)

MIN_FRAMES, MAX_FRAMES = 4, 8


@dataclass
class AgentConfig:
    """How to run the agent."""

    codex_bin: str = field(default_factory=codex_bin)
    model: str = "gpt-6-astra"
    reasoning: str = "xhigh"
    retries: int = 2  # resumptions while a stage's output is invalid
    timeout_s: float = 4 * 3600  # per codex call, a safety net only


class RepositoryChanged(RuntimeError):
    """The agent changed files outside its run."""


class AgentMethod:
    """The agent does stages 2, 3, 4 and 6, with the tools."""

    name = "agentic"
    stages: tuple[str, ...] = ("2", "3", "4", "6")

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    def run_stage(self, ws: Workspace, key: str, stage_dir: Path) -> dict[str, Any]:
        """One agent session on the stage (resumed while its product is invalid); the
        session and how often it was resumed."""
        cfg = self.config
        (ws.root / "AGENTS.md").write_text(
            render_brief("AGENTS.md", ws), encoding="utf-8"
        )
        (stage_dir / "BRIEF.md").write_text(
            render_brief(f"stage{key}.md", ws), encoding="utf-8"
        )
        before = repo_state()
        prompt = (
            f"Read {ws.root / 'AGENTS.md'} (the rules of this workspace, the "
            f"recording, the tools) and then {stage_dir / 'BRIEF.md'}, and carry out "
            f"that stage. Work in {stage_dir}."
        )
        session = _codex(prompt, stage_dir, cfg, None, "codex_0")
        found = problems(ws, self, key)
        attempt = 0
        while found and attempt < cfg.retries:
            attempt += 1
            logger.warning("stage %s output invalid: %s", key, found)
            _codex(
                "The stage's output is not complete yet:\n- "
                + "\n- ".join(found)
                + "\nFix this and finish the stage.",
                stage_dir,
                cfg,
                session,
                f"codex_{attempt}",
            )
            found = problems(ws, self, key)
        changed = diff_states(before, repo_state())
        if changed:
            raise RepositoryChanged(
                f"stage {key} changed files outside its workspace: {changed}"
            )
        return {"session": session, "resumptions": attempt}

    def check(self, ws: Workspace, key: str, stage_dir: Path) -> list[str]:
        """Stage 2 beyond the shared checks: 4 to 8 frames, each with depth; a support
        with a positive extent; the objects listed for stage 3."""
        if key != "2":
            return []
        try:
            out = read_json(stage_dir / "output.json")
            frames = list(out.get("frames") or [])
        except (ValueError, AttributeError, TypeError):
            return []  # the shared check says what is wrong
        support = out.get("support")
        support = support.get("file") if isinstance(support, dict) else None
        found = []
        if not MIN_FRAMES <= len(frames) <= MAX_FRAMES:
            found.append(
                f"choose {MIN_FRAMES} to {MAX_FRAMES} frames, not {len(frames)}"
            )
        for fid in frames:
            try:
                if ws.frame(fid).depth_image is None:
                    found.append(f"{fid} has no depth")
            except (AttributeError, KeyError, ValueError):
                pass  # the shared check says so
        if support:
            try:
                if load_support(stage_dir / support)[1] is None:
                    found.append(f"support {support}: no extent")
            except (OSError, KeyError, TypeError, ValueError):
                pass  # the shared check says what is wrong
        if not isinstance(out.get("objects"), list):
            found.append("output.json lists no objects")
        return found


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
    """A brief with the run's facts filled in."""
    brief = resources.files("r2s2r.pipeline.agentic").joinpath("briefs", name)
    text = brief.read_text()
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


# ------------------------------------------------------------------ repository guard
def repo_state() -> dict[str, tuple[str, str]]:
    """A fingerprint of the repository's and the SimFoundry submodule's working trees:

    their status, and a digest of the tracked changes and the untracked files.
    """
    state = {}
    for repo in (REPO_ROOT, SIMFOUNDRY_DIR):

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
