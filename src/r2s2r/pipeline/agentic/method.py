"""The agentic method: a coding agent does stages 2, 3, 4 and 6 (see
:mod:`r2s2r.pipeline.agentic`); :mod:`r2s2r.pipeline.run` runs them.

The agent is Codex (``codex exec``, ``gpt-6-astra``: "astra") or Claude Code
(``claude -p``, ``claude-opus-5-5``); a run keeps the one that last worked on it unless
told otherwise. Each stage runs it in the stage's own directory, with the stage's brief
and the run's rules (``AGENTS.md``), unsandboxed (the tools need the GPU): its events go
to ``<agent>_<n>.jsonl``, its last message to ``<agent>_<n>_final.md``. While the stage's
product is missing or invalid, the same session is resumed with what is wrong. The agent
must not change the repository: its state (tracked changes and untracked files, the
SimFoundry submodule's too) is compared before and after every stage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from r2s2r.capture.stereo import DEPTH_NOTE
from r2s2r.paths import (
    CLAUDE_MODEL,
    CODEX_MODEL,
    REPO_ROOT,
    SIMFOUNDRY_DIR,
    claude_bin,
    codex_bin,
)
from r2s2r.pipeline.run import problems
from r2s2r.pipeline.stages import STAGE_DIRS, read_json, support_file
from r2s2r.tools.geometry import load_support
from r2s2r.workspace import Workspace

logger = logging.getLogger(__name__)

MIN_FRAMES, MAX_FRAMES = 4, 8
SIM_NAMES = {"isaac": "Isaac Lab", "mujoco": "MuJoCo"}  # r2s2r.sim.SIMS, as briefs say
# The agents, with their CLIs and default models.
AGENTS: dict[str, tuple[Callable[[], str], str]] = {
    "codex": (codex_bin, CODEX_MODEL),
    "claude": (claude_bin, CLAUDE_MODEL),
}
# How long a command Claude Code runs may take when the agent gives no limit (its own
# default is 2 minutes; generating meshes takes longer).
CLAUDE_COMMAND_S = 3600
STOP_GRACE_S = 10.0  # how long a stopped agent may take to end before it is killed


@dataclass
class AgenticConfig:
    """How to run the agent. Left None: the agent is the one that last worked on the
    run (Codex for a new run); the model and the executable are the agent's own."""

    agent: str | None = None  # "codex" or "claude" (Claude Code)
    model: str | None = None
    reasoning: str = "xhigh"  # Codex's reasoning effort, Claude Code's --effort
    executable: str | None = None
    retries: int = 2  # resumptions while a stage's output is invalid
    timeout_s: float = 4 * 3600  # per agent call, a safety net only

    def __post_init__(self) -> None:
        if self.agent is not None and self.agent not in AGENTS:
            raise ValueError(f"no agent {self.agent!r}: {sorted(AGENTS)}")


@dataclass(frozen=True)
class _Agent:
    """The agent working on one stage: which, run how, where."""

    name: str
    executable: str
    model: str
    reasoning: str
    timeout_s: float
    cwd: Path  # the stage's directory
    root: Path  # the run's


class RepositoryChanged(RuntimeError):
    """The agent changed files outside its run; ``record``: what the stage would have
    recorded about the agent."""

    def __init__(self, message: str, record: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.record = record or {}


class AgenticMethod:
    """The agent does stages 2, 3, 4 and 6, with the tools."""

    name = "agentic"
    stages: tuple[str, ...] = ("2", "3", "4", "6")

    def __init__(self, config: AgenticConfig | None = None) -> None:
        self.config = config or AgenticConfig()

    def run_stage(self, ws: Workspace, key: str, stage_dir: Path) -> dict[str, Any]:
        """One agent session on the stage (resumed while its product is invalid): the
        agent, its model, the session and how often it was resumed."""
        cfg = self.config
        name = cfg.agent or run_agent(ws)
        find_executable, default_model = AGENTS[name]
        agent = _Agent(
            name,
            cfg.executable or find_executable(),
            cfg.model or default_model,
            cfg.reasoning,
            cfg.timeout_s,
            stage_dir,
            ws.root,
        )
        ask = _codex if name == "codex" else _claude
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
        record: dict[str, Any] = {"agent": name, "model": agent.model}
        try:
            session, record["model"] = ask(agent, prompt, None, f"{name}_0")
            record.update(session=session, resumptions=0)
            found = problems(ws, self, key)
            while found and record["resumptions"] < cfg.retries:
                record["resumptions"] += 1
                logger.warning("stage %s output invalid: %s", key, found)
                ask(
                    agent,
                    "The stage's output is not complete yet:\n- "
                    + "\n- ".join(found)
                    + "\nFix this and finish the stage.",
                    session,
                    f"{name}_{record['resumptions']}",
                )
                found = problems(ws, self, key)
        except BaseException as exc:  # the repository is compared however it ends
            changed = diff_states(before, repo_state())
            if changed:
                logger.error("stage %s changed the repository: %s", key, changed)
                raise RepositoryChanged(
                    f"stage {key} changed files outside its workspace: {changed} "
                    f"(and failed: {type(exc).__name__}: {exc})",
                    record,
                ) from exc
            raise
        changed = diff_states(before, repo_state())
        if changed:
            raise RepositoryChanged(
                f"stage {key} changed files outside its workspace: {changed}", record
            )
        return record

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
        support = support_file(out)
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


def run_agent(ws: Workspace) -> str:
    """The agent that last worked on the run, by its logs (``<agent>_<n>.jsonl``);
    Codex when none has."""
    logs = [
        log
        for d in STAGE_DIRS.values()
        for agent in AGENTS
        for log in (ws.root / d).glob(f"{agent}_[0-9]*.jsonl")
    ]
    if not logs:
        return "codex"
    return max(logs, key=lambda p: p.stat().st_mtime).name.split("_")[0]


def _codex(
    agent: _Agent, prompt: str, session: str | None, name: str
) -> tuple[str, str]:
    """One ``codex exec`` (or a resumption of ``session``); the session id and the
    model."""
    common = [
        "-m",
        agent.model,
        "-c",
        f'model_reasoning_effort="{agent.reasoning}"',
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
        "-o",
        str(agent.cwd / f"{name}_final.md"),
    ]
    if session is None:
        cmd = [agent.executable, "exec", *common, "-C", str(agent.cwd), prompt]
    else:
        cmd = [agent.executable, "exec", "resume", *common, session, prompt]
    log, events, code = _run(agent, cmd, name)
    found = session
    for event in events:
        if event.get("type") == "thread.started":
            found = event.get("thread_id", found)
        if event.get("type") == "turn.failed":
            raise RuntimeError(
                f"codex turn failed ({log}): {event.get('error', {}).get('message')}"
            )
    if code != 0 and not any(e.get("type") == "turn.completed" for e in events):
        raise RuntimeError(f"codex exited {code}; {_tail(log)}")
    if found is None:
        raise RuntimeError(f"codex started no session; {_tail(log)}")
    return found, agent.model


def _claude(
    agent: _Agent, prompt: str, session: str | None, name: str
) -> tuple[str, str]:
    """One ``claude -p`` (or a resumption of ``session``); the session id and the model
    that ran. Its last message is written to ``<name>_final.md``."""
    cmd = [agent.executable, "-p"]
    if session is not None:
        cmd += ["--resume", session]
    cmd += [
        "--model",
        agent.model,
        "--effort",
        agent.reasoning,
        "--dangerously-skip-permissions",
        "--add-dir",  # the run's rules and inputs, above the stage's directory
        str(agent.root),
        "--output-format",
        "stream-json",
        "--verbose",
        prompt,
    ]
    env = {
        "BASH_DEFAULT_TIMEOUT_MS": str(CLAUDE_COMMAND_S * 1000),
        "BASH_MAX_TIMEOUT_MS": str(round(agent.timeout_s * 1000)),
        # Nothing it learns of one scene is remembered in the next run.
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    }
    log, events, code = _run(agent, cmd, name, env)
    found, model = session, agent.model
    for event in events:
        if event.get("type") == "system" and event.get("subtype") == "init":
            found = event.get("session_id", found)
            model = event.get("model", model)
        if event.get("type") == "result":
            if event.get("is_error"):
                error = event.get("result") or event.get("errors")
                raise RuntimeError(f"claude failed ({log}): {error}")
            (agent.cwd / f"{name}_final.md").write_text(
                str(event.get("result", "")), encoding="utf-8"
            )
    if code != 0 and not any(e.get("type") == "result" for e in events):
        raise RuntimeError(f"claude exited {code}; {_tail(log)}")
    if found is None:
        raise RuntimeError(f"claude started no session; {_tail(log)}")
    return found, model


def _run(
    agent: _Agent, cmd: list[str], name: str, env: dict[str, str] | None = None
) -> tuple[Path, list[dict[str, Any]], int]:
    """Run the agent's CLI in the stage's directory, with nothing on its stdin and its
    output (JSON lines) in ``<name>.jsonl``; that log, its events and its exit code.

    It runs in a session of its own: when it times out or this process is interrupted,
    it is stopped with everything it started (the tools' GPU jobs among them).
    """
    full_env = {**os.environ, **(env or {})}
    # The agent's `python` and `r2s2r` are this environment's.
    full_env["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), os.environ["PATH"]]
    )
    log = agent.cwd / f"{name}.jsonl"
    logger.info("%s: %s (events: %s)", agent.name, agent.cwd.name, log)
    with open(log, "w", encoding="utf-8") as out:
        with subprocess.Popen(
            cmd,
            cwd=agent.cwd,
            env=full_env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as proc:
            try:
                code = proc.wait(timeout=agent.timeout_s)
            except BaseException:
                _stop_group(proc)
                raise
    events = []
    for line in log.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return log, events, code


def _stop_group(proc: subprocess.Popen[bytes]) -> None:
    """Stop the process group ``proc`` leads: SIGTERM, then SIGKILL to whatever of it
    is left after a grace period."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _tail(log: Path) -> str:
    return f"{log}:\n" + "\n".join(log.read_text(errors="replace").splitlines()[-20:])


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
        if cap.metadata.get("depth") == DEPTH_NOTE
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
        "sim": SIM_NAMES[ws.sim],
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
        untracked = git("ls-files", "--others", "--exclude-standard", "-z")
        for name in untracked.split("\0"):  # raw paths (status quotes odd ones)
            path = repo / name
            if name and path.is_file():
                digest.update(name.encode() + b"\0" + path.read_bytes())
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
