"""A coding agent: one ``codex exec`` session in a workspace it may write to.

The local Codex CLI (``gpt-6-astra``, reasoning ``medium``) runs sandboxed: it can read
anything, write only inside the workspace, and has no network. The r2s2r tools are on
its ``PATH``; it sees attached images, and can open image files it finds in the
workspace.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from r2s2r.vlm import default_codex_bin

FINAL_MESSAGE = "agent_final.md"
TRANSCRIPT = "agent_transcript.txt"


class Agent(Protocol):  # pylint: disable=too-few-public-methods
    """``agent(prompt, workdir, images) -> final message``."""

    def __call__(self, prompt: str, workdir: Path, images: Sequence[Path]) -> str: ...


@dataclass
class CodexAgent:
    """Runs a task in ``workdir`` with the Codex CLI and returns its final message; the
    transcript is kept in ``workdir``."""

    model: str = "gpt-6-astra"
    reasoning: str = "medium"
    codex_bin: str | None = None
    timeout_s: float = 3600.0

    def __call__(self, prompt: str, workdir: Path, images: Sequence[Path] = ()) -> str:
        workdir = Path(workdir).resolve()
        final = workdir / FINAL_MESSAGE
        cmd = [
            self.codex_bin or default_codex_bin(),
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "-C",
            str(workdir),
            "-s",
            "workspace-write",
            "-m",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.reasoning}"',
            *[f"--image={Path(p).resolve()}" for p in images],
            "-o",
            str(final),
            "-",
        ]
        env = dict(os.environ)
        env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
        env.setdefault("MUJOCO_GL", "egl")
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                env=env,
                check=False,
            )
            transcript = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout or b""
            partial = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
            transcript = f"timed out after {self.timeout_s:.0f} s\n{partial}"
        (workdir / TRANSCRIPT).write_text(transcript, encoding="utf-8")
        return final.read_text(encoding="utf-8") if final.exists() else ""
