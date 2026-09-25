"""Ask a vision-language model about images, through the local Codex CLI.

The same CLI SimFoundry's fork uses (``simfoundry/models/codex_vlm.py``): one read-only
``codex exec`` call with the images attached; the final message is the answer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

DESKTOP_APP_CODEX = "/usr/lib/chatgpt/resources/codex"

PROMPT_TEMPLATE = """You are serving as a vision-language model behind a program's API.
Answer the request below using only the attached image(s) and the text.
Do not run shell commands, read or write files, or use any tool.
Your final message is passed verbatim to a parser, so follow the requested
output format exactly and add nothing before or after it.

<request>
{prompt}
</request>
"""


def default_codex_bin() -> str:
    """``SIMFOUNDRY_CODEX_BIN``, the desktop app's bundled CLI, or ``codex``."""
    explicit = os.environ.get("SIMFOUNDRY_CODEX_BIN")
    if explicit:
        return explicit
    if os.access(DESKTOP_APP_CODEX, os.X_OK):
        return DESKTOP_APP_CODEX
    found = shutil.which("codex")
    if found is None:
        raise FileNotFoundError("no codex executable; set SIMFOUNDRY_CODEX_BIN")
    return found


@dataclass
class CodexVLM:
    """``vlm(prompt, images) -> answer`` through ``codex exec``."""

    model: str = "gpt-6-astra"
    reasoning: str = "medium"
    codex_bin: str | None = None
    timeout_s: float = 600.0
    retries: int = 3

    def __call__(self, prompt: str, images: list[Path]) -> str:
        last_error = ""
        for _ in range(self.retries):
            with tempfile.TemporaryDirectory(prefix="r2s2r_vlm_") as tmp:
                answer = Path(tmp) / "answer.txt"
                cmd = [
                    self.codex_bin or default_codex_bin(),
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "-C",
                    tmp,
                    "-s",
                    "read-only",
                    "-m",
                    self.model,
                    "-c",
                    f'model_reasoning_effort="{self.reasoning}"',
                    *[f"--image={Path(p).resolve()}" for p in images],
                    "-o",
                    str(answer),
                    "-",
                ]
                try:
                    proc = subprocess.run(
                        cmd,
                        input=PROMPT_TEMPLATE.format(prompt=prompt),
                        text=True,
                        capture_output=True,
                        timeout=self.timeout_s,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    last_error = f"timed out after {self.timeout_s:.0f} s"
                    continue
                text = answer.read_text() if answer.exists() else ""
                if proc.returncode == 0 and text.strip():
                    return text.strip()
                last_error = (proc.stderr or proc.stdout)[-500:]
        raise RuntimeError(f"Codex [{self.model}] gave no answer: {last_error}")
