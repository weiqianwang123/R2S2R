"""Jobs that run in another conda environment.

SAM3 and CoACD run in ``simfoundry``, Hunyuan3D in ``hunyuan``; each job is a script
under ``scripts/agentic/`` that reads a job JSON and writes a result JSON. The
SimFoundry submodule is on their path, and its models are used from there.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from r2s2r.reconstruct.simfoundry import DEFAULT_SIMFOUNDRY_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

JOBS_DIR = REPO_ROOT / "scripts" / "agentic"
ENV_SIMFOUNDRY = "simfoundry"
ENV_MESH = "hunyuan"


def mamba_exe() -> str:
    """The mamba (or conda) executable."""
    for name in ("mamba", "conda"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (Path.home() / "miniforge3" / "bin" / "mamba",):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("neither mamba nor conda is on PATH")


def run_env_job(
    script: str, job: dict[str, Any], workdir: Path, env_name: str
) -> dict[str, Any]:
    """Run ``scripts/agentic/<script>`` in ``env_name`` on ``job``; its result.

    The job's output goes to ``workdir/logs/``; on failure, its tail is raised.
    """
    workdir = Path(workdir)
    logs = workdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{Path(script).stem}_{stamp}_{os.getpid()}"
    job_path, result_path = logs / f"{stem}.json", logs / f"{stem}_result.json"
    log_path = logs / f"{stem}.log"
    job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(DEFAULT_SIMFOUNDRY_DIR), env.get("PYTHONPATH", "")) if p
    )
    env.pop("VIRTUAL_ENV", None)  # r2s2r's venv must not shadow the conda env
    cmd = [
        mamba_exe(),
        "run",
        "-n",
        env_name,
        "python",
        str(JOBS_DIR / script),
        str(job_path),
        str(result_path),
    ]
    logger.info("running %s in %s (log: %s)", script, env_name, log_path)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0 or not result_path.exists():
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(
            f"{script} failed in env {env_name} (exit {proc.returncode}); "
            f"log {log_path}:\n{tail}"
        )
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    return result
