"""Scoring and saving a pick rollout, the same way for every deployment target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r2s2r.policy.robot import CommandLog

LIFT_SUCCESS_M = 0.05  # the target must rise this much to count as picked


def score_lift(
    before: dict[str, NDArray], after: dict[str, NDArray], target: str
) -> dict[str, Any]:
    """Per-object height change and displacement, and whether ``target`` rose."""
    lift = {n: float(after[n][2] - before[n][2]) for n in before}
    return {
        "object_lift_m": lift,
        "object_shift_m": {
            n: float(np.linalg.norm(after[n] - before[n])) for n in before
        },
        "success": bool(lift[target] > LIFT_SUCCESS_M),
    }


def summarize(result: dict[str, Any]) -> str:
    """One line: verdict, what the policy picked, how far every object rose."""
    verdict = "SUCCESS" if result["success"] else "FAILURE"
    lift = ", ".join(
        f"{n} {dz * 100:+.1f} cm" for n, dz in result["object_lift_m"].items()
    )
    return f"{verdict}: picked {result['policy']['target']!r}; lifted {lift}"


def save_rollout(out_dir: Path, result: dict[str, Any], log: CommandLog) -> None:
    """``result.json`` and ``commands.json`` (every command sent)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "commands.json").write_text(json.dumps(log.as_dict()), encoding="utf-8")
