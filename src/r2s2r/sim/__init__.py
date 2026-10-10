"""Simulation of a scene, in either simulator (:data:`SIMS`), with the same calls and
the same files: settling it and replaying a capture's recording in it.

Isaac Lab runs in its own process (:mod:`~r2s2r.sim.isaac`, the code inside it in
:mod:`~r2s2r.sim.isaaclab`); MuJoCo in this one (:mod:`~r2s2r.sim.mujoco`, the scene in
:mod:`~r2s2r.sim.mjscene`). Either holds a cloth as it lies; once the bodies are at
rest, the cloths settle in Newton (:mod:`~r2s2r.sim.cloth`). What both do the same is in
:mod:`~r2s2r.sim.world`; a replay is compared with the recording by
:mod:`~r2s2r.sim.compare`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SIMS = ("isaac", "mujoco")


def _backend(sim: str) -> Any:
    # pylint: disable=import-outside-toplevel
    if sim == "isaac":
        from r2s2r.sim import isaac

        return isaac
    if sim == "mujoco":
        from r2s2r.sim import mujoco

        return mujoco
    raise ValueError(f"unknown simulator {sim!r} (one of {', '.join(SIMS)})")


def settle(
    scene_dir: str | Path,
    capture_dir: str | Path,
    out_dir: str | Path,
    sim: str,
    seconds: float | None = None,
) -> dict[str, Any]:
    """The scene's objects at rest in ``sim`` (``out_dir/scene.json``,
    ``out_dir/settle.json``), its cloths then in Newton, and how far each moved."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.sim import cloth
    from r2s2r.sim.world import SETTLE_SECONDS

    backend = _backend(sim)
    seconds = SETTLE_SECONDS if seconds is None else seconds
    result = dict(backend.settle(scene_dir, capture_dir, out_dir, seconds))
    report = cloth.settle(out_dir, capture_dir, seconds)
    return result if report is None else {**result, **report}


def replay(
    scene_dir: str | Path,
    capture_dir: str | Path,
    out_dir: str | Path,
    sim: str,
    cameras: list[str] | None = None,
    every: int = 1,
) -> dict[str, Any]:
    """The capture's static period replayed in the scene in ``sim``, every view
    compared with the recording (``out_dir/compare/``); the residuals."""
    return dict(_backend(sim).replay(scene_dir, capture_dir, out_dir, cameras, every))
