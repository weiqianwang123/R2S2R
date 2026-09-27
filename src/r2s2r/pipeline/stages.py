"""The stages of a run, where each leaves its product, and the checks every method's
products must pass (a method may add its own, :class:`r2s2r.pipeline.run.Method`).

=====  ==============  ==============================================================
stage  directory       product
=====  ==============  ==============================================================
1      ``inputs/``     the capture (:mod:`r2s2r.pipeline.workspace`)
2      ``s2_frames``   ``output.json``: ``frames`` (ids of frames of the static
                       period) and ``support`` (``file``: a support JSON, its
                       ``T_base_support`` rigid)
3      ``s3_objects``  ``objects.json``: an objects file (:mod:`r2s2r.tools.objects`)
4      ``s4_scene``    ``scene/``: a simulation-ready scene (SceneSpec) with its
                       objects inside; ``output.json``: ``objects`` (mass, friction,
                       why)
5      ``s5_settle``   ``scene/``: the scene settled in Isaac Lab, its objects stage
                       4's; ``output.json``: how far each moved. The same for every
                       method (:mod:`r2s2r.pipeline.run`)
6      ``s6_refine``   ``scene/`` and ``report.md`` (a method may leave it out)
=====  ==============  ==============================================================

A stage is done when its product passes the checks and is newer than the product of the
stage before, so redoing a stage makes the ones after it stale. After the last stage,
the final scene (stage 6's, else stage 5's) is replayed against the recording, into
``<its stage dir>/final_replay``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from r2s2r.pipeline.workspace import Workspace
from r2s2r.structs import SceneSpec
from r2s2r.tools.geometry import UP_ROTATIONS, load_support
from r2s2r.transforms import is_rigid

STAGES = ("2", "3", "4", "5", "6")
STAGE_NAMES = {
    "2": "frames",
    "3": "objects",
    "4": "scene",
    "5": "settle",
    "6": "refine",
}
STAGE_DIRS = {
    "2": "s2_frames",
    "3": "s3_objects",
    "4": "s4_scene",
    "5": "s5_settle",
    "6": "s6_refine",
}
SETTLE = "5"  # the stage every method shares
# The files of each stage's product (stage 2's also the support file it names).
PRODUCTS = {
    "2": ("output.json",),
    "3": ("objects.json",),
    "4": ("scene/scene.json", "output.json"),
    "5": ("scene/scene.json", "output.json"),
    "6": ("scene/scene.json", "report.md"),
}


def read_json(path: str | Path) -> Any:
    """The JSON in ``path``; ``ValueError`` saying what is wrong when it is missing or
    not JSON."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path.name} not written yet")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc


# -------------------------------------------------------------------------- checks
def check_frames(ws: Workspace, d: Path) -> list[str]:
    """Problems with stage 2's product."""
    try:
        out = read_json(d / "output.json")
    except ValueError as exc:
        return [str(exc)]
    if not isinstance(out, dict):
        return ["output.json is not a JSON object"]
    problems = []
    frames = out.get("frames")
    if not isinstance(frames, list) or not frames:
        problems.append("output.json names no frames")
        frames = []
    for fid in frames:
        try:
            frame = ws.frame(str(fid))
        except (KeyError, ValueError) as exc:
            problems.append(str(exc))
            continue
        if not ws.capture.in_static(frame.step):
            problems.append(f"{fid} is not a frame of the static period")
    support = out.get("support")
    support = support.get("file") if isinstance(support, dict) else None
    if not support:
        problems.append("output.json names no support file")
    else:
        problems += check_support(d / support)
    return problems


def check_support(source: Path | dict[str, Any]) -> list[str]:
    """Problems with a support (a support JSON, or its dict): ``T_base_support`` must be
    rigid, and the ``extent``, if given, positive."""
    where = "the support" if isinstance(source, dict) else f"support {source}"
    try:
        T, extent = load_support(source)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return [f"{where}: {exc}"]
    if not is_rigid(T):
        return [f"{where}: T_base_support is not a rigid transform"]
    if extent is not None and min(extent) <= 0:
        return [f"{where}: its extent is not positive"]
    return []


def check_objects(_: Workspace, d: Path) -> list[str]:
    """Problems with stage 3's product."""
    try:
        spec = read_json(d / "objects.json")
    except ValueError as exc:
        return [str(exc)]
    if not isinstance(spec, dict):
        return ["objects.json is not a JSON object"]
    base = d.resolve()  # what relative paths in it start from
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
        if not is_rigid(obj.get("T_base_obj", np.zeros((4, 4)))):
            problems.append(f"{name}: T_base_obj must be a rigid 4x4 transform")
        if obj.get("up", "z") not in UP_ROTATIONS:
            problems.append(f"{name}: up must be one of {sorted(UP_ROTATIONS)}")
    support = spec.get("support")
    if isinstance(support, str):
        problems += check_support(base / support)
    elif isinstance(support, dict):
        problems += check_support(support)
    else:
        problems.append("objects.json has no support")
    return problems


def check_scene(extra: str) -> Callable[[Workspace, Path], list[str]]:
    """The check of a stage whose product is ``scene/`` and the file ``extra``."""

    def check(_: Workspace, d: Path) -> list[str]:
        problems = [] if (d / extra).exists() else [f"{extra} missing"]
        try:
            scene = SceneSpec.load(d / "scene")
        except (OSError, KeyError, ValueError) as exc:
            return problems + [f"{d / 'scene'}: {exc}"]
        if not scene.objects:
            problems.append("the scene has no objects")
        for obj in scene.objects:
            if not Path(obj.asset_path).exists():
                problems.append(f"{obj.name}: asset {obj.asset_path} missing")
            if not obj.mass or obj.mass <= 0:
                problems.append(f"{obj.name}: no mass")
        return problems

    return check


VALIDATORS: dict[str, Callable[[Workspace, Path], list[str]]] = {
    "2": check_frames,
    "3": check_objects,
    "4": check_scene("output.json"),
    "5": check_scene("output.json"),
    "6": check_scene("report.md"),
}


# ----------------------------------------------------------------------- staleness
def product_files(root: Path, key: str) -> list[Path]:
    """The files of a stage's product (whether they exist or not)."""
    d = root / STAGE_DIRS[key]
    files = [d / name for name in PRODUCTS[key]]
    if key == "2":
        try:
            support = read_json(files[0]).get("support").get("file")
        except (ValueError, AttributeError):
            support = None
        if isinstance(support, str):
            files.append(d / support)
    return files


def newer_than_previous(root: Path, key: str) -> bool:
    """Whether every file of the stage's product was written after all of the previous
    stage's (stage 2 has none before it)."""
    k = STAGES.index(key)
    if k == 0:
        return True
    ours, theirs = product_files(root, key), product_files(root, STAGES[k - 1])
    if not all(p.exists() for p in ours + theirs):
        return False
    return min(p.stat().st_mtime for p in ours) >= max(
        p.stat().st_mtime for p in theirs
    )
