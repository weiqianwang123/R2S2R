"""What the viewer shows of a run, read afresh on every request: the recording, each
stage's status, and the products worth a look: the chosen frames, the support, the
objects, their physical parameters, and the latest replay.

A stage's status is what ``run.json`` says (:mod:`r2s2r.pipeline.run`), unless its
product has since become invalid or stale (then "stopped"), or it has been "running"
without writing anything for a long time (the run was killed: "stopped" too). A running
stage's activity is an agent's last message (``codex_*.jsonl``), or the SimFoundry stage
the fixed method is at (the last ``[Stage N]`` line of ``simfoundry.log``).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from r2s2r.pipeline.stages import (
    STAGE_DIRS,
    STAGES,
    VALIDATORS,
    fresh,
    read_json,
    support_file,
)
from r2s2r.tools.geometry import load_support
from r2s2r.workspace import RUN_FILENAME, Workspace

ACTIVE_S = 15 * 60  # a running stage that wrote nothing for longer has stopped
ACTIVITY_CHARS = 140


def recording(ws: Workspace) -> dict[str, Any]:
    """The capture: cameras, frames, static period, the task."""
    cap = ws.capture
    cameras = [
        {
            "serial": c.serial,
            "role": c.role,
            "width": c.width,
            "height": c.height,
            "K": c.K.tolist(),
            "static": c.is_static,
            "stereo": c.stereo_baseline is not None,
        }
        for c in cap.cameras.values()
    ]
    return {
        "name": cap.name,
        "source": cap.source,
        "embodiment": cap.embodiment,
        "instruction": cap.instruction,
        "static_steps": list(cap.static_steps),
        "cameras": cameras,
        "frames": ws.frame_index(),
    }


def run_state(ws: Workspace) -> dict[str, Any]:
    """Stage statuses and products (everything but the recording and the robot)."""
    run = _json(ws.root / RUN_FILENAME) or {}
    dirs = {key: ws.root / name for key, name in STAGE_DIRS.items()}
    return {
        "run": str(ws.root),
        "method": run.get("method"),
        "now": time.time(),
        "stages": [
            _stage(ws, key, (run.get("stages") or {}).get(key) or {}) for key in STAGES
        ],
        "frames": _frames(dirs["2"]),
        "support": _support(ws, dirs["2"], dirs["3"]),
        "objects": _objects(ws, dirs["3"]),
        "physics": _physics(dirs["4"]),
        "replay_panels": _replay_panels(ws, dirs["6"]) or _replay_panels(ws, dirs["5"]),
        "scenes": _scenes(ws, dirs),
    }


# ------------------------------------------------------------------------- stages
def _stage(ws: Workspace, key: str, entry: dict[str, Any]) -> dict[str, Any]:
    status = entry.get("status", "pending")
    out: dict[str, Any] = {
        "key": key,
        "status": status,
        "started": entry.get("started"),
        "seconds": entry.get("seconds"),
    }
    if status == "failed":
        out["error"] = entry.get("error")
    d = ws.root / STAGE_DIRS[key]
    if status not in ("running", "done") or not d.exists():
        return out
    if status == "running":
        out["updated"] = max(
            (p.stat().st_mtime for p in d.rglob("*") if p.is_file()),
            default=d.stat().st_mtime,
        )
        if time.time() - out["updated"] > ACTIVE_S:
            out["status"] = "stopped"
        message = _agent_message(d)
        if message:
            out["activity"] = _first_sentence(message)
        elif (d / "simfoundry.log").exists():
            out["activity"] = _simfoundry_stage(d / "simfoundry.log")
    elif status == "done":
        try:
            valid = fresh(ws.root, key) and not VALIDATORS[key](ws, d)
        except Exception:  # pylint: disable=broad-except
            valid = False
        if not valid:
            out["status"] = "stopped"
    return out


def _agent_message(d: Path) -> str | None:
    """The agent's last message in the stage, if an agent works there."""
    message = None
    for path in sorted(d.glob("codex_*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") == (
                "agent_message"
            ):
                message = item.get("text")
    return message


def _simfoundry_stage(log: Path) -> str | None:
    """The SimFoundry stage a log's last ``[Stage N] <description>`` line names."""
    stage = None
    for line in log.read_text(errors="replace").splitlines():
        if re.match(r"\[Stage \w+\] (?!cmd:)", line):
            stage = line.strip()
    return None if stage is None else f"SimFoundry {stage[1:].replace(']', ':', 1)}"


def _first_sentence(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", " ".join(text.split()))
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    if len(sentence) > ACTIVITY_CHARS:
        sentence = sentence[: ACTIVITY_CHARS - 1].rstrip() + "…"
    return sentence


# ----------------------------------------------------------------------- products
def _frames(d: Path) -> list[dict[str, Any]]:
    out = _json(d / "output.json")
    if not isinstance(out, dict):
        return []
    notes = out.get("frame_notes") or {}
    return [
        {
            "id": fid,
            "note": notes.get(fid, ""),
            "selected": fid == out.get("selected"),
        }
        for fid in out.get("frames", [])
    ]


def _support(ws: Workspace, d: Path, objects_dir: Path) -> dict[str, Any] | None:
    """Stage 2's support; its extent, when stage 2 left it out, the objects file's."""
    out = _json(d / "output.json")
    name = support_file(out)
    candidates = [d / name] if name else sorted(d.glob("support*.json"))
    objects = _json(objects_dir / "objects.json")
    inline = objects.get("support") if isinstance(objects, dict) else None
    for path in candidates:
        support = _json(path)
        if isinstance(support, dict) and "T_base_support" in support:
            return {
                "extent": support.get("extent")
                or (inline.get("extent") if isinstance(inline, dict) else None),
                "tilt_deg": support.get("tilt_deg"),
                "rms_m": support.get("inlier_rms_m"),
                "description": (
                    (out.get("support") or {}).get("description")
                    if isinstance(out, dict)
                    else None
                ),
                "overlays": [
                    _rel(ws, p)
                    for p in sorted(path.parent.glob(f"{path.stem}_*@*.png"))
                ],
            }
    return None


def _objects(ws: Workspace, d: Path) -> list[dict[str, Any]]:
    """The objects file's objects (or, before it exists, every generated mesh), each
    with its preview, its up axis and its latest fit; an articulated one with its
    number of joints and the objects file to show it whole from (``model``); whether
    it is a cloth."""
    fits = []  # (mesh, fit directory, summary), oldest first
    for path in sorted(d.rglob("fit.json"), key=lambda p: p.stat().st_mtime):
        fit = _json(path)
        if not (isinstance(fit, dict) and fit.get("mesh")):
            continue
        summary = {
            "iou": fit.get("mean_iou"),
            "overlays": [
                {
                    "frame": fid,
                    "iou": row.get("iou"),
                    "path": _rel(ws, path.parent / f"{fid}.png"),
                }
                for fid, row in (fit.get("frames") or {}).items()
            ],
        }
        fits.append(
            (str(Path(fit["mesh"]).resolve()), str(path.parent.relative_to(d)), summary)
        )
    spec = _json(d / "objects.json")
    if isinstance(spec, dict):
        meshes = []
        for obj in spec.get("objects", []):
            mesh = Path(obj.get("mesh", ""))
            meshes.append(
                (
                    str(obj.get("name")),
                    mesh if mesh.is_absolute() else d / mesh,
                    str(obj.get("up", "z")),
                    len(obj.get("joints") or []),
                    obj.get("cloth") is not None,
                )
            )
    else:  # generated meshes are y-up
        previews = sorted(d.rglob("preview.png"), key=lambda p: p.stat().st_mtime)
        latest = {p.parent.name: p.parent / "mesh.glb" for p in previews}
        meshes = [(name, mesh, "y", 0, False) for name, mesh in latest.items()]
    out = []
    for name, mesh, up, joints, cloth in meshes:
        mesh = mesh.resolve()
        preview = mesh.parent / "preview.png"
        # The latest fit of this mesh, else the latest whose directory names the object.
        fit = next((f for m, _, f in reversed(fits) if m == str(mesh)), None) or next(
            (f for _, where, f in reversed(fits) if name in where), {}
        )
        out.append(
            {
                "name": name,
                "glb": _rel(ws, mesh) if mesh.exists() else None,
                "up": up,
                "preview": _rel(ws, preview) if preview.exists() else None,
                "iou": fit.get("iou"),
                "overlays": fit.get("overlays", []),
                # An articulated object is shown whole (all its parts, its joints
                # movable) from the objects file.
                "joints": joints,
                "model": _rel(ws, d / "objects.json") if joints else None,
                "cloth": cloth,
            }
        )
    return out


def _physics(d: Path) -> list[dict[str, Any]]:
    """Every object's mass, friction and why (stage 4's ``output.json``, else what its
    objects file has set so far), with an articulated object's joints and a cloth's
    material."""
    spec = _json(d / "objects.json")
    written = {
        o.get("name"): o
        for o in (spec.get("objects", []) if isinstance(spec, dict) else [])
    }
    out = _json(d / "output.json")
    if isinstance(out, dict) and isinstance(out.get("objects"), dict):
        rows = list(out["objects"].items())
    else:  # before output.json: what the objects file has set so far
        rows = [
            (o.get("name"), {**o, "why": None})
            for o in (spec.get("objects", []) if isinstance(spec, dict) else [])
            if o.get("mass") is not None
        ]
    return [
        {
            "name": name,
            "mass": o.get("mass"),
            "friction": o.get("friction"),
            "why": o.get("why"),
            "joints": [
                {
                    "name": j.get("name"),
                    "type": j.get("type"),
                    "limits": j.get("limits"),
                    "position": j.get("position"),
                    "why": _joint_why(o.get("joints"), j.get("name")),
                }
                for j in written.get(name, {}).get("joints") or []
            ],
            "cloth": written.get(name, {}).get("cloth"),
        }
        for name, o in rows
    ]


def _joint_why(written: Any, joint: str) -> str | None:
    """The reason the method wrote for ``joint``, if any (output.json's ``joints``: a
    reason per joint name)."""
    entry = written.get(joint) if isinstance(written, dict) else None
    return entry if isinstance(entry, str) else None


def _replay_panels(ws: Workspace, d: Path) -> list[str]:
    """The latest replay's panels (real | sim | blend | depth residual per frame)."""
    compares = sorted(d.rglob("compare/compare.json"), key=lambda p: p.stat().st_mtime)
    if not compares:
        return []
    return [_rel(ws, p) for p in sorted((compares[-1].parent / "frames").glob("*.png"))]


def _scenes(ws: Workspace, dirs: dict[str, Path]) -> list[dict[str, Any]]:
    """What the 3D view can show, newest first: the stages' scenes and objects files,
    and the support alone (when its file has an extent: a fixed run's holds only the
    plane)."""
    found = []
    for key, d in dirs.items():
        if not d.exists():
            continue
        for name in ("scene/scene.json", "parsed/scene.json", "objects.json"):
            if (d / name).exists():
                found.append((key, d / name))
    support_name = support_file(_json(dirs["2"] / "output.json"))
    path = dirs["2"] / support_name if support_name else None
    if path is not None and path.exists() and load_support(path)[1] is not None:
        found.append(("2", path))
    out: list[dict[str, Any]] = [
        {
            "label": f"stage {key}: {p.relative_to(ws.root)}",
            "path": _rel(ws, p.parent if p.name == "scene.json" else p),
            "mtime": p.stat().st_mtime,
        }
        for key, p in found
    ]
    out.sort(key=lambda s: s["mtime"], reverse=True)
    return out


# ---------------------------------------------------------------------- helpers
def _rel(ws: Workspace, path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ws.root))
    except ValueError:
        return str(path)


def _json(path: Path) -> Any:
    """:func:`read_json`, or None when there is nothing to read yet."""
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None
