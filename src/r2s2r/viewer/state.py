"""What the viewer shows of a workspace, read afresh on every request: the recording,
each stage's status, and the products worth a look: the chosen frames, the support, the
objects, their physical parameters, and the latest replay."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from r2s2r.agentic.runner import STAGE_DIRS, VALIDATORS
from r2s2r.agentic.workspace import Workspace

ACTIVE_S = 15 * 60  # a stage written to this recently counts as running
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
    frames = [
        {
            "id": ws.frame_id(f),
            "camera": cap.cameras[f.camera].role,
            "step": f.step,
            "image": _rel(ws, ws.image_path(f)),
            "depth": f.depth_image is not None,
            "static": ws.is_static(f),
            "T_base_cam": np.round(f.T_base_cam, 5).tolist(),
        }
        for f in sorted(cap.frames, key=lambda f: (f.step, f.camera))
    ]
    return {
        "name": cap.name,
        "source": cap.source,
        "embodiment": cap.embodiment,
        "instruction": cap.instruction,
        "static_steps": list(cap.static_steps),
        "cameras": cameras,
        "frames": frames,
    }


def workspace_state(ws: Workspace) -> dict[str, Any]:
    """Stage statuses and products (everything but the recording and the robot)."""
    run = _json(ws.root / "run.json") or {}
    dirs = {key: ws.root / name for key, name in STAGE_DIRS.items()}
    return {
        "workspace": str(ws.root),
        "now": time.time(),
        "stages": [
            _stage(
                ws,
                key,
                d,
                run.get("stages", {}).get(key),
                run.get("started", {}).get(key),
            )
            for key, d in dirs.items()
        ],
        "frames": _frames(dirs["2"]),
        "support": _support(ws, dirs["2"]),
        "objects": _objects(ws, dirs["3"]),
        "physics": _physics(dirs["4"]),
        "replay_panels": _replay_panels(ws, dirs["6"]) or _replay_panels(ws, dirs["5"]),
        "scenes": _scenes(ws, dirs),
    }


# ------------------------------------------------------------------------- stages
def _stage(
    ws: Workspace,
    key: str,
    d: Path,
    run_entry: dict[str, Any] | None,
    started: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"key": key, "status": "pending"}
    if not d.exists():
        return out
    mtimes = [p.stat().st_mtime for p in d.rglob("*") if p.is_file()] or [
        d.stat().st_mtime
    ]
    # The runner records when it starts a stage; files copied into the stage keep older
    # times, so the oldest file is only a fallback.
    out["started"], out["updated"] = started or min(mtimes), max(mtimes)
    failed, open_turn, message = _agent(d)
    try:
        problems = VALIDATORS[key](ws, d)
    except Exception as exc:  # pylint: disable=broad-except
        problems = [f"{type(exc).__name__}: {exc}"]
    recent = time.time() - out["updated"] < ACTIVE_S
    if failed:
        out["status"] = "failed"
    elif (open_turn or (key == "5" and problems)) and recent:
        out["status"] = "running"  # the agent is still at it, whatever it wrote
    elif not problems:
        out["status"] = "done"
    else:
        out["status"] = "stopped"
    if out["status"] == "running" and message:
        out["activity"] = _first_sentence(message)
    if run_entry:
        out["seconds"] = run_entry.get("seconds")
    return out


def _agent(d: Path) -> tuple[bool, bool, str | None]:
    """Whether a turn failed, whether the last turn is still open, and the agent's last
    message."""
    failed = open_turn = False
    message = None
    for path in sorted(d.glob("codex_*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind in ("thread.started", "turn.started"):
                open_turn = True
            elif kind == "turn.completed":
                open_turn = False
            elif kind == "turn.failed":
                failed, open_turn = True, False
            elif kind == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    message = item.get("text")
    return failed, open_turn, message


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
    return [{"id": fid, "note": notes.get(fid, "")} for fid in out.get("frames", [])]


def _support(ws: Workspace, d: Path) -> dict[str, Any] | None:
    out = _json(d / "output.json")
    name = (out.get("support") or {}).get("file") if isinstance(out, dict) else None
    candidates = [d / name] if name else sorted(d.glob("support*.json"))
    for path in candidates:
        support = _json(path)
        if isinstance(support, dict) and "T_base_support" in support:
            return {
                "extent": support.get("extent"),
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
    with its preview and its latest fit."""
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
                (str(obj.get("name")), mesh if mesh.is_absolute() else d / mesh)
            )
    else:
        previews = sorted(d.rglob("preview.png"), key=lambda p: p.stat().st_mtime)
        latest = {p.parent.name: p.parent / "mesh.glb" for p in previews}
        meshes = list(latest.items())
    out = []
    for name, mesh in meshes:
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
                "preview": _rel(ws, preview) if preview.exists() else None,
                "iou": fit.get("iou"),
                "overlays": fit.get("overlays", []),
            }
        )
    return out


def _physics(d: Path) -> list[dict[str, Any]]:
    out = _json(d / "output.json")
    if isinstance(out, dict) and isinstance(out.get("objects"), dict):
        return [
            {
                "name": name,
                "mass": o.get("mass"),
                "friction": o.get("friction"),
                "why": o.get("why"),
            }
            for name, o in out["objects"].items()
        ]
    spec = _json(d / "objects.json")  # before output.json: what it has set so far
    if isinstance(spec, dict):
        return [
            {
                "name": o.get("name"),
                "mass": o.get("mass"),
                "friction": o.get("friction"),
                "why": None,
            }
            for o in spec.get("objects", [])
            if o.get("mass") is not None
        ]
    return []


def _replay_panels(ws: Workspace, d: Path) -> list[str]:
    """The latest replay's panels (real | sim | blend | depth residual per frame)."""
    compares = sorted(d.rglob("compare/compare.json"), key=lambda p: p.stat().st_mtime)
    if not compares:
        return []
    return [_rel(ws, p) for p in sorted((compares[-1].parent / "frames").glob("*.png"))]


def _scenes(ws: Workspace, dirs: dict[str, Path]) -> list[dict[str, Any]]:
    """What the 3D view can show, newest first: the stages' scenes and objects files,
    and the support alone."""
    found = []
    for key, d in dirs.items():
        if not d.exists():
            continue
        for name in ("scene/scene.json", "objects.json"):
            if (d / name).exists():
                found.append((key, d / name))
    output = _json(dirs["2"] / "output.json")
    support_name = (
        (output.get("support") or {}).get("file") if isinstance(output, dict) else None
    )
    if support_name and (dirs["2"] / support_name).exists():
        found.append(("2", dirs["2"] / support_name))
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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
