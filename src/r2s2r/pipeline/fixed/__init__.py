"""The fixed method: SimFoundry reconstructs the scene from one frame (stage 2,
:mod:`~r2s2r.pipeline.fixed.simfoundry`); the depth of every candidate frame then says
which objects exist (:mod:`~r2s2r.pipeline.fixed.existence`), how they are turned about
the support normal (:mod:`~r2s2r.pipeline.fixed.orientation`, a Codex VLM on ties,
:mod:`~r2s2r.pipeline.fixed.vlm`) and where they and the support's plane and outline
are (:mod:`~r2s2r.pipeline.fixed.refine`) (stage 3); the shared assembly makes them
simulation-ready with SimFoundry's collision hulls, masses and frictions (stage 4,
:func:`r2s2r.tools.objects.assemble`).

Every stage starts from an empty directory. Besides the products
(:mod:`r2s2r.pipeline.stages`)::

    s2_frames/   output.json   its frames are SimFoundry's candidates; also says
                               which one SimFoundry rebuilt from (selected,
                               decided_by, frame_notes) and which gave no objects
                               (retried)
                 parsed/       SimFoundry's scene as it made it (its stage 11
                               URDFs)
                 simfoundry/   SimFoundry's own directory; simfoundry.log its output
    s3_objects/  <object>/     its mesh, texture, collision hulls and preview.png
                 refinement.json  what the existence and orientation checks and
                               the refinement found; orientation/ what the VLM saw
    s4_scene/    output.json   every mass and friction: SimFoundry's estimates
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from r2s2r.pipeline.fixed.existence import drop_unseen
from r2s2r.pipeline.fixed.orientation import check_orientations
from r2s2r.pipeline.fixed.refine import refine_scene
from r2s2r.pipeline.fixed.simfoundry import (
    SimFoundry,
    SimFoundryConfig,
    candidate_view,
    candidates,
    copy_object,
)
from r2s2r.pipeline.fixed.vlm import CodexVLM
from r2s2r.pipeline.stages import STAGE_DIRS, read_json
from r2s2r.structs import SceneSpec
from r2s2r.tools.objects import DEFAULT_DENSITY, assemble, render_preview
from r2s2r.transforms import tilt_deg
from r2s2r.workspace import Workspace

WHY = "SimFoundry stage 11 estimate"
NO_MASS = f"no estimate: {DEFAULT_DENSITY:.0f} kg/m^3 of its convex hull"


class FixedMethod:
    """SimFoundry and multi-view refinement do stages 2, 3 and 4 (see the module
    doc)."""

    name = "fixed"
    stages: tuple[str, ...] = ("2", "3", "4")

    def __init__(self, config: SimFoundryConfig | None = None) -> None:
        self.config = config or SimFoundryConfig()

    def run_stage(self, ws: Workspace, key: str, stage_dir: Path) -> dict[str, Any]:
        """Do stage ``key`` in ``stage_dir`` (emptied first)."""
        shutil.rmtree(stage_dir)
        stage_dir.mkdir()
        stage = {"2": self.frames, "3": self.objects, "4": self.scene}[key]
        return stage(ws, stage_dir)

    def check(  # pylint: disable=unused-argument
        self, ws: Workspace, key: str, stage_dir: Path
    ) -> list[str]:
        """Nothing beyond the shared checks."""
        return []

    # ---------------------------------------------------------------- stage 2
    def frames(self, ws: Workspace, d: Path) -> dict[str, Any]:
        """SimFoundry on the candidates (again from other frames while it finds no
        objects)."""
        frames = candidates(ws.capture, self.config.max_frames)
        sf = SimFoundry(ws.capture, d / "simfoundry", self.config, d / "simfoundry.log")
        sf.prepare(ws, frames)
        sf.run()
        scene = sf.parse()
        if not scene.objects and self.config.retry_frames:
            scene = sf.retry_empty() or scene
        if not scene.objects:
            raise RuntimeError(f"SimFoundry found no objects (log: {sf.log})")
        return write_frames(ws, d, sf, [ws.frame_id(f) for f in frames], scene)

    # ---------------------------------------------------------------- stage 3
    def objects(self, ws: Workspace, d: Path) -> dict[str, Any]:
        """Stage 2's scene checked and refined against every candidate's depth, as
        SimFoundry saw it; its objects exported."""
        s2 = ws.root / STAGE_DIRS["2"]
        frames = read_json(s2 / "output.json")
        scene = SceneSpec.load(s2 / "parsed")
        views = [candidate_view(ws, ws.frame(fid)) for fid in frames["frames"]]
        scene, existence = drop_unseen(scene, views)
        vlm = CodexVLM(reasoning=self.config.codex_reasoning)
        scene, orientation = check_orientations(
            scene, views, vlm, image_dir=d / "orientation"
        )
        scene, refinement = refine_scene(scene, views)
        (d / "refinement.json").write_text(
            json.dumps(
                {
                    "existence": existence,
                    "orientation": orientation,
                    "refinement": refinement,
                },
                indent=1,
                default=_plain,
            ),
            encoding="utf-8",
        )
        export_objects(scene, d, frames["selected"])
        return {"objects": [o.name for o in scene.objects]}

    # ---------------------------------------------------------------- stage 4
    def scene(self, ws: Workspace, d: Path) -> dict[str, Any]:
        """The shared assembly, SimFoundry's collision hulls kept."""
        objects = ws.root / STAGE_DIRS["3"] / "objects.json"
        masses = assemble(ws, objects, d / "scene")["objects"]
        out = {
            "scene": "scene/scene.json",
            "objects": {  # the names are file-name safe already
                o["name"]: {
                    "mass": masses[o["name"]]["mass_kg"],
                    "friction": o.get("friction"),
                    "why": WHY if o.get("mass") else NO_MASS,
                }
                for o in read_json(objects)["objects"]
            },
        }
        (d / "output.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return {}


def write_frames(
    ws: Workspace, d: Path, sf: SimFoundry, frames: list[str], scene: SceneSpec
) -> dict[str, Any]:
    """Stage 2's product from SimFoundry's ``scene`` (see the module doc); what
    ``run.json`` records."""
    selection = sf.selection()
    selected = ws.frame_id(
        ws.capture.frame(scene.reference_camera, scene.reference_step)
    )
    empty = [
        ws.frame_id(sf.frame_of(i))
        for i in scene.provenance.get("empty_candidates", [])
    ]
    decided_by = selection.get("decided_by")
    notes = {fid: "SimFoundry found no objects from this frame" for fid in empty}
    notes[selected] = selection.get("vlm_note") or f"selected ({decided_by})"
    T = scene.T_base_support
    support = {
        "T_base_support": T.tolist(),
        "tilt_deg": round(tilt_deg(T[:3, 2]), 2),
    }
    (d / "support.json").write_text(json.dumps(support, indent=1), encoding="utf-8")
    scene.save(d / "parsed")
    output: dict[str, Any] = {
        "frames": frames,
        "selected": selected,
        "decided_by": decided_by,
        "frame_notes": notes,
        "support": {"file": "support.json"},
    }
    description = (selection.get("support") or {}).get("description")
    if description:  # what Codex's frame selection took for the support
        output["support"]["description"] = description
    if empty:
        output["retried"] = empty
    (d / "output.json").write_text(json.dumps(output, indent=1), encoding="utf-8")
    return {
        "selected": selected,
        "decided_by": decided_by,
        "objects": len(scene.objects),
    }


def export_objects(scene: SceneSpec, d: Path, reference_frame: str) -> None:
    """``objects.json`` of a refined scene of SimFoundry objects: each object's mesh,
    texture and collision hulls copied into ``<name>/`` (with a preview), placed as is,
    its mass and friction SimFoundry's; the support inline."""
    assert scene.support_extent is not None
    objects = []
    for obj in scene.objects:
        mesh, hulls = copy_object(Path(obj.asset_path), d / obj.name)
        render_preview(mesh, d / obj.name / "preview.png", up="z")
        objects.append(
            {
                "name": obj.name,
                "category": obj.category,
                "mesh": os.path.relpath(mesh, d),
                "scale": 1.0,
                "up": "z",
                "T_base_obj": obj.T_base_obj.tolist(),
                "mass": obj.mass,
                "friction": obj.friction,
                "collision": [os.path.relpath(h, d) for h in hulls],
            }
        )
    spec = {
        "support": {
            "T_base_support": scene.T_base_support.tolist(),
            "extent": [round(v, 4) for v in scene.support_extent],
        },
        "reference_frame": reference_frame,
        "objects": objects,
    }
    (d / "objects.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")


def _plain(value: Any) -> Any:
    """A numpy value as plain JSON."""
    return value.tolist() if hasattr(value, "tolist") else str(value)
