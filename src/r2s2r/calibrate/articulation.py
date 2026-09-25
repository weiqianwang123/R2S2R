"""Recalibrate articulated objects whose joints do not explain the video.

For every articulated object, the backend's model is evaluated against the video
(:func:`r2s2r.calibrate.tools.evaluate`). A model that passes the check is kept, with
its limits widened to the range seen. For one that fails, a coding agent gets a
workspace with the evidence and the tools; it proposes joint models (other axes, types,
hinge positions, which geometry moves) and evaluates them. Then, outside the agent's
reach, the agent's choice and the best candidates of its ledger are evaluated again, and
a candidate replaces the model only if it passes the check and its score beats the
backend's model by ``min_gain``. Otherwise the backend's model stays.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Collection

from r2s2r.calibrate.agent import Agent
from r2s2r.calibrate.tools import (
    LEDGER_FILE,
    Workspace,
    create_workspace,
    dumps,
    evaluate,
)
from r2s2r.io.rgbd import save_depth_steps
from r2s2r.reconstruct.joints import JointFitConfig, scale_view
from r2s2r.structs import DepthView, ObjectSpec, SceneSpec

CHOICE_FILE = "choice.json"
TASK_FILE = "TASK.md"
STEPS_FILE = "steps.npz"


@dataclass
class CalibrationConfig:
    """Knobs of :func:`calibrate_articulation`."""

    budget: int = 10  # evaluations the agent may run
    # A candidate is accepted when it passes the check and its score is at least this
    # share below the backend model's.
    min_gain: float = 0.05
    recheck: int = 3  # the ledger's best candidates re-evaluated besides the choice
    evidence_steps: int = 4
    fit: JointFitConfig = field(default_factory=JointFitConfig)


def calibrate_articulation(
    scene_dir: str | Path,
    capture_dir: str | Path,
    steps: dict[int, list[DepthView]],
    rest_steps: Collection[int],
    out_dir: str | Path,
    agent: Agent | None,
    config: CalibrationConfig | None = None,
    force: bool = False,
) -> tuple[SceneSpec, dict[str, Any]]:
    """Calibrate every articulated object of the scene saved in ``scene_dir`` against
    the video's depth views ``steps`` (``rest_steps``: the static period), one workspace
    per object in ``out_dir``. ``force`` hands models that pass the check to the agent
    too.

    Returns the scene with each articulated object's final URDF (limits widened to the
    range seen when its check passes) and a report per object.
    """
    cfg = config or CalibrationConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The tools read the views from disk, at the fit's resolution.
    steps_file = out_dir / STEPS_FILE
    save_depth_steps(
        steps_file,
        {t: [scale_view(v, cfg.fit.view_scale) for v in vs] for t, vs in steps.items()},
    )
    scene = SceneSpec.load(scene_dir)
    objects = list(scene.objects)
    report: dict[str, Any] = {}
    for i, obj in enumerate(scene.objects):
        if not obj.articulated:
            continue
        ws = create_workspace(
            out_dir / obj.name, scene_dir, capture_dir, obj.name, steps_file, rest_steps
        )
        baseline = evaluate(
            ws, ws.urdf, ws.root / "baseline", cfg.fit, cfg.evidence_steps
        )
        entry: dict[str, Any] = {"workspace": str(ws.root), "baseline": brief(baseline)}
        final = baseline
        if baseline["consistent"] and not force:
            entry["decision"] = "kept: the backend's joints explain the video"
        elif agent is None:
            entry["decision"] = "kept: no agent to recalibrate"
        else:
            prompt = _prompt(ws, obj, baseline, cfg)
            (ws.root / TASK_FILE).write_text(prompt, encoding="utf-8")
            images = [Path(e["image"]) for e in baseline["evidence"]]
            message = agent(prompt, ws.root, images)
            verified = _verify(ws, cfg)
            entry["agent"] = {
                "choice": _read_json(ws.root / CHOICE_FILE),
                "final_message": message.strip()[-3000:],
            }
            entry["verified"] = {name: brief(r) for name, r in verified.items()}
            best = _accept(baseline, verified, cfg)
            if best is None:
                entry["decision"] = (
                    "kept: no candidate passed the check and beat the backend's "
                    f"score by {cfg.min_gain:.0%}"
                )
            else:
                final = verified[best]
                entry["decision"] = f"accepted {best}"
        entry["final"] = brief(final) | {"report": final["report"]}
        objects[i] = replace(obj, asset_path=final["fitted_urdf"])
        report[obj.name] = entry
    return replace(scene, objects=objects), report


def final_fits(report: dict[str, Any]) -> dict[str, Any]:
    """The joint fit of each object's final model, from the report of
    :func:`calibrate_articulation` (the per-object ``joints`` and ``check`` of
    :func:`~r2s2r.reconstruct.joints.fit_joints`)."""
    return {
        name: json.loads(Path(e["final"]["report"]).read_text(encoding="utf-8"))["fit"]
        for name, e in report.items()
    }


def brief(summary: dict[str, Any]) -> dict[str, Any]:
    """The gist of an evaluation: score, verdict and joints."""
    return {
        "urdf": summary["urdf"],
        "score_mm": summary["score_mm"],
        "consistent": summary["consistent"],
        "unexplained_steps": summary["unexplained_steps"],
        "collision_mismatch": summary["collision_mismatch"],
        "joints": {
            name: {
                k: j.get(k)
                for k in ("type", "axis_base", "origin_base", "observed_range")
            }
            for name, j in summary["joints"].items()
            if j.get("type") != "fixed"
        },
    }


def _verify(ws: Workspace, cfg: CalibrationConfig) -> dict[str, dict[str, Any]]:
    """Evaluate the agent's choice and the ledger's best consistent candidates again,
    each into ``verified/`` (the agent's own numbers are only a hint of what to
    recheck)."""
    paths = []
    choice = _read_json(ws.root / CHOICE_FILE)
    if isinstance(choice, dict) and isinstance(choice.get("candidate"), str):
        paths.append(ws.root / choice["candidate"])
    ledger = []
    if (ws.root / LEDGER_FILE).exists():
        for line in (ws.root / LEDGER_FILE).read_text(encoding="utf-8").splitlines():
            try:
                ledger.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    ranked = sorted(
        (
            e
            for e in ledger
            if isinstance(e, dict)
            and e.get("consistent") is True
            and isinstance(e.get("score_mm"), (int, float))
            and isinstance(e.get("urdf"), str)
        ),
        key=lambda e: e["score_mm"],
    )
    paths += [Path(e["urdf"]) for e in ranked[: cfg.recheck]]
    model_dir, original = ws.model_dir.resolve(), Path(ws.urdf).resolve()
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        path = path.resolve()
        name = path.stem
        if (
            name in out
            or path == original
            or not path.is_file()
            or model_dir not in path.parents
        ):
            continue
        try:
            out[name] = evaluate(
                ws, path, ws.root / "verified" / name, cfg.fit, cfg.evidence_steps
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"candidate {path} could not be evaluated: {exc}", file=sys.stderr)
    return out


def _accept(
    baseline: dict[str, Any],
    verified: dict[str, dict[str, Any]],
    cfg: CalibrationConfig,
) -> str | None:
    """The best-scoring candidate that passes the check, beats the baseline, and whose
    collision meshes match its visual ones."""
    base = baseline["score_mm"]
    needed = float("inf") if base is None else base * (1.0 - cfg.min_gain)
    ok = {
        name: r["score_mm"]
        for name, r in verified.items()
        if r["consistent"]
        and not r["collision_mismatch"]
        and r["score_mm"] is not None
        and r["score_mm"] <= needed
    }
    return min(ok, key=ok.__getitem__) if ok else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


PROMPT = """\
You are calibrating the articulation model of one object in a scene reconstructed for
robot simulation. Work in the current directory, a workspace set up for this.

## The situation
A reconstruction pipeline modelled the object "{category}" (scene object {name}) as
an articulated URDF, model/{urdf_name}, guessing its joints from a single static
view. A video of the scene recorded afterwards shows the robot interacting with it.
A fixed verifier fitted the model's joints to every sampled step of the video, and
checked, step by step, how well the object's surroundings are explained with the
joints at their fitted positions, against the static period at the start of the
video. {verdict}

The verifier's evaluation of this model (also in baseline/report.json):
{baseline}

The attached images are its worst steps (baseline/step_*.png): the camera photo, the
measured depth with the robot removed, the model's depth with its joints as fitted,
and what is left unexplained (red: surfaces the camera sees that the model lacks;
blue: model surfaces the camera sees past). The same appears in "evidence" as boxes
of points in the object frame (box_obj) and the robot base frame (box_base).

## Your task
Find the joint model that best explains the video: propose candidates, evaluate each,
and pick the best. A candidate may change a joint's type (prismatic, revolute,
continuous, fixed), its axis, its origin (where a hinge line runs), its limits, the
number of joints, and which geometry belongs to which link (splitting or regrouping
meshes). Keep the object's pose, overall shape and size. Reason from the evidence
and the geometry; do not decide what the object is from its name.

## Tools (run them from this directory)
- `{r2s2r} urdf-info model/<file>.urdf --workspace .` prints the links' boxes and the
  joints' axes and origins in the object frame and the base frame.
- `{r2s2r} joints-eval . model/<file>.urdf` fits the candidate's joints to the video
  (about half a minute), prints a JSON summary like the one above, and writes
  evals/<file>/ with report.json and images of the worst steps. score_mm: lower
  explains the video better. consistent: no step is explained much worse than the
  static period.
- `{python}` has numpy, scipy and trimesh, for editing meshes.

## Frames
Base frame: x forward from the robot, y to its left, z up; metres. Object frame: the
URDF's root link, with T_base_obj = {T_base_obj}. A joint's <axis> is in its own
frame, after its <origin>; urdf-info reports the resulting axes in both frames, so
check a candidate there after editing it. A joint's <origin> is also the frame of its
child link: moving it moves the child's geometry, unless the child's <visual> and
<collision> origins are moved back by the same amount.

## Rules
- Write each candidate as a new file model/cand_<short_name>.urdf, starting from a
  copy of model/{urdf_name}. Do not edit model/{urdf_name}, workspace.json, the steps
  file, baseline/ or {ledger}. New meshes go under model/parts/<short_name>/; a link
  whose geometry changes needs matching <collision> meshes (convex pieces) too.
- Evaluate every candidate with joints-eval, at most {budget} evaluations in all.
- Finish by writing choice.json: {{"candidate": "model/cand_<short_name>.urdf",
  "reason": "..."}}, or {{"candidate": null, "reason": "..."}} if nothing beats the
  original. End with a short summary of what you tried and the scores.

The verifier then evaluates your choice and your best candidates again itself: one
is accepted only if it is consistent, its score is at least {gain:.0%} below the
original's ({baseline_score} mm), and every link's collision meshes match its visual
ones (joints-eval lists those that do not under collision_mismatch).
"""


def _prompt(
    ws: Workspace, obj: ObjectSpec, baseline: dict[str, Any], cfg: CalibrationConfig
) -> str:
    if baseline["consistent"]:
        verdict = (
            "The model passed: every step is explained about as well as the static "
            "period. Look for a model that explains the video even better, or keep it."
        )
    else:
        verdict = (
            "It found steps the model cannot explain at any joint position: "
            f"{baseline['unexplained_steps']}."
        )
    shown = json.loads(json.dumps(baseline))  # a copy to edit
    for key in ("fitted_urdf", "report"):
        shown.pop(key, None)
    for e in shown["evidence"]:
        e["image"] = str(Path(e["image"]).relative_to(ws.root))
    bin_dir = Path(sys.executable).parent
    return PROMPT.format(
        category=obj.category,
        name=obj.name,
        urdf_name=Path(ws.urdf).name,
        verdict=verdict,
        baseline=dumps(shown),
        r2s2r=bin_dir / "r2s2r",
        python=bin_dir / "python",
        T_base_obj=json.dumps(obj.T_base_obj.round(4).tolist()),
        ledger=LEDGER_FILE,
        budget=cfg.budget,
        gain=cfg.min_gain,
        baseline_score=baseline["score_mm"],
    )
