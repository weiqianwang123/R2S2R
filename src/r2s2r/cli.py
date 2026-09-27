"""Command-line entry points.

Captures: the robot's calibrated cameras, poses in its base frame, with joint states::

    r2s2r capture droid EPISODE_DIR --calib CALIB_DIR --out CAPTURE_DIR
    r2s2r capture mujoco --world panda_table|physcoder_box_block --out CAPTURE_DIR
        [--seed 0] [--block corner|beside] [--every 5]

Reconstruction, a run of a method on a capture (see :mod:`r2s2r.pipeline`), and the
tools a method (or anyone) can use on a run::

    r2s2r run CAPTURE_DIR --method fixed|agentic --out RUN_DIR [--cameras ext1 wrist]
        [--stages 2 3 4 5 6] [--force]
        fixed: [--frame-selection codex] [--max-frames 12] [--retry-frames 3]
               [--codex-reasoning medium] [--override HYDRA_OVERRIDE ...]
        agentic: [--model gpt-6-astra] [--reasoning xhigh]
    r2s2r run RUN_DIR --method fixed            # resumes: done stages are skipped
    r2s2r tool <frames|segment|crop|points|support|generate|fit|check|assemble|settle|
                replay> ...
    r2s2r viewer outputs/agentic [--port 8765]   # live progress in the browser

The MuJoCo testbed (:mod:`r2s2r.testbed`), scoring against a MuJoCo capture's ground
truth and running the pick test in MuJoCo and Isaac Lab::

    r2s2r eval RUN_DIR|SCENE_DIR --capture CAPTURE_DIR
    r2s2r pick (SCENE_DIR | --oracle) --capture CAPTURE_DIR --out OUT_DIR
        [--video-camera ext1]

The Isaac Lab side runs as scripts, since the Omniverse app must start first
(``scripts/isaaclab/``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

from r2s2r.capture.droid import DEFAULT_ROLES, load_droid_episode
from r2s2r.pipeline.agentic.method import AgentConfig, AgentMethod
from r2s2r.pipeline.fixed import FixedMethod
from r2s2r.pipeline.fixed.simfoundry import FRAME_SELECTIONS, SimFoundryConfig
from r2s2r.pipeline.run import Method, run
from r2s2r.pipeline.stages import PRODUCTS, STAGE_DIRS, STAGES
from r2s2r.structs import SCENE_FILENAME, Capture, SceneSpec
from r2s2r.tools.cli import add_tool_parser, print_json


def _fixed(args: argparse.Namespace) -> Method:
    return FixedMethod(
        SimFoundryConfig(
            frame_selection=args.frame_selection,
            max_frames=args.max_frames,
            retry_frames=args.retry_frames,
            codex_reasoning=args.codex_reasoning,
            overrides=args.override,
        )
    )


def _agentic(args: argparse.Namespace) -> Method:
    return AgentMethod(AgentConfig(model=args.model, reasoning=args.reasoning))


METHODS: dict[str, Callable[[argparse.Namespace], Method]] = {
    "fixed": _fixed,
    "agentic": _agentic,
}


def _droid_capture(args: argparse.Namespace) -> None:
    capture = load_droid_episode(
        args.episode_dir,
        args.out,
        calib_dir=args.calib,
        roles=args.roles,
        stride=args.stride,
        gripper_threshold=args.gripper_threshold,
    )
    print(
        f"capture {capture.name}: {len(capture.frames)} frames from "
        f"{len(capture.cameras)} cameras, static steps {capture.static_steps}, "
        f"'{capture.instruction}' -> {capture.root}"
    )


def _run(args: argparse.Namespace) -> None:
    stages = tuple(args.stages) if args.stages else None
    method = METHODS[args.method](args)
    print_json(run(args.source, args.out, method, stages, args.force, args.cameras))


def _mujoco_capture(args: argparse.Namespace) -> None:
    # The testbed is for local tests only; imported when used.
    # pylint: disable=import-outside-toplevel
    from r2s2r.testbed.record import record_capture
    from r2s2r.testbed.worlds import build_world

    params = {
        k: getattr(args, k) for k in ("seed", "block") if getattr(args, k) is not None
    }
    world = build_world(args.world, **params)
    capture = record_capture(world, args.out, args.every)
    world.close()
    views = [v["view"] for v in capture.metadata["scan"] if "step" in v]
    print(
        f"capture {capture.name}: {len(capture.frames)} RGB-D frames from "
        f"{len(capture.cameras)} cameras, wrist views {', '.join(views)}, "
        f"'{capture.instruction}' -> {capture.root}"
    )


def _viewer(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.viewer.server import serve

    serve(args.paths, args.host, args.port)


def _scenes_to_score(path: Path) -> list[Path]:
    """A scene directory, or the scenes a run's stages left."""
    if (path / SCENE_FILENAME).exists():
        return [path]
    scenes = [
        (path / STAGE_DIRS[key] / name).parent
        for key, files in PRODUCTS.items()
        for name in files
        if Path(name).name == SCENE_FILENAME
    ]
    found = [s for s in scenes if (s / SCENE_FILENAME).exists()]
    if not found:
        raise SystemExit(f"eval: {path} is neither a scene nor a run with scenes")
    return found


def _eval(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.testbed.evaluate import evaluate, summary

    capture = Capture.load(args.capture)
    for scene_dir in _scenes_to_score(args.path):
        print(f"{scene_dir}:")
        print("\n".join(summary(evaluate(SceneSpec.load(scene_dir), capture))))


def _pick(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.sim import isaac
    from r2s2r.testbed.evaluate import evaluate, summary
    from r2s2r.testbed.pick import match_target, oracle_scene, run_pick
    from r2s2r.testbed.policy import summarize
    from r2s2r.testbed.worlds import world_from_capture

    if args.oracle == (args.scene_dir is not None):
        raise SystemExit("pick: give a SCENE_DIR or --oracle, not both")
    capture = Capture.load(args.capture)
    if args.oracle:
        world = world_from_capture(capture)
        scene_dir = oracle_scene(world, capture, args.out / "oracle_scene")
        world.close()
    else:
        scene_dir = args.scene_dir
    scene = SceneSpec.load(scene_dir)
    print("\n".join(summary(evaluate(scene, capture))))
    target = match_target(scene, capture)
    rollouts = {
        "mujoco": run_pick(
            capture, scene, target, args.out / "mujoco", args.video_camera
        ),
        "isaaclab": isaac.pick(
            scene_dir, target, args.out / "isaaclab", args.video_camera
        ),
    }
    result = {
        "scene": str(Path(scene_dir).resolve()),
        "scene_method": scene.provenance.get("method"),
        "target": target,
        **rollouts,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    for sim, rollout in rollouts.items():
        print(f"{sim}: {summarize(rollout)}")
    print(f"-> {args.out / 'result.json'}")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="r2s2r")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="recorded data -> a capture")
    sources = capture.add_subparsers(dest="source", required=True)
    p = sources.add_parser("droid", help="a raw DROID episode -> a capture")
    p.add_argument("episode_dir", type=Path)
    p.add_argument("--calib", type=Path, required=True, help="KarlP/droid JSON dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--roles", nargs="+", default=list(DEFAULT_ROLES))
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--gripper-threshold", type=float, default=0.05)
    p.set_defaults(func=_droid_capture)
    p = sources.add_parser("mujoco", help="record a capture in a MuJoCo world")
    p.add_argument("--world", required=True, help="panda_table, physcoder_box_block")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, help="the layout's seed (physcoder_box_block)")
    p.add_argument("--block", help="corner or beside the box (physcoder_box_block)")
    p.add_argument("--every", type=int, default=5, help="frames every n steps")
    p.set_defaults(func=_mujoco_capture)

    p = sub.add_parser("run", help="a method's run on a capture (new, or resumed)")
    p.add_argument("source", type=Path, help="a capture (a new run), or a run")
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--out", type=Path, help="the new run's directory")
    p.add_argument(
        "--stages", nargs="+", choices=STAGES, help="default: all of the method's"
    )
    p.add_argument("--force", action="store_true", help="rerun stages already done")
    p.add_argument(
        "--cameras", nargs="+", help="only these cameras (roles), for a new run"
    )
    fixed = p.add_argument_group("fixed")
    fixed.add_argument(
        "--frame-selection",
        default=SimFoundryConfig.frame_selection,
        choices=FRAME_SELECTIONS,
        help="how SimFoundry picks the frame to rebuild from; codex: Codex (xhigh) sees "
        "every candidate and the task and picks the frame and the support surface",
    )
    fixed.add_argument(
        "--max-frames",
        type=int,
        default=SimFoundryConfig.max_frames,
        help="candidate frames offered to SimFoundry",
    )
    fixed.add_argument(
        "--retry-frames",
        type=int,
        default=SimFoundryConfig.retry_frames,
        help="other frames to rebuild from while one gives no objects",
    )
    fixed.add_argument("--codex-reasoning", default=SimFoundryConfig.codex_reasoning)
    fixed.add_argument(
        "--override", action="append", default=[], help="SimFoundry Hydra override"
    )
    agentic = p.add_argument_group("agentic")
    agentic.add_argument("--model", default=AgentConfig.model)
    agentic.add_argument("--reasoning", default=AgentConfig.reasoning)
    p.set_defaults(func=_run)
    add_tool_parser(sub)

    p = sub.add_parser("viewer", help="live web viewer of runs")
    p.add_argument("paths", type=Path, nargs="+", help="runs, or their parents")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=_viewer)

    p = sub.add_parser("eval", help="score scenes against a MuJoCo capture's truth")
    p.add_argument("path", type=Path, help="a scene, or a run (its stage 4-6 scenes)")
    p.add_argument(
        "--capture", type=Path, required=True, help="the original MuJoCo capture"
    )
    p.set_defaults(func=_eval)

    p = sub.add_parser(
        "pick", help="the pick test on a scene, in MuJoCo and in Isaac Lab"
    )
    p.add_argument("scene_dir", type=Path, nargs="?")
    p.add_argument(
        "--capture", type=Path, required=True, help="the original MuJoCo capture"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--oracle", action="store_true", help="pick in the ground truth's own scene"
    )
    p.add_argument("--video-camera", default="ext1", help="a static camera, '' none")
    p.set_defaults(func=_pick)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
