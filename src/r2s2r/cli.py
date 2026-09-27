"""Command-line entry points.

Captures: the robot's calibrated cameras, poses in its base frame, with joint states::

    r2s2r capture droid EPISODE_DIR --calib CALIB_DIR --out CAPTURE_DIR
    r2s2r mujoco-capture --out CAPTURE_DIR [--cameras ext1 wrist]

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

MuJoCo as the real world::

    r2s2r mujoco-eval SCENE_DIR --capture CAPTURE_DIR
    r2s2r mujoco-deploy (SCENE_DIR | --oracle) --capture CAPTURE_DIR --target NAME
        --out OUT_DIR

The Isaac Lab side runs as scripts, since the Omniverse app must start first
(``scripts/isaaclab/``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

from r2s2r.capture.droid import DEFAULT_ROLES, load_droid_episode
from r2s2r.pipeline.agentic.method import AgentConfig, AgentMethod
from r2s2r.pipeline.fixed import FixedMethod
from r2s2r.pipeline.fixed.simfoundry import FRAME_SELECTIONS, SimFoundryConfig
from r2s2r.pipeline.run import Method, run
from r2s2r.pipeline.stages import STAGES
from r2s2r.policy.scoring import summarize
from r2s2r.structs import Capture, SceneSpec
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
    # Imported here so the other subcommands work without MuJoCo and a GL driver.
    # pylint: disable=import-outside-toplevel
    from r2s2r.real.mujoco.capture import record_capture
    from r2s2r.real.mujoco.world import MujocoWorldConfig

    capture = record_capture(
        args.out,
        MujocoWorldConfig.preset(args.world),
        name=args.name,
        every=args.every,
        cameras=args.cameras,
    )
    print(
        f"capture {capture.name}: {len(capture.frames)} RGB-D frames from "
        f"{len(capture.cameras)} cameras -> {capture.root}"
    )


def _print_scene_errors(scene: SceneSpec, capture: Capture) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.real.mujoco.deploy import scene_errors

    for name, err in scene_errors(scene, capture).items():
        print(
            f"{name} ~ {err['ground_truth']}: centre off by "
            f"{err['center_error_m'] * 100:.1f} cm, size {err['size_m']} "
            f"(true {err['ground_truth_size_m']})"
        )


def _viewer(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.viewer.server import serve

    serve(args.paths, args.host, args.port)


def _mujoco_eval(args: argparse.Namespace) -> None:
    scene, capture = SceneSpec.load(args.scene_dir), Capture.load(args.capture)
    if args.match_target:
        # pylint: disable=import-outside-toplevel
        from r2s2r.real.mujoco.deploy import match_target

        print(match_target(scene, capture))
        return
    _print_scene_errors(scene, capture)


def _mujoco_deploy(args: argparse.Namespace) -> None:
    from r2s2r.real.mujoco.deploy import (  # pylint: disable=import-outside-toplevel
        match_target,
        oracle_scene,
        run_pick,
    )

    capture = Capture.load(args.capture)
    if args.oracle:
        scene = oracle_scene(capture, args.out)
        scene.save(args.out / "oracle_scene")
    else:
        scene = SceneSpec.load(args.scene_dir)
        _print_scene_errors(scene, capture)
    target = args.target or match_target(scene, capture)
    result = run_pick(capture, scene, target, args.out, args.video_camera)
    print(f"{summarize(result)} -> {args.out}")


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

    p = sub.add_parser("mujoco-capture", help="record a capture in the MuJoCo world")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="mujoco_pick")
    p.add_argument("--every", type=int, default=5, help="save every n control steps")
    p.add_argument(
        "--cameras", nargs="+", help="cameras to record (default: ext1 wrist)"
    )
    p.add_argument("--world", default="pick", help="world preset (pick)")
    p.set_defaults(func=_mujoco_capture)

    p = sub.add_parser("mujoco-eval", help="score a scene against MuJoCo ground truth")
    p.add_argument("scene_dir", type=Path)
    p.add_argument("--capture", type=Path, required=True, help="MuJoCo capture dir")
    p.add_argument(
        "--match-target",
        action="store_true",
        help="only print the scene object that is the world's task target",
    )
    p.set_defaults(func=_mujoco_eval)

    p = sub.add_parser("mujoco-deploy", help="run the pick policy in the MuJoCo world")
    p.add_argument("scene_dir", type=Path, nargs="?")
    p.add_argument("--capture", type=Path, required=True, help="MuJoCo capture dir")
    p.add_argument(
        "--target",
        help="object name or unique substring (default: the world's task target)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video-camera", default="ext1")
    p.add_argument(
        "--oracle", action="store_true", help="use ground-truth objects, not SCENE_DIR"
    )
    p.set_defaults(func=_mujoco_deploy)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
