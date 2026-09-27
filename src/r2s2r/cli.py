"""Command-line entry points.

Captures: the robot's calibrated cameras, poses in its base frame, with joint states::

    r2s2r capture droid EPISODE_DIR --calib CALIB_DIR --out CAPTURE_DIR
    r2s2r mujoco-capture --out CAPTURE_DIR [--cameras ext1 wrist]

Reconstruction, a run of a method on a capture (see :mod:`r2s2r.pipeline`), and the
tools a method (or anyone) can use on a run::

    r2s2r run CAPTURE_DIR --method agentic --out RUN_DIR [--cameras wrist]
        [--stages 2 3 4 5 6] [--force]
    r2s2r run RUN_DIR --method agentic            # resumes: done stages are skipped
    r2s2r tool <frames|segment|crop|points|support|generate|fit|check|assemble|settle|
                replay> ...
    r2s2r viewer outputs/agentic [--port 8765]   # live progress in the browser

The fixed method's steps, until they are a method of ``r2s2r run``::

    r2s2r stereo-depth CAPTURE_DIR --out CAPTURE_DIR [--all-steps]
    r2s2r reconstruct CAPTURE_DIR --workdir WORKDIR [--cameras ...] [--stages 2,3]
    r2s2r refine SCENE_DIR --capture CAPTURE_DIR (--views WORKDIR... | --capture-depth)
        [--keep-unseen] [--no-vlm | --no-orientation-check]

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

import cv2
import numpy as np

from r2s2r.capture.droid import DEFAULT_ROLES, load_droid_episode
from r2s2r.pipeline.agentic.method import AgentConfig, AgentMethod
from r2s2r.pipeline.fixed.refine import refine_scene
from r2s2r.pipeline.fixed.simfoundry import (
    SimFoundryBackend,
    SimFoundryConfig,
    load_stage2_views,
)
from r2s2r.pipeline.run import Method, run
from r2s2r.pipeline.stages import STAGES
from r2s2r.policy.scoring import summarize
from r2s2r.structs import Capture, DepthView, SceneSpec, read_depth
from r2s2r.tools.cli import add_tool_parser, print_json


def _agentic(args: argparse.Namespace) -> Method:
    return AgentMethod(AgentConfig(model=args.model, reasoning=args.reasoning))


METHODS: dict[str, Callable[[argparse.Namespace], Method]] = {"agentic": _agentic}


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


def _reconstruct(args: argparse.Namespace) -> None:
    capture = Capture.load(args.capture_dir)
    workdir = Path(args.workdir)
    config = SimFoundryConfig(
        cameras=tuple(args.cameras) if args.cameras else None,
        max_frames=args.max_frames,
        mask_robot=not args.no_robot_mask,
        frame_selection=args.frame_selection,
        vlm_backend=args.vlm_backend,
        codex_reasoning=args.codex_reasoning,
        overrides=args.override,
    )
    if args.stages:
        config.stages = tuple(s.strip() for s in args.stages.split(","))
    backend = SimFoundryBackend(config)
    if not args.parse_only:
        backend.prepare_inputs(capture, workdir)
        if args.prepare_only:
            print(f"inputs written to {backend.scene_dir(capture, workdir)}")
            return
        backend.run(capture, workdir)
        if config.stages[-1] != "12":
            print(f"ran stages {','.join(config.stages)}; parse needs stage 12")
            return
    scene = backend.parse(capture, workdir)
    if not scene.objects and config.retry_frames and not args.parse_only:
        scene = backend.retry_empty(capture, workdir) or scene
    path = scene.save(args.scene_out or workdir / capture.name / "scene")
    print(f"scene with {len(scene.objects)} objects -> {path}")


def _refine(args: argparse.Namespace) -> None:
    scene = SceneSpec.load(args.scene_dir)
    capture = Capture.load(args.capture)
    steps = None if args.step is None else {args.step}
    views = _capture_depth_views(capture, args, steps) if args.capture_depth else []
    for workdir in args.views:
        views += load_stage2_views(capture, workdir, steps=steps)
    if not views:
        raise SystemExit("no depth to refine with (--capture-depth or --views)")
    if not args.keep_unseen:
        # MuJoCo renders the objects; imported only when used.
        # pylint: disable=import-outside-toplevel
        from r2s2r.pipeline.fixed.existence import drop_unseen

        scene, seen = drop_unseen(scene, views)
        for name, entry in seen["objects"].items():
            if entry["dropped"]:
                print(
                    f"{name}: dropped (no measured points match it, and the cameras "
                    f"see through {entry['seen_through']:.0%} of it)"
                )
            elif entry.get("unexplained_points_under"):
                under = entry["unexplained_points_under"]
                print(
                    f"{name}: kept, though the cameras see through "
                    f"{entry['seen_through']:.0%} of it: {under} measured points under "
                    "it match no object (its shape is likely off)"
                )
        for name, fall in seen["settled"].items():
            print(f"{name}: rested on a dropped object, lowered {fall * 100:.1f} cm")
    if not args.no_orientation_check:
        # MuJoCo renders the candidates; imported only when used.
        # pylint: disable=import-outside-toplevel
        from r2s2r.pipeline.fixed.orientation import check_orientations
        from r2s2r.pipeline.fixed.vlm import CodexVLM

        vlm = None if args.no_vlm else CodexVLM(reasoning=args.codex_reasoning)
        scene, turns = check_orientations(
            scene, views, vlm, image_dir=args.out / "orientation"
        )
        for name, entry in turns.items():
            print(
                f"{name}: turned {entry['turn_deg']:.0f} deg "
                f"(candidates {entry['candidates']}, {entry['decided_by']})"
            )
    refined, report = refine_scene(scene, views)
    path = refined.save(args.out)
    for name, entry in report["objects"].items():
        if entry["matched"]:
            print(
                f"{name}: moved {entry['shift_m'] * 100:.1f} cm, "
                f"yaw {entry['yaw_change_deg']:+.0f} deg, footprint IoU "
                f"{entry['footprint_iou_before']:.2f} -> "
                f"{entry['footprint_iou_after']:.2f}"
                + ("" if entry["applied"] else " (kept backend pose)")
            )
        else:
            print(f"{name}: no matching point cluster, left as is")
    w, h = report["support"]["size"]
    print(f"support {w:.2f} x {h:.2f} m from {len(views)} views -> {path}")


def _capture_depth_views(
    capture: Capture, args: argparse.Namespace, steps: set[int] | None
) -> list[DepthView]:
    """The capture's own depth (robot cut out unless ``--no-robot-mask``) of up to
    ``--max-views`` static frames, spread over the cameras like SimFoundry's
    candidates."""
    masker = None
    if not args.no_robot_mask:
        # MuJoCo renders the robot; imported only when used.
        # pylint: disable=import-outside-toplevel
        from r2s2r.robots.mask import robot_masker

        masker = robot_masker(capture.embodiment)
    views = []
    for frame in capture.select_frames(
        args.cameras,
        args.max_views,
        lambda f: f.depth_image is not None and (steps is None or f.step in steps),
    ):
        assert frame.depth_image is not None
        cam = capture.cameras[frame.camera]
        depth = read_depth(capture.root / frame.depth_image)
        if masker is not None:
            depth, _ = masker.mask_depth(
                depth,
                cam.K,
                frame.T_base_cam,
                frame.joint_positions,
                frame.gripper_position,
            )
        bgr = cv2.imread(str(capture.root / frame.left_image))
        rgb = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        views.append(
            DepthView(
                frame.camera,
                frame.step,
                depth.astype(float),
                cam.K,
                frame.T_base_cam,
                None if rgb is None else np.asarray(rgb, np.uint8),
            )
        )
    return views


def _stereo_depth(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.capture.stereo import add_stereo_depth

    capture = add_stereo_depth(
        Capture.load(args.capture_dir),
        args.out,
        SimFoundryConfig(),
        static_only=not args.all_steps,
    )
    with_depth = sum(f.depth_image is not None for f in capture.frames)
    print(f"{with_depth} of {len(capture.frames)} frames have depth -> {args.out}")


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


def _robot_mask_argument(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--no-robot-mask",
        action="store_true",
        help="keep the robot in depth (default: cut out by rendering its model at the "
        "recorded joints)",
    )


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

    p = sub.add_parser("reconstruct", help="capture -> scene spec (SimFoundry)")
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--scene-out", type=Path)
    p.add_argument("--stages", help="comma-separated SimFoundry stage ids")
    p.add_argument(
        "--cameras", nargs="+", help="roles or serials to draw candidates from (all)"
    )
    _robot_mask_argument(p)
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument(
        "--frame-selection",
        default="hybrid",
        choices=["hybrid", "heuristic", "vlm", "codex"],
        help="how SimFoundry picks the frame to rebuild from; codex: Codex (xhigh) sees "
        "every candidate and the task and picks the frame and the support surface",
    )
    p.add_argument("--vlm-backend", default="codex", choices=["codex", "gemini"])
    p.add_argument("--codex-reasoning", default="medium")
    p.add_argument("--override", action="append", default=[], help="Hydra override")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--parse-only", action="store_true")
    p.set_defaults(func=_reconstruct)

    p = sub.add_parser("refine", help="multi-view refinement of a scene spec")
    p.add_argument("scene_dir", type=Path)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument(
        "--views",
        type=Path,
        nargs="+",
        default=[],
        help="SimFoundry workdirs whose stage-2 depth to fuse (one per camera)",
    )
    p.add_argument(
        "--capture-depth",
        action="store_true",
        help="fuse the capture's own depth (RGB-D cameras) of every static camera",
    )
    p.add_argument("--step", type=int, help="only this step (default: static period)")
    p.add_argument("--cameras", nargs="+", help="capture-depth cameras (all)")
    p.add_argument("--max-views", type=int, default=12, help="capture-depth views")
    _robot_mask_argument(p)
    p.add_argument(
        "--keep-unseen",
        action="store_true",
        help="keep objects the depth sees through and no measured points match",
    )
    p.add_argument(
        "--no-orientation-check",
        action="store_true",
        help="keep every object's turn about the support normal",
    )
    p.add_argument(
        "--no-vlm",
        action="store_true",
        help="orientation check from geometry only (ties keep the backend's turn)",
    )
    p.add_argument("--codex-reasoning", default="medium")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_refine)

    p = sub.add_parser(
        "stereo-depth",
        help="a capture with FoundationStereo depth on its stereo frames",
    )
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--all-steps", action="store_true", help="every frame, not only the static ones"
    )
    p.set_defaults(func=_stereo_depth)

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
