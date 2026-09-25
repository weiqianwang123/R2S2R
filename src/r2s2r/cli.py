"""Command-line entry points.

Subcommands::

    r2s2r droid-capture EPISODE_DIR --calib CALIB_DIR --out CAPTURE_DIR
    r2s2r mujoco-capture --out CAPTURE_DIR
    r2s2r reconstruct CAPTURE_DIR --workdir WORKDIR [--stages 2,3] [--prepare-only]
    r2s2r refine SCENE_DIR --capture CAPTURE_DIR (--views WORKDIR... | --capture-depth)
    r2s2r mujoco-deploy (SCENE_DIR | --oracle) --capture CAPTURE_DIR --target NAME
        --out OUT_DIR

The Isaac Lab side runs as scripts, since the Omniverse app must start first:
``scripts/isaaclab/run_pick.py`` and ``scripts/isaaclab/render_overlay.py``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from r2s2r.io.droid import load_droid_episode
from r2s2r.io.rgbd import capture_depth_views
from r2s2r.policy.scoring import summarize
from r2s2r.reconstruct import make_backend, registered_backends
from r2s2r.reconstruct.refine import refine_scene
from r2s2r.reconstruct.simfoundry import (
    SimFoundryBackend,
    SimFoundryConfig,
    load_stage2_views,
)
from r2s2r.structs import Capture, SceneSpec


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


def _reconstruct(args: argparse.Namespace) -> None:
    capture = Capture.load(args.capture_dir)
    workdir = Path(args.workdir)
    if args.backend != "simfoundry":
        scene = make_backend(args.backend).reconstruct(capture, workdir)
    else:
        mamba = args.mamba or shutil.which("mamba") or "mamba"
        config = SimFoundryConfig(
            mamba_exe=mamba,
            camera_role=args.camera_role,
            max_frames=args.max_frames,
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
    path = scene.save(args.scene_out or workdir / capture.name / "scene")
    print(f"scene with {len(scene.objects)} objects -> {path}")


def _refine(args: argparse.Namespace) -> None:
    scene = SceneSpec.load(args.scene_dir)
    capture = Capture.load(args.capture)
    step = scene.reference_step if args.step is None else args.step
    views = []
    if args.capture_depth:
        views += capture_depth_views(capture, step)
    for workdir in args.views:
        views += load_stage2_views(capture, workdir, steps={step})
    if not views:
        raise SystemExit(f"no depth at step {step} (capture depth or --views)")
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


def _mujoco_capture(args: argparse.Namespace) -> None:
    # Imported here so the other subcommands work without MuJoCo and a GL driver.
    from r2s2r.real.mujoco.capture import (  # pylint: disable=import-outside-toplevel
        record_capture,
    )

    capture = record_capture(args.out, name=args.name, every=args.every)
    print(
        f"capture {capture.name}: {len(capture.frames)} RGB-D frames from "
        f"{len(capture.cameras)} cameras -> {capture.root}"
    )


def _mujoco_deploy(args: argparse.Namespace) -> None:
    from r2s2r.real.mujoco.deploy import (  # pylint: disable=import-outside-toplevel
        oracle_scene,
        run_pick,
        scene_errors,
    )

    capture = Capture.load(args.capture)
    if args.oracle:
        scene = oracle_scene(capture, args.out)
        scene.save(args.out / "oracle_scene")
    else:
        scene = SceneSpec.load(args.scene_dir)
        for name, err in scene_errors(scene, capture).items():
            print(
                f"{name} ~ {err['ground_truth']}: centre off by "
                f"{err['center_error_m'] * 100:.1f} cm, size {err['size_m']} "
                f"(true {err['ground_truth_size_m']})"
            )
    result = run_pick(capture, scene, args.target, args.out, args.video_camera)
    print(f"{summarize(result)} -> {args.out}")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="r2s2r")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("droid-capture", help="raw DROID episode -> capture")
    p.add_argument("episode_dir", type=Path)
    p.add_argument("--calib", type=Path, required=True, help="KarlP/droid JSON dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--roles", nargs="+", default=["ext1", "ext2", "wrist"])
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--gripper-threshold", type=float, default=0.05)
    p.set_defaults(func=_droid_capture)

    p = sub.add_parser("reconstruct", help="capture -> scene spec")
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--backend", default="simfoundry", choices=registered_backends())
    p.add_argument("--scene-out", type=Path)
    p.add_argument("--stages", help="comma-separated SimFoundry stage ids")
    p.add_argument("--camera-role", default="ext1")
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--mamba", help="path to the mamba executable")
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
    p.add_argument("--step", type=int, help="default: the scene's reference step")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_refine)

    p = sub.add_parser("mujoco-capture", help="record a capture in the MuJoCo world")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="mujoco_pick")
    p.add_argument("--every", type=int, default=5, help="save every n control steps")
    p.set_defaults(func=_mujoco_capture)

    p = sub.add_parser("mujoco-deploy", help="run the pick policy in the MuJoCo world")
    p.add_argument("scene_dir", type=Path, nargs="?")
    p.add_argument("--capture", type=Path, required=True, help="MuJoCo capture dir")
    p.add_argument("--target", required=True, help="object name (or unique substring)")
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
