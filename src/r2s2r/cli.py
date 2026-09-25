"""Command-line entry points.

Subcommands::

    r2s2r droid-capture EPISODE_DIR --calib CALIB_DIR --out CAPTURE_DIR
    r2s2r mujoco-capture --out CAPTURE_DIR [--cameras ext1 ext2 wrist]
    r2s2r reconstruct CAPTURE_DIR --workdir WORKDIR [--cameras ...] [--stages 2,3]
    r2s2r refine SCENE_DIR --capture CAPTURE_DIR (--views WORKDIR... | --capture-depth)
        [--no-vlm | --no-orientation-check]
    r2s2r calibrate-joints SCENE_DIR --capture CAPTURE_DIR --out OUT_DIR [--no-agent]
    r2s2r joints-eval WORKSPACE URDF      (these two: the calibrating agent's tools)
    r2s2r urdf-info URDF [--workspace WORKSPACE]
    r2s2r mujoco-eval SCENE_DIR --capture CAPTURE_DIR
    r2s2r mujoco-deploy (SCENE_DIR | --oracle) --capture CAPTURE_DIR --target NAME
        --out OUT_DIR

The Isaac Lab side runs as scripts, since the Omniverse app must start first:
``scripts/isaaclab/run_pick.py`` and ``scripts/isaaclab/render_overlay.py``.
"""

from __future__ import annotations

import argparse
import json
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
            cameras=tuple(args.cameras) if args.cameras else None,
            max_frames=args.max_frames,
            mask_robot=not args.no_robot_mask,
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
    steps = None if args.step is None else {args.step}
    views = []
    if args.capture_depth:
        masker = None
        if not args.no_robot_mask:
            # MuJoCo only when masking.
            from r2s2r.robots.mask import (  # pylint: disable=import-outside-toplevel
                RobotMasker,
            )

            masker = RobotMasker(capture.embodiment)
        views += capture_depth_views(
            capture, args.cameras, args.max_views, steps, masker=masker
        )
    for workdir in args.views:
        views += load_stage2_views(capture, workdir, steps=steps)
    if not views:
        raise SystemExit("no depth to refine with (--capture-depth or --views)")
    if not args.no_orientation_check:
        # MuJoCo renders the candidates; imported only when used.
        # pylint: disable=import-outside-toplevel
        from r2s2r.reconstruct.orientation import check_orientations
        from r2s2r.vlm import CodexVLM

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


def _calibrate_joints(args: argparse.Namespace) -> None:
    # MuJoCo renders the joint positions; imported only when used.
    # pylint: disable=import-outside-toplevel
    from r2s2r.calibrate.agent import CodexAgent
    from r2s2r.calibrate.articulation import (
        CalibrationConfig,
        calibrate_articulation,
        final_fits,
    )
    from r2s2r.io.rgbd import capture_depth_steps
    from r2s2r.robots.mask import RobotMasker

    scene = SceneSpec.load(args.scene_dir)
    capture = Capture.load(args.capture)
    if not any(o.articulated for o in scene.objects):
        (args.out / "joints_report.json").unlink(missing_ok=True)
        print(f"no articulated objects -> {scene.save(args.out)}")
        return
    masker = None if args.no_robot_mask else RobotMasker(capture.embodiment)
    steps = capture_depth_steps(capture, args.cameras, args.max_steps, masker)
    start, end = capture.static_steps
    agent = None
    if not args.no_agent:
        agent = CodexAgent(model=args.codex_model, reasoning=args.codex_reasoning)
    calibrated, report = calibrate_articulation(
        args.scene_dir,
        args.capture,
        steps,
        [t for t in steps if start <= t < end],
        args.out / "calibration",
        agent,
        CalibrationConfig(budget=args.budget),
        force=args.force,
    )
    path = calibrated.save(args.out)
    for name, data in (("calibration", report), ("joints_report", final_fits(report))):
        (args.out / f"{name}.json").write_text(json.dumps(data, indent=2), "utf-8")
    for name, entry in report.items():
        base, final = entry["baseline"], entry["final"]
        print(
            f"{name}: {entry['decision']}; score {base['score_mm']} -> "
            f"{final['score_mm']} mm, consistent {base['consistent']} -> "
            f"{final['consistent']}"
        )
        for joint, j in final["joints"].items():
            print(f"  {joint}: {j}")
    print(f"{len(steps)} steps -> {path}")


def _joints_eval(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.calibrate.articulation import brief
    from r2s2r.calibrate.tools import LEDGER_FILE, Workspace, dumps, evaluate

    ws = Workspace.load(args.workspace)
    name = args.name or args.urdf.stem
    summary = evaluate(ws, args.urdf, ws.root / "evals" / name)
    with open(ws.root / LEDGER_FILE, "a", encoding="utf-8") as ledger:
        ledger.write(json.dumps({"name": name, **brief(summary)}) + "\n")
    print(dumps({k: v for k, v in summary.items() if k != "fitted_urdf"}))


def _urdf_info(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.calibrate.tools import Workspace, dumps, urdf_info

    T_base_obj = None
    if args.workspace is not None:
        ws = Workspace.load(args.workspace)
        scene = SceneSpec.load(ws.scene_dir)
        T_base_obj = next(
            o.T_base_obj for o in scene.objects if o.name == ws.object_name
        )
    print(dumps(urdf_info(args.urdf, T_base_obj)))


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
    from r2s2r.real.mujoco.deploy import articulation_errors, scene_errors

    for name, err in scene_errors(scene, capture).items():
        print(
            f"{name} ~ {err['ground_truth']}: centre off by "
            f"{err['center_error_m'] * 100:.1f} cm, size {err['size_m']} "
            f"(true {err['ground_truth_size_m']})"
        )
    for name, entry in articulation_errors(scene, capture).items():
        for joint in entry["joints"]:
            print(f"{name} joint {joint['name']}: {joint}")
        if not entry["joints"]:
            print(f"{name}: articulated but no movable joint")


def _mujoco_eval(args: argparse.Namespace) -> None:
    scene, capture = SceneSpec.load(args.scene_dir), Capture.load(args.capture)
    if args.match_target:
        # pylint: disable=import-outside-toplevel
        from r2s2r.real.mujoco.deploy import match_target

        print(match_target(scene, capture))
        return
    _print_scene_errors(scene, capture)
    report = args.scene_dir / "joints_report.json"  # written by calibrate-joints
    if report.exists():
        # pylint: disable=import-outside-toplevel
        from r2s2r.real.mujoco.deploy import trajectory_errors

        fitted = json.loads(report.read_text(encoding="utf-8"))
        for name, rows in trajectory_errors(scene, capture, fitted).items():
            for row in rows:
                print(f"{name} joint trajectory: {row}")


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
    p.add_argument(
        "--cameras", nargs="+", help="roles or serials to draw candidates from (all)"
    )
    p.add_argument(
        "--no-robot-mask", action="store_true", help="keep the robot in depth"
    )
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
    p.add_argument("--step", type=int, help="only this step (default: static period)")
    p.add_argument("--cameras", nargs="+", help="capture-depth cameras (all)")
    p.add_argument("--max-views", type=int, default=12, help="capture-depth views")
    p.add_argument(
        "--no-robot-mask", action="store_true", help="keep the robot in depth"
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
        "calibrate-joints",
        help="fit joints to the video; a Codex agent remodels those that fail the check",
    )
    p.add_argument("scene_dir", type=Path)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--cameras", nargs="+", help="cameras to use (all)")
    p.add_argument("--max-steps", type=int, default=48, help="video steps sampled")
    p.add_argument(
        "--no-robot-mask", action="store_true", help="keep the robot in depth"
    )
    p.add_argument("--budget", type=int, default=10, help="evaluations per object")
    p.add_argument(
        "--force", action="store_true", help="call the agent even when the check passes"
    )
    p.add_argument("--no-agent", action="store_true", help="only fit and check")
    p.add_argument("--codex-model", default="gpt-6-astra")
    p.add_argument("--codex-reasoning", default="medium")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_calibrate_joints)

    p = sub.add_parser(
        "joints-eval",
        help="evaluate a candidate joint model in a calibration workspace",
    )
    p.add_argument("workspace", type=Path)
    p.add_argument("urdf", type=Path)
    p.add_argument("--name", help="evaluation name (default: the URDF's stem)")
    p.set_defaults(func=_joints_eval)

    p = sub.add_parser("urdf-info", help="links and joints of a URDF, in both frames")
    p.add_argument("urdf", type=Path)
    p.add_argument("--workspace", type=Path, help="calibration workspace (base frame)")
    p.set_defaults(func=_urdf_info)

    p = sub.add_parser("mujoco-capture", help="record a capture in the MuJoCo world")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="mujoco_pick")
    p.add_argument("--every", type=int, default=5, help="save every n control steps")
    p.add_argument(
        "--cameras", nargs="+", help="cameras to record (default: ext1 ext2 wrist)"
    )
    p.add_argument(
        "--world", default="pick", help="world preset: pick, or cabinet (articulated)"
    )
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
