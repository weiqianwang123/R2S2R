"""Command-line entry points.

r2s2r droid-capture EPISODE_DIR --calib CALIB_DIR --out CAPTURE_DIR r2s2r reconstruct
CAPTURE_DIR --workdir WORKDIR [--stages 2,3] [--prepare-only]
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from r2s2r.io.droid import load_droid_episode
from r2s2r.reconstruct import make_backend, registered_backends
from r2s2r.reconstruct.simfoundry import SimFoundryBackend, SimFoundryConfig
from r2s2r.structs import Capture


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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
