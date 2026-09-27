"""``r2s2r tool ...``: the tools, on a run's directory (the agent's, and anyone's)."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from r2s2r.sim import isaac
from r2s2r.workspace import Workspace

NUMBER_LIST = re.compile(r"\[\s*((?:-?[\d.eE+-]+,?\s*)+)\]")


def _ws(args: argparse.Namespace) -> Workspace:
    return Workspace.load(args.ws) if args.ws else Workspace.find()


def print_json(summary: Any) -> None:
    """JSON, with lists of numbers kept on one line."""
    text = json.dumps(summary, indent=1)
    print(NUMBER_LIST.sub(lambda m: "[" + " ".join(m.group(1).split()) + "]", text))


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


# -------------------------------------------------------------------------- tools
def _frames(args: argparse.Namespace) -> None:
    ws = _ws(args)
    rows = [
        {k: r[k] for k in ("id", "static", "depth", "gripper_closed", "image")}
        for r in ws.frame_index()
        if (args.all or (r["static"] and r["depth"]))
        and (args.camera is None or r["camera"] == args.camera)
    ]
    print_json(rows)


def _segment(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.tools.segment import segment

    points = None
    if args.point:
        points = []
        for text in args.point:
            v = _floats(text)
            points.append((v[0], v[1], int(v[2]) if len(v) > 2 else 1))
    box = tuple(_floats(args.box)) if args.box else None
    print_json(
        segment(
            _ws(args),
            args.frames,
            args.out,
            texts=args.text,
            points=points,
            box=box,  # type: ignore[arg-type]
            name=args.name,
            threshold=args.threshold,
        )
    )


def _crop(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.tools.segment import crop

    print_json(crop(_ws(args), args.frame, args.mask, args.out, args.pad, args.size))


def _points(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.tools.geometry import parse_masks, points

    print_json(
        points(
            _ws(args),
            args.frames or [],
            parse_masks(args.mask),
            args.out,
            args.support,
            args.stride,
        )
    )


def _support(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.tools.geometry import parse_masks, support

    print_json(support(_ws(args), parse_masks(args.mask), args.out, args.frames))


def _generate(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.tools.geometry import parse_masks
    from r2s2r.tools.objects import generate

    print_json(generate(_ws(args), parse_masks(args.images), args.out, args.seed))


def _fit(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.tools.geometry import fit, parse_masks

    print_json(
        fit(
            _ws(args),
            args.mesh,
            args.support,
            parse_masks(args.mask),
            args.out,
            up=args.up,
            scale=args.scale,
            yaw_deg=args.yaw,
            rest=not args.no_rest,
            refine=not args.no_refine,
        )
    )


def _check(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.tools.check import check

    print_json(check(_ws(args), args.source, args.frames, args.out))


def _assemble(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from r2s2r.tools.objects import assemble

    print_json(
        assemble(_ws(args), args.objects, args.out, args.collision, args.max_hulls)
    )


def _settle(args: argparse.Namespace) -> None:
    print_json(isaac.settle(args.scene, _ws(args).capture.root, args.out, args.seconds))


def _replay(args: argparse.Namespace) -> None:
    capture = _ws(args).capture.root
    print_json(isaac.replay(args.scene, capture, args.out, args.cameras, args.every))


def add_tool_parser(sub: Any) -> None:
    """``r2s2r tool <name>``."""
    tool = sub.add_parser("tool", help="the tools, on a run's directory")
    tools = tool.add_subparsers(dest="tool", required=True)

    def add(name: str, func: Any, help_: str) -> argparse.ArgumentParser:
        p = tools.add_parser(name, help=help_, description=help_)
        p.add_argument(
            "--ws", help="run directory (default: the one containing the cwd)"
        )
        p.set_defaults(func=func)
        return p

    p = add("frames", _frames, "list the frames (default: static, with depth)")
    p.add_argument("--camera", help="only this camera (role)")
    p.add_argument("--all", action="store_true", help="every frame")

    p = add("segment", _segment, "SAM3 masks from text, or from points / a box")
    p.add_argument("--frames", nargs="+", required=True)
    p.add_argument("--text", action="append", help="what to find (repeatable)")
    p.add_argument("--point", action="append", help="x,y[,label]: 1 in, 0 out")
    p.add_argument("--box", help="x0,y0,x1,y1")
    p.add_argument("--name", help="output sub-directory (default: from the text)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", required=True)

    p = add("crop", _crop, "cut a masked object out, RGBA, for generation")
    p.add_argument("--frame", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pad", type=float, default=0.15, help="margin, share of the size")
    p.add_argument("--size", type=int, default=1024)

    p = add("points", _points, "fused robot-free points, as a PLY")
    p.add_argument("--frames", nargs="+", help="frames used whole")
    p.add_argument("--mask", action="append", help="FRAME=MASK (repeatable)")
    p.add_argument("--support", help="support JSON: also say where points lie on it")
    p.add_argument("--stride", type=int, default=2, help="every n-th pixel")
    p.add_argument("--out", required=True)

    p = add("support", _support, "fit the support plane, outline and frame")
    p.add_argument("--mask", action="append", help="FRAME=MASK (repeatable)")
    p.add_argument("--frames", nargs="+", help="frames used whole")
    p.add_argument("--out", required=True, help="support JSON")

    p = add("generate", _generate, "Hunyuan3D-2.1 mesh from one image per object")
    p.add_argument("images", nargs="+", help="NAME=IMAGE (RGBA)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)

    p = add("fit", _fit, "scale and place a mesh to match masks in several frames")
    p.add_argument("mesh")
    p.add_argument("--support", required=True)
    p.add_argument("--mask", action="append", required=True, help="FRAME=MASK")
    p.add_argument(
        "--up", default="y", help="the mesh's up axis: y (generated), z, x, -y"
    )
    p.add_argument("--scale", type=float, help="start scale")
    p.add_argument("--yaw", type=float, help="yaw (deg, about the support normal)")
    p.add_argument("--no-rest", action="store_true", help="not standing on the support")
    p.add_argument("--no-refine", action="store_true", help="keep the start values")
    p.add_argument("--out", required=True)

    p = add("check", _check, "render a scene's objects into frames (fast, MuJoCo)")
    p.add_argument("source", help="objects JSON or scene directory")
    p.add_argument("--frames", nargs="+", required=True)
    p.add_argument("--out", required=True)

    p = add("assemble", _assemble, "objects JSON -> simulation-ready scene")
    p.add_argument("objects")
    p.add_argument("--collision", default="coacd", choices=["coacd", "hull", "none"])
    p.add_argument("--max-hulls", type=int, default=16)
    p.add_argument("--out", required=True)

    p = add("settle", _settle, "let the objects come to rest (Isaac Lab)")
    p.add_argument("scene")
    p.add_argument("--seconds", type=float, default=isaac.SETTLE_SECONDS)
    p.add_argument("--out", required=True)

    p = add("replay", _replay, "replay the recording in the scene and compare (Isaac)")
    p.add_argument("scene")
    p.add_argument("--cameras", nargs="+", help="roles (default: all)")
    p.add_argument("--every", type=int, default=1, help="every n-th frame step")
    p.add_argument("--out", required=True)
