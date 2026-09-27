"""SimFoundry reconstructs the scene from one frame, for the fixed method.

SimFoundry (third_party/SimFoundry, a git submodule) is never imported: it runs in its
own conda environments, driven through its reconstruction orchestrator. This module

1. offers SimFoundry candidate frames: static frames with depth, from every camera of
   the run, each with its own intrinsics and pose. A candidate's depth is the run's
   robot-free depth, written where SimFoundry's stage 2 (FoundationStereo) leaves its
   own, at FoundationStereo's resolution: at most ``CANDIDATE_WIDTH`` wide;
2. runs SimFoundry stages 3-8 and 10-12 (9 makes objects articulated, and every object
   here is rigid; 13-14 import into OmniGibson), their VLM calls through Codex;
3. rebuilds from other frames when a frame gives no objects;
4. reads the stage outputs back as a scene in the robot base frame, and copies an
   object's mesh, texture and collision hulls out of its stage-11 URDF.

SimFoundry reconstructs from the one candidate its frame selection picks, and defines
its world frame on the support plane seen there. The candidate's index is mapped back to
(camera, step) through ``r2s2r_frames.json``, and that frame's known ``T_base_cam``
anchors the scene to the robot.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from r2s2r.paths import (
    ENV_MESH,
    ENV_NERFSTUDIO,
    ENV_SIMFOUNDRY,
    SIMFOUNDRY_DIR,
    codex_bin,
    mamba_exe,
)
from r2s2r.pipeline.workspace import Workspace
from r2s2r.structs import Capture, DepthView, FrameRecord, ObjectSpec, SceneSpec
from r2s2r.tools.envjobs import simfoundry_env
from r2s2r.tools.segment import slug
from r2s2r.transforms import invert, pos_quat_to_matrix, quat_xyzw_to_wxyz

logger = logging.getLogger(__name__)

FRAME_MAP_FILENAME = "r2s2r_frames.json"
STAGES = ("3", "4", "5", "6", "7", "8", "10", "11", "12")
# FoundationStereo runs at half of the 1280-pixel images SimFoundry was made for.
CANDIDATE_WIDTH = 640
CODEX_MODEL = "gpt-6-astra"
FRAME_SELECTIONS = ("codex", "hybrid", "heuristic", "vlm")
SETTINGS = (
    "s3_ground.use_fs=true",  # depth from s2_fs/
    # No generated image of the whole scene: nothing generated enters it but meshes.
    "s5_scene.use_upsampled_source_image=false",
    "s7_mesh.low_vram=true",  # CPU offload: ~6 GB of VRAM instead of ~29
)


@dataclass
class SimFoundryConfig:
    """What a fixed run can choose."""

    # How SimFoundry stage 3 picks the frame to rebuild from: "codex" (Codex, at xhigh
    # effort, sees every candidate and the capture's instruction, and picks the frame and
    # the support surface in it), or SimFoundry's "hybrid" (geometric scores, then a VLM
    # among the best few), "heuristic" or "vlm".
    frame_selection: str = "codex"
    max_frames: int = 12  # candidates offered
    # When the frame gives no objects, rebuild from up to this many others; 0: never.
    retry_frames: int = 3
    # Of Codex's VLM answers, in SimFoundry and in the orientation check.
    codex_reasoning: str = "medium"
    overrides: list[str] = field(default_factory=list)  # extra Hydra overrides


def candidates(capture: Capture, max_frames: int) -> list[FrameRecord]:
    """The frames offered to SimFoundry: up to ``max_frames`` static frames with depth,
    spread over the cameras (in the run's camera order)."""
    frames = capture.select_frames(
        None, max_frames, lambda f: f.depth_image is not None
    )
    if not frames:
        raise ValueError(f"capture {capture.name} has no static frames with depth")
    return frames


def candidate_view(ws: Workspace, frame: FrameRecord) -> DepthView:
    """The frame as SimFoundry sees it: its robot-free depth and its colour, scaled to
    at most ``CANDIDATE_WIDTH`` wide (``K`` in the pixel-centre convention)."""
    view = ws.depth_view(frame)
    assert view.image is not None
    h, w = view.depth.shape
    scale = min(1.0, CANDIDATE_WIDTH / w)
    size = (int(round(w * scale)), int(round(h * scale)))
    K = view.K.copy()
    K[:2, :2] *= scale
    K[:2, 2] = (K[:2, 2] + 0.5) * scale - 0.5
    return replace(
        view,
        depth=np.asarray(
            cv2.resize(view.depth, size, interpolation=cv2.INTER_NEAREST), np.float64
        ),
        K=K,
        image=np.asarray(
            cv2.resize(view.image, size, interpolation=cv2.INTER_AREA), np.uint8
        ),
    )


class SimFoundry:
    """SimFoundry's directory for a capture (``<root>/<capture name>``), and how to
    drive it; its output goes to ``log`` too."""

    def __init__(
        self, capture: Capture, root: Path, config: SimFoundryConfig, log: Path
    ) -> None:
        self.capture = capture
        self.root = Path(root).resolve()
        self.scene_dir = self.root / capture.name
        self.config = config
        self.log = log

    # ----------------------------------------------------------------- inputs
    def prepare(self, ws: Workspace, frames: list[FrameRecord]) -> None:
        """Write the candidates where SimFoundry's stages 1 and 2 leave them: the
        colour image in ``s1_zed/``, and in ``s2_fs/`` what FoundationStereo writes."""
        s1, fs = self.scene_dir / "s1_zed", self.scene_dir / "s2_fs"
        for d in (s1, fs):
            d.mkdir(parents=True, exist_ok=True)
        frame_map = []
        for i, frame in enumerate(frames):
            view = candidate_view(ws, frame)
            assert view.image is not None
            bgr = cv2.cvtColor(ws.image(frame), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(s1 / f"image_{i}_l.png"), bgr)
            cv2.imwrite(
                str(fs / f"image_{i}_rgb.png"),
                cv2.cvtColor(view.image, cv2.COLOR_RGB2BGR),
            )
            np.save(fs / f"image_{i}_rgb.npy", view.image)
            np.save(fs / f"image_{i}_depth_meter.npy", view.depth.astype(np.float32))
            np.save(fs / f"image_{i}_K.npy", view.K.astype(np.float32))
            frame_map.append({"index": i, "camera": frame.camera, "step": frame.step})
        (s1 / FRAME_MAP_FILENAME).write_text(
            json.dumps(frame_map, indent=2), encoding="utf-8"
        )

    def frame_of(self, index: int) -> FrameRecord:
        """The capture frame of candidate ``index``."""
        for m in _read_json(self.scene_dir / "s1_zed" / FRAME_MAP_FILENAME):
            if int(m["index"]) == index:
                return self.capture.frame(m["camera"], int(m["step"]))
        raise KeyError(f"no candidate {index} in {self.scene_dir}")

    # -------------------------------------------------------------------- run
    def command(
        self, stages: tuple[str, ...], extra: tuple[str, ...] = ()
    ) -> tuple[list[str], dict[str, str]]:
        """The orchestrator call and its environment; ``extra`` Hydra overrides go
        last."""
        cfg = self.config
        task = self.capture.instruction
        overrides = [
            f"root_dir={self.root}",
            f"scene_name={self.capture.name}",
            *SETTINGS,
            f"s3_ground.frame_selection.mode={cfg.frame_selection}",
            *(
                [f"s3_ground.frame_selection.task={json.dumps(task)}"]
                if cfg.frame_selection == "codex" and task
                else []
            ),
            *cfg.overrides,
            *extra,
        ]
        cmd = [
            mamba_exe(),
            "run",
            "-n",
            ENV_SIMFOUNDRY,
            "python",
            "scripts/pipeline/A_reconstruction/run_reconstruction.py",
            "--input-mode",
            "stereo",
            "--include",
            ",".join(stages),
            "--exec-mode",
            "mamba",
            "--env-simfoundry",
            ENV_SIMFOUNDRY,
            "--env-mesh",
            ENV_MESH,
            "--env-nerfstudio",
            ENV_NERFSTUDIO,
            "--env-b1k",
            ENV_SIMFOUNDRY,
            *overrides,
        ]
        env = simfoundry_env()
        env.update(
            PYTHONUNBUFFERED="1",  # the log follows the stages as they go
            SIMFOUNDRY_VLM_BACKEND="codex",
            SIMFOUNDRY_CODEX_BIN=codex_bin(),
            SIMFOUNDRY_CODEX_MODEL=CODEX_MODEL,
            SIMFOUNDRY_CODEX_REASONING=cfg.codex_reasoning,
        )
        return cmd, env

    def run(
        self, stages: tuple[str, ...] = STAGES, extra: tuple[str, ...] = ()
    ) -> None:
        """Run SimFoundry's ``stages`` (after :meth:`prepare`), its output teed to the
        log."""
        cmd, env = self.command(stages, extra)
        logger.info(
            "running SimFoundry stages %s (log: %s)", ",".join(stages), self.log
        )
        with open(self.log, "a", encoding="utf-8") as log:
            with subprocess.Popen(
                cmd,
                cwd=SIMFOUNDRY_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                    log.flush()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    def retry_empty(self) -> SceneSpec | None:
        """After a frame that gave no objects: stages 3-12 again, pinned to other frames
        until one gives objects (the camera with the fewest empty frames first, then the
        best frame score); the scene (its provenance says which frames gave nothing), or
        None if none does.

        Stage 3 fits the support surface in one frame, choosing the largest surface it
        detects, and from some viewpoints that is not the one the objects stand on (on
        DROID, the robot's own mounting table, whose clamps also pass for objects in the
        frame scores). Another camera, or another moment, sees it differently. Every
        rebuild starts from stage 3 on a clean slate, so nothing of the empty one stays;
        a frame that gives no objects costs little, since stage 5 then ends at once.
        """
        selection = self.selection()
        scores = [s for s in selection.get("scores", []) if s.get("eligible")]
        camera = {
            int(m["index"]): m["camera"]
            for m in _read_json(self.scene_dir / "s1_zed" / FRAME_MAP_FILENAME)
        }
        failed = [int(selection["selected_idx"])]
        for _ in range(self.config.retry_frames):
            left = [s for s in scores if int(s["idx"]) not in failed]
            if not left:
                return None
            fails = Counter(camera[i] for i in failed)
            _, _, idx = max(
                (-fails[camera[int(s["idx"])]], s["score"], int(s["idx"])) for s in left
            )
            logger.warning(
                "no objects from candidate(s) %s; rebuilding from candidate %d",
                failed,
                idx,
            )
            for d in self.scene_dir.iterdir():
                prefix = d.name.split("_", 1)[0]
                if d.is_dir() and prefix[1:].isdigit() and 3 <= int(prefix[1:]) <= 12:
                    shutil.rmtree(d)
            self.run(STAGES, (f"s3_ground.img_idx={idx}",))
            scene = self.parse()
            if scene.objects:
                scene.provenance["empty_candidates"] = failed
                return scene
            failed.append(idx)
        return None

    # ------------------------------------------------------------------ parse
    def selection(self) -> dict[str, Any]:
        """SimFoundry stage 3's frame choice: its scored selection, or the frame it was
        pinned to (``s3_ground.img_idx``), for which it writes only that frame's floor
        info."""
        ground = self.scene_dir / "s3_ground"
        if (ground / "frame_selection.json").exists():
            selection: dict[str, Any] = _read_json(ground / "frame_selection.json")
            return selection
        pinned = sorted(ground.glob("image_*_floor_info.json"))
        if len(pinned) != 1:
            raise FileNotFoundError(f"no frame selection in {ground}: did stage 3 run?")
        return {
            "selected_idx": int(pinned[0].name.split("_")[1]),
            "decided_by": "pinned",
        }

    def parse(self) -> SceneSpec:
        """SimFoundry's scene in the robot base frame. Each object's asset is its
        stage-11 URDF as SimFoundry wrote it, and its name its category's (numbered
        when several share one)."""
        selection = self.selection()
        idx = int(selection["selected_idx"])
        ref = self.frame_of(idx)
        # SimFoundry's world: the support plane's frame of the frame rebuilt from.
        T_world_cam = np.load(
            self.scene_dir / "s4_frame" / f"image_{idx}_cam2world.npy"
        )
        T_base_world = ref.T_base_cam @ invert(T_world_cam)
        info = _read_json(self.scene_dir / "s11_sim" / "scene_objects_info.json")
        poses = _read_json(self.scene_dir / "s12_physics" / "pb_scene_poses.json")
        objects: list[ObjectSpec] = []
        named: Counter[str] = Counter()
        for obj in info.values():
            if obj["name"] not in poses:
                logger.warning("object %s has no stabilized pose; skipped", obj["name"])
                continue
            pos, quat_xyzw = poses[obj["name"]]
            T_world_obj = pos_quat_to_matrix(pos, quat_xyzw_to_wxyz(quat_xyzw))
            category, model = obj["category"], obj["model"]
            urdf = self.scene_dir / "s11_sim" / "objects" / category / model / "urdf"
            urdf = urdf / f"{model}.urdf"
            name = slug(category)
            named[name] += 1
            objects.append(
                ObjectSpec(
                    name=f"{name}_{named[name]}" if named[name] > 1 else name,
                    category=category,
                    asset_path=str(urdf),
                    T_base_obj=T_base_world @ T_world_obj,
                    mass=_urdf_mass(urdf),
                    friction=obj.get("friction"),
                )
            )
        return SceneSpec(
            name=self.capture.name,
            embodiment=self.capture.embodiment,
            objects=objects,
            T_base_support=T_base_world,
            cameras=self.capture.cameras,
            reference_camera=ref.camera,
            reference_step=ref.step,
            joint_positions=ref.joint_positions,
            provenance={
                "method": "fixed",
                "simfoundry_scene": str(self.scene_dir),
                "selected_candidate": idx,
                "decided_by": selection.get("decided_by"),
            },
        )


def copy_object(urdf: Path, out_dir: Path) -> tuple[Path, list[Path]]:
    """A stage-11 object's visual mesh (with its textures) and collision hulls, copied
    into ``out_dir`` in its link frame (each hull's scale baked in); their paths.

    A SimFoundry object is one link, its visual mesh and collision hulls at the link's
    origin, the hulls scaled, the visual mesh not."""
    links = ET.parse(urdf).getroot().findall("link")
    visuals = links[0].findall("visual/geometry/mesh") if len(links) == 1 else []
    if (
        len(visuals) != 1
        or not np.allclose(_numbers(visuals[0], "scale", "1 1 1"), 1)
        or not all(
            _at_origin(el) for el in links[0] if el.tag in ("visual", "collision")
        )
    ):
        raise ValueError(
            f"{urdf}: not one link with one unscaled visual mesh, all at the link"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = _copy_obj(urdf.parent / visuals[0].attrib["filename"], out_dir)
    hulls = []
    for k, el in enumerate(links[0].findall("collision/geometry/mesh")):
        hull = trimesh.load(urdf.parent / el.attrib["filename"], force="mesh")
        assert isinstance(hull, trimesh.Trimesh)
        hull.apply_transform(np.diag([*_numbers(el, "scale", "1 1 1"), 1.0]))
        hulls.append(out_dir / "collision" / f"hull_{k}.obj")
        hulls[-1].parent.mkdir(exist_ok=True)
        hull.export(hulls[-1])
    return mesh, hulls


def _copy_obj(source: Path, out_dir: Path) -> Path:
    """An OBJ, its material file and its textures, into ``out_dir`` side by side."""
    mesh = out_dir / source.name
    shutil.copy(source, mesh)
    for line in source.read_text(errors="ignore").splitlines():
        if not line.startswith("mtllib"):
            continue
        mtl = source.parent / line.split(maxsplit=1)[1].strip()
        lines = []
        for mline in mtl.read_text(errors="ignore").splitlines():
            key, _, value = mline.strip().partition(" ")
            if key.startswith("map_") and value:
                texture = mtl.parent / value.strip()
                shutil.copy(texture, out_dir / texture.name)
                mline = f"{key} {texture.name}"
            lines.append(mline)
        (out_dir / mtl.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mesh


def _numbers(el: ET.Element, key: str, default: str) -> np.ndarray:
    return np.array(el.attrib.get(key, default).split(), float)


def _at_origin(part: ET.Element) -> bool:
    """Whether a visual or collision part sits at its link's origin."""
    origin = part.find("origin")
    return origin is None or not (
        np.any(_numbers(origin, "xyz", "0 0 0"))
        or np.any(_numbers(origin, "rpy", "0 0 0"))
    )


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: did the SimFoundry stage run?")
    return json.loads(path.read_text(encoding="utf-8"))


def _urdf_mass(urdf: Path) -> float | None:
    """Total mass over all links, if the URDF declares any."""
    masses = [float(m.attrib["value"]) for m in ET.parse(urdf).getroot().iter("mass")]
    return sum(masses) if masses else None
