"""SimFoundry as a reconstruction backend (single-frame baseline).

SimFoundry (third_party/SimFoundry, a git submodule) is never imported: it runs
in its own conda environments, driven through its reconstruction orchestrator.
This module only

1. offers SimFoundry candidate frames from any mix of cameras (exterior, wrist;
   one or many images each), every frame with its own intrinsics and pose. A stereo
   frame is written in the layout of stage 1a (``image_<i>_{l,r}.png`` plus its own
   ``image_<i>_intrinsic.txt``) for FoundationStereo (stage 2); an RGB-D frame's
   measured depth is written where FoundationStereo would put it (``s2_fs/``);
2. removes the robot from every candidate's depth (:mod:`r2s2r.robots.mask`), so
   frame selection and refinement do not take the arm for an object;
3. runs stages 2-12 (13-14 import into OmniGibson, which r2s2r does not use);
4. reads the stage outputs back and re-expresses them in the robot base frame.

SimFoundry reconstructs from the one candidate its frame selection picks, and
defines its world frame on the support plane seen there. The frame index is mapped
back to (camera, step) through ``r2s2r_frames.json`` so its known ``T_base_cam``
anchors the scene to the robot.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from r2s2r.assets import make_sim_ready
from r2s2r.io.rgbd import read_depth
from r2s2r.reconstruct.base import ReconstructionBackend, register_backend
from r2s2r.structs import Capture, DepthView, FrameRecord, ObjectSpec, SceneSpec
from r2s2r.transforms import invert, pos_quat_to_matrix, quat_xyzw_to_wxyz

if TYPE_CHECKING:  # MuJoCo is imported only when masking
    from r2s2r.robots.mask import RobotMasker

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SIMFOUNDRY_DIR = REPO_ROOT / "third_party" / "SimFoundry"
FRAME_MAP_FILENAME = "r2s2r_frames.json"
# 13-14 import into OmniGibson. Stage 9 (articulated objects) runs when
# articulate-anything is installed (scripts/setup/install_articulation.sh).
DEFAULT_STAGES = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12")
# Stages that call a VLM (Gemini upstream; Codex with the r2s2r fork).
GEMINI_STAGES = {"3", "5", "6", "8", "9", "11"}


@dataclass
class SimFoundryConfig:
    """How to find and drive a SimFoundry checkout."""

    repo_dir: Path = DEFAULT_SIMFOUNDRY_DIR
    mamba_exe: str = "mamba"
    env_simfoundry: str = "simfoundry"
    # Stage 2 env. Upstream always maps it to "da3", but FoundationStereo (the
    # stereo depth backend used here) is installed in the simfoundry env.
    env_depth: str = "simfoundry"
    env_mesh: str = "hunyuan"
    env_nerfstudio: str = "nerfstudio_simfoundry"
    # Cameras (serials or roles) to draw candidates from; None: every camera.
    cameras: tuple[str, ...] | None = None
    max_frames: int = 12  # candidates offered to SimFoundry's frame selection
    stages: tuple[str, ...] = DEFAULT_STAGES
    # "codex" routes SimFoundry's Gemini calls through the local Codex CLI (needs
    # the r2s2r branch of the fork); "gemini" keeps upstream Vertex / API-key use.
    vlm_backend: str = "codex"
    codex_bin: str | None = None  # None: SimFoundry picks the desktop app's CLI
    codex_model: str = "gpt-6-astra"
    codex_reasoning: str = "medium"
    # Stage 5 can first re-generate the whole scene image at a higher resolution
    # with the image model; off by default so no generated content enters the scene.
    upsample_source_image: bool = False
    mesh_low_vram: bool = True  # stage 7 CPU offload (~29 GB -> ~6 GB VRAM)
    # RGB-D input is downscaled like FoundationStereo's output (s2_fs.fs.scale), so
    # stages 3-12 see the same resolution whichever depth source was used.
    rgbd_scale: float = 0.5
    mask_robot: bool = True  # remove the robot from candidate depth (MuJoCo models)
    overrides: list[str] = field(default_factory=list)  # extra Hydra overrides


@register_backend("simfoundry")
class SimFoundryBackend(ReconstructionBackend):
    """Single-frame SimFoundry reconstruction, re-anchored to the robot base."""

    def __init__(self, config: SimFoundryConfig | None = None, **kwargs: Any) -> None:
        self.config = config or SimFoundryConfig(**kwargs)

    # ------------------------------------------------------------------ paths
    def scene_dir(self, capture: Capture, workdir: Path) -> Path:
        """SimFoundry's ``${root_dir}/${scene_name}`` for this capture."""
        return Path(workdir).resolve() / capture.name

    # ----------------------------------------------------------------- inputs
    def candidates(self, capture: Capture) -> list[FrameRecord]:
        """Frames offered to SimFoundry: static, with stereo or measured depth."""

        def has_depth(frame: FrameRecord) -> bool:
            stereo = capture.cameras[frame.camera].stereo_baseline is not None
            return frame.depth_image is not None or (
                stereo and frame.right_image is not None
            )

        frames = capture.select_frames(
            self.config.cameras, self.config.max_frames, has_depth
        )
        if not frames:
            raise ValueError(
                f"no static frames with stereo or depth from cameras "
                f"{self.config.cameras or 'any'} in capture {capture.name}"
            )
        return frames

    def prepare_inputs(self, capture: Capture, workdir: Path) -> Path:
        """Write every candidate where SimFoundry reads it (see the module doc)."""
        frames = self.candidates(capture)
        scene_dir = self.scene_dir(capture, workdir)
        s1_dir, fs_dir = scene_dir / "s1_zed", scene_dir / "s2_fs"
        for d in (s1_dir, fs_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        masker = self._masker(capture)
        frame_map = []
        for i, frame in enumerate(frames):
            cam = capture.cameras[frame.camera]
            left = _imread(capture.root / frame.left_image)
            cv2.imwrite(str(s1_dir / f"image_{i}_l.png"), left)
            if frame.depth_image is not None:
                depth = read_depth(capture.root / frame.depth_image)
                if masker is not None:
                    depth, mask = masker.mask_depth(
                        depth,
                        cam.K,
                        frame.T_base_cam,
                        frame.joint_positions,
                        frame.gripper_position,
                    )
                    _write_mask(fs_dir / f"image_{i}_robot_mask.png", mask)
                _write_fs_outputs(
                    fs_dir / f"image_{i}", left, depth, cam.K, self.config.rgbd_scale
                )
                source = "rgbd"
            else:
                assert frame.right_image is not None
                right = _imread(capture.root / frame.right_image)
                cv2.imwrite(str(s1_dir / f"image_{i}_r.png"), right)
                # FoundationStereo's format: row-major K, then the baseline (m).
                k_line = " ".join(f"{v:.8f}" for v in cam.K.reshape(-1))
                (s1_dir / f"image_{i}_intrinsic.txt").write_text(
                    f"{k_line}\n{cam.stereo_baseline:.8f}\n", encoding="utf-8"
                )
                source = "stereo"
            frame_map.append(
                {
                    "index": i,
                    "camera": frame.camera,
                    "step": frame.step,
                    "depth": source,
                }
            )
        stereo = [m for m in frame_map if m["depth"] == "stereo"]
        if stereo:
            # Stage 2's shared fallback; every stereo frame also has its own file.
            shutil.copy(
                s1_dir / f"image_{stereo[0]['index']}_intrinsic.txt",
                s1_dir / "intrinsic.txt",
            )
        (s1_dir / FRAME_MAP_FILENAME).write_text(
            json.dumps(frame_map, indent=2), encoding="utf-8"
        )
        return s1_dir

    def mask_stage2_outputs(self, capture: Capture, workdir: Path) -> None:
        """Remove the robot from FoundationStereo's depth of the stereo candidates.

        Idempotent: the unmasked depth is kept as ``image_<i>_depth_meter_raw.npy``.
        """
        masker = self._masker(capture)
        if masker is None:
            return
        scene_dir = self.scene_dir(capture, workdir)
        fs_dir = scene_dir / "s2_fs"
        for m in _read_json(scene_dir / "s1_zed" / FRAME_MAP_FILENAME):
            if m["depth"] != "stereo":
                continue
            stem = fs_dir / f"image_{m['index']}"
            raw = Path(f"{stem}_depth_meter_raw.npy")
            if not raw.exists():
                shutil.copy(f"{stem}_depth_meter.npy", raw)
            frame = _frame(capture, m)
            depth, mask = masker.mask_depth(
                np.load(raw),
                np.load(f"{stem}_K.npy"),
                frame.T_base_cam,
                frame.joint_positions,
                frame.gripper_position,
            )
            np.save(f"{stem}_depth_meter.npy", depth)
            _write_mask(Path(f"{stem}_robot_mask.png"), mask)

    def _masker(self, capture: Capture) -> RobotMasker | None:
        if not self.config.mask_robot:
            return None
        # MuJoCo (and an EGL context) only when masking is on.
        from r2s2r.robots.mask import (  # pylint: disable=import-outside-toplevel
            RobotMasker,
        )
        from r2s2r.robots.mujoco_models import (  # pylint: disable=import-outside-toplevel
            EMBODIMENTS,
        )

        if capture.embodiment not in EMBODIMENTS:
            logger.warning(
                "no robot model for %s: depth not masked", capture.embodiment
            )
            return None
        return RobotMasker(capture.embodiment)

    # -------------------------------------------------------------------- run
    def build_command(
        self, capture: Capture, workdir: Path, stages: tuple[str, ...] | None = None
    ) -> tuple[list[str], dict[str, str], Path]:
        """The orchestrator call, its environment and working directory."""
        cfg = self.config
        stages = stages or cfg.stages
        upsample = _hydra_bool(cfg.upsample_source_image)
        overrides = [
            f"root_dir={Path(workdir).resolve()}",
            f"scene_name={capture.name}",
            "s2_depth.backend=fs",
            "s3_ground.use_fs=true",
            f"s5_scene.use_upsampled_source_image={upsample}",
            f"s7_mesh.low_vram={_hydra_bool(cfg.mesh_low_vram)}",
            *cfg.overrides,
        ]
        cmd = [
            cfg.mamba_exe,
            "run",
            "-n",
            cfg.env_simfoundry,
            "python",
            "scripts/pipeline/A_reconstruction/run_reconstruction.py",
            "--input-mode",
            "stereo",
            "--include",
            ",".join(stages),
            "--exec-mode",
            "mamba",
            "--env-simfoundry",
            cfg.env_simfoundry,
            "--env-da3",
            cfg.env_depth,
            "--env-mesh",
            cfg.env_mesh,
            "--env-nerfstudio",
            cfg.env_nerfstudio,
            "--env-b1k",
            cfg.env_simfoundry,
            # Stage 9 is only planned with this flag; --include alone cannot add it.
            *(["--detect-articulation"] if "9" in stages else []),
            *overrides,
        ]
        env = dict(os.environ)
        # Make every stage import the submodule, not another SimFoundry install.
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(cfg.repo_dir), env.get("PYTHONPATH", "")) if p
        )
        env["SIMFOUNDRY_VLM_BACKEND"] = cfg.vlm_backend
        if cfg.vlm_backend == "codex":
            env["SIMFOUNDRY_CODEX_MODEL"] = cfg.codex_model
            env["SIMFOUNDRY_CODEX_REASONING"] = cfg.codex_reasoning
            if cfg.codex_bin:
                env["SIMFOUNDRY_CODEX_BIN"] = cfg.codex_bin
        return cmd, env, cfg.repo_dir

    def run(
        self,
        capture: Capture,
        workdir: Path,
        stages: tuple[str, ...] | None = None,
    ) -> None:
        """Run the requested SimFoundry stages (after :meth:`prepare_inputs`).

        With stereo candidates, stage 2 runs on its own so the robot can be removed from
        its depth before stages 3+ read it.
        """
        stages = stages or self.config.stages
        if "9" in stages and not self.articulation_installed():
            logger.warning(
                "stage 9 skipped: articulate-anything is not installed "
                "(scripts/setup/install_articulation.sh); objects stay rigid"
            )
            stages = tuple(s for s in stages if s != "9")
        vlm_stages = sorted(GEMINI_STAGES & set(stages))
        if (
            vlm_stages
            and self.config.vlm_backend == "gemini"
            and not _gemini_configured()
        ):
            raise RuntimeError(
                f"stages {vlm_stages} call Gemini: set GCLOUD_PROJECT (Vertex AI + "
                "gcloud ADC) or GEMINI_API_KEY first."
            )
        frame_map = _read_json(
            self.scene_dir(capture, workdir) / "s1_zed" / FRAME_MAP_FILENAME
        )
        has_stereo = any(m["depth"] == "stereo" for m in frame_map)
        ran_stage2 = "2" in stages and has_stereo
        if ran_stage2:
            self._run_stages(capture, workdir, ("2",))
        stages = tuple(s for s in stages if s != "2")
        if has_stereo and (ran_stage2 or stages):
            self.mask_stage2_outputs(capture, workdir)
        if stages:
            self._run_stages(capture, workdir, stages)

    def articulation_installed(self) -> bool:
        """Whether stage 9's articulate-anything checkout exists."""
        return (
            self.config.repo_dir / "deps" / "articulate-anything" / "simfoundry"
        ).is_dir()

    def _run_stages(
        self, capture: Capture, workdir: Path, stages: tuple[str, ...]
    ) -> None:
        cmd, env, cwd = self.build_command(capture, workdir, stages)
        logger.info("running SimFoundry stages %s in %s", ",".join(stages), cwd)
        subprocess.run(cmd, cwd=cwd, env=env, check=True)

    # ------------------------------------------------------------------ parse
    def parse(self, capture: Capture, workdir: Path) -> SceneSpec:
        """Read stages 3-12 and express the scene in the robot base frame."""
        scene_dir = self.scene_dir(capture, workdir)
        selection = _read_json(scene_dir / "s3_ground" / "frame_selection.json")
        idx = int(selection["selected_idx"])
        frame_map = _read_json(scene_dir / "s1_zed" / FRAME_MAP_FILENAME)
        ref_frame = _frame(
            capture, next(m for m in frame_map if int(m["index"]) == idx)
        )

        # SimFoundry world = support-plane frame of the canonical camera image.
        T_world_cam = np.load(scene_dir / "s4_frame" / f"image_{idx}_cam2world.npy")
        T_base_world = ref_frame.T_base_cam @ invert(T_world_cam)

        info = _read_json(scene_dir / "s11_sim" / "scene_objects_info.json")
        poses = _read_json(scene_dir / "s12_physics" / "pb_scene_poses.json")
        objects = []
        for obj in info.values():
            name, category, model = obj["name"], obj["category"], obj["model"]
            if name not in poses:
                logger.warning("object %s has no stabilized pose; skipped", name)
                continue
            pos, quat_xyzw = poses[name]
            T_world_obj = pos_quat_to_matrix(pos, quat_xyzw_to_wxyz(quat_xyzw))
            urdf = scene_dir / "s11_sim" / "objects" / category / model / "urdf"
            urdf = urdf / f"{model}.urdf"
            articulated = bool(obj.get("is_articulated", False))
            asset = urdf
            if urdf.exists():
                # SimFoundry's world frame is the support plane.
                asset = make_sim_ready(urdf, T_world_obj, articulated)
            objects.append(
                ObjectSpec(
                    name=name,
                    category=category,
                    asset_path=str(asset),
                    T_base_obj=T_base_world @ T_world_obj,
                    mass=_urdf_mass(urdf),
                    friction=obj.get("friction"),
                    articulated=articulated,
                )
            )

        return SceneSpec(
            name=capture.name,
            embodiment=capture.embodiment,
            objects=objects,
            T_base_support=T_base_world,
            cameras=capture.cameras,
            reference_camera=ref_frame.camera,
            reference_step=ref_frame.step,
            joint_positions=ref_frame.joint_positions,
            provenance={
                "backend": self.name,
                "simfoundry_scene_dir": str(scene_dir),
                "selected_idx": idx,
                "selection_decided_by": selection.get("decided_by"),
            },
        )

    def reconstruct(self, capture: Capture, workdir: Path) -> SceneSpec:
        self.prepare_inputs(capture, workdir)
        self.run(capture, workdir)
        return self.parse(capture, workdir)


def load_stage2_views(
    capture: Capture, workdir: Path, steps: set[int] | None = None
) -> list[DepthView]:
    """Depth of every SimFoundry candidate (stereo or measured, robot removed)."""
    scene_dir = Path(workdir).resolve() / capture.name
    fs = scene_dir / "s2_fs"
    views = []
    for m in _read_json(scene_dir / "s1_zed" / FRAME_MAP_FILENAME):
        if steps is not None and int(m["step"]) not in steps:
            continue
        views.append(
            DepthView(
                camera=m["camera"],
                step=int(m["step"]),
                depth=np.load(fs / f"image_{m['index']}_depth_meter.npy").astype(float),
                K=np.load(fs / f"image_{m['index']}_K.npy").astype(float),
                T_base_cam=_frame(capture, m).T_base_cam,
                image=np.load(fs / f"image_{m['index']}_rgb.npy"),
            )
        )
    return views


def _imread(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def _write_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def _frame(capture: Capture, entry: dict[str, Any]) -> FrameRecord:
    """The capture frame a frame-map entry points at."""
    return next(
        f
        for f in capture.frames
        if f.camera == entry["camera"] and f.step == int(entry["step"])
    )


def _write_fs_outputs(
    stem: Path, bgr: np.ndarray, depth: np.ndarray, K: np.ndarray, scale: float
) -> None:
    """What FoundationStereo writes per frame: rgb (png + npy), depth, K."""
    h, w = depth.shape
    size = (int(round(w * scale)), int(round(h * scale)))
    rgb = cv2.cvtColor(
        cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
    )
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    K = K.astype(np.float32).copy()
    K[:2, :2] *= scale
    K[:2, 2] = (K[:2, 2] + 0.5) * scale - 0.5  # pixel-centre convention
    cv2.imwrite(f"{stem}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(f"{stem}_rgb.npy", rgb)
    np.save(f"{stem}_depth_meter.npy", depth.astype(np.float32))
    np.save(f"{stem}_K.npy", K)


def _hydra_bool(value: bool) -> str:
    return "true" if value else "false"


def _gemini_configured() -> bool:
    return any(
        os.environ.get(k)
        for k in ("GCLOUD_PROJECT", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    )


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: did the SimFoundry stage run?")
    return json.loads(path.read_text(encoding="utf-8"))


def _urdf_mass(urdf: Path) -> float | None:
    """Total mass over all links, if the URDF exists and declares any."""
    if not urdf.exists():
        return None
    masses = [float(m.attrib["value"]) for m in ET.parse(urdf).getroot().iter("mass")]
    return sum(masses) if masses else None
