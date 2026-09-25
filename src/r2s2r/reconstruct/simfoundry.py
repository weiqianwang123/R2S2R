"""SimFoundry as a reconstruction backend (single-frame baseline).

SimFoundry (third_party/SimFoundry, a git submodule) is never imported: it runs
in its own conda environments, driven through its reconstruction orchestrator.
This module only

1. writes a capture in the layout of SimFoundry's stereo capture stage (1a), so
   its FoundationStereo path gives metric depth from calibrated stereo;
2. runs stages 2-12 (13-14 import into OmniGibson, which r2s2r does not use);
3. reads the stage outputs back and re-expresses them in the robot base frame.

SimFoundry reconstructs from a single canonical frame that it picks itself, and
defines its world frame on the support plane of that frame. The frame index is
mapped back to (camera, step) through ``r2s2r_frames.json`` so its known
``T_base_cam`` can anchor the scene to the robot.
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
from typing import Any

import cv2
import numpy as np

from r2s2r.reconstruct.base import ReconstructionBackend, register_backend
from r2s2r.structs import Capture, ObjectSpec, SceneSpec
from r2s2r.transforms import invert, pos_quat_to_matrix, quat_xyzw_to_wxyz

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SIMFOUNDRY_DIR = REPO_ROOT / "third_party" / "SimFoundry"
FRAME_MAP_FILENAME = "r2s2r_frames.json"
# Stage 9 (articulation) is optional upstream; 13-14 target OmniGibson.
DEFAULT_STAGES = ("2", "3", "4", "5", "6", "7", "8", "10", "11", "12")
# Stages that call a VLM (Gemini upstream; Codex with the r2s2r fork).
GEMINI_STAGES = {"3", "5", "6", "8", "11"}


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
    camera_role: str = "ext1"  # which stream SimFoundry reconstructs from
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
    def prepare_inputs(self, capture: Capture, workdir: Path) -> Path:
        """Write stereo pairs + ``intrinsic.txt`` where stage 1a would.

        Files are named ``image_<i>_{l,r}.png`` (not stage 1a's ``zed_<i>``), which is
        the prefix stages 3-8 read back from the FoundationStereo output.
        """
        cam = capture.camera_by_role(self.config.camera_role)
        if cam.stereo_baseline is None:
            raise ValueError(f"camera {cam.serial} is not stereo")
        frames = capture.frames_of(cam.serial)
        if not frames:
            raise ValueError(f"capture has no frames of camera {cam.serial}")
        pick = np.linspace(0, len(frames) - 1, min(self.config.max_frames, len(frames)))
        frames = [frames[i] for i in sorted({int(round(p)) for p in pick})]

        s1_dir = self.scene_dir(capture, workdir) / "s1_zed"
        if s1_dir.exists():
            shutil.rmtree(s1_dir)
        s1_dir.mkdir(parents=True)
        frame_map = []
        for i, frame in enumerate(frames):
            assert frame.right_image is not None
            for side, rel in (("l", frame.left_image), ("r", frame.right_image)):
                img = cv2.imread(str(capture.root / rel))
                if img is None:
                    raise IOError(f"cannot read {capture.root / rel}")
                cv2.imwrite(str(s1_dir / f"image_{i}_{side}.png"), img)
            frame_map.append({"index": i, "camera": cam.serial, "step": frame.step})
        np.save(s1_dir / "K.npy", cam.K)
        # FoundationStereo's format: row-major K on line 1, baseline (m) on line 2.
        k_line = " ".join(f"{v:.8f}" for v in cam.K.reshape(-1))
        (s1_dir / "intrinsic.txt").write_text(
            f"{k_line}\n{cam.stereo_baseline:.8f}\n", encoding="utf-8"
        )
        (s1_dir / FRAME_MAP_FILENAME).write_text(
            json.dumps(frame_map, indent=2), encoding="utf-8"
        )
        return s1_dir

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
        """Run the requested SimFoundry stages."""
        stages = stages or self.config.stages
        needs_vlm = GEMINI_STAGES & set(stages)
        if (
            needs_vlm
            and self.config.vlm_backend == "gemini"
            and not _gemini_configured()
        ):
            raise RuntimeError(
                f"stages {sorted(GEMINI_STAGES & set(stages))} call Gemini: set "
                "GCLOUD_PROJECT (Vertex AI + gcloud ADC) or GEMINI_API_KEY first."
            )
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
        ref = next(m for m in frame_map if int(m["index"]) == idx)
        ref_frame = next(
            f
            for f in capture.frames
            if f.camera == ref["camera"] and f.step == int(ref["step"])
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
            objects.append(
                ObjectSpec(
                    name=name,
                    category=category,
                    asset_path=str(urdf),
                    T_base_obj=T_base_world @ T_world_obj,
                    mass=_urdf_mass(urdf),
                    friction=obj.get("friction"),
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
    """Mass of the first link that declares one, if the URDF exists."""
    if not urdf.exists():
        return None
    mass = ET.parse(urdf).getroot().find(".//inertial/mass")
    return float(mass.attrib["value"]) if mass is not None else None
