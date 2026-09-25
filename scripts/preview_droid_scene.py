"""Write an object-free preview SceneSpec for a DROID capture.

Uses SimFoundry's stage-2 (FoundationStereo) depth of one candidate frame and
the end-effector height at the first gripper closure to fit the support plane.

    python scripts/preview_droid_scene.py outputs/iris/capture \
        outputs/iris/simfoundry --index 0 --out outputs/iris/preview
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from r2s2r.preview import preview_scene
from r2s2r.reconstruct.simfoundry import FRAME_MAP_FILENAME
from r2s2r.structs import Capture


def main() -> None:
    """Build and save the preview scene."""
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("workdir", type=Path, help="SimFoundry root_dir")
    parser.add_argument("--index", type=int, default=0, help="stage-2 frame index")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    capture = Capture.load(args.capture_dir)
    scene_dir = args.workdir / capture.name
    frame_map = json.loads((scene_dir / "s1_zed" / FRAME_MAP_FILENAME).read_text())
    ref = frame_map[args.index]
    frame = next(
        f
        for f in capture.frames
        if f.camera == ref["camera"] and f.step == int(ref["step"])
    )
    depth = np.load(scene_dir / "s2_fs" / f"image_{args.index}_depth_meter.npy")
    K = np.load(scene_dir / "s2_fs" / f"image_{args.index}_K.npy")

    with h5py.File(Path(capture.metadata["episode_dir"]) / "trajectory.h5") as f:
        ee = f["observation/robot_state/cartesian_position"][:]
    grasp_z = float(ee[min(capture.static_steps[1], len(ee) - 1), 2])

    scene = preview_scene(capture, frame, depth, K, grasp_z=grasp_z)
    path = scene.save(args.out)
    height = scene.T_base_support[2, 3]
    print(f"support plane z={height:.3f} m (grasp z={grasp_z:.3f}) -> {path}")


if __name__ == "__main__":
    main()
