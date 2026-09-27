"""FoundationStereo depth for stereo pairs, run in the simfoundry env by
r2s2r.capture.stereo.

    python stereo_job.py JOB.json RESULT.json

JOB: {"work": DIR, "pairs": [{"left": PNG, "right": PNG, "K": 3x3, "baseline": M}]}
RESULT: {"depth": [each pair's metric depth (.npy) at the backend's scale]}

SimFoundry's own stage-2 backend does the work, with its configuration's settings, on
the pairs laid out in ``DIR`` as its stage 1 leaves them (an intrinsics file per pair).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

# pylint: disable=import-error
from hydra import compose, initialize_config_dir
from simfoundry import CFG_DIR
from simfoundry.pipeline.depth_backends import create_backend
from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir

# pylint: enable=import-error

SCENE = "stereo"


def main(job_path, result_path):
    """Run the job in ``job_path``; write its result to ``result_path``."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    work = Path(job["work"]).resolve()
    images = work / SCENE / "s1_zed"
    if images.parent.exists():
        shutil.rmtree(images.parent)
    images.mkdir(parents=True)
    for i, pair in enumerate(job["pairs"]):
        shutil.copy(pair["left"], images / f"image_{i}_l.png")
        shutil.copy(pair["right"], images / f"image_{i}_r.png")
        k_line = " ".join(f"{v:.8f}" for v in np.asarray(pair["K"]).reshape(-1))
        (images / f"image_{i}_intrinsic.txt").write_text(
            f"{k_line}\n{pair['baseline']:.8f}\n", encoding="utf-8"
        )
    bootstrap_hydra_workdir(__file__)  # as a stage: the backend's paths start there
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        cfg = compose("real2sim_cfg", [f"root_dir={work}", f"scene_name={SCENE}"])
    create_backend("fs").run(cfg)
    out = Path(cfg.s2_fs.out_dir)
    depth = [str(out / f"image_{i}_depth_meter.npy") for i in range(len(job["pairs"]))]
    Path(result_path).write_text(
        json.dumps({"depth": depth}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
