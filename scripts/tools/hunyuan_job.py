"""Hunyuan3D-2.1 single-image mesh generation, run in the hunyuan env by
r2s2r.tools.envjobs.

    python hunyuan_job.py JOB.json RESULT.json

JOB: {"repo": HUNYUAN_REPO, "low_vram": true,
      "items": [{"image": RGBA_PNG, "out": OUT.glb, "seed": 1}]}

Writes ``OUT.glb`` (textured) and ``OUT_untextured.glb`` per item; one failed item does
not stop the others.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import torch


def main(job_path, result_path):
    """Run the job in ``job_path``; write its result to ``result_path``."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    repo = job["repo"]
    sys.path.insert(0, repo)
    # Both need the Hunyuan repository on the path first.
    # pylint: disable=import-outside-toplevel,import-error
    from torchvision_fix import apply_fix

    apply_fix()
    from simfoundry.models.mesh_generator import Hunyuan, make_generator

    Hunyuan.set_repo_path(repo_path=repo)
    generator = make_generator(
        Hunyuan,
        low_vram=bool(job.get("low_vram", True)),
        create_shape_pipeline=True,
        create_texture_pipeline=True,
    )
    results = []
    for item in job["items"]:
        Path(item["out"]).parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        try:
            generator.generate_mesh(
                out_fpath=item["out"],
                shape_image_path=item["image"],
                texture_image_path=item["image"],
                seed=int(item.get("seed", 1)),
            )
            results.append({**item, "ok": True, "seconds": round(time.time() - start)})
        except Exception as exc:  # pylint: disable=broad-except
            traceback.print_exc()
            results.append(
                {**item, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
        torch.cuda.empty_cache()
    Path(result_path).write_text(
        json.dumps({"results": results}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
