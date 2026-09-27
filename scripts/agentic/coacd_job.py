"""Convex decomposition (CoACD) of one mesh, run in the simfoundry env by
r2s2r.agentic.envjobs.

    python coacd_job.py JOB.json RESULT.json

JOB: {"mesh": PATH, "out_dir": DIR, "threshold": 0.05, "max_hulls": 16}

Writes ``DIR/hull_<k>.obj``.
"""

import json
import sys
from pathlib import Path

import coacd
import numpy as np
import trimesh


def main(job_path, result_path):
    """Run the job in ``job_path``; write its result to ``result_path``."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    mesh = trimesh.load(job["mesh"], force="mesh", process=True)
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        threshold=float(job.get("threshold", 0.05)),
        max_convex_hull=int(job.get("max_hulls", 16)),
    )
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    hulls = []
    for k, (vertices, faces) in enumerate(parts):
        path = out_dir / f"hull_{k}.obj"
        trimesh.Trimesh(vertices, faces).export(path)
        hulls.append(str(path))
    Path(result_path).write_text(
        json.dumps({"hulls": hulls}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
