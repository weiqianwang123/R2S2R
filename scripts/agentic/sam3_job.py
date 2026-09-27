"""SAM3 segmentation, run in the simfoundry env by r2s2r.agentic.envjobs.

    python sam3_job.py JOB.json RESULT.json

JOB: {"threshold": 0.5, "requests": [{"image": PATH, "out": PREFIX, "text": STR | null,
      "points": [[x, y, label], ...] | null, "box": [x0, y0, x1, y1] | null}]}

A text request returns every instance SAM3 finds above the threshold; a point / box
request returns the one mask it outlines. Masks go to ``<PREFIX>_<k>.png`` (0/255),
best score first.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from simfoundry.models.sam_v3_gmask import SAM3


def _instances(masks, scores, prefix):
    masks = np.asarray(masks)
    masks = masks.reshape(-1, *masks.shape[-2:]) if masks.size else masks
    scores = np.asarray(scores, float).reshape(-1)
    out = []
    for i in np.argsort(-scores):
        mask = masks[i] > 0.5 if masks.dtype != bool else masks[i]
        if not mask.any():
            continue
        path = f"{prefix}_{len(out)}.png"
        cv2.imwrite(path, mask.astype(np.uint8) * 255)
        ys, xs = np.nonzero(mask)
        out.append(
            {
                "mask": path,
                "score": round(float(scores[i]), 4),
                "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "area": int(mask.sum()),
            }
        )
    return out


def main(job_path, result_path):
    """Run the job in ``job_path``; write its result to ``result_path``."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    sam = SAM3(confidence_threshold=float(job.get("threshold", 0.5)))
    results = []
    for req in job["requests"]:
        Path(req["out"]).parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(req["image"]).convert("RGB")
        if req.get("text"):
            masks, _, scores = sam.predict_segmentation(image, req["text"])
        else:
            kwargs = {}
            if req.get("points"):
                pts = np.asarray(req["points"], np.float32).reshape(-1, 3)
                kwargs["point_coords"] = pts[:, :2]
                kwargs["point_labels"] = pts[:, 2].astype(np.int32)
            if req.get("box"):
                kwargs["box"] = np.asarray(req["box"], np.float32)
            with sam._image_inference_context():  # pylint: disable=protected-access
                state = sam.model.set_image(image)
                sam.model.reset_all_prompts(state)
                masks, scores, _ = sam.model.model.predict_inst(
                    inference_state=state, multimask_output=False, **kwargs
                )
        results.append(
            {"request": req, "instances": _instances(masks, scores, req["out"])}
        )
    Path(result_path).write_text(
        json.dumps({"results": results}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
