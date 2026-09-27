"""A run's directory, whichever method makes it::

    RUN/
      run.json            the method, the capture, and every stage's status
                          (:mod:`r2s2r.pipeline.run`)
      inputs/
        capture/          the capture, copied (only the chosen cameras), its metadata
                          dropped but how its depth was made (a simulated capture's
                          metadata holds the true scene); stereo cameras' frames of the
                          static period get FoundationStereo depth here
                          (:mod:`r2s2r.capture.stereo`)
        frames.json       every frame: id, camera, step, image, depth, pose, ...
        sheets/<role>.png contact sheets of the frames that can be reconstructed from
      cache/depth/        robot-free depth per frame; cache/jobs/: model job logs
      s2_frames/ ...      one directory per stage (:mod:`r2s2r.pipeline.stages`)

Frames are named ``<role>@<step>`` (``ext2@12``, ``wrist@30``).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from r2s2r.capture.stereo import add_stereo_depth
from r2s2r.robots import get_robot
from r2s2r.structs import Capture, DepthView, FrameRecord, read_depth, read_rgb

if TYPE_CHECKING:  # MuJoCo is imported only when the robot is cut out
    from r2s2r.robots.mask import RobotMasker

RUN_FILENAME = "run.json"
CAPTURE_DIR = "inputs/capture"
# Metadata a capture keeps in the run's copy: how its depth was made.
KEPT_METADATA = ("depth",)
SHEET_MAX = 24  # thumbnails per contact sheet
SHEET_WIDTH = 320


@dataclass
class Workspace:
    """A run's directory and the capture copy the methods and the tools read."""

    root: Path
    capture: Capture
    _masker: RobotMasker | None = field(default=None, repr=False)

    # ----------------------------------------------------------------- set-up
    @classmethod
    def create(
        cls,
        capture: Capture,
        root: str | Path,
        method: str,
        cameras: list[str] | None = None,
    ) -> Workspace:
        """Copy ``capture`` into a new run of ``method`` at ``root`` and index its
        frames.

        Only ``cameras`` (roles or serials; default: all) are kept, and only the files
        their frames refer to are copied. Stereo frames of the static period without
        depth get FoundationStereo's.
        """
        root = Path(root).resolve()
        if (root / RUN_FILENAME).exists():
            raise FileExistsError(f"{root} already holds a run")
        serials = capture.resolve_cameras(cameras)
        frames = [f for f in capture.frames if f.camera in serials]
        copy = root / CAPTURE_DIR
        for frame in frames:
            for rel in (frame.left_image, frame.right_image, frame.depth_image):
                if rel is not None:
                    (copy / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(capture.root / rel, copy / rel)
        clean = replace(
            capture,
            cameras={s: capture.cameras[s] for s in serials},
            frames=frames,
            root=copy,
            metadata={k: v for k, v in capture.metadata.items() if k in KEPT_METADATA},
        )
        clean.save()
        add_stereo_depth(clean, root / "cache" / "jobs")
        ws = cls(root, Capture.load(copy))
        (root / "inputs" / "frames.json").write_text(
            json.dumps(ws.frame_index(), indent=1), encoding="utf-8"
        )
        ws.write_sheets()
        ws.write_run({"method": method, "capture": CAPTURE_DIR, "stages": {}})
        return ws

    @classmethod
    def load(cls, root: str | Path) -> Workspace:
        """The run at ``root``."""
        root = Path(root).resolve()
        info = json.loads((root / RUN_FILENAME).read_text(encoding="utf-8"))
        return cls(root, Capture.load(root / info["capture"]))

    @classmethod
    def find(cls, start: str | Path | None = None) -> Workspace:
        """The run containing ``start`` (default: the working directory)."""
        here = Path(start or Path.cwd()).resolve()
        for d in (here, *here.parents):
            if (d / RUN_FILENAME).exists():
                return cls.load(d)
        raise FileNotFoundError(f"no {RUN_FILENAME} in {here} or above: pass --ws RUN")

    # -------------------------------------------------------------- run.json
    def read_run(self) -> dict[str, Any]:
        """``run.json``."""
        run: dict[str, Any] = json.loads(
            (self.root / RUN_FILENAME).read_text(encoding="utf-8")
        )
        return run

    def write_run(self, run: dict[str, Any]) -> None:
        """Replace ``run.json`` at once (the viewer and the tools read it meanwhile)."""
        path = self.root / RUN_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(run, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    # ----------------------------------------------------------------- frames
    def frame_id(self, frame: FrameRecord) -> str:
        """``<role>@<step>``."""
        return f"{self.capture.cameras[frame.camera].role}@{frame.step}"

    def frame(self, frame_id: str) -> FrameRecord:
        """The frame named ``<role>@<step>`` (or ``<serial>@<step>``)."""
        name, _, step = frame_id.partition("@")
        if not step.isdigit():
            raise ValueError(f"frame id {frame_id!r} is not <camera role>@<step>")
        return self.capture.frame(name, int(step))

    def frame_index(self) -> list[dict[str, Any]]:
        """What ``inputs/frames.json`` lists for every frame."""
        rows = []
        for f in sorted(self.capture.frames, key=lambda f: (f.camera, f.step)):
            cam = self.capture.cameras[f.camera]
            rows.append(
                {
                    "id": self.frame_id(f),
                    "camera": cam.role,
                    "step": f.step,
                    "static": self.capture.in_static(f.step),
                    "image": str(self.image_path(f).relative_to(self.root)),
                    "depth": f.depth_image is not None,
                    "gripper_closed": round(float(f.gripper_position), 3),
                    "width": cam.width,
                    "height": cam.height,
                    "K": cam.K.tolist(),
                    "T_base_cam": np.round(f.T_base_cam, 6).tolist(),
                }
            )
        return rows

    def image_path(self, frame: FrameRecord) -> Path:
        """The frame's (left) colour image."""
        return self.capture.root / frame.left_image

    def image(self, frame: FrameRecord) -> NDArray[np.uint8]:
        """The frame's colour image, RGB."""
        return read_rgb(self.image_path(frame))

    def reconstructable(self) -> list[FrameRecord]:
        """Frames of the static period that have depth."""
        return [
            f
            for f in sorted(self.capture.frames, key=lambda f: (f.camera, f.step))
            if self.capture.in_static(f.step) and f.depth_image is not None
        ]

    def write_sheets(self) -> None:
        """A contact sheet per camera of the frames that can be reconstructed from, each
        labelled with its id."""
        out = self.root / "inputs" / "sheets"
        out.mkdir(parents=True, exist_ok=True)
        frames = self.reconstructable()
        for serial, cam in self.capture.cameras.items():
            pool = [f for f in frames if f.camera == serial]
            if not pool:
                continue
            idx = np.linspace(0, len(pool) - 1, min(SHEET_MAX, len(pool)))
            thumbs = []
            for i in sorted(set(idx.round().astype(int))):
                frame = pool[i]
                image = self.image(frame)
                h = int(round(image.shape[0] * SHEET_WIDTH / image.shape[1]))
                thumb = np.asarray(
                    cv2.resize(image, (SHEET_WIDTH, h), interpolation=cv2.INTER_AREA),
                    np.uint8,
                )
                _label(thumb, self.frame_id(frame))
                thumbs.append(thumb)
            cv2.imwrite(
                str(out / f"{cam.role}.png"),
                cv2.cvtColor(_grid(thumbs, 4), cv2.COLOR_RGB2BGR),
            )

    # ------------------------------------------------------------------ depth
    def depth_view(self, frame: FrameRecord) -> DepthView:
        """The frame's metric depth with the robot cut out (cached)."""
        if frame.depth_image is None:
            raise ValueError(f"frame {self.frame_id(frame)} has no depth")
        depth, _ = self._robot_free(frame)
        cam = self.capture.cameras[frame.camera]
        return DepthView(
            camera=frame.camera,
            step=frame.step,
            depth=depth.astype(np.float64),
            K=cam.K,
            T_base_cam=frame.T_base_cam,
            image=self.image(frame),
        )

    def robot_mask(self, frame: FrameRecord) -> NDArray[np.bool_]:
        """Pixels the robot covers in the frame."""
        return self._robot_free(frame)[1]

    def _robot_free(
        self, frame: FrameRecord
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        path = self.root / "cache" / "depth" / f"{self.frame_id(frame)}.npz"
        if path.exists():
            with np.load(path) as data:
                return data["depth"], data["robot"]
        cam = self.capture.cameras[frame.camera]
        if frame.depth_image is not None:
            depth = read_depth(self.capture.root / frame.depth_image)
        else:
            depth = np.zeros((cam.height, cam.width), np.float32)
        depth, robot = self.masker.mask_depth(
            depth,
            cam.K,
            frame.T_base_cam,
            frame.joint_positions,
            frame.gripper_position,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, depth=depth, robot=robot)
        return depth, robot

    @property
    def masker(self) -> RobotMasker:
        """The capture's robot, for cutting it out of depth."""
        if self._masker is None:
            # pylint: disable=import-outside-toplevel
            from r2s2r.robots.mask import RobotMasker

            self._masker = RobotMasker(get_robot(self.capture.embodiment))
        return self._masker

    def close(self) -> None:
        """Free the robot renderer."""
        if self._masker is not None:
            self._masker.close()
            self._masker = None


def _label(image: NDArray[np.uint8], text: str) -> None:
    cv2.rectangle(image, (0, 0), (9 * len(text) + 8, 20), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (4, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )


def _grid(images: list[NDArray[np.uint8]], columns: int) -> NDArray[np.uint8]:
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    rows = int(np.ceil(len(images) / columns))
    out = np.zeros((rows * h, min(columns, len(images)) * w, 3), np.uint8)
    for k, image in enumerate(images):
        r, c = divmod(k, columns)
        out[r * h : r * h + image.shape[0], c * w : c * w + image.shape[1]] = image
    return out
