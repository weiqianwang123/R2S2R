"""Backend-independent data structures.

A :class:`Capture` is what the robot recorded (images, calibration, robot state);
a :class:`SceneSpec` is what a reconstruction backend produced from it. Both are
plain data saved as JSON next to their files, so every stage can be run, inspected
and re-run on its own. Frame and quaternion conventions are those of
:mod:`r2s2r.transforms`; every pose in a SceneSpec is in the robot base frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

CAPTURE_FILENAME = "capture.json"
SCENE_FILENAME = "scene.json"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _array(value: Any) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


@dataclass
class CameraSpec:
    """A calibrated camera.

    ``K`` belongs to the left (reference) image.
    """

    serial: str
    role: str  # e.g. "ext1", "ext2", "wrist"
    width: int
    height: int
    K: NDArray[np.float64]
    stereo_baseline: float | None = None  # meters, None for a mono camera
    is_static: bool = True  # static cameras also carry T_base_cam
    T_base_cam: NDArray[np.float64] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraSpec:
        """Inverse of ``asdict``."""
        d = dict(d)
        d["K"] = _array(d["K"])
        if d.get("T_base_cam") is not None:
            d["T_base_cam"] = _array(d["T_base_cam"])
        return cls(**d)


@dataclass
class FrameRecord:
    """One synchronized camera frame and the robot state at that step."""

    step: int
    camera: str  # CameraSpec.serial
    left_image: str  # path relative to the capture root
    right_image: str | None
    T_base_cam: NDArray[np.float64]
    joint_positions: NDArray[np.float64]
    gripper_position: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrameRecord:
        """Inverse of ``asdict``."""
        d = dict(d)
        d["T_base_cam"] = _array(d["T_base_cam"])
        d["joint_positions"] = _array(d["joint_positions"])
        return cls(**d)


@dataclass
class Capture:
    """Everything recorded for one scene, saved under ``root``."""

    name: str
    source: str  # e.g. "droid", "mujoco"
    embodiment: str  # e.g. "droid_franka"
    instruction: str
    cameras: dict[str, CameraSpec]
    frames: list[FrameRecord]
    static_steps: tuple[int, int]  # [start, end): objects assumed not to move
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def frames_of(self, camera: str) -> list[FrameRecord]:
        """Frames of one camera, in step order."""
        return sorted(
            (f for f in self.frames if f.camera == camera), key=lambda f: f.step
        )

    def camera_by_role(self, role: str) -> CameraSpec:
        """Look up a camera by role (e.g. ``"ext1"``)."""
        for cam in self.cameras.values():
            if cam.role == role:
                return cam
        raise KeyError(f"no camera with role {role!r} in capture {self.name}")

    def save(self) -> Path:
        """Write ``capture.json`` into ``root``."""
        payload = {
            "name": self.name,
            "source": self.source,
            "embodiment": self.embodiment,
            "instruction": self.instruction,
            "cameras": {k: asdict(v) for k, v in self.cameras.items()},
            "frames": [asdict(f) for f in self.frames],
            "static_steps": list(self.static_steps),
            "metadata": self.metadata,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / CAPTURE_FILENAME
        path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, root: str | Path) -> Capture:
        """Read a capture saved by :meth:`save`."""
        root = Path(root)
        d = json.loads((root / CAPTURE_FILENAME).read_text(encoding="utf-8"))
        return cls(
            name=d["name"],
            source=d["source"],
            embodiment=d["embodiment"],
            instruction=d["instruction"],
            cameras={k: CameraSpec.from_dict(v) for k, v in d["cameras"].items()},
            frames=[FrameRecord.from_dict(f) for f in d["frames"]],
            static_steps=(int(d["static_steps"][0]), int(d["static_steps"][1])),
            root=root,
            metadata=d.get("metadata", {}),
        )


@dataclass
class ObjectSpec:
    """One rigid object, placed in the robot base frame."""

    name: str
    category: str
    asset_path: str  # URDF produced by the backend (absolute path)
    T_base_obj: NDArray[np.float64]
    mass: float | None = None
    friction: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObjectSpec:
        """Inverse of ``asdict``."""
        d = dict(d)
        d["T_base_obj"] = _array(d["T_base_obj"])
        return cls(**d)


@dataclass
class SceneSpec:
    """A reconstructed scene, simulator-agnostic, in the robot base frame."""

    name: str
    embodiment: str
    objects: list[ObjectSpec]
    T_base_support: NDArray[np.float64]  # support plane: z = normal, origin on it
    cameras: dict[str, CameraSpec]
    reference_camera: str  # the frame the backend reconstructed from
    reference_step: int
    joint_positions: NDArray[np.float64]  # robot state at the reference step
    provenance: dict[str, Any] = field(default_factory=dict)
    # Support outline as a rectangle centred on T_base_support (x, y sizes in its
    # frame); None when only the plane is known.
    support_extent: tuple[float, float] | None = None

    def save(self, root: str | Path) -> Path:
        """Write ``scene.json`` into ``root``."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["cameras"] = {k: asdict(v) for k, v in self.cameras.items()}
        path = root / SCENE_FILENAME
        path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, root: str | Path) -> SceneSpec:
        """Read a scene saved by :meth:`save`."""
        d = json.loads((Path(root) / SCENE_FILENAME).read_text(encoding="utf-8"))
        return cls(
            name=d["name"],
            embodiment=d["embodiment"],
            objects=[ObjectSpec.from_dict(o) for o in d["objects"]],
            T_base_support=_array(d["T_base_support"]),
            cameras={k: CameraSpec.from_dict(v) for k, v in d["cameras"].items()},
            reference_camera=d["reference_camera"],
            reference_step=int(d["reference_step"]),
            joint_positions=_array(d["joint_positions"]),
            provenance=d.get("provenance", {}),
            support_extent=(
                None
                if d.get("support_extent") is None
                else (float(d["support_extent"][0]), float(d["support_extent"][1]))
            ),
        )
