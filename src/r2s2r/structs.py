"""Method-independent data structures.

A :class:`Capture` is what the robot recorded (images, calibration, robot state);
a :class:`SceneSpec` is what a reconstruction method produced from it; a
:class:`DepthView` is one calibrated metric depth image. Captures and scenes are
plain data saved as JSON next to their files, so every stage can be run, inspected
and re-run on its own. Frame and quaternion conventions are those of
:mod:`r2s2r.transforms`; every pose in a SceneSpec is in the robot base frame.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

CAPTURE_FILENAME = "capture.json"
TRAJECTORY_FILENAME = "trajectory.npz"
SCENE_FILENAME = "scene.json"
DEPTH_PNG_SCALE = 0.001  # meters per unit of uint16 depth PNGs
# A support is simulated and drawn as a slab this thick (m), its top face the plane.
SUPPORT_THICKNESS = 0.02


def read_rgb(path: str | Path) -> NDArray[np.uint8]:
    """A colour image, RGB."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise IOError(f"cannot read {path}")
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.uint8)


def read_depth(path: str | Path) -> NDArray[np.float32]:
    """Metric depth from a uint16 PNG (:data:`DEPTH_PNG_SCALE` metres per unit)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"cannot read {path}")
    return raw.astype(np.float32) * DEPTH_PNG_SCALE


def write_depth(path: str | Path, depth: NDArray) -> None:
    """Metric depth as a uint16 PNG, the inverse of :func:`read_depth` (clipped to its
    range; 0 is no depth)."""
    units = np.clip(np.round(np.asarray(depth) / DEPTH_PNG_SCALE), 0, 65535)
    cv2.imwrite(str(path), units.astype(np.uint16))


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
    T_base_cam: NDArray[np.float64]  # a static camera's frames may leave it out
    joint_positions: NDArray[np.float64]
    gripper_position: float  # 0 open, 1 closed
    # Metric depth of the left image: uint16 PNG, DEPTH_PNG_SCALE meters per unit.
    depth_image: str | None = None

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], camera: CameraSpec | None = None
    ) -> FrameRecord:
        """Inverse of ``asdict``; a frame without ``T_base_cam`` takes that of its
        ``camera``, which must be static and calibrated."""
        d = dict(d)
        if d.get("T_base_cam") is None:
            if camera is None or not camera.is_static or camera.T_base_cam is None:
                raise ValueError(
                    f"frame {d.get('step')} of {d.get('camera')} has no T_base_cam, "
                    "and its camera is not a static, calibrated one"
                )
            d["T_base_cam"] = camera.T_base_cam
        d["T_base_cam"] = _array(d["T_base_cam"])
        d["joint_positions"] = _array(d["joint_positions"])
        return cls(**d)


@dataclass
class RobotTrajectory:
    """The robot's state at every control step of the recording (the frames keep only
    some steps), for replaying it."""

    steps: NDArray[np.int64]  # the capture's step indices
    times: NDArray[np.float64]  # seconds since the first step
    joint_positions: NDArray[np.float64]  # (N, dof) arm joints
    gripper_position: NDArray[np.float64]  # (N,) 0 open, 1 closed

    def save(self, path: str | Path) -> None:
        """Write the arrays to ``path`` (``.npz``)."""
        np.savez(
            path,
            steps=self.steps,
            times=self.times,
            joint_positions=self.joint_positions,
            gripper_position=self.gripper_position,
        )

    @classmethod
    def load(cls, path: str | Path) -> RobotTrajectory:
        """Inverse of :meth:`save`."""
        with np.load(path) as data:
            return cls(
                steps=np.asarray(data["steps"], np.int64),
                times=np.asarray(data["times"], np.float64),
                joint_positions=np.asarray(data["joint_positions"], np.float64),
                gripper_position=np.asarray(data["gripper_position"], np.float64),
            )


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
    # The robot's state at every step, when recorded (``trajectory.npz``).
    trajectory: RobotTrajectory | None = None

    def frames_of(self, camera: str) -> list[FrameRecord]:
        """Frames of one camera, in step order."""
        return sorted(
            (f for f in self.frames if f.camera == camera), key=lambda f: f.step
        )

    def frame(self, camera: str, step: int) -> FrameRecord:
        """The frame of ``camera`` (serial or role) at ``step``."""
        serial = self.resolve_cameras([camera])[0]
        for f in self.frames:
            if f.camera == serial and f.step == step:
                return f
        raise KeyError(f"no frame of {camera} at step {step} in capture {self.name}")

    def in_static(self, step: int) -> bool:
        """Whether ``step`` is in the static period (objects not yet touched)."""
        start, end = self.static_steps
        return start <= step < end

    def camera_by_role(self, role: str) -> CameraSpec:
        """Look up a camera by role (e.g. ``"ext1"``)."""
        for cam in self.cameras.values():
            if cam.role == role:
                return cam
        raise KeyError(f"no camera with role {role!r} in capture {self.name}")

    def resolve_cameras(self, names: Iterable[str] | None = None) -> list[str]:
        """Serials of the cameras named by serial or role; all cameras for None."""
        if names is None:
            return list(self.cameras)
        serials = []
        for name in names:
            serial = name if name in self.cameras else self.camera_by_role(name).serial
            if serial not in serials:
                serials.append(serial)
        return serials

    def select_frames(
        self,
        max_frames: int | None = None,
        keep: Callable[[FrameRecord], bool] | None = None,
    ) -> list[FrameRecord]:
        """Frames of the static period from every camera.

        At most ``max_frames`` in total, shared evenly between the cameras (a camera
        with fewer frames leaves its share to the others) and spread evenly over each
        camera's frames. ``keep`` filters frames first.
        """
        pools = {}
        for serial in self.cameras:
            pool = [
                f
                for f in self.frames_of(serial)
                if self.in_static(f.step) and (keep is None or keep(f))
            ]
            if pool:
                pools[serial] = pool
        quota = {s: len(p) for s, p in pools.items()}
        if max_frames is not None:
            quota = dict.fromkeys(pools, 0)
            left, open_ = max_frames, list(pools)
            while left > 0 and open_:
                share = max(1, left // len(open_))
                for serial in list(open_):
                    take = min(share, len(pools[serial]) - quota[serial], left)
                    quota[serial] += take
                    left -= take
                    if quota[serial] == len(pools[serial]):
                        open_.remove(serial)
                    if left == 0:
                        break
        picked = []
        for serial, pool in pools.items():
            if quota[serial] == 0:
                continue
            idx = np.linspace(0, len(pool) - 1, quota[serial]).round().astype(int)
            picked += [pool[i] for i in sorted(set(idx))]
        return picked

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
        if self.trajectory is not None:
            self.trajectory.save(self.root / TRAJECTORY_FILENAME)
        return path

    @classmethod
    def load(cls, root: str | Path) -> Capture:
        """Read a capture saved by :meth:`save`."""
        root = Path(root)
        d = json.loads((root / CAPTURE_FILENAME).read_text(encoding="utf-8"))
        cameras = {k: CameraSpec.from_dict(v) for k, v in d["cameras"].items()}
        return cls(
            name=d["name"],
            source=d["source"],
            embodiment=d["embodiment"],
            instruction=d["instruction"],
            cameras=cameras,
            frames=[
                FrameRecord.from_dict(f, cameras.get(f["camera"])) for f in d["frames"]
            ],
            static_steps=(int(d["static_steps"][0]), int(d["static_steps"][1])),
            root=root,
            metadata=d.get("metadata", {}),
            trajectory=(
                RobotTrajectory.load(root / TRAJECTORY_FILENAME)
                if (root / TRAJECTORY_FILENAME).exists()
                else None
            ),
        )


@dataclass
class DepthView:
    """One calibrated metric depth image."""

    camera: str
    step: int
    depth: NDArray[np.float64]  # metres, <= 0 where invalid
    K: NDArray[np.float64]
    T_base_cam: NDArray[np.float64]
    image: NDArray[np.uint8] | None = None  # RGB at the depth's resolution


@dataclass
class ObjectSpec:
    """One object, rigid, articulated or cloth, placed in the robot base frame."""

    name: str
    category: str
    # The simulation-ready URDF: absolute in memory, saved relative to the scene
    # directory.
    asset_path: str
    T_base_obj: NDArray[np.float64]
    mass: float | None = None
    friction: float | None = None
    # The object as one USD file for Isaac Lab (its rigid body or articulation,
    # colliders and friction material, or a cloth's deformable surface and material;
    # written by settling), with ``metadata.yaml``
    # beside it. Paths as for ``asset_path``.
    usd: str | None = None
    # An articulated object's joint positions (rad or m) by the URDF's joint names: as
    # recorded, or where settling left them; None for a rigid object.
    joints: dict[str, float] | None = None
    # A cloth's material (``thickness`` m, ``youngs_modulus`` Pa, ``poissons_ratio``);
    # its visual mesh is its surface as it lies, unstretched there, bending back toward
    # flat. None for a body.
    cloth: dict[str, float] | None = None

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
    reference_camera: str  # the frame the method reconstructed from
    reference_step: int
    joint_positions: NDArray[np.float64]  # robot state at the reference step
    provenance: dict[str, Any] = field(default_factory=dict)
    # Support outline as a rectangle centred on T_base_support (x, y sizes in its
    # frame); None when only the plane is known.
    support_extent: tuple[float, float] | None = None
    # The support's colour, sRGB 0 to 1, from the capture (settling); None: unknown.
    support_color: tuple[float, float, float] | None = None

    def save(self, root: str | Path) -> Path:
        """Write ``scene.json`` into ``root``, asset paths relative to it (a run can be
        moved as a whole)."""
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["cameras"] = {k: asdict(v) for k, v in self.cameras.items()}
        for obj in payload["objects"]:
            for key in ("asset_path", "usd"):
                if obj[key] is not None:
                    obj[key] = os.path.relpath(Path(obj[key]).resolve(), root)
        path = root / SCENE_FILENAME
        path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, root: str | Path) -> SceneSpec:
        """Read a scene saved by :meth:`save`; asset paths come back absolute."""
        root = Path(root).resolve()
        d = json.loads((root / SCENE_FILENAME).read_text(encoding="utf-8"))
        objects = []
        for o in d["objects"]:
            obj = ObjectSpec.from_dict(o)
            obj.asset_path = str((root / obj.asset_path).resolve())
            if obj.usd is not None:
                obj.usd = str((root / obj.usd).resolve())
            objects.append(obj)
        return cls(
            name=d["name"],
            embodiment=d["embodiment"],
            objects=objects,
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
            support_color=(
                None
                if d.get("support_color") is None
                else (
                    float(d["support_color"][0]),
                    float(d["support_color"][1]),
                    float(d["support_color"][2]),
                )
            ),
        )
