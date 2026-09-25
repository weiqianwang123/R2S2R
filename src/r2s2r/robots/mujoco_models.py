"""MuJoCo models of the supported embodiments, built from mujoco_menagerie.

``franka_panda`` is the Panda with the Franka Hand; ``droid_franka`` is the Panda with a
Robotiq 2F-85 on the flange, as on the DROID platform. Assets are fetched into
``~/.cache/r2s2r`` by ``scripts/setup/fetch_mujoco_assets.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r2s2r.mjrender import mujoco
from r2s2r.robots.franka import FLANGE_OFFSET, FRANKA_HAND_MAX_WIDTH

CACHE_DIR = Path(os.environ.get("R2S2R_CACHE", Path.home() / ".cache" / "r2s2r"))
MENAGERIE_DIR = CACHE_DIR / "mujoco_menagerie"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
EMBODIMENTS = ("franka_panda", "droid_franka")
ROBOTIQ_PREFIX = "gripper/"
ROBOTIQ_MAX_DRIVER = 0.8  # rad, fully closed
# Robotiq mount on DROID's flange, about link7's z axis. Fitted to the fingers in the
# wrist-camera images of a DROID episode (IRIS); the Franka Hand sits at -45 deg.
DROID_ROBOTIQ_YAW = np.pi / 2


def robot_spec(embodiment: str, menagerie_dir: str | Path = MENAGERIE_DIR) -> Any:
    """The robot alone as an ``mujoco.MjSpec``; its base is the world origin."""
    root = Path(menagerie_dir)
    if embodiment == "franka_panda":
        return mujoco.MjSpec.from_file(str(root / "franka_emika_panda" / "panda.xml"))
    if embodiment == "droid_franka":
        spec = mujoco.MjSpec.from_file(
            str(root / "franka_emika_panda" / "panda_nohand.xml")
        )
        gripper = mujoco.MjSpec.from_file(str(root / "robotiq_2f85" / "2f85.xml"))
        frame = spec.body("link7").add_frame(
            pos=[0.0, 0.0, FLANGE_OFFSET],
            quat=[
                np.cos(DROID_ROBOTIQ_YAW / 2),
                0.0,
                0.0,
                np.sin(DROID_ROBOTIQ_YAW / 2),
            ],
        )
        frame.attach_body(gripper.body("base_mount"), ROBOTIQ_PREFIX, "")
        return spec
    raise NotImplementedError(f"no MuJoCo model for embodiment {embodiment!r}")


class GripperPoser:
    """Sets the finger joints for a gripper opening (0 open, 1 closed).

    The Robotiq's four-bar linkage is closed by equality constraints, so its joint
    values are solved once by simulation at a few openings and interpolated.
    """

    def __init__(self, model: Any, embodiment: str) -> None:
        self.model = model
        self.embodiment = embodiment
        if embodiment == "franka_panda":
            self.qadr = [model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
            return
        names = [
            model.joint(i).name
            for i in range(model.njnt)
            if model.joint(i).name.startswith(ROBOTIQ_PREFIX)
        ]
        self.qadr = [model.joint(n).qposadr[0] for n in names]
        self.levels = np.linspace(0.0, 1.0, 9)
        self.table = np.stack([self._solve(level) for level in self.levels])

    def _solve(self, level: float) -> NDArray[np.float64]:
        data = mujoco.MjData(self.model)
        driver = self.model.joint(f"{ROBOTIQ_PREFIX}right_driver_joint").id
        act = next(
            a
            for a in range(self.model.nu)
            if self.model.actuator(a).name.endswith("fingers_actuator")
        )
        data.ctrl[act] = 255.0 * level
        gravity = self.model.opt.gravity.copy()
        self.model.opt.gravity[:] = 0.0
        for _ in range(1500):
            data.qpos[: len(ARM_JOINTS)] = 0.0
            data.qvel[: len(ARM_JOINTS)] = 0.0
            mujoco.mj_step(self.model, data)
        self.model.opt.gravity[:] = gravity
        assert data.qpos[self.model.jnt_qposadr[driver]] <= ROBOTIQ_MAX_DRIVER + 0.05
        return data.qpos[self.qadr].copy()

    def set(self, data: Any, gripper_position: float) -> None:
        """Write the finger joints for ``gripper_position`` into ``data.qpos``."""
        level = float(np.clip(gripper_position, 0.0, 1.0))
        if self.embodiment == "franka_panda":
            data.qpos[self.qadr] = (1.0 - level) * FRANKA_HAND_MAX_WIDTH / 2
            return
        data.qpos[self.qadr] = [
            np.interp(level, self.levels, self.table[:, j])
            for j in range(self.table.shape[1])
        ]
