"""The Isaac Lab robot behind :class:`~r2s2r.policy.robot.RobotInterface`.

Import after ``isaaclab.app.AppLauncher`` has started.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from numpy.typing import NDArray

from r2s2r.policy.robot import RobotInterface
from r2s2r.robots.franka import PandaKinematics
from r2s2r.sim.isaaclab.scene import PANDA_JOINTS

FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
FINGER_OPEN = 0.04


class IsaacLabRobot(RobotInterface):
    """A Panda with the Franka Hand in an interactive scene (env 0)."""

    control_dt = 0.02

    def __init__(
        self,
        sim: SimulationContext,
        scene: InteractiveScene,
        on_step: Callable[[IsaacLabRobot], None] | None = None,
        render_every: int = 0,  # render on every n-th control step (0: never)
    ) -> None:
        super().__init__(PandaKinematics())
        self.sim = sim
        self.scene = scene
        self.robot = scene["robot"]
        self.on_step = on_step
        self.render_every = render_every
        self.arm_ids, _ = self.robot.find_joints(PANDA_JOINTS, preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(FINGER_JOINTS, preserve_order=True)
        self.decimation = max(1, int(round(self.control_dt / sim.get_physics_dt())))
        self.steps = 0

    def joint_positions(self) -> NDArray[np.float64]:
        return self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy().astype(float)

    def gripper_width(self) -> float:
        return float(self.robot.data.joint_pos[0, self.finger_ids].sum().item())

    def _hold(self, q: NDArray[np.float64], gripper_closed: bool) -> None:
        target = self.robot.data.joint_pos_target.clone()
        target[0, self.arm_ids] = target.new_tensor(q)
        target[0, self.finger_ids] = 0.0 if gripper_closed else FINGER_OPEN
        self.robot.set_joint_position_target(target)
        render = self.render_every > 0 and (self.steps + 1) % self.render_every == 0
        dt = self.sim.get_physics_dt()
        for i in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=render and i == self.decimation - 1)
            self.scene.update(dt)
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self)
