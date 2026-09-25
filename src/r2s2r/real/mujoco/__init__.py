"""MuJoCo standing in for the real world, at both ends of the loop.

* real -> sim: :func:`~r2s2r.real.mujoco.capture.record_capture` writes a Capture
  exactly like a real recording (RGB, metric depth, calibration, joint states);
* sim -> real: :func:`~r2s2r.real.mujoco.deploy.run_pick` runs a program policy on
  :class:`~r2s2r.real.mujoco.world.MujocoRobot` and scores it with ground truth.

Submodules import MuJoCo (and pick an EGL context) when imported.
"""
