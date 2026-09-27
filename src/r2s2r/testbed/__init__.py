"""MuJoCo standing in for the real world, for local tests only.

:mod:`~r2s2r.testbed.worlds` builds the worlds, :mod:`~r2s2r.testbed.record` records a
capture in one exactly like a real rig would (with the ground truth hidden in its
metadata), :mod:`~r2s2r.testbed.evaluate` scores a reconstruction against that truth,
and :mod:`~r2s2r.testbed.policy` and :mod:`~r2s2r.testbed.pick` run the pick test: a
program policy on the reconstruction, in the world (and, through
:mod:`r2s2r.sim.isaac`, in Isaac Lab).
"""
