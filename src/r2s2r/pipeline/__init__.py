"""A run: a capture in, a simulation-ready scene out, through the same stages and the
same run directory whichever method does the work.

:mod:`~r2s2r.workspace` lays out a run's directory,
:mod:`~r2s2r.pipeline.stages` says what each stage leaves there and checks it, and
:mod:`~r2s2r.pipeline.run` runs a method's stages, the shared settling (stage 5) and the
final replay. The methods: :mod:`~r2s2r.pipeline.agentic` (a coding agent with the
tools) and :mod:`~r2s2r.pipeline.fixed` (SimFoundry and multi-view refinement).
"""
