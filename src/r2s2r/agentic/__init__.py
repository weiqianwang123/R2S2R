"""Agent-driven reconstruction: a coding agent ("astra", Codex ``gpt-6-astra``) builds
the scene from the robot's own recording, with tools.

1. capture (as before: the robot's calibrated cameras, poses in its base frame, joints);
2. astra picks several frames (4-8), the support surface and the scene's frame on it;
3. astra decomposes the scene: every object segmented in those frames, generated from
   its best single view (Hunyuan3D-2.1), and fitted to all of them;
4. astra assembles the objects into a simulation-ready scene (mass, friction,
   collision parts);
5. the scene settles under physics in Isaac Lab;
6. astra refines the result against every recorded view, replaying the robot's
   recording in Isaac Lab (geometry only; the stage can be left out).

The final scene is then replayed once more against every recorded view, to be looked at
in the viewer (:mod:`r2s2r.viewer`).

:mod:`r2s2r.agentic.workspace` lays out a run. The tools (``r2s2r tool ...``, in
:mod:`r2s2r.agentic.cli`) are in :mod:`~r2s2r.agentic.segment` (SAM3 masks, crops),
:mod:`~r2s2r.agentic.geometry` (points, the support, fitting meshes to several views),
:mod:`~r2s2r.agentic.objects` (Hunyuan3D generation, assembly),
:mod:`~r2s2r.agentic.check` (quick renders) and :mod:`~r2s2r.agentic.isaac` (settling,
replays); :mod:`r2s2r.agentic.runner` drives the agent through the stages
(``r2s2r agent ...``).
"""
