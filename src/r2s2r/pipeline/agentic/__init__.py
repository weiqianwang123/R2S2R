"""The agentic method: a coding agent (Codex ``gpt-6-astra``, "astra", or Claude Code)
builds the scene from the robot's own recording, with the tools (:mod:`r2s2r.tools`).

2. the agent picks several frames (4-8), the support surface and the scene's frame on
   it;
3. the agent decomposes the scene: every object segmented in those frames, generated
   from its best single view (Hunyuan3D-2.1), and fitted to all of them (a cloth's
   surface built from the depth);
4. the agent assembles the objects into a simulation-ready scene (mass, friction, joint
   dynamics, a cloth's material, collision parts);
5. (shared by every method) the scene settles under physics in the run's simulator
   (Isaac Lab or MuJoCo), its cloths then in Newton;
6. the agent refines the result against every recorded view, replaying the robot's
   recording in the same simulator (geometry only; the stage can be left out).

:mod:`~r2s2r.pipeline.agentic.method` drives the agent through its stages
(``r2s2r run CAPTURE --method agentic ...``); ``briefs/`` holds what it is told.
"""
