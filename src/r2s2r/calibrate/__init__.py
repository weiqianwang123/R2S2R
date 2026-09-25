"""Calibrating a reconstructed scene against its video with a coding agent.

Measurement and verification are fixed code (:mod:`r2s2r.calibrate.tools`); an agent
(:mod:`r2s2r.calibrate.agent`) proposes changes where judgement is needed, and a change
is accepted only when the fixed verifiers, rerun outside the agent's reach, say it
explains the video better (:mod:`r2s2r.calibrate.articulation`).
"""
