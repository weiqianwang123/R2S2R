"""The fixed method: SimFoundry reconstructs the scene from one frame
(:mod:`~r2s2r.pipeline.fixed.simfoundry`), then several views' depth check which objects
exist (:mod:`~r2s2r.pipeline.fixed.existence`), their turn about the support normal
(:mod:`~r2s2r.pipeline.fixed.orientation`, with a Codex VLM on ties,
:mod:`~r2s2r.pipeline.fixed.vlm`) and where they stand
(:mod:`~r2s2r.pipeline.fixed.refine`).
"""
