"""A live web viewer of reconstruction runs, whichever method made them
(``r2s2r viewer outputs/agentic``).

The page is a progress axis over the stages (a stage lights up when done and blinks
while running; stages the run's method leaves out are skipped). A click on a stage opens
what it made: the recording (every camera's frames on one timeline, beside the robot's
model moving along the recorded trajectory, with the cameras and the latest scene in 3D,
and the final replay against the real frames), the chosen frames, the support, the
objects (preview, fit, 3D model), and their physical parameters. The page polls the
server, which reads the runs afresh, so a stage's results show up as soon as they are
written.
"""
