# Stage 6: refine against the recording

The scene of stage 4 (`../s4_scene/objects.json`, assembled in `../s4_scene/scene`) has
been settled under physics in {{sim}}: `../s5_settle/scene`, with how far each object
moved in `../s5_settle/scene/settle.json` (and each articulated object's joints:
`joints_moved`; a joint moves as its dynamics say, so a part its friction does not hold
falls, and a spring pulls its part toward its rest; a cloth drapes over what is under
it, in Newton, and how far its furthest point moved). Now compare it with everything the
robot recorded, and make it right. Work in this directory.

Your main tool is `r2s2r tool replay SCENE_DIR --out DIR`: {{sim}} replays the static
period with the robot at each recorded state and renders every camera at every frame.
`DIR/compare/frames/` has, per frame, the real image with the objects' outlines | the
sim render | a blend | the depth residual (black 0, white 5 cm or more, blue unknown),
`DIR/compare/sheet_<camera>.png` a contact sheet per camera, and
`DIR/compare/compare.json` the numbers (median depth residual per frame and per
object). A replay takes up to a minute; `--every N` renders every N-th frame step only.

This stage is about geometry: where the objects and the support are, their orientation,
size and shape. Colours, textures, materials and lighting are out of scope: do not edit
them, and do not let colour differences between the render and the images drive
changes.

1. Replay the settled scene. Look at every camera's sheet and at single frames, the
   frames not used for reconstruction included: they are the fairest test.
2. Find what is wrong: an object off in position, height, orientation or scale; a
   wrong shape (generate it again from another view); a missing or a spurious object;
   the support at the wrong height or too small; objects that moved while settling
   (not resting, or intersecting); an articulated part in the wrong place.
3. Fix it on a copy of the objects file here (start from `../s4_scene/objects.json`,
   paths rewritten). Use `check` to try changes (seconds); then `assemble`, `settle`
   and `replay` the changed scene once to confirm them.
4. Stop when the replay matches every camera's frames well, or after that one
   confirming round: a second one only if it showed a clear geometric error that you
   can fix. If the settled scene already matches well, keep it as it is.

Leave the final scene in `scene/`: assembled, then settled into it
(`r2s2r tool settle <assembled dir> --out scene`). Write `report.md`: what you changed
and why, the numbers before and after, and what is still off.
