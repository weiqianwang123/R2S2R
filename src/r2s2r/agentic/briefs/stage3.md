# Stage 3: objects

Make every object of stage 2 a mesh, scaled and placed in the base frame so that it
matches every chosen frame. Work in this directory.

Inputs: `../s2_frames/output.json` (the chosen frames, the objects, notes) and
`../s2_frames/support.json`.

For each object:

1. Segment it in every chosen frame (`segment`), and pick the right instance per frame
   from the overlays (or use point / box prompts). A mask must cover all that is
   visible of the object and nothing else: not its neighbours, not the robot. Keep the
   masks you settle on.
2. Choose the one best view to generate it from: the object whole and unoccluded, large
   in the image, its 3D shape visible (an oblique view of its sides beats a view from
   straight above). `crop` it and look at the crop.
3. `generate` the meshes (all objects in one call). Look at every `preview.png`: it must
   be the object, whole, alone, with a plausible shape. If not, improve the crop
   (another view, a cleaner mask) and generate again.
4. `fit` each mesh to the object's masks in every chosen frame that shows it. Look at
   the overlays and numbers (IoU per frame; mask pixels the mesh misses; mesh pixels
   outside the mask). Iterate where it is off: other start values for yaw or scale, a
   better mask, leaving out a frame where the object is mostly hidden, or your own
   optimisation.

Then write `objects.json` (the objects file of `../AGENTS.md`) with every object and
`"support": "../s2_frames/support.json"`, run
`r2s2r tool check objects.json --frames <the chosen frames> --out check`, and look at
every panel: each object's outline must sit on it in every frame. Fix what is off.

Add objects stage 2 missed; drop anything that is not a separate object. Finish with
`notes.md`: per object, the view it was generated from, the frames it was fitted to,
its final IoUs, and any doubts.
