# Stage 3: objects

Make every object of stage 2 a mesh, scaled and placed in the base frame so that it
matches every chosen frame. Work in this directory.

Inputs: `../s2_frames/output.json` (the chosen frames, the objects, notes) and the
support file it names (`support.file`, in `../s2_frames/`).

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
`"support": "../s2_frames/<that support file>"`, run
`r2s2r tool check objects.json --frames <the chosen frames> --out check`, and look at
every panel: each object's outline must sit on it in every frame. Fix what is off.

Articulated objects: an object with parts that move against each other (a lid, a
cover, a door, a drawer) is one object with `parts` and `joints` (see the objects file
in `../AGENTS.md`). The recording may never show them move: judge from what the object
is and what you see (hinges, seams, gaps, handles) where the joint is and how far it
goes, the way you would expect such an object to work. Split the part from the fitted
mesh (or build it) in the mesh's own coordinates, as it was recorded; give each joint
its axis, origin, limits and recorded position. Then `check` it at the recorded
positions and with `--joint` at the ends of its limits: the part must turn or slide
the way the real one would, without passing through the rest of the object. Only
model joints you believe the object has; a rigid object stays rigid. Its joints'
dynamics come in stage 4.

Cloths: a towel, a napkin or a cloth lying on the scene is a `cloth` (see the objects
file in `../AGENTS.md`), not a generated mesh: build its surface from the recorded depth
(`points` with its masks gives the cloth's points in the base frame; mesh them into an
even, open triangle surface that follows them, with no holes where the cloth is
whole), as it lies. Place it with `T_base_obj` as you like (the identity and the mesh in
the base frame is fine), give it its `cloth` material (a first estimate; stage 4 looks
again) and `check` it like any other object.

Add objects stage 2 missed; drop anything that is not a separate object. Finish with
`notes.md`: per object, the view it was generated from, the frames it was fitted to,
its final IoUs, its joints and why, how a cloth was built, and any doubts.
