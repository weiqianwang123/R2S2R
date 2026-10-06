# Stage 4: assembly, simulation-ready

Turn the objects of stage 3 into a simulation-ready scene. Work in this directory.

1. Copy `../s3_objects/objects.json` to `objects.json` here, with every relative path
   rewritten so that it still points at the same file.
2. Give every object a mass (kg) and a friction coefficient, from what it is: its
   material and its size (the fit's `size_m`). An articulated object's mass is the
   whole object's; check its joints' limits once more (what a real one of its kind
   allows), and give each joint its dynamics, from how a real one of its kind moves
   when let go (see the objects file in `../AGENTS.md`): does its part stay where it is
   left, fall, or spring back, and how quickly. Where a value is a guess, give its
   range too (`<name>_range`, see `../AGENTS.md`), as wide as you are unsure.
3. Check the poses: an object standing on the support must touch it (lowest point at
   about 0 in the support frame), objects must not intersect each other, and every
   object must stand within the support's `extent` (enlarge the extent if not; it is
   only what the cameras saw of the surface).
4. `r2s2r tool assemble objects.json --out scene`, and read its report: the number
   of collision hulls, their volume as a share of the mesh's convex hull
   (`hull_volume_share`: well below 1 keeps hollows such as a mug's opening; near 1
   means solid), the lowest point above the support.
5. `r2s2r tool check scene --frames <the chosen frames> --out check` for a last look.

Write `output.json`:

```
{
  "scene": "scene/scene.json",
  "objects": {"mug": {"mass": 0.3, "friction": 0.6, "why": "..."},
              "box": {"mass": 0.2, "mass_range": [0.15, 0.3], "friction": 0.6,
                      "why": "... and why each range is as wide as it is",
                      "joints": {"hinge": "why its limits and dynamics are what they
                                           are"}}},
  "notes": "..."
}
```
