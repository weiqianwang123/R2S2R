# Stage 2: frames, support surface, scene frame

Choose the frames the scene will be reconstructed from, find the surface the objects
stand on, and set up the scene's frame on it. Work in this directory.

1. Look at `../inputs/sheets/*.png` and at single frames until you understand the
   scene: which objects there are, which ones the task involves, what they stand on.
2. Choose 4 to 8 frames (static, with depth) that together show every object on the
   support well: different viewpoints (every camera where it helps; oblique views that
   show objects' sides, not only views from above), objects not hidden by the arm or by
   each other, sharp images. Several moments of one camera help only if it moved.
3. Find the support surface the objects stand on (not the floor, not the robot's own
   mount, unless the objects stand there). Segment it in your chosen frames, fit it
   with `r2s2r tool support`, and check the overlays: the rectangle must lie on that
   surface in every frame, with a small residual and a plausible tilt. Fix bad masks
   (objects or the robot included, another surface) and fit again as needed.
4. List every object on the support that belongs in the simulation (anything the robot
   could touch), each with a short description good for segmenting it (colour,
   material, shape).

Write `output.json`:

```
{
  "frames": ["ext1@0", ...],
  "frame_notes": {"ext1@0": "why this frame"},
  "support": {"file": "support.json", "description": "what the surface is"},
  "objects": [
    {"name": "snake_case_name", "prompt": "text for segmenting it",
     "description": "...", "visible_in": ["ext1@0", ...], "on_support": true}
  ],
  "notes": "anything the next stages should know"
}
```
