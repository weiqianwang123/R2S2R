# Workspace rules

You are reconstructing a robot's real scene as a simulation scene, from the robot's own
recording. This directory is one run's workspace: `{{ws}}`. Each stage has its own
directory with a `BRIEF.md`.

## Rules

- Write files only inside this workspace. Never modify anything outside it: not the
  r2s2r or SimFoundry code, not other directories, not the environments (no pip or
  conda installs).
- Use only the data in `inputs/` and what this run produces. Other directories on this
  machine hold other runs and other methods' results: do not read them.
- You may write and run your own Python (keep it in the stage directory). `python` is
  r2s2r's environment: numpy, scipy, cv2, trimesh, mujoco and the `r2s2r` package.
- The GPU is shared: run the tools that use it (`segment`, `generate`, `settle`,
  `replay`) one at a time.
- Look at images whenever it helps (you can open PNG files), and check every result by
  looking, not only by its numbers.
- Learn what is in the scene only from the recording in `inputs/`: nothing else on
  this machine describes it (the scene may be real or simulated). Assume nothing beyond
  what the recording shows.

## The recording

- Robot: {{embodiment}}. The task it was recorded doing: "{{instruction}}".
- Cameras:
{{cameras}}
- Frames are named `<camera>@<step>`. `inputs/frames.json` lists every frame: whether
  it is in the static period (steps {{static_start}} to {{static_last}}: the objects
  have not been touched yet), whether it has metric depth, its image, `K` and
  `T_base_cam`. Only static frames with depth can be reconstructed from:
  `inputs/sheets/<camera>.png` shows them. {{frame_counts}}
- Depth: {{depth_note}} The tools cut the robot out of the depth with its known model.

## Conventions

- Metres; radians unless a name ends in `_deg`.
- `T_a_b` is a 4x4 transform from frame b coordinates to frame a. Everything is in the
  robot base frame (z up).
- Cameras follow OpenCV (x right, y down, z forward); `T_base_cam` is the camera pose.
- The support frame (`T_base_support`) has z along the support's normal (towards the
  cameras) and its origin on the support surface.

## Tools

`r2s2r tool <name> --help` shows every option. Each tool prints a JSON summary. Paths
may be relative to the directory you run them in.

- `r2s2r tool frames [--camera ROLE] [--all]`: list the frames.
- `r2s2r tool segment --frames F [F ...] --text "white mug" [--text ...] --out DIR`:
  SAM3 finds every instance of each text in each frame: masks
  `DIR/<name>/<frame>_<k>.png` (best score first) and an overlay numbering them. With
  `--point x,y[,label]` or `--box x0,y0,x1,y1` (one frame) instead: the one thing they
  pick out. `--name` names the output directory.
- `r2s2r tool points --frames F [F ...] [--mask F=MASK ...] [--support JSON] --out PLY`:
  fused robot-free points in the base frame, and where they lie.
- `r2s2r tool support --mask F=MASK [--mask ...] --out support.json`: the plane
  through the masked depth of several frames, its outline as a rectangle, and the
  support frame at its centre; an overlay per frame (magenta: the rectangle, a 10 cm
  grid; red: support x, green: support y).
- `r2s2r tool crop --frame F --mask MASK --out obj.png`: the object cut out on a
  transparent background, for generation.
- `r2s2r tool generate NAME=obj.png [NAME=obj.png ...] --out DIR`: Hunyuan3D-2.1 makes
  a textured mesh from each single image: `DIR/NAME/mesh.glb`, `mesh.obj`,
  `preview.png`. Generation takes a few minutes per object; objects in one call share
  the model loading. Its meshes are y-up.
- `r2s2r tool fit MESH --support JSON --mask F=MASK [--mask ...] --out DIR`: scales and
  places a mesh so that it matches the object's masks and depth in every given frame
  (mean silhouette IoU): `DIR/fit.json` (`scale`, `T_base_obj`, per frame IoU) and an
  overlay per frame (green: the mask, red: the fitted mesh). Options: `--scale`,
  `--yaw` (start values), `--no-rest` (for an object not standing on the support),
  `--up`.
- `r2s2r tool check (OBJECTS_JSON | SCENE_DIR) --frames F [F ...] --out DIR`: renders
  the scene's objects into frames in seconds: real image with outlines | render |
  blend | depth residual, and numbers. `--joint OBJECT:JOINT=VALUE [...]` renders an
  articulated object with those joints moved, to see where its parts go.
- `r2s2r tool assemble OBJECTS_JSON --out SCENE_DIR`: the simulation-ready scene
  (`scene.json`): collision parts (CoACD), inertia, flat bases for resting objects.
- `r2s2r tool settle SCENE_DIR --out SCENE_DIR2`: the objects come to rest under
  gravity in Isaac Lab (robot held still); how far each moved.
- `r2s2r tool replay SCENE_DIR --out DIR`: Isaac Lab replays the static period (the
  robot at every recorded state, the objects held) and renders every camera at every
  frame; `DIR/compare/` compares each render with the real frame.

## The objects file

The scene as you build it, before assembly (JSON):

```
{
  "support": "path/to/support.json"  or  {"T_base_support": 4x4, "extent": [x, y]},
  "reference_frame": "ext1@0",           (optional) the robot state the scene keeps
  "objects": [
    {"name": "mug", "category": "mug", "mesh": "path/to/mesh.glb",
     "scale": 0.052,                     or [sx, sy, sz]: applied to the mesh file first
     "T_base_obj": 4x4,                  then this rigid pose, in the base frame
     "up": "y",                          the mesh file's up axis (z if left out)
     "mass": 0.3, "friction": 0.6}       kg; needed from assembly on
  ]
}
```

An articulated object (a box and its lid, a book and its cover, a cabinet and its door
or drawer) also lists the parts that move and the joints that move them; its `mesh` is
then the part that does not, and `mass` is the whole object's:

```
     "parts": [{"name": "lid", "mesh": "lid.obj"}],      same coordinates as "mesh"
     "joints": [{"name": "hinge", "type": "revolute",    or "prismatic" (a drawer)
                 "parent": "base", "child": "lid",       "base": the object's "mesh"
                 "origin": [x, y, z],                    a point on the axis, and
                 "axis": [x, y, z],                      its direction: object frame, m
                 "limits": [lower, upper],               rad or m
                 "position": 0.0}]                       where the joint is as recorded
```

Everything is as recorded: the parts' meshes where the parts were, the joints'
`origin` and `axis` in the object's frame (the frame `T_base_obj` places, in metres,
`scale` applied), each joint at its recorded `position` within its `limits`. The
direction of `axis` and the sign of `position` follow the right-hand rule.

Relative paths are relative to the objects file. `fit.json` gives `scale`,
`T_base_obj` and `up` in this form. The support's `extent` is the size of the simulated
table top, centred on the support frame: every object must stand within it.
