# R2S2R: Real-to-Sim-to-Real

R2S2R turns recordings from a robot's own cameras into a simulation of its workspace,
and deploys what is learned there back to the robot. The input is what the robot
records while it runs: teleoperated demonstrations, play data, or just a few views
from its exterior camera, its wrist camera, or both. Every camera is calibrated, every
pose is in the robot's base frame, and every frame carries the robot's joint state. The
robot's model is known. The output is an Isaac Lab scene of rigid objects on their
support, in the robot's base frame. MuJoCo can stand in for the real world: it records
such data with ground truth and runs the deployed policy.

The module is shared between the code-for-gen project and
[PhysCoder](https://github.com/Jaraxxus-Me/physcoder). The first platform is the
DROID Franka (Panda + Robotiq 2F-85, two ZED 2 exterior cameras, a ZED Mini wrist
camera).

## Status

| Piece | State |
|---|---|
| Captures from DROID episodes, from MuJoCo, or any `capture.json` | done |
| One exterior camera, the wrist camera, or both; one to many images each | done: MuJoCo and DROID (see [Results](#results) for which setups were run) |
| Robot removed from depth (its model rendered at the recorded joints) | done: Panda + Franka Hand, DROID's Panda + Robotiq |
| SimFoundry backend (Codex as the VLM), stereo or RGB-D input | done |
| Refinement: objects not there dropped, orientation check, position, support outline | done |
| Isaac Lab scene, program policy, deployment to MuJoCo | done |
| Replay of the recording in Isaac Lab, compared with every view | done (geometry; physics mode runs, grasps not yet aligned) |
| Agentic reconstruction (astra + tools, branch `agentic`), live viewer | done: MuJoCo and DROID, one exterior + wrist camera |
| Articulated objects | removed for now (see [Roadmap](#roadmap)) |
| Robot-side alignment (controller and dynamics) | **not started, see below** |

## Pipeline

```
Capture ─► robot removed ─► reconstruction ─► sim-ready ─► refinement ─► SceneSpec ─► Isaac Lab ─► deploy
(robot's    from depth       (SimFoundry,      assets       (ghosts,      (robot base  + program   (MuJoCo as real;
cameras,    (model at its    one frame)                     orientation,  frame)       policy      real robot)
joints)     joints)                                         position,
                                                            outline)
```

```
src/r2s2r/
  structs.py        Capture (what was recorded), SceneSpec (what was reconstructed),
                    DepthView; plain data saved as JSON
  transforms.py     frames, quaternions, projection / backprojection
  assets.py         sim-ready URDFs (baked scales, flat resting bases), URDF readers
  mjrender.py       MuJoCo rendering from calibrated pinhole cameras
  vlm.py            a VLM through the local Codex CLI
  cli.py            the r2s2r command
  io/               recordings -> Capture: droid.py, rgbd.py (depth PNGs, depth views),
                    stereo.py (FoundationStereo depth for a stereo capture)
  reconstruct/      Capture -> SceneSpec: base.py (pluggable backends), simfoundry.py,
                    render.py (scene renders), existence.py, orientation.py, refine.py
  robots/           franka.py (Panda FK / IK in numpy), mujoco_models.py (robot models),
                    mask.py (robot removal)
  policy/           robot.py (RobotInterface + motion primitives), pick.py, scoring.py
  sim/isaaclab/     scene.py (SceneSpec -> Isaac Lab scene), robot.py, rollout.py,
                    replay.py (recording replayed, scene settled)
  sim/compare.py    replay renders against the real frames
  agentic/          the agent's tools, its runner and briefs (see below)
  viewer/           live web viewer of agentic runs
  real/mujoco/      MuJoCo as the real world: world.py, capture.py, deploy.py
  video.py          rollout videos and contact sheets
scripts/
  setup/            install.sh, link_simfoundry_resources.sh, fetch_mujoco_assets.sh
  isaaclab/         entry points that start the Omniverse app: render_overlay.py, run_pick.py,
                    replay.py, settle.py
  agentic/          jobs run in SimFoundry's conda envs (SAM3, Hunyuan3D, CoACD)
  mujoco_demo.sh    the whole MuJoCo loop
third_party/SimFoundry   git submodule of our fork (branch r2s2r); SimFoundry changes go there
```

Conventions: `T_a_b` maps frame-b points into frame a, camera frames are OpenCV
(+z forward, +y down), quaternions are `(w, x, y, z)`, and every pose is in the robot
base frame. Nothing in the pipeline depends on the kind of object.

### Capture: the robot's cameras, poses in its base frame

A capture is a set of calibrated cameras whose poses are in the robot base frame. The
setup assumed throughout is one exterior camera and the wrist camera, or either alone;
more cameras work too (DROID episodes have two exterior cameras; `droid-capture` takes
ext2, the exterior view used, and the wrist unless `--roles` says otherwise). Each
camera contributes one image or many; the recording can be a demonstration, play, or a
few still views. A frame needs its image, its pose `T_base_cam`, the robot's state
(`joint_positions`, `gripper_position`), and depth: metric depth (`depth_image`, uint16
PNG in millimetres, from an RGB-D camera) or a stereo partner (`right_image`, camera
`stereo_baseline`; FoundationStereo computes the depth, as for DROID's ZEDs). A wrist
camera's pose comes from the arm's kinematics at each frame; a static camera's pose can
be given once, on the camera.

`r2s2r droid-capture` and `r2s2r mujoco-capture` write `capture.json`
(`r2s2r.structs.Capture`); other data can be brought in by writing the same file:

```json
{"name": "my_scene", "source": "my_rig", "embodiment": "franka_panda",
 "instruction": "pick up the red block", "static_steps": [0, 100],
 "cameras": {"side_cam": {"serial": "side_cam", "role": "ext1", "width": 1280,
             "height": 720, "K": [[...]], "stereo_baseline": null, "is_static": true,
             "T_base_cam": [[...]]},
             "wrist_cam": {"serial": "wrist_cam", "role": "wrist", "width": 1280,
             "height": 720, "K": [[...]], "stereo_baseline": null, "is_static": false}},
 "frames": [{"step": 0, "camera": "side_cam", "left_image": "rgb/side_0000.png",
             "right_image": null, "depth_image": "depth/side_0000.png",
             "joint_positions": [...], "gripper_position": 0.0},
            {"step": 0, "camera": "wrist_cam", "left_image": "rgb/wrist_0000.png",
             "right_image": null, "depth_image": "depth/wrist_0000.png",
             "T_base_cam": [[...]], "joint_positions": [...], "gripper_position": 0.0}]}
```

`static_steps` is the period in which the objects stay where they are (before the robot
touches anything). `Capture.select_frames` shares a frame budget evenly between cameras
and spreads each camera's share over that period; reconstruction and refinement both
use it (`--cameras` limits them to some cameras, by role or serial).

### Robot removal

A robot standing on the table is geometry above the support plane: frame selection
counts an arm that leaves the image as a clipped object, and refinement clusters it.
`robots/mask.py` renders the robot's MuJoCo model (mujoco_menagerie) at the recorded
joint state from each calibrated camera, with the exact pinhole intrinsics, and sets
the depth to zero (no measurement) wherever the robot is the visible surface
(silhouette grown by 0.6 % of the image diagonal; surfaces over 3 cm in front of the
robot are kept). In the MuJoCo world this matches the simulator's own segmentation at
IoU ≥ 0.98; on DROID the Robotiq's mounting yaw (+90° on the flange) was fitted to the
fingers in the wrist-camera images. `--no-robot-mask` turns it off.

### Reconstruction: the SimFoundry backend

[SimFoundry](third_party/SimFoundry) runs in its own conda envs; r2s2r drives its
orchestrator with `PYTHONPATH` pointing at the submodule. The backend

1. writes up to 12 candidate frames from any cameras where SimFoundry reads them: a
   stereo frame as stage 1a would (`s1_zed/image_<i>_{l,r}.png` with its own
   `image_<i>_intrinsic.txt`), an RGB-D frame's depth as FoundationStereo would
   (`s2_fs/`, at its 0.5 scale). Without stereo frames stage 2 is skipped;
2. removes the robot from every candidate's depth (stereo frames right after stage 2,
   from a kept raw copy);
3. runs stages 3–8 and 10–12 (stage 9 makes objects articulated, which r2s2r leaves
   out for now);
4. anchors SimFoundry's support-plane world frame to the robot with the chosen frame's
   `T_base_cam`, and writes each object's URDF sim-ready (below);
5. if SimFoundry found no object at all, reruns stages 3–12 pinned to other frames
   (up to 3; the camera with the fewest empty frames first, then the best frame
   score) until one gives objects; stage 2's depth is kept. Stage 3 fits the support
   surface in one frame, choosing the largest surface it detects, and from some
   viewpoints that is not the one the objects stand on: on DROID, the robot's own
   mounting table, whose clamps also pass for objects in the frame scores. Another
   camera, or another moment, sees it differently.

SimFoundry reconstructs from the one frame its selection picks, so most of the video
goes unused; refinement uses the rest. By default (`--frame-selection hybrid`) stage 3
scores every candidate geometrically (the largest table-like surface SAM3 finds, and the
geometry standing on it), and a VLM picks among the best four; the support surface is
the largest one in the chosen frame. With `--frame-selection codex` (on the fork),
Codex at xhigh reasoning effort sees every candidate from every camera and the capture's
instruction. It ranks the usable frames (preferring views that show each object's sides:
a mesh generated from a near-overhead view comes out too wide or too tall), and names the
support surface the task's objects stand on, with its box in each ranked frame. The first
frame in that ranking whose support SAM3 can segment there, with a plane stage 3 accepts,
is used, and stage 3 segments that surface instead of the largest. If Codex fails,
selection falls back to hybrid.

Every VLM call goes through the local Codex CLI (`gpt-6-astra`, reasoning `medium`):
the fork's `Gemini.__new__` returns a `CodexVLM` when `SIMFOUNDRY_VLM_BACKEND=codex`,
which r2s2r sets. Text/vision requests take about 7 s, image edits about 75 s. Runs are
not bit-reproducible (no sampling controls); `--vlm-backend gemini` restores upstream.

### Simulation-ready assets

`assets.make_sim_ready` writes one `<model>_r2s2r.urdf` per object:

- `<mesh scale>` is baked into the meshes (Isaac Sim 5.1's importer left a 1 m collider
  beside each scaled one);
- an object resting on the support gets a flat 5 mm base cut from its own
  cross-section. An object seen standing still must stand in simulation too, but
  generated meshes round off the edges it stands on (see [Results](#results)).

### Refinement

`r2s2r refine` fuses metric depth from many views (capture depth, or SimFoundry's
stage-2 depth), with the robot removed. The points above the support are clustered,
and each cluster is matched to at most one object: the one whose surface its points lie
closest to on average, so a flat object does not take a tall one's points. Then it

1. **drops objects that are not there** (`reconstruct/existence.py`). SimFoundry finds
   objects one at a time in an image it edits (each one found is erased and the gap
   inpainted), and a smudge left where one was erased can be found as another object.
   An object is dropped when no cluster is matched to it and, rendered with the rest of
   the scene, most of its visible surface lies more than 2 cm in front of the measured
   depth: the cameras see through it. Whatever rested on it settles on what is beneath.
   Objects too flat for the depth to rule on, or seen by no view, stay, and so does one
   standing over measured points no object explains: a real object whose generated
   shape is off (flagged in the report) is better kept than lost (`--keep-unseen`
   keeps all);
2. **checks every object's turn about the support normal** (`reconstruct/orientation.py`).
   Each candidate (0, 90, 180, 270°) is rendered into every view and scored by its depth
   residual; turns the geometry rules out are dropped. The remaining turns go to the VLM,
   which sees the photo and a render of each candidate and looks for features that fix
   the orientation, such as handles, openings or printed text. The backend's pose is
   kept unless the VLM is confident. `--no-vlm` keeps ties as they are;
3. **re-registers every object** to the fused points by its top-down footprint (yaw and
   in-plane shift), only when the fit clearly improves;
4. **fits the support surface's outline** (a rectangle in `SceneSpec.support_extent`).

### Simulation and deployment

`sim/isaaclab/scene.py` places the robot at the origin, the support, every object, and
each static camera with its real intrinsics. Two Isaac pitfalls are handled there:
`PinholeCameraCfg.from_intrinsic_matrix` needs an explicit focal length (else the render
is magnified about 1.3×), and Omniverse ignores principal-point offsets, so each camera
renders a larger image centred on (cx, cy) that is cropped back.

Policies are programs written against `policy.robot.RobotInterface` (hold joint targets
and a gripper command for one control period). Cartesian interpolation and IK happen
above it, in numpy (`robots/franka.py`, checked against MuJoCo's kinematics), so Isaac
Lab, MuJoCo and the real arm receive the same command stream and only the plant differs.
The test policy (`policy/pick.py`) plans a top-down grasp across the target's narrowest
side from the SceneSpec mesh, then approaches, closes and lifts 15 cm.

### Replay: a scene against every recorded view

Captures keep the robot's state at every control step (`trajectory.npz`), not only at
the frames. `scripts/isaaclab/replay.py` replays it in a reconstructed scene in Isaac
Lab and renders every camera at every frame: a static camera where it is calibrated,
the wrist camera at the frame's recorded pose. In `geometry` mode (the default) the
robot is set to each recorded state of the static period and the objects are held
where the scene puts them; `physics` mode drives the arm along the whole recording and
simulates the objects. `sim/compare.py` then puts each render next to its real frame
(real image with the objects' outlines | render | blend | depth residual), with the
median depth residual per frame and per object, and a contact sheet per camera.
`r2s2r stereo-depth` gives a stereo capture (DROID) FoundationStereo depth to compare
with; `r2s2r compare-replay` redoes the comparison.

### Agentic reconstruction (branch `agentic`)

A coding agent, astra (Codex `gpt-6-astra` at xhigh effort, `r2s2r agent`), builds the
scene with tools (`r2s2r tool ...`, `src/r2s2r/agentic/`) instead of a fixed pipeline:

| Stage | Who | What |
|---|---|---|
| 1 | — | the capture, as above |
| 2 | astra | 4-8 frames to reconstruct from; the support surface (segmented, plane fitted across the frames) and the scene frame on it |
| 3 | astra | every object segmented in those frames; one best view per object cut out and generated by Hunyuan3D-2.1; the mesh scaled and placed to match all frames |
| 4 | astra | masses and friction; assembly into a simulation-ready scene (CoACD collision parts, inertia, flat bases) |
| 5 | — | objects settled under gravity in Isaac Lab |
| 6 | astra | the scene compared with every recorded view by replaying the recording in Isaac Lab, and fixed until they agree |

Each stage runs in its own directory of a workspace (`outputs/.../WS/s2_frames`, ...)
with a brief (`agentic/briefs/`) and the workspace's rules (`AGENTS.md`: the
recording, conventions, tools). The agent sees a copy of the capture without its
metadata, since a MuJoCo capture's metadata holds the true scene. It runs unsandboxed,
since the tools need the GPU. The runner checks every stage's output and resumes the
session with what is missing. It also fingerprints the repository and the SimFoundry
submodule before and after each stage, and stops if anything outside the workspace
changed.

| Tool | Does |
|---|---|
| `frames` | lists the frames (static, with depth) |
| `segment` | SAM3 masks from text (every instance) or from points / a box |
| `points` | fused robot-free points of masked frames (PLY), and where they lie |
| `support` | plane through masked depth of several frames (RANSAC), its outline, the support frame; overlays with a 10 cm grid |
| `crop` | an object cut out on a transparent background, for generation |
| `generate` | Hunyuan3D-2.1 mesh from one image (textured GLB + OBJ + a four-view preview) |
| `fit` | scale, yaw and position of a mesh maximising its silhouette IoU with the object's masks in every frame, from a footprint match; overlays |
| `check` | the objects rendered into frames (MuJoCo, seconds) |
| `assemble` | objects file → simulation-ready scene |
| `settle` | objects come to rest in Isaac Lab (`scripts/isaaclab/settle.py`) |
| `replay` | the geometry replay above, summarised |

SAM3 and CoACD run in the `simfoundry` conda env and Hunyuan3D in `hunyuan`
(`scripts/agentic/*_job.py`), from the SimFoundry submodule. Fitting renders one
unscaled mesh, and scales it by moving the camera instead, so a scale change needs no
recompile. After stage 6 the final scene is replayed once more (`s6_refine/final_replay`),
so that its replay against the real frames is kept whatever the agent did.

`r2s2r viewer outputs/agentic` serves a live page of every run under that directory
(`src/r2s2r/viewer/`; port 8765, reachable from other machines). The page is a progress
axis over the stages: a finished stage lights up, and the running one blinks. Clicking a
stage opens what it made:

- **Capture**: every camera's frames on one timeline, beside the robot's model moving
  along the recorded trajectory, in sync. The model is the one used to cut the robot
  out of depth, so it is the Robotiq on DROID. The 3D view also shows the cameras and
  the latest scene. The *replay* switch puts the final replay's panels (real | sim |
  blend | depth residual) in place of the raw frames.
- **Frames**: the frames chosen in stage 2.
- **Support**: the support fit, with overlays.
- **Objects**: each object's preview, its fit in every frame, and its 3D model.
- **Physics**: each object's mass and friction, and the agent's reasoning for them.

It polls the workspaces, so each stage shows up once it has been written.

Results (2026-09-26):

| Run | Stages 2 / 3 / 4 / 5 / 6 (min) | Final replay, median depth residual |
|---|---|---|
| DROID IRIS, ext2 + wrist (FoundationStereo depth) | 5.5 / 21 / 6 / 0.2 / 13 | ext2 0.2 cm, wrist 0.4 cm; mug 0.5 / 1.1 cm, marker 0.3 / 0.2 cm (ext2 / wrist) |
| DROID IRIS, wrist only | 7 / 21 / 9.5 / 0.2 / 18 | wrist; on the held-out ext2: mug 0.75 cm, marker 0.3 cm |
| MuJoCo, ext1 + wrist | ~6 / 28 / 6 / 0.1 / 16 | ext1 0.1 cm, wrist 0.0 cm; objects 0.1–0.4 cm |
| MuJoCo, wrist only | 4 / 22 / 7 / 0.1 / 12 | wrist; objects within 0.3 cm of the ground truth |

Against the MuJoCo ground truth, the object centres are 0.2, 0.0 and 0.2 cm off (crayon
box, mug, figurine). The SimFoundry pipeline on the same capture was 0.4–1.7 cm off. The
sizes are within 4 mm (figurine height 8.1 cm vs 8.0 true; SimFoundry had 4.5–5.8). On
DROID, stage 2 picked the small white table the objects stand on, not the robot's mounting
table. The wrist camera alone is enough in both scenes:

- MuJoCo: centres 0.3 / 0.1 / 0.1 cm off, sizes within 4.6 mm.
- DROID, replayed on ext2, which the wrist-only run never saw: the mug is off by
  0.75 cm (median depth). SimFoundry's wrist-only scene was off by 7.45 cm there, with
  a closed can on a pedestal where the mug is.

Stage 6 gains little once stage 3 has done well: millimetres, except the support's extent,
which it extended where wrist views had not covered the table. When it had no scope, it
also spent time on colours; its brief now keeps it to geometry and one confirming
round. Every run took longer than the 30 minutes aimed for, mostly in stage 3 (segment,
generate, fit, repeat). Most of the gain over SimFoundry is probably the multi-view
`fit` (silhouette IoU over 4-8 views, instead of fitting a point cloud from one frame).
Running the same tools in a fixed script, with no agent, would show how much the agent
itself adds.

## Install

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/weiqianwang123/R2S2R.git
cd R2S2R
uv venv --python 3.11 .venv && uv pip install -e ".[develop]"   # core
bash scripts/setup/install.sh                # + torch 2.7 (cu128), Isaac Sim 5.1, Isaac Lab v2.3.2
bash scripts/setup/fetch_mujoco_assets.sh    # MuJoCo robots and scanned objects (~50 MB)
```

SimFoundry needs its own install (conda envs, model repos, checkpoints); follow its
README once, then link its git-ignored parts into the submodule:

```bash
bash scripts/setup/link_simfoundry_resources.sh ~/SimFoundry
```

On first use SimFoundry downloads Hunyuan3D-2.1 (~14 GB), Depth Anything 3 (~7 GB),
Prior-DA and bria-rmbg (~2 GB), and SAM3 (~3.5 GB) unless it is already cached. The
Codex CLI of the ChatGPT desktop app (`/usr/lib/chatgpt/resources/codex`) is used when
present; older CLIs reject `gpt-6-astra`. MuJoCo is pinned below 3.4, whose
`websockets>=13` conflicts with Isaac Sim 5.1.

## Quickstart: a DROID episode

```bash
# Improved calibration + language annotations (~175 MB)
mkdir -p data/droid/calib && for f in intrinsics.json cam2base_extrinsic_superset.json \
    episode_id_to_path.json droid_language_annotations.json; do
  curl -L -o data/droid/calib/$f https://huggingface.co/KarlP/droid/resolve/main/$f; done

# One episode, without trajectory_im128.h5 (~30 MB)
E=gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-03-07/Tue_Mar__7_16_19_59_2023
D=data/droid/episodes/IRIS+ef107c48+2023-03-07-16h-19m-59s
mkdir -p $D/recordings && gsutil -m cp $E/metadata_*.json $E/trajectory.h5 $D/ \
  && gsutil -m cp -r $E/recordings/MP4 $E/recordings/SVO $D/recordings/

r2s2r droid-capture $D --calib data/droid/calib --out outputs/iris/capture
r2s2r reconstruct outputs/iris/capture --workdir outputs/iris/simfoundry \
    --cameras ext2                                   # or: wrist, or: ext2 wrist
r2s2r refine outputs/iris/simfoundry/<scene>/scene --capture outputs/iris/capture \
    --views outputs/iris/simfoundry --out outputs/iris/scene
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/render_overlay.py outputs/iris/scene \
    outputs/iris/capture --out outputs/iris/overlay --headless
```

## Quickstart: MuJoCo as real

A Franka Panda with the Franka Hand stands on a table with Google Scanned Objects (a
crayon box to pick, a mug, a figurine), watched by a RealSense-like RGB-D camera in
front of the table (1280×720, f = 910 px) and a wrist camera (1280×720, f = 800 px);
depth has no return nearer than 0.1 m or beyond 10 m. While recording, the arm points
the wrist camera at the table from three directions. The pipeline sees only what a
real rig would record (images, depth, calibration, joint states); ground truth goes
into `capture.metadata` for scoring.

```bash
CAMERAS="ext1 wrist" bash scripts/mujoco_demo.sh outputs/mujoco_demo   # all of:

r2s2r mujoco-capture --out outputs/mujoco_demo/capture --cameras ext1 wrist
r2s2r reconstruct outputs/mujoco_demo/capture --workdir outputs/mujoco_demo/simfoundry
r2s2r refine outputs/mujoco_demo/simfoundry/mujoco_pick/scene \
    --capture outputs/mujoco_demo/capture --capture-depth --out outputs/mujoco_demo/scene_refined
r2s2r mujoco-eval outputs/mujoco_demo/scene_refined --capture outputs/mujoco_demo/capture
T=$(r2s2r mujoco-eval outputs/mujoco_demo/scene_refined \
    --capture outputs/mujoco_demo/capture --match-target)
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/run_pick.py outputs/mujoco_demo/scene_refined \
    --target $T --out outputs/mujoco_demo/isaac --headless --video-camera ""
r2s2r mujoco-deploy outputs/mujoco_demo/scene_refined --capture outputs/mujoco_demo/capture \
    --target $T --out outputs/mujoco_demo/deploy
r2s2r mujoco-deploy --oracle --capture outputs/mujoco_demo/capture \
    --out outputs/mujoco_demo/deploy_oracle          # ground-truth baseline
```

Which reconstructed object is "the crayon box" depends on the VLM's naming ("crayon
box", "yellow box"). Grounding the task in the scene's object list is the policy
writer's job (in code-for-gen, an agent); in this harness the ground truth stands in for
it (`--match-target`), and the policy only sees the reconstruction.

## Quickstart: agentic reconstruction

```bash
# The capture as before; stereo captures need depth for the tools (FoundationStereo)
r2s2r stereo-depth outputs/iris/capture --out outputs/iris/capture_fs
r2s2r agent outputs/iris/capture_fs --out outputs/agentic/iris [--cameras wrist] \
    [--stages 2 3 4 5]                                # leave out stage 6
r2s2r viewer outputs/agentic                          # http://localhost:8765
```

A workspace can be resumed: `r2s2r agent --out WORKSPACE` skips the stages whose output
is already valid (`--force` reruns them). The agent is Codex (`gpt-6-astra`, xhigh),
run through the ChatGPT desktop app's CLI.

## Results

One exterior camera throughout (ext1 in MuJoCo, whose world has one; ext2 on DROID, the
exterior view used in practice), with SimFoundry's default frame selection (hybrid) and
with `--frame-selection codex`. Codex runs marked "first prompt" predate the prompt's
preference for views that show object sides, and the check that the support can be
segmented in the chosen frame (2026-09-26).

**MuJoCo as real.** Error is the distance between the centres of the reconstructed and
true objects' boxes, in the base frame. Isaac and MuJoCo are how far the same program
lifted the crayon box, with the same commands; with the ground-truth objects it lifts
14.9 cm. Every scene has the three objects and nothing else.

| Cameras | Frame selection | Frame used | Error: crayon box / mug / figurine | Isaac | MuJoCo |
|---|---|---|---|---|---|
| ext1 | hybrid | ext1 | 0.2 / 0.3 / 1.1 cm | 14.8 cm | 14.9 cm |
| ext1 | codex (first prompt) | ext1 | 0.6 / 0.6 / 1.3 cm | 14.8 cm | 14.9 cm |
| wrist | hybrid | wrist | 1.0 / 0.3 / 1.1 cm | 14.8 cm | 14.7 cm |
| ext1 + wrist | hybrid | wrist | 0.6 / 0.4 / 1.7 cm | 14.8 cm | 14.9 cm |
| ext1 + wrist | codex (first prompt) | wrist | 0.5 / 0.4 / 1.1 cm | 14.8 cm | 14.9 cm |

SimFoundry takes about 13 minutes per scene, mostly Codex image edits and Hunyuan.

**DROID (IRIS episode, "place the marker in the red mug").** No ground truth. Each scene
is rendered in Isaac Lab from the real exterior cameras and laid over their images, and
checked against the depth of ext1, which no run below used: "seen through" is the share
of the mug's visible surface that ext1's depth lies more than 2 cm behind (the marker,
flat and mostly behind the mug from ext1, scores high in every scene).

| Cameras | Frame selection | Frame used | Objects | Mug seen through | Table outline |
|---|---|---|---|---|---|
| ext2 | hybrid | ext2 | marker, red mug | 0.05 | 0.37 × 0.43 m |
| ext2 | codex (first prompt) | ext2 | marker, red mug | 0.13 | 0.37 × 0.43 m |
| wrist | hybrid | wrist | marker, mug 14.8 × 13.4 cm | 0.70 | 0.40 × 0.47 m |
| wrist | codex | wrist, 22° off vertical | marker, mug 14 cm tall (kept, flagged) | 0.93 | 0.39 × 0.47 m |
| ext2 + wrist | hybrid | ext2 | marker, red mug | 0.10 | 0.40 × 0.44 m |
| ext2 + wrist | codex (first prompt) | wrist, near overhead | marker, mug 14.7 × 13.7 cm | 0.51 | 0.40 × 0.44 m |
| ext2 + wrist | codex | ext2 | marker, red mug | 0.03 | 0.40 × 0.44 m |

With ext2, objects and table line up with the real images. DROID's wrist camera looks
down steeply during the episode (16–34° off vertical), and a mug generated from one such
frame comes out wrong: too wide from the most overhead ones, twice as tall from a more
oblique one; the marker, flat anyway, comes out right. Choosing among these frames
cannot fix that; generating meshes from several frames would (see the roadmap). The
first Codex prompt asked for objects as large as possible and so chose an overhead
wrist frame; asked to prefer views that show the objects' sides, it chose ext2.

From ext1, the other exterior view, hybrid selection took the robot's mounting table
for the support in every frame and the scene came out empty after three retries; Codex
chose the side table, but from there the marker hides behind the mug, so only the mug
was reconstructed.

What the runs exposed:

- **The robot as an object.** Before robot removal, SimFoundry's frame selection
  rejected every exterior frame because the arm runs off the image.
- **Ghost objects.** SimFoundry's erase-and-inpaint loop can leave a smudge where it
  erased an object, and then reconstruct the smudge. In one MuJoCo run a 31 × 23 × 4 cm
  slab appeared under the mug, which physics then set on top of it, 4 cm up. The
  existence check dropped the slab and the mug settled back (4.7 → 0.8 cm off). A real
  mug generated 14 cm tall then looked like a ghost to that check too; objects standing
  over measured points that no object explains are now kept, and flagged.
- **Two tables.** On DROID, SimFoundry often chose an ext1 frame and took the robot's
  mounting table for the support (stage 3 takes the largest surface it detects, and the
  clamps counted as four objects in the frame scores); the detector then found nothing
  on it. With the arm over the side table, an ext2 frame did the same. Such runs are
  rebuilt from other frames, the camera with the fewest empty frames first, but with
  ext1 as the only exterior camera every frame failed. Codex frame selection avoids it:
  it names the surface the objects stand on, and stage 3 segments that one.
- **One view per mesh.** Each object's mesh is generated from the one chosen frame, so a
  near-overhead view gives a mug the wrong width or height. Asked only for large
  objects, Codex chose such a view; asked to prefer views that show the objects' sides,
  it chose well.
- **A surface that will not segment.** In a close-up wrist frame the table Codex named
  scored below SAM3's threshold; stage 3 fell back to the floor, 0.57 m lower, and the
  scene landed 67 cm off. Another frame tripped stage 3's roll limit. Codex's frame is
  now used only if stage 3 can segment its surface there, with a plane stage 3 accepts;
  otherwise the next frame in Codex's ranking is.
- **Holes in the depth.** Upstream SimFoundry always has dense depth; r2s2r's has none
  where the robot is cut out. SimFoundry back-projected those pixels onto the camera
  centre: support planes were fitted through it (a frame with the arm over the table
  lost both objects as "below the surface"), and an object detected where the depth is
  missing crashed stage 5. Fixed on the fork.
- **Rounded bases topple.** In a wrist-only scene the box's contact patch was 8.8 cm²
  of an 18 cm² footprint, with the centre of mass 3.3 mm inside it; it toppled in Isaac
  within 0.4 s while standing in MuJoCo. The flat resting base fixed it everywhere.
- **Generative shape errors.** The figurine came out as a pot or teapot, 25–30 % too
  small; the box and mug up to 5–25 % too large. The grasp depends only on the
  target's narrow side (3.1 cm reconstructed, 3.0 cm true).
- **Coarse physics estimates.** The VLM put the mug at 0.45–0.65 kg; it is 0.35 kg.
- **Names vary with the view.** From the wrist, the crayon box is a "yellow box".
- **Isaac's initial state.** `sim.reset()` leaves the robot in the USD's joint state, so
  the rollout writes the initial joint state explicitly.
- **Near clip planes.** MuJoCo scales its near plane with the scene's size (8.5 cm in
  the scene renderer), so objects vanished from renders when the wrist camera came
  close. Renderers now use 5 mm / 50 m, and the MuJoCo cameras report no depth under
  0.1 m, like a real sensor, instead of seeing through what is too close.

## Roadmap

1. **Articulated objects.** Removed for now: SimFoundry's stage 9 and a joint fit to a
   warmup video worked in MuJoCo (last in commit `4f2d5af`), but how they should enter
   the pipeline is open.
2. **Fixed verifiers, agent calibration** for object grounding, physical parameters
   (rest stability and sim-vs-real replay as verifiers) and geometry.
3. **Isaac Lab env**: wrap the scene builder into a `ManagerBasedRLEnvCfg`.
4. **Better shapes**: generate meshes from several views. A single frame cannot give a
   wrist-only DROID capture a right mug: every wrist frame looks down steeply.
5. **Deployment** to the real DROID Franka: a `RobotInterface` over its controller.

Scans without poses (a phone video, a hand-held RGB-D sweep, the robot located in them)
were tried and set aside: on estimated depth the robot's heading came out 10° off.

## TODO: robot-side alignment

Everything above aligns the **scene**. The **robot** is not yet aligned between sim and
real, and must be before policies can be trusted to transfer:

- **Same controller in sim and real**: control law, gains and rate. DROID runs a
  Cartesian/joint impedance controller on the Franka.
- **System identification of the arm**: joint armature, friction and actuation delay
  from excitation runs, validated on held-out trajectories.
  [FrankaTwin](https://github.com/tsrobcvai/frankatwin) does this for FR3/Panda in Isaac
  Lab and is the starting point.
- **Gripper**: the Robotiq 2F-85's actuation and contact parameters. Its mount is fixed:
  Isaac Sim's `franka.usd` mounts the Robotiq like the Franka Hand (10.7 cm out, -45 deg),
  and DROID's is 1.1 cm further out and turned +45 deg from that. `sim/isaaclab/scene.py`
  (`make_scene`) re-mounts it where the MuJoCo model fitted to the wrist-camera images
  has it. Before that, the fingers showed up 45 deg off in every wrist-camera render.
- **Camera timing**: DROID videos lag their step timestamps by about 40 ms.

The MuJoCo loop covers the first point only for its own setup (Franka Hand in both
simulators, one joint-target interface with shared IK).

## Data caveats

- DROID's `cartesian_position` sits about 0.17 m above the Robotiq fingertips.
- DROID 1.0.1 raw videos are face-blurred, and the blur sometimes fires on objects;
  SimFoundry's frame selection scores sharpness and prefers the clean frames.

## Changes on the fork's `r2s2r` branch

- Codex as the VLM: `simfoundry/models/codex_vlm.py` (with `Gemini.__new__`) and
  `codex_genai.py` (a genai client stand-in for articulate-anything's agents).
- Stage 2 (FoundationStereo) reads per-image intrinsics and only complete stereo pairs,
  so frames from different stereo cameras, and RGB-D frames, share one run.
- Stages 5 and 9 read stereo / RGB-D captures (upstream: video mode only); stage 6
  accepts the Codex stand-in.
- Frame selection mode `codex`: Codex ranks all candidates, given the task, and names
  the support surface; the first ranked frame whose surface segments, with a plane
  stage 3 accepts, is used, and stage 3 segments that surface.
- Stages 3 and 5 and the frame selection ignore pixels without depth (holes in RGB-D
  input, the robot cut out), which would back-project onto the camera centre and pull
  support planes through it; stage 5 skips an object whose points span no volume
  instead of crashing in qhull.
- Stage 7: shape-only runs publish a face-reduced untextured mesh for stages 8 and 11.
- `patches/articulate-anything-r2s2r.patch`: Codex routing, and GLB scene-graph fixes
  for stage 9 (which r2s2r does not run for now).

Worked around on the r2s2r side: the orchestrator runs stage 2 in the `da3` env even
for FoundationStereo (r2s2r passes `--env-da3 simfoundry`), and stage 1a names stereo
pairs `zed_<i>` where later stages read `image_<i>` (r2s2r writes `image_<i>`
directly).

## Development

```bash
./run_ci_checks.sh   # autoformat, mypy, pylint, pytest
```

Tests need neither a GPU, SimFoundry nor Codex: they build synthetic DROID episodes and
SimFoundry outputs, and stub the robot masker. Tests that render (MuJoCo world,
orientation and existence checks; marked `gl`) are skipped without an offscreen GL
context, and the MuJoCo world's without its assets.
