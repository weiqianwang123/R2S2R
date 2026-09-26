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
  io/               recordings -> Capture: droid.py, rgbd.py (depth PNGs, depth views)
  reconstruct/      Capture -> SceneSpec: base.py (pluggable backends), simfoundry.py,
                    render.py (scene renders), existence.py, orientation.py, refine.py
  robots/           franka.py (Panda FK / IK in numpy), mujoco_models.py (robot models),
                    mask.py (robot removal)
  policy/           robot.py (RobotInterface + motion primitives), pick.py, scoring.py
  sim/isaaclab/     scene.py (SceneSpec -> Isaac Lab scene), robot.py, rollout.py
  real/mujoco/      MuJoCo as the real world: world.py, capture.py, deploy.py
  video.py          rollout videos and contact sheets
scripts/
  setup/            install.sh, link_simfoundry_resources.sh, fetch_mujoco_assets.sh
  isaaclab/         entry points that start the Omniverse app: render_overlay.py, run_pick.py
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

## Results

Every run below uses the same code (2026-09-26); the camera setup is the only change.
The exterior rows were run with **two** exterior cameras (the MuJoCo world had a second
one then, and DROID has two); the setup assumed now is a single exterior camera, whose
results (ext1 alone, ext1 + wrist) are still to be run. The wrist-only rows are
unaffected.

**MuJoCo as real.** Error is the distance between the centres of the reconstructed and
true objects' boxes, in the base frame. Isaac and MuJoCo are how far the same program
lifted the crayon box, with the same commands; with the ground-truth objects it lifts
14.9 cm in every setup.

| Cameras | Frame SimFoundry picked | Error: crayon box / mug / figurine | Isaac | MuJoCo |
|---|---|---|---|---|
| two exterior (ext1 + ext2) | ext1 | 0.7 / 0.3 / 0.9 cm | 14.6 cm | 14.9 cm |
| wrist | wrist | 1.0 / 0.3 / 1.1 cm | 14.8 cm | 14.7 cm |
| two exterior + wrist | wrist | 0.7 / 0.5 / 1.5 cm | 14.8 cm | 14.9 cm |

The two-exterior scene also has a 6 mm flat disc beside the figurine, a ghost too thin for
the existence check to rule on; the others have exactly the three objects. SimFoundry
takes about 13 minutes per scene, mostly Codex image edits and Hunyuan.

**DROID (IRIS episode, "place the marker in the red mug").** No ground truth: each scene
is rendered in Isaac Lab from both exterior cameras and laid over their images, which
for the wrist-only scene is a check by cameras it never used. Distances are between the
objects' box centres and those of the exterior-camera scene.

| Cameras | Frame SimFoundry used | Objects | Mug / marker vs exterior | Table outline |
|---|---|---|---|---|
| two exterior (ext1 + ext2) | ext2, after 1 empty ext1 frame | marker, red mug | – | 0.37 × 0.44 m |
| wrist | wrist | marker, mug | 4.5 / 0.7 cm | 0.40 × 0.47 m |
| two exterior + wrist | ext2, after 3 empty frames | marker, red mug | 0.4 / 0.1 cm | 0.39 × 0.48 m |

With exterior cameras, objects and table line up with the real images in both views.
The wrist-only scene places the marker as well, but its mug, generated from close-up
wrist views, came out pale and too large (14.8 × 13.4 cm against 8.6 × 10.7 cm), and its
table outline overhangs by about 3 cm. "Empty" frames are ext1 frames, and one ext2
frame with the arm over the table, where stage 3 took the robot's mounting table for
the support; the retry (above) rebuilt from other frames.

What the runs exposed:

- **The robot as an object.** Before robot removal, SimFoundry's frame selection
  rejected every exterior frame because the arm runs off the image.
- **Ghost objects.** SimFoundry's erase-and-inpaint loop can leave a smudge where it
  erased an object, and then reconstruct the smudge. In one MuJoCo run a 31 × 23 × 4 cm
  slab appeared under the mug, which physics then set on top of it, 4 cm up. The
  existence check dropped the slab and the mug settled back (4.7 → 0.8 cm off).
- **Two tables.** On DROID, SimFoundry often chose an ext1 frame and took the robot's
  mounting table for the support (stage 3 takes the largest surface it detects, and the
  clamps counted as four objects in the frame scores); the detector then found nothing
  on it. With the arm over the side table, an ext2 frame did the same. Such runs are
  now rebuilt from other frames, the camera with the fewest empty frames first. With
  ext1 as the only exterior camera, the retry has only ext1's other frames and the
  wrist camera's to fall back on: that setup is the one to run next.
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
4. **Better shapes**: generate meshes from several views.
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
- **Gripper**: the Robotiq 2F-85 model, its actuation and contact parameters.
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
- Frame selection mode `codex`: Codex chooses the frame and the support surface among all
  candidates, given the task; stage 3 segments that surface.
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
