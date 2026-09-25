# R2S2R: Real-to-Sim-to-Real

R2S2R turns a recording of a robot's workspace into a simulation of that scene, and
deploys what is learned there back to the robot. The input is the robot's
parameters, calibrated cameras (any mix of exterior and wrist cameras, stereo or
RGB-D) and a video; the output is an Isaac Lab scene with every object, rigid or
articulated, in the robot's own base frame. MuJoCo can stand in for the real world:
it records captures with ground truth and runs the deployed policy.

The module is shared between the code-for-gen project and
[PhysCoder](https://github.com/Jaraxxus-Me/physcoder). The first platform is the
DROID Franka (Panda + Robotiq 2F-85, two ZED 2 exterior cameras, a ZED Mini wrist
camera).

## Status

| Piece | State |
|---|---|
| Captures from DROID episodes, from MuJoCo, or any `capture.json` | done |
| Any camera mix (exterior and/or wrist, one to many images each) | done: MuJoCo ext + wrist and wrist-only, DROID 3 stereo cameras |
| Robot removed from depth before reconstruction | done: Panda + Franka Hand, DROID's Panda + Robotiq |
| SimFoundry backend (Codex as the VLM), stereo or RGB-D input | done |
| Articulated objects (SimFoundry stage 9, articulate-anything) | runs end to end; joint limits are guesses (see [Results](#results)) |
| Refinement: orientation check, position, support outline | done |
| Isaac Lab scene, program policy, deployment to MuJoCo | done: pick succeeds in Isaac Lab and MuJoCo in every test world |
| Robot-side alignment (controller and dynamics) | **not started, see below** |

## Pipeline

```
Capture ─► robot removed ─► reconstruction ─► sim-ready ─► refinement ─► SceneSpec ─► Isaac Lab ─► deploy
(any       from depth       backend           assets       (orientation,  (robot base   + program   (MuJoCo as real;
cameras)                    (SimFoundry)                   position,      frame)        policy      real robot)
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
                    orientation.py, refine.py
  robots/           franka.py (Panda FK / IK in numpy), mujoco_models.py (robot models),
                    mask.py (robot removal)
  policy/           robot.py (RobotInterface + motion primitives), pick.py, scoring.py
  sim/isaaclab/     scene.py (SceneSpec -> Isaac Lab scene), robot.py, rollout.py
  real/mujoco/      MuJoCo as the real world: world.py, capture.py, deploy.py
  video.py          rollout videos and contact sheets
scripts/
  setup/            install.sh, install_articulation.sh, link_simfoundry_resources.sh,
                    fetch_mujoco_assets.sh
  isaaclab/         entry points that start the Omniverse app: render_overlay.py, run_pick.py
  mujoco_loop.sh    the whole MuJoCo loop
third_party/SimFoundry   git submodule of our fork (branch r2s2r); SimFoundry changes go there
```

Conventions: `T_a_b` maps frame-b points into frame a, camera frames are OpenCV
(+z forward, +y down), quaternions are `(w, x, y, z)`, and every `SceneSpec` pose is in
the robot base frame. Nothing in the pipeline depends on the kind of object.

### Capture: any cameras, with poses in one frame

A capture is any set of calibrated cameras whose poses are in the robot base frame: two
exterior cameras (DROID), one exterior camera and the wrist camera, or only the wrist
camera. Each camera contributes one image or many, and a moving camera has its own
`T_base_cam` on every frame. A frame needs its image, its pose, and either a stereo
partner (`right_image`, camera `stereo_baseline`) or metric depth (`depth_image`, uint16
PNG in millimetres). Joint positions and gripper opening are needed to remove the robot.

`r2s2r droid-capture` and `r2s2r mujoco-capture` write `capture.json`
(`r2s2r.structs.Capture`); other data can be brought in by writing the same file:

```json
{"name": "my_scene", "source": "my_rig", "embodiment": "franka_panda",
 "instruction": "pick up the red block", "static_steps": [0, 100],
 "cameras": {"wrist_cam": {"serial": "wrist_cam", "role": "wrist", "width": 1280,
             "height": 720, "K": [[...]], "stereo_baseline": null, "is_static": false}},
 "frames": [{"step": 0, "camera": "wrist_cam", "left_image": "rgb/0000.png",
             "right_image": null, "depth_image": "depth/0000.png",
             "T_base_cam": [[...]], "joint_positions": [...], "gripper_position": 0.0}]}
```

`Capture.select_frames` shares a frame budget evenly between cameras and spreads each
camera's share over the static period; reconstruction and refinement both use it
(`--cameras` limits them to some cameras, by role or serial).

### Robot removal

A robot standing on the table is geometry above the support plane: frame selection
counts an arm that leaves the image as a clipped object, and refinement clusters it.
`robots/mask.py` renders the robot's MuJoCo model (mujoco_menagerie) at the recorded
joint state from each calibrated camera, with the exact pinhole intrinsics, and sets
the depth to zero wherever the robot is the visible surface (silhouette grown by 0.6 %
of the image diagonal; surfaces over 3 cm in front of the robot are kept). In the
MuJoCo world the rendered robot matches the simulator's own segmentation at IoU ≥ 0.98;
on DROID the Robotiq's mounting yaw (+90° on the flange) was fitted to the fingers in
the wrist-camera images. `--no-robot-mask` turns it off.

### Reconstruction: the SimFoundry backend

[SimFoundry](third_party/SimFoundry) runs in its own conda envs; r2s2r drives its
orchestrator with `PYTHONPATH` pointing at the submodule. The backend

1. writes up to 12 candidate frames from any cameras where SimFoundry reads them: a
   stereo frame as stage 1a would (`s1_zed/image_<i>_{l,r}.png` with its own
   `image_<i>_intrinsic.txt`), an RGB-D frame's measured depth as FoundationStereo would
   (`s2_fs/`, at its 0.5 scale). Without stereo frames stage 2 is skipped;
2. removes the robot from every candidate's depth (stereo frames right after stage 2,
   from a kept raw copy);
3. runs stages 3–12, with stage 9 when articulate-anything is installed;
4. anchors SimFoundry's support-plane world frame to the robot with the chosen frame's
   `T_base_cam`, and writes each object's URDF sim-ready (below).

SimFoundry reconstructs from the one frame its selection picks, so most of the video
goes unused; refinement uses the rest.

Every VLM call goes through the local Codex CLI (`gpt-6-astra`, reasoning `medium`):
the fork's `Gemini.__new__` returns a `CodexVLM` when `SIMFOUNDRY_VLM_BACKEND=codex`,
which r2s2r sets. Text/vision requests take about 7 s, image edits about 75 s. Runs are
not bit-reproducible (no sampling controls); `--vlm-backend gemini` restores upstream.

### Articulated objects

Stage 9 asks the VLM which objects are articulated and hands their meshes to
[articulate-anything](https://github.com/nadunRanawaka1/articulate-anything-sf): render
the object, let the VLM name its parts and build an articulation tree, segment the mesh
(Hunyuan3D-Part's P3-SAM), merge the parts into a URDF, and fit each joint with a VLM
actor–critic loop. Stage 11 gives every part a mass and friction. Such objects carry
`ObjectSpec.articulated`; Isaac Lab spawns them as articulations with a fixed base and
passive, damped joints.

From one static view the joint type and axis are inferred from appearance; how far a
closed drawer slides or a door swings cannot be seen, so the limits are guesses. They
are kept as the backend gives them.

`scripts/setup/install_articulation.sh` installs articulate-anything without sudo
(git-lfs and Blender into `~/.local`, the Hunyuan3D-Part env, the P3-SAM weights) and
applies the fork's `patches/articulate-anything-r2s2r.patch`: Codex for its VLM calls
(video parts go as a few frames) and two fixes for GLB scene graphs. Without it, stage 9
is skipped with a warning and every object stays rigid.

### Simulation-ready assets

`assets.make_sim_ready` writes one `<model>_r2s2r.urdf` per object:

- `<mesh scale>` is baked into the meshes (Isaac Sim 5.1's importer left a 1 m collider
  beside each scaled one);
- a rigid object resting on the support gets a flat 5 mm base cut from its own
  cross-section. An object seen standing still must stand in simulation too, but
  generated meshes round off the edges it stands on (see [Results](#results)).

### Refinement

`r2s2r refine` fuses metric depth from many views (capture depth, or SimFoundry's
stage-2 depth), with the robot removed, and

1. **checks every object's turn about the support normal** (`reconstruct/orientation.py`).
   Each candidate (0, 90, 180, 270°) is rendered into every view and scored by its depth
   residual; turns the geometry rules out are dropped. The remaining turns go to the VLM,
   which sees the photo and a render of each candidate and looks for features that fix
   the orientation, such as handles, openings or printed text. The backend's pose is
   kept unless the VLM is confident. `--no-vlm` keeps ties as they are;
2. **re-registers every object** to the fused points by its top-down footprint (yaw and
   in-plane shift), only when the fit clearly improves;
3. **fits the support surface's outline** (a rectangle in `SceneSpec.support_extent`).

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
bash scripts/setup/install_articulation.sh   # optional: articulated objects (stage 9)
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
r2s2r reconstruct outputs/iris/capture --workdir outputs/iris/simfoundry --cameras ext1 ext2
r2s2r refine outputs/iris/simfoundry/<scene>/scene --capture outputs/iris/capture \
    --views outputs/iris/simfoundry --out outputs/iris/scene
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/render_overlay.py outputs/iris/scene \
    outputs/iris/capture --out outputs/iris/overlay --headless
```

## Quickstart: MuJoCo as real

A Franka Panda with the Franka Hand stands on a table with Google Scanned Objects,
watched by two RealSense-like RGB-D cameras (1280×720, f = 910 px) and a wrist camera
(1280×720, f = 800 px). While recording, the arm points the wrist camera at the table
from three directions. The pipeline sees only what a real rig would record; ground truth
goes into `capture.metadata` for scoring. World presets: `pick` (a crayon box to pick, a
mug, a figurine) and `cabinet` (the mug replaced by a cabinet with a sliding drawer).

```bash
CAMERAS="ext2 wrist" WORLD=pick bash scripts/mujoco_loop.sh outputs/mujoco_pick   # all of:

r2s2r mujoco-capture --out outputs/mujoco_pick/capture --cameras ext2 wrist --world pick
r2s2r reconstruct outputs/mujoco_pick/capture --workdir outputs/mujoco_pick/simfoundry
r2s2r refine outputs/mujoco_pick/simfoundry/mujoco_pick/scene \
    --capture outputs/mujoco_pick/capture --capture-depth --out outputs/mujoco_pick/scene_refined
r2s2r mujoco-eval outputs/mujoco_pick/scene_refined --capture outputs/mujoco_pick/capture
T=$(r2s2r mujoco-eval outputs/mujoco_pick/scene_refined \
    --capture outputs/mujoco_pick/capture --match-target)
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/run_pick.py outputs/mujoco_pick/scene_refined \
    --target $T --out outputs/mujoco_pick/isaac --headless --video-camera ""
r2s2r mujoco-deploy outputs/mujoco_pick/scene_refined --capture outputs/mujoco_pick/capture \
    --out outputs/mujoco_pick/deploy
r2s2r mujoco-deploy --oracle --capture outputs/mujoco_pick/capture \
    --out outputs/mujoco_pick/deploy_oracle          # ground-truth baseline
```

Which reconstructed object is "the crayon box" depends on the VLM's naming ("crayon
box", "yellow box"). Grounding the task in the scene's object list is the policy
writer's job (in code-for-gen, an agent); in this harness the ground truth stands in for
it (`--match-target`), and the policy only sees the reconstruction.

## Results

**DROID (IRIS episode).** SimFoundry picked an ext2 frame, separated the marker and the
mug and placed both on the table. Refinement with both exterior cameras fitted the
table outline (0.36 × 0.42 m) and kept both poses (within about 2 mm and 1°). Rendered
from the real camera poses in Isaac Lab, the robot, the objects and the table outline
line up with the real images in both cameras.

**MuJoCo as real** (2026-09-25). Error is the distance between the centres of the
reconstructed and true objects' boxes, in the base frame; Isaac and MuJoCo columns are
how far the same program lifted the crayon box, with the same commands.

| World, cameras | Frame SimFoundry picked | Error: crayon box / mug or cabinet / figurine | Isaac | MuJoCo |
|---|---|---|---|---|
| pick, ext2 | ext2 | 0.5 / 1.0 / 1.4 cm | 14.8 cm | 14.9 cm |
| pick, ext2 + wrist | wrist | 0.5 / 0.5 / 1.0 cm | 14.8 cm | 15.0 cm |
| pick, wrist only | wrist | 0.6 / 0.4 / 1.3 cm | 14.8 cm | 14.9 cm |
| cabinet, ext2 + wrist | wrist | 0.4 / 1.2 / 1.3 cm | 14.8 cm | 15.0 cm |

The ground-truth scene succeeds in every world (14.9 cm). SimFoundry takes about 13
minutes per scene (3 more for stage 9), mostly Codex image edits and Hunyuan.

In the cabinet world, stage 9 found the cabinet articulated and made its drawer front a
prismatic joint. The cabinet first came out turned by 90° (the drawer front on a side
face): it is nearly square and its generated texture lost the brown sides, so the pose
fit could not tell the faces apart. The orientation check turned it back (the VLM chose
among 0, 180 and 270°, which the geometry could not separate); the joint axis is now 1°
from the truth. The drawer's travel is 3 cm (true 14 cm), unobservable while it is
closed.

What the runs exposed:

- **The robot as an object.** Before robot removal, SimFoundry's frame selection
  rejected every exterior frame because the arm runs off the image.
- **Rounded bases topple.** In the wrist-only scene the box's contact patch was 8.8 cm²
  of an 18 cm² footprint, with the centre of mass 3.3 mm inside it; it toppled in Isaac
  within 0.4 s while standing in MuJoCo. The flat resting base fixed it everywhere.
- **Generative shape errors.** The figurine came out as a pot or teapot, 25–30 % too
  small; the box, mug and cabinet up to 5–25 % too large. The grasp depends only on the
  target's narrow side (3.1 cm reconstructed, 3.0 cm true).
- **Coarse physics estimates.** The VLM put the mug at 0.45–0.65 kg; it is 0.35 kg.
- **Names vary with the view.** From the wrist, the crayon box is a "yellow box".
- **Isaac's initial state.** `sim.reset()` leaves the robot in the USD's joint state, so
  the rollout writes the initial joint state explicitly.

## Roadmap

1. **Use the whole warmup video.** Record the robot interacting (opening the drawer,
   pushing an object) and measure from the motion what one static view cannot show:
   joint axes and ranges from part motion, masses and friction from how objects move
   under a known push. Try it in MuJoCo first.
2. **Fixed verifiers, agent calibration.** Keep measurement and scoring fixed (depth
   and render residuals, rest stability, motion consistency, sim-vs-real replay), and let
   a coding agent calibrate what needs judgement: object grounding, articulation
   structure, physical parameters, and repairs when a verifier fails, accepting a change
   only when the verifiers improve.
3. **Isaac Lab env**: wrap the scene builder into a `ManagerBasedRLEnvCfg`.
4. **Better shapes**: generate meshes from several views.
5. **Deployment** to the real DROID Franka: a `RobotInterface` over its controller.

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
- Stage 7: shape-only runs publish a face-reduced untextured mesh for stages 8 and 11.
- `patches/articulate-anything-r2s2r.patch`: Codex routing, and GLB scene-graph fixes
  (node and geometry names that differ; node transforms when rendering).

Worked around on the r2s2r side: the orchestrator runs stage 2 in the `da3` env even
for FoundationStereo (r2s2r passes `--env-da3 simfoundry`), plans stage 9 only with
`--detect-articulation`, and stage 1a names stereo pairs `zed_<i>` where later stages
read `image_<i>` (r2s2r writes `image_<i>` directly).

## Development

```bash
./run_ci_checks.sh   # autoformat, mypy, pylint, pytest
```

Tests need neither a GPU nor SimFoundry: they build synthetic DROID episodes and
SimFoundry outputs. Tests that render (MuJoCo world, orientation check) are skipped
without an offscreen GL context or the MuJoCo assets.
