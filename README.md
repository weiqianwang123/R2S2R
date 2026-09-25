# R2S2R: Real-to-Sim-to-Real

R2S2R turns a recording of a robot's workspace into a simulation of that scene, and
deploys what is learned there back to the robot. The input is the robot's
parameters, calibrated camera intrinsics and extrinsics, an RGB-D (stereo) video and
optionally CAD models of the objects. The output is an Isaac Lab environment with
every rigid object in place, in the robot's own base frame. MuJoCo can stand in for
the real world: it records captures with ground truth and runs the deployed policy.

The module is shared between the code-for-gen project and
[PhysCoder](https://github.com/Jaraxxus-Me/physcoder). The first platform is the
DROID Franka (Panda + Robotiq 2F-85, two ZED 2 exterior cameras, a ZED Mini wrist
camera).

## Status

| Piece | State |
|---|---|
| DROID raw episode → `Capture` (`r2s2r droid-capture`) | done, tested on `IRIS+ef107c48+2023-03-07-16h-19m-59s` |
| SimFoundry backend, stages 2–12 (Codex as the VLM), textured meshes | done on IRIS: mug + marker on a table plane at z = −0.189 m |
| Multi-view refinement (`r2s2r refine`): support outline + object footprint check | done: table 0.36 × 0.42 m from both exterior cameras; both objects already within ~2 mm, poses kept |
| Isaac Lab scene builder + real-camera overlay | done (`scripts/isaaclab/render_overlay.py`): robot, mug, marker and table line up with the real images in both exterior cameras |
| MuJoCo as real: RGB-D capture → SimFoundry → Isaac Lab pick → MuJoCo deploy | done, see [MuJoCo loop](#the-mujoco-loop-mujoco-as-real) |
| Robot-side alignment (controller and dynamics) | **not started, see below** |

## How it fits together

```
Capture ──► ReconstructionBackend ──► SceneSpec ──► simulator builder ──► policy ──► deploy
(io/)       (reconstruct/, pluggable)  (robot base    (Isaac Lab; MuJoCo)             (real robot;
                                        frame)                                         MuJoCo as real)
```

```
src/r2s2r/
  structs.py        Capture (what was recorded), SceneSpec (what was reconstructed),
                    DepthView; plain data saved as JSON
  transforms.py     frames, quaternions, projection / backprojection
  assets.py         URDF helpers: bake mesh scales, sample visual surfaces
  cli.py            the r2s2r command
  io/               recordings -> Capture: droid.py (raw DROID episodes), rgbd.py
  reconstruct/      Capture -> SceneSpec: base.py (pluggable backends),
                    simfoundry.py, refine.py (multi-view pose and outline check)
  robots/franka.py  Panda FK / Jacobian / IK in numpy, shared by every target
  policy/           robot.py (RobotInterface + motion primitives), pick.py (the pick
                    program), scoring.py (lift scoring, result files)
  sim/isaaclab/     scene.py (SceneSpec -> Isaac Lab scene), robot.py, rollout.py
  real/mujoco/      MuJoCo as the real world: world.py, capture.py, deploy.py
  video.py          rollout videos and contact sheets
scripts/
  setup/            install.sh, link_simfoundry_resources.sh, fetch_mujoco_assets.sh
  isaaclab/         entry points that start the Omniverse app: render_overlay.py,
                    run_pick.py
  mujoco_loop.sh    the whole MuJoCo loop
```

- `io/droid.py` uses the improved DROID calibration and keeps only the steps before
  the gripper first closes, when the objects are static.
- `reconstruct/simfoundry.py` drives [SimFoundry](third_party/SimFoundry) and
  re-expresses its output in the robot base frame.
- A deployment target implements `policy.robot.RobotInterface`: hold joint targets
  and a gripper command for one control period. Isaac Lab (`sim/isaaclab/robot.py`)
  and MuJoCo (`real/mujoco/world.py`) do; the real DROID arm will.
- `third_party/SimFoundry`: git submodule of
  [our fork](https://github.com/weiqianwang123/SimFoundry). All SimFoundry changes go
  there, never into this repository.

Conventions: `T_a_b` maps frame-b points into frame a, camera frames are OpenCV
(+z forward, +y down), quaternions are `(w, x, y, z)`, and every `SceneSpec` pose is in
the robot base frame.

### The SimFoundry backend (single-frame baseline)

SimFoundry is run, not imported. It keeps its own conda envs, and r2s2r calls its
orchestrator with `PYTHONPATH` pointing at the submodule. For a DROID capture, the
backend:

1. writes the stereo pairs of one exterior camera in the layout of SimFoundry's
   stereo capture stage (`image_<i>_{l,r}.png` plus `intrinsic.txt` holding K and the
   baseline);
2. runs stages 2–12 in stereo mode. FoundationStereo gives metric depth, and stages
   13–14 (OmniGibson import) are skipped;
3. reads back which frame SimFoundry chose. It then anchors SimFoundry's
   support-plane world frame to the robot with that frame's `T_base_cam` and
   converts every object pose from stage 12.

For an RGB-D camera (no stereo pair, `depth_image` on every frame), the backend
writes the measured depth where FoundationStereo would (`s2_fs/image_<i>_*`, at
FoundationStereo's 0.5 scale) and drops stage 2; stages 3–12 cannot tell the
difference.

SimFoundry reconstructs from one frame of one camera, so most of the video goes
unused. That single frame is the baseline. Using the rest of the video is on the
roadmap below.

## Install

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/weiqianwang123/R2S2R.git
cd R2S2R
uv venv --python 3.11 .venv && uv pip install -e ".[develop]"   # core only
bash scripts/setup/install.sh    # + torch 2.7 (cu128), Isaac Sim 5.1, Isaac Lab v2.3.2
```

SimFoundry needs its own install (conda envs, model repos, checkpoints). Follow its
README once. Then link the heavy, git-ignored parts into the submodule:

```bash
bash scripts/setup/link_simfoundry_resources.sh ~/SimFoundry
```

SimFoundry's stages 3, 5, 6, 8 and 11 call Gemini upstream. The `r2s2r` branch of
our fork adds a Codex backend (`simfoundry/models/codex_vlm.py`). With
`SIMFOUNDRY_VLM_BACKEND=codex`, which r2s2r sets by default, every `Gemini(...)`
the stages construct becomes a `codex exec` call:

- Text/vision requests (detection, frame choice, front picking, mass/friction) are
  read-only calls with the images attached. They take about 7 s at reasoning effort
  medium.
- Image requests (removing an object to see behind it, upsampling an object crop)
  enable Codex's `image_generation` feature. They take about 75 s per image.

The default model is `gpt-6-astra` at reasoning effort `medium`
(`r2s2r reconstruct --codex-reasoning ...` to change). SimFoundry picks the ChatGPT
desktop app's bundled CLI (`/usr/lib/chatgpt/resources/codex`) when it exists.
Older CLIs reject `gpt-6-astra`. Sampling controls (temperature, seed) have no
Codex equivalent, so runs are not bit-reproducible. `--vlm-backend gemini` restores
upstream behaviour (needs `GCLOUD_PROJECT` or `GEMINI_API_KEY`).

On first use SimFoundry downloads its model weights into the usual caches:
Hunyuan3D-2.1 (~14 GB, `~/.cache/hy3dgen`), Depth Anything 3 (~7 GB), Prior-DA and
bria-rmbg (~2 GB). It also downloads SAM3 (~3.5 GB, `facebook/sam3`) unless that is
already in the Hugging Face cache.

## Quickstart: one public DROID episode

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
    --camera-role ext2 --stages 2            # depth only, no Gemini needed
r2s2r reconstruct outputs/iris/capture --workdir outputs/iris/simfoundry \
    --camera-role ext2                       # stages 2-12, then writes scene.json
r2s2r reconstruct outputs/iris/capture --workdir outputs/iris/simfoundry_ext1 \
    --camera-role ext1 --stages 2            # depth from the second camera
r2s2r refine outputs/iris/simfoundry/<scene>/scene --capture outputs/iris/capture \
    --views outputs/iris/simfoundry outputs/iris/simfoundry_ext1 --out outputs/iris/scene
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/render_overlay.py outputs/iris/scene \
    outputs/iris/capture --out outputs/iris/overlay --headless
```

## The MuJoCo loop (MuJoCo as real)

MuJoCo plays the real world at both ends. A Franka Panda with the Franka Hand
(mujoco_menagerie) stands on a table with three Google Scanned Objects, watched by
two RealSense-like RGB-D cameras (1280×720, f = 910 px) and a wrist camera. The
camera layout follows DROID's 2 exterior + 1 wrist. The pipeline only ever sees what a
real rig would record: images, metric depth, intrinsics, extrinsics and joint
states. Ground truth goes into `capture.metadata` and is used for scoring only.

```bash
bash scripts/setup/fetch_mujoco_assets.sh   # Panda + 3 scanned objects, ~45 MB
bash scripts/mujoco_loop.sh                 # everything below, into outputs/mujoco_pick

r2s2r mujoco-capture --out outputs/mujoco_pick/capture
r2s2r reconstruct outputs/mujoco_pick/capture --workdir outputs/mujoco_pick/simfoundry \
    --camera-role ext2 --override s3_ground.frame_selection.max_clipped_frac=0.9
r2s2r refine outputs/mujoco_pick/simfoundry/mujoco_pick/scene \
    --capture outputs/mujoco_pick/capture --capture-depth --out outputs/mujoco_pick/scene_refined
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaaclab/run_pick.py outputs/mujoco_pick/scene_refined \
    --target crayon --out outputs/mujoco_pick/isaac --headless
r2s2r mujoco-deploy outputs/mujoco_pick/scene_refined --capture outputs/mujoco_pick/capture \
    --target crayon --out outputs/mujoco_pick/deploy
r2s2r mujoco-deploy --oracle --capture outputs/mujoco_pick/capture \
    --target crayon --out outputs/mujoco_pick/deploy_oracle   # ground-truth baseline
```

The policy (`r2s2r.policy.pick.pick_up`) is a short program. It goes to a fixed
ready pose, plans a top-down grasp across the target's narrowest side from the
SceneSpec mesh, then approaches, descends, closes and lifts 15 cm. Every target runs
it through the same `RobotInterface`. Cartesian interpolation and IK happen above
that interface, in numpy (`r2s2r.robots.franka`, checked against MuJoCo's own
kinematics). Isaac Lab and MuJoCo therefore receive the same joint-target stream,
and only the plant differs.

First run (2026-09-25):

| | Result |
|---|---|
| Capture | 77 RGB-D frames per camera × 3 cameras while the arm sweeps over the table; 5 s |
| SimFoundry (ext2, RGB-D, Codex) | 12 min 56 s; Codex picked frame 6 and found "teal mug", "crayon box", "orange figurine" |
| Reconstruction vs ground truth (box centre) | crayon box 0.5 cm, mug 1.0 cm, figurine 1.5 cm; table 1.20 × 1.20 m from `refine` (exact) |
| Isaac Lab, reconstructed scene | success: crayon box lifted 14.8 cm |
| MuJoCo as real, same program | **success**: crayon box lifted 14.9 cm; the 411 commands after the ready pose are identical to Isaac's |
| MuJoCo, ground-truth scene (oracle) | success: 14.9 cm |

What the run exposed:

- Generative shape errors. Codex's image model redrew the Android figurine as a small
  orange pot with a handle, and the mesh came out 25–30 % too small. The crayon box and
  the mug are 5–15 % too large. The grasp only depends on the target's narrow side,
  which came out at 3.1 cm (true 3.0 cm).
- The VLM's mass estimates are rough: 0.65 kg for a 0.35 kg mug.
- The arm stands on the table and runs off the image edge. SimFoundry's frame selection
  counts it as a clipped object and rejected every frame until `max_clipped_frac` was
  raised. Masking the robot out of the depth would be the proper fix.
- Isaac Lab's `sim.reset()` leaves the robot in the USD's joint state, not the
  config's, so `sim/isaaclab/rollout.py` writes the initial joint state explicitly.
- MuJoCo 3.4+ requires `websockets>=13`, which conflicts with Isaac Sim 5.1's pin. The
  project pins `mujoco<3.4`.

## Development

```bash
./run_ci_checks.sh   # autoformat, mypy, pylint, pytest (src/ and tests/ only)
```

Tests need neither a GPU nor SimFoundry. They build a tiny synthetic DROID episode,
and synthetic SimFoundry outputs where those are needed. The MuJoCo tests are skipped
until `scripts/setup/fetch_mujoco_assets.sh` has run.

## Roadmap

1. **Isaac Lab env**: wrap the scene builder into a `ManagerBasedRLEnvCfg`, adding
   actions, observations and resets, so policies can run in it.
2. **Use more of the video**: add the moving wrist camera to `r2s2r refine`, generate
   meshes from several views (to fix shapes like the over-thick marker), and replay
   the interaction steps to validate physics.
3. **MuJoCo as real, harder**: DROID's Robotiq gripper in MuJoCo, randomized
   layouts and lighting, many objects, and a robot mask so the arm never enters the
   reconstruction.
4. **Deployment** to the real DROID Franka: a `RobotInterface` over its controller.

## TODO: robot-side alignment

Everything above aligns the **scene** (objects, support surface, cameras). The
**robot** is not yet aligned between sim and real, and this has to be done before
policies can be trusted to transfer:

- **Same controller in sim and real.** The simulated arm must run the same control
  law, gains and rate as the real one. DROID runs a Cartesian/joint impedance
  controller on the Franka.
- **System identification of the arm.** Fit joint armature, friction and actuation
  delay from real excitation runs, and validate on held-out trajectories.
  [FrankaTwin](https://github.com/tsrobcvai/frankatwin) does exactly this for FR3/Panda
  in Isaac Lab and is the starting point.
- **Gripper.** The Robotiq 2F-85 model, its actuation and its contact parameters.
- **Camera timing.** Latency between the camera and robot state streams. DROID videos
  lag their step timestamps by about 40 ms.

The MuJoCo loop covers the first point only for its own setup: Franka Hand in both
simulators, and one joint-target interface with shared IK. DROID's Robotiq, real arm
dynamics and camera timing are still open.

## Data caveats

- DROID's `cartesian_position` sits about 0.17 m above the Robotiq fingertips
  (IRIS: end effector at −0.008 m while grasping a marker at −0.179 m).
- DROID 1.0.1 raw videos are face-blurred, and the blur sometimes fires on objects:
  the red mug in the IRIS episode is smeared in some frames and sharp in others.
  SimFoundry's frame selection scores sharpness, so it prefers the clean frames.

## First result (IRIS episode)

SimFoundry picked frame 2 of ext2 (step 10) via Codex, separated the marker and the mug,
generated textured meshes and placed both on the table. Fusing FoundationStereo depth
from both exterior cameras (`r2s2r refine`) fits the table outline (0.36 × 0.42 m) and
checks every object's top-down footprint against the fused points. Both were already
within about 2 mm and 1°, so the backend's poses were kept. Rendered from the real
camera poses in Isaac Lab, the robot, the mug, the marker and the table outline line up
with the real images in both cameras. ext1 was never used by SimFoundry.

Two Isaac-side pitfalls made the first overlays look centimetres off. Both are fixed in
this repository:

- **Camera field of view.** Without a focal length, Isaac Lab's
  `PinholeCameraCfg.from_intrinsic_matrix` sets a 1 mm aperture and a sub-millimetre
  focal length, and the render came out magnified by about 1.3×. The builder passes
  24 mm. Omniverse also ignores principal-point offsets, so each camera renders a
  slightly larger image centred on (cx, cy), which is then cropped back
  (`centered_render_size`).
- **Scaled URDF meshes.** SimFoundry's collision hulls are unit meshes scaled in the
  URDF. Isaac Sim's importer left a 1 m collider beside each scaled one.
  `r2s2r.assets.bake_mesh_scales` writes `<model>_r2s2r.urdf` with the scales baked
  into the meshes, and the SceneSpec points at that copy.

## Changes on the fork's `r2s2r` branch

- `simfoundry/models/codex_vlm.py` + `Gemini.__new__`: Codex backend (see Install).
- Stage 5: supports stereo (FoundationStereo) inputs. Upstream it only read the
  video-mode frames and Depth Anything outputs.
- Stage 6: accepts the Codex stand-in where it asserted a `Gemini` instance.
- Stage 7: shape-only runs publish a face-reduced (40k) untextured mesh where stages 8
  and 11 look for the textured one, so geometry can be checked before texturing.

Worked around on the r2s2r side, not yet fixed in the fork:

- The orchestrator runs stage 2 in the `da3` env even for the FoundationStereo backend,
  which is installed in the `simfoundry` env. r2s2r passes `--env-da3 simfoundry`.
- Stage 1a writes stereo pairs as `zed_<i>_{l,r}.png`, while stages 3–8 read
  `image_<i>_*` from the FoundationStereo output. r2s2r writes `image_<i>_*` directly.
