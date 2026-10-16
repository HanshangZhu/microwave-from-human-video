# Putting a pot in a microwave, from three human videos

Three 40-second RGB-D videos (one fixed RealSense) of me putting a Heinz Snap Pot into my
microwave become a Franka Panda doing the same task in a physics simulation of that
microwave and that pot. A scripted solver turns the videos' object keyframes into robot
demonstrations; a **camera-based policy** trained on them runs closed-loop from pot
positions it has not seen, on two camera images and its own joints, nothing privileged.
It succeeds **74 %** on a plain grey floor and **68 %** with both cameras looking at a
**3D Gaussian-splat reconstruction of my kitchen** it never saw in training (128 start
states each). Nothing here has run on a real robot.

## Demo

| human take (left) vs. simulation rendered from the same camera (right) |
|---|
| ![replay](docs/media/take0_replay.gif) |

| me (take 0, the real camera) vs. the camera policy, same camera, in the splatted kitchen, at a pot position it never saw |
|---|
| ![me vs the policy](docs/media/human_vs_policy_splat.gif) |

| robot, solved from take 0 | the policy, a novel view of the splatted kitchen |
|---|---|
| ![solver](docs/media/solver_take0.gif) | ![camera policy](docs/media/policy_splat_kitchen.gif) |

| the same keyframes on a Unitree G1 upper body -- scripted, not a policy, not evaluated |
|---|
| ![G1](docs/media/g1_scripted_take0.gif) |

The policy's inputs are the front camera (the real RealSense's registered pose and
intrinsics, 224 x 126) and a wrist camera (inset in the second GIF); in the splatted kitchen
each is the splat rendered from the camera's pose every step with the simulated robot,
microwave and pot drawn over it. Novel views are not inputs. The policy takes about 100 s for
what I do in 40 (GIFs: 3x human side, 8x robot side).
**Video (1080p, 18 s): [docs/media/policy_splat_kitchen_1080p.mp4](docs/media/policy_splat_kitchen_1080p.mp4).**

**Key result.** The submitted policy (v6-long), 128 unseen start states per column, the same
starts in both columns; success rate reached at each stage of the task:

| stage reached | grey floor | splatted kitchen (both cameras) |
|---|---|---|
| door opened | 100 % | 100 % |
| pot grasped and lifted | 90 % | 77 % |
| pot inside the microwave | 84 % | 72 % |
| door shut with the pot inside | 75 % | 69 % |
| **task success** | **74 %** | **68 %** |

Grey floor vs kitchen, paired: 21 / 13 discordant, p = 0.23. The kitchen was never a
training scene; it costs the policy at the grasp and nothing at the door. The evaluation's
own floor: the grey-floor run repeated with another sampling-noise seed gives 79 % on the
same starts and 78 % on 128 new starts, so single numbers here carry about +-5 points and
only the paired differences are read.

## Method

![method](docs/media/method.png)

\* with the lid texture the policy was trained on (orange rim); with the corrected pot,
74 % / 68 % (Results).

The human video contributes the task's keyframes and the camera; the measured twin and the
solver contribute the motion; the room enters only at evaluation, never in training. The G1
branch is a single scripted run outside the evaluation.

What the front camera sees -- the real frame, the splat composite, the earlier mesh scan:

![front views](docs/media/front_views.png)

**The splatted kitchen** (evaluation only): 300 frames of a RealSense sweep of the room,
posed by Pi3X in batches anchored to a 60-frame global solve; the microwave painted out of
every frame with LaMa (its known box projected at each pose); a gsplat 3DGS with per-view
pose refinement (held-out PSNR 26.5 dB); rendered per camera pose and composited by
segmentation (`recon/sweep_dense.py`, `gs_inpaint.py`, `gs_kitchen.py`; `sim/gs_demo.py`;
`dit_vis.py eval --real-bg --splat-wrist`; `sim/EXPERIMENTS.md` 7e).

Code: `capture/` (recording), `recon/` (scan -> assets, take -> keyframes, room -> splat),
`sim/` (environment, solver, data generation, rendering, policy, splat compositing),
`archive/` (demoted but kept as evidence, indexed in `archive/README.md`).

## Conclusions, statements and future work

**Statement.** I found this challenge on 13 October, four days before the deadline, and
chose the route that could reach a complete, measured result in that time: a digital twin,
a scripted solver, an imitation policy, and a thorough evaluation of each. The world-model
route the brief invites -- the collected video as the training data, the simulator only as
the exam -- is the one I would take with the full two weeks; its design, and why this
repo's own results point to it, is below and in `FUTURE_DIRECTIONS.md` 10. After the
challenge I will continue this project and carry out those world-model experiments at
[github.com/HanshangZhu/V-JEPA_kitchen](https://github.com/HanshangZhu/V-JEPA_kitchen) --
check it if you are interested. Thank you for the challenge; it was a good four days.

**What the experiments say, in short.** Steps, model size and demonstration count are not
what limits this policy (~80 %, all n.s.); one random scene per demonstration costs ~20
points; in the real-looking room the two cameras disagreeing cost 21 points and the room
itself ~6 (n.s.); a 7 mm band of lid colour the asset had wrong costs 9; and wherever a
scripted oracle exists, human motion used as action labels loses to it (59 % vs 88 % at
matched demo counts). Numbers and tests: Full results below and `sim/EXPERIMENTS.md`.

**What they point to.** The last two results are the ones that matter, and together they
say the human data was used on the wrong side of the pipeline.

Used as *action labels* -- retargeted and imitated -- a human hand trajectory is noise
relative to a solver that already produces perfect actions for this task, in this region,
for free. That is structural, not a tuning problem: the hand is not a gripper, the cavity
hides it, and the simulator is a better demonstrator of robot actions than any human can
be. This repo measured that loss and could not have avoided it by any amount of retargeting.

Used as *video*, the same takes carry exactly what the simulator cannot produce: how a real
door, a real pot and a real room look and move under manipulation. No action labels are
needed to learn that -- it is what the video head of a world model learns -- and the
measured sensitivities of the policy here (to the room, to a 7 mm band of print, to the two
cameras disagreeing) are sensitivities a model trained on real video need not have.

So the design this project should have had, and the one I would build with the time:

1. **Scale ego-motion video.** Head- or chest-mounted RGB(-D), hundreds of takes rather
   than three, many objects, many rooms; the wrist pose tracked as the action proxy. It is
   the cheapest data in robotics to collect, and the only kind that is unlimited.
2. **Train the video head of a world-action model on it** (V-JEPA 2-style predictor, or the
   video branch of a WAM such as LingBot-VA): the model learns scene dynamics and
   appearance from human video alone. Where the data are few, the object-centric variant
   (`FUTURE_DIRECTIONS.md` 10: predict pot pose and door angle from wrist motion) is the
   minimal instance of the same idea and fits three takes. This is where the project
   continues: [V-JEPA_kitchen](https://github.com/HanshangZhu/V-JEPA_kitchen).
3. **Align the action head separately, with the robot's own embodiment**: SFT on simulated
   or teleoperated robot demonstrations, then RL in the twin -- where success is measurable
   and the scripted oracle is worth something as a critic and a reward, not as a
   demonstrator. The human never labels robot actions; the robot's actions are learned
   where they can be scored.
4. **Use the simulator as the exam, not the data source.** The twin and the splatted room
   are where a policy is tested across starts, rooms and bodies; the real world is where
   the model learns what the world looks like. This inverts the pipeline in this repo, in
   which simulation produced the data and the real room appeared only as a backdrop.
5. **Then change the body.** With the dynamics in the video head and the actions aligned
   per embodiment, the Panda -> G1 step is an action-head problem, not a data problem:
   a whole-body G1 in Isaac Lab on an existing whole-body controller
   (`FUTURE_DIRECTIONS.md` 9).

What this repo contributes to that plan is the part that is hard to skip: a registered
digital twin, per-frame object extraction (the object-centric state the world model would
predict), a splatted room that closes the visual gap, the finding that the cameras must
agree, and a measured demonstration of why the alternative -- human motion as labels --
does not work. None of the plan above is tested here; it stands on a negative result and
on the brief's own weighting of world modelling over policy performance. That is where I
would put the two weeks, and where the work goes next: [V-JEPA_kitchen](https://github.com/HanshangZhu/V-JEPA_kitchen).

## Design choices, and why

**The human video gives task keyframes, not motion.** Hand tracking (MediaPipe + depth) is
good while the hand reaches and carries, but the hand is invisible inside the dark cavity,
exactly where the place happens, and it is not a parallel gripper. What the video gives
reliably is what happened to the *objects*: the door angle per frame (fitted from depth
against the known door), where the pot was picked and put down, and the order.

**Measured, not reconstructed, geometry.** A microwave is a box with a door; a tape
measure takes ten minutes. The scan supplies *where* it is and *what colour* it is.
Reconstructing it failed usefully: round a box, Pi3X confuses back and front views and the
fused cloud grows a second door (`recon/ASSET_ISSUES.md`). Each metric RealSense frame is
registered to the known model instead; colours are baked on with ray-cast visibility.

**No markers.** The camera pose of each take comes from registering the take's last seconds
(scene at rest) to the model; a knocked camera is detected from the background's image
shift and the earlier segment re-registered.

**A solver, not a motion planner.** Every motion is IK along a dense Cartesian path,
executed in physics, so a run either works physically or shows where it fails. The door is
carried round its hinge 1 deg at a time and shut with up to three regrasps.

**DART noise.** With clean solver data both state policies leave the demonstrations and
never come back (0 %). Re-solving with smooth random offsets on the executed joint targets
while recording the clean ones gives recovery examples: 0 -> 65 %.

**Training length was the bug, found by a control.** Three camera policies (small CNN,
ResNet-18, frozen DINOv2) all failed at 12 k steps; so did a control fed the *privileged
state* through the same pipeline, which ruled out vision. 4x the steps (+ EMA) gave v4,
5 -> 97 %. The same happened again with randomisation (v6 65 % -> v6-long 83 % at 2x steps).

**Domain randomisation, DoorMan-style.** v4 learned the scene: 97 -> 0 % with the scanned
kitchen as a backdrop. v6 renders every demo in two random scenes (`sim/dr.py`: 40 Poly
Haven environment maps as background and light, random floor / walls / lights, tinted
objects, a jittered camera) with colour jitter, blur and random erasing in training. The
kitchen is never a training scene.

## Full results

**v6-long across backgrounds and lid textures.** "Orange" is the lid texture it was trained
with; "white" is the corrected asset in this repo and in the GIFs.

| background seen by the cameras | lid | success | paired against |
|---|---|---|---|
| grey floor | orange | 83 % | -- |
| scanned mesh of the room, uncoloured (noisy) | orange | 81 % | grey: 14 / 12, p = 0.85 |
| splat, front camera only; wrist sees the grey floor | orange | 62 % | grey: 34 / 8, p = 0.0001 |
| splat, both cameras | orange | 77 % | grey: 21 / 13, p = 0.23 |
| grey floor | white | 74 % | orange grey: 19 / 8, p = 0.05 |
| splat, both cameras | white | 68 % | orange splat: 18 / 7, p = 0.04 |
| grey floor / splat, both -- fine-tuned 12 k steps on the corrected pot | white | 81 / 72 % | white: 21 / 12, p = 0.16; 16 / 11, p = 0.44 |

**How the policy got here** (orange lid, grey floor / scanned mesh):

| policy | input | grey | mesh |
|---|---|---|---|
| DiT + action chunking, clean data | privileged state | 0 % (n = 20) | -- |
| + DART | privileged state | 70 % | -- |
| v4: ResNet-18 + DiT, one clean scene, 49 k steps | 2 images + 25 robot numbers | **97 %** | **0 %** |
| v6: + domain randomisation, 2 scenes per demo, 49 k steps | same | 65 % | 70 % |
| **v6-long: 102 k steps (submitted)** | same | **83 %** | **81 %** |

**Ablations on v6-long** (grey / mesh, paired with 83 / 81 %): 3x steps 80 / 72 %; 7x larger
transformer 80 / 77 %; 2.1x demos 82 / 80 % -- all n.s. One scene per demo: 64 / 58 %
(96 demos, p <= 0.001), 38 / 45 % (205 demos).

**Human paths as solver constraints** (`sim/human_paths.py`, trained as v4, same starts):
solver vs human approach + carry, 86 demos, 88 % vs 59 % (p < 0.0001); solver vs human carry
only, 81 demos, 78 % vs 66 % (p = 0.03); solver data at 71 / 81 / 94 demos, 48 / 78 / 90 %.
Smoothed human paths: 59 %; one take only: 65 %.

## Appendix

### Pipeline stage outcomes

| stage | what | result |
|---|---|---|
| capture | 3 takes x 40 s, RealSense D435i, fixed camera, no markers | 0 depth frames dropped |
| twin | microwave: tape-measured URDF + scanned colours; pot: measured mesh | per-frame registration 2.0-2.2 mm rmse, 79 / 80 frames kept |
| extraction | per take: door angle per frame, pot pick / place, hand | all 3 takes (`recon/DEMOS.md`) |
| solver | each take reproduced by the Panda in physics | 3 / 3 |
| data | new pot / place positions in the region the takes span | 58 / 60 clean; 34 / 80 at DART sigma 0.02 rad; 4 / 40 at 0.04 = 96 demos |

### What did not work, or only partly

- **Day 1 (archived):** a finger pressing a bottle cap -> RoboTwin `click_bell` ->
  post-training LingBot-VLA-2 made it *worse* (75 -> 50 %, p = 0.0003); a BF16 time grid
  in its sampler cost 18 points, reported upstream
  ([Robbyant/lingbot-vla-v2#59](https://github.com/Robbyant/lingbot-vla-v2/pull/59)).
- **Residual RL (PPO) on the state policy:** +5.4 points at best, not significant; the same
  checkpoint scores 70 % in PhysX CPU and 35 % in PhysX GPU, so everything stays in CPU
  physics.
- **DAgger with the solver as expert:** 69.5 -> 15.6 %; the "restart" labels contradict the
  demos near the pot.
- **The scanned mesh as a backdrop:** a noisy reconstruction (speckled near-white table,
  chairs as scattered points, partly my own cropping; `EXPERIMENTS.md` 7d); superseded by the
  splat. Its 81 % tested robustness to an unseen noisy background, not to the room.
- **NVIDIA Newton** (`archive/newton_spike/`): reproduces the solver at a 0.002 s timestep,
  not at ManiSkill's 0.01 s; five importer traps; not adopted (`EXPERIMENTS.md` 10).

### The humanoid run

`sim/g1_spike/`, `EXPERIMENTS.md` 9: left hand hooks the handle and opens the door to 76 deg,
right hand carries the pot onto the turntable, left hand shuts the door. ManiSkill's
upper-body G1 has no legs and its pelvis sits at counter height, so re-filmed in the splat
(`sim/g1_spike/replay_splat.py`: recorded states replayed, the part below the counter hidden
by the table) it looks mounted at the table. Its waist does what legs would do: with the
pelvis fixed and 5-DoF arms, the four hand tasks (hook at 0 and 60 deg, pick, place) are all
reachable only if the waist may swing +-45 deg (`stance_scan.py`); at +-7, 15, 25 or 35 deg
no stance in the grid reaches them all, and the run above used -107 to +38 deg.

### Limitations

- Simulation only. One task, one microwave, one pot, one training seed per variant.
- The pot's start region is the one the takes span (10 x 14 cm); v4 falls to 47 % within
  2.5 cm beyond its edge and 0 % beyond 5 cm; with the pot on top of the microwave the door
  is never opened. v6-long was not tested outside it.
- The splatted kitchen is the room's look, not a real image: the robot, microwave and pot
  are simulator renders over it, without shadows on the real table; the splat is blurry
  close up and where the sweep did not look; the ceiling is a flat grey.
- The object's appearance was never randomised (conclusion 4).
- Hinge friction / damping and the pot's mass are set, not measured.
- Rollouts are not bit-identical across machines (x86 vs aarch64 PhysX); every comparison is
  paired on one machine. Offline validation error does not predict closed-loop success.

### Running it

[`REPRODUCE.md`](REPRODUCE.md): environments, the released demonstrations and checkpoints
([Release v1.0](https://github.com/HanshangZhu/microwave-from-human-video/releases/tag/v1.0)),
a 40-minute route that re-runs the headline evaluation from the released policy, and the
full route from the captures. The raw recordings of my home (takes, room sweep, splat) are
not published; for those, write to zcabhz4@ucl.ac.uk.

### Where to look

- `REPRODUCE.md` -- how to run it, quick and full.
- `sim/EXPERIMENTS.md` -- every data / policy / RL run with its command and numbers.
- `docs/MICROWAVE_LOG.md` -- the same story in short, as it happened.
- `recon/DEMOS.md` -- what was extracted from each take, and how.
- `assets/microwave/README.md`, `recon/ASSET_ISSUES.md` -- the assets: measured vs guessed,
  and the pipeline's problems as they were found and fixed.
- `FUTURE_DIRECTIONS.md` -- when human video is needed at all, a wrist IMU for the occluded
  part, ego video as a video-prediction target, the whole-body G1 in Isaac Lab, the
  world-model design this should have had.
- [V-JEPA_kitchen](https://github.com/HanshangZhu/V-JEPA_kitchen) -- the continuation: the world-model experiments, starting after the challenge.
- `docs/viz/` -- step-by-step visualisation of solver vs human-guided demonstrations.
- `archive/` -- demoted code kept as evidence (state-policy test bed, mesh backdrop, Newton,
  day 1), indexed in `archive/README.md`.
