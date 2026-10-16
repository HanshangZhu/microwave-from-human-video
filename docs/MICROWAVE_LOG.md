# Microwave line: what was done, what was found (14 Oct afternoon -> 15 Oct 06:00 BST)

Context: Humanoid (London) robot-learning internship challenge -- collect real-world
manipulation data, use it to drive a robot manipulator in simulation, public repo + README.
Deadline 16 Oct 23:59 BST. Day 1 (13 Oct, finger press -> RoboTwin -> LingBot-VLA-2) is
archived; this is day 2 onwards.

Task: **open the microwave door by its bar handle, pick a Heinz Snap Pot off the counter,
put it on the turntable, shut the door.** Robot: Franka Panda. Simulator: ManiSkill3 /
SAPIEN 3.0.3, PhysX **CPU** backend, 100 Hz physics, 20 Hz joint-position control, episodes
up to 3000 control steps. Success = pot inside on the turntable AND door < 5 deg.

Detailed per-experiment numbers: `sim/EXPERIMENTS.md`. Asset problems: `recon/ASSET_ISSUES.md`.
Demo extraction: `recon/DEMOS.md`. Ideas not done: `FUTURE_DIRECTIONS.md`.

---

## 1. Digital twin (done)

- **Microwave** = tape-measured articulated URDF (`assets/microwave/build_microwave.py`):
  26 parameters, 17 measured, 2 from a photo, 7 guessed (door mass, hinge damping/friction
  0.05 -- kept light on purpose, turntable diameter, max door angle ...).
- **Colour** baked from a RealSense D435i scan onto the measured geometry, from per-frame
  registered camera poses (2.0-2.2 mm rmse, 79/80 frames kept); front panel from a
  straight-on photo through a homography.
- Things that went wrong and were fixed: fused Pi3X clouds of a box scanned all the way
  round grew a second door on the back (front/back confusion) -> geometry from the tape,
  scan only for pose + colour; the first dimensions disagreed with single depth frames
  (height 279 vs 268 mm, depth incl. door) -> re-measured; a 180 deg registration flip that
  scored as well as the right one -> 360 deg yaw search + visual overlay check.
- **Pot**: measured mesh (`assets/heinz_snap_pot/`), ~5 mm mean
  Chamfer vs its scan, mass 0.25 kg (guess), friction 1.0 / 0.8.

## 2. Human demonstrations (done)

- 3 takes x 40 s, one fixed RealSense at the end of the table, no markers. Take 2 is
  two-handed.
- Extracted from depth + RGB, all in the microwave frame: camera pose per take (registered
  still end; knocked camera detected and re-registered), door angle per frame (fitted to the
  known door), pot start / pick / place (lid colour + depth), hand keypoints (MediaPipe +
  depth -- good outside the cavity, lost inside it).
- Replay videos (recorded frame | simulation from the same camera) confirm alignment.
- **Decision: the video supplies task keyframes (door, pick, place, order), not motion.**
  The hand is invisible exactly where the place happens, and a hand is not a gripper.

## 3. Robot demonstrations by a solver (done)

- `sim/solve_microwave.py`: IK continuation along dense Cartesian paths (mplib segfaults
  with numpy 2), executed in physics -- open the door 1 deg at a time, pick+carry+insert
  as one continuous joint path, shut with up to three regrasps. Refactored into a `Solver`
  class (verified bit-identical) so it can also take over mid-episode.
- 3/3 takes reproduced; 58/60 new start states from the takes' region (97 %).
- **DART**: smooth random offsets on the executed joint targets, clean planned targets
  recorded; 34/80 kept at sigma 0.02 rad, 4/40 at 0.04.

## 4. Policies on privileged state (done)

State = joints (pos + vel), tcp pose, **pot pose, door angle** (33-d).

| policy | success (n=20 / 128) |
|---|---|
| MLP + action chunking | 0 % |
| DiT + action chunking, clean data | 0 % (15 % grasp) |
| DiT + chunking, clean + DART | 65 % (n=20); **69.5 %** (n=128) |

DART (recovery data) was the decisive change. Base failures (128 episodes): door 100 %,
lifted 80 %, inside 72 %, success 70 %. Of 26 not-lifted: ~17 pushed the pot 3-17 cm
along the counter, ~7 knocked it over; failures somewhat more at the edges of the region.

**CPU vs GPU physics**: the same checkpoint scores 70 % in PhysX CPU, 35 % in PhysX GPU
(n=128 each). Everything since stays in CPU physics.

## 5. Residual RL (stopped; a documented negative result)

Frozen DiT + Gaussian residual (<= 0.03 rad per joint target), PPO, stage-aware dense
reward (stage bonuses + progress to the current stage's goal), optional style terms.

| run | outcome |
|---|---|
| GPU physics | 40 -> 23 % (exploration noise and reward balance; physics gap) |
| CPU, stage 1 (task reward) | best 73.4 % at 1 M vs base 68.0 % on the same 128 starts -- +5.4, z ~ 0.95, n.s.; declines after |
| CPU, stage 2 (+ time / acceleration / stop penalties) | 70 -> 48 %; stops fell, acceleration rose |
| CPU, gamma 0.9998 | every checkpoint below base (53-63 %) |

Diagnosis, each hypothesis tested:
- *Reward hacking of a dense term* (human's hypothesis): the progress term telescopes
  within a stage; under stage 1 the discounted objective PPO maximises did rise while
  success fell, but stage 2 did not even raise its own objective.
- *Discount too short* (agent's hypothesis): refuted by the gamma 0.9998 run.
- *Policy adapts to its own exploration noise*: refuted by evaluating with noise.
- What holds: no checkpoint improves measurably by any measure (3 of 14 above base, all
  within noise; the best is a max-of-many). Suspected: ~2200-step episodes, ~1400
  episodes per 3 M steps, ~3 finished episodes per PPO update, a per-step residual with
  tiny effect next to outcome variance, and the gripper not under the residual's control.
- Throughput fix along the way: several environments per worker process (a SAPIEN
  process is ~1 GB; one per env swapped the machine): 4.8 k -> 12 k steps/s at 256 envs.

## 6. DAgger with the solver as privileged expert (one round; negative so far)

`sim/dagger.py`: run the student, hand the episode to the solver at 5 branch times between
turning to the pot and the lift, keep successful continuations. Expert options tried most
local first, each from a deterministic replay (drift 0.00 mm): hold / continue / regrasp /
restart. 96 starts x 5 = 480 hand-overs, 403 kept.

| retrain on gen + DART + ... | success (same 128 starts) |
|---|---|
| (base, none) | 69.5 % |
| all 403 hand-overs | **15.6 %** |
| only local corrections (149: hold / continue / regrasp) | **57.0 %** (paired vs base: 38 vs 22 discordant, p ~ 0.04) |

The "restart" labels (retreat to tuck, 254 of 403, 165 of them from episodes the student
would have finished) contradict the original demos near the pot. Even without them, one
round did not help. Unchecked suspects: the "continue" straight line approaches at angles
the demos never use; hand-over data outweighs the original demos after the grasp.

## 7. Camera-based student: works once trained long enough

Plan agreed: the privileged state policy is a test bed; the deployable policy sees only
images + the robot's own joints. Teacher for its data = the solver (97 %).

- **Cameras** (`sim/views.py`): *front* = the real RealSense pose from the human take's
  registration with its real intrinsics (projected microwave corners agree to 0.001 of
  the image); *wrist* = ManiSkill panda_wristcam mount.
- **Rendering** (`sim/render_views.py`): re-render every 4th step of every demo from the
  stored simulator states (no re-simulation; cameras and scenes can change later).
- **Model** (`sim/dit_vis.py`): image tokens + joint tokens + noisy action chunk, same DiT.
  No pot pose, no door angle.

What happened, in order (same 128 start states for every row):

| policy | input / encoder | gradient steps | door | lifted | inside | success |
|---|---|---|---|---|---|---|
| state DiT (`dit.py`) | privileged state | 37 k | 100 % | 80 % | 72 % | 70 % |
| v1 | small CNN, minimal shader | 12 k | 84 % | 25 % | 4 % | 1 % |
| v2 | ResNet-18 (ImageNet), hq renders | 12 k | 87 % | 21 % | 5 % | 5 % |
| v3 | DINOv2 ViT-S/14 frozen, all patch tokens | 12 k | 88 % | 10 % | 1 % | 0 % |
| control 1 | **privileged state** through `dit_vis.py` | 12 k | 49 % | 6 % | 0 % | 0 % |
| (b) | privileged state through `dit_vis.py` | **49 k** | 100 % | 79 % | 40 % | 38 % |
| **v4** | **v2 + EMA** | **49 k** | **100 %** | **100 %** | **97 %** | **97 %** |

- Three encoders failing alike, then the control with privileged state failing too, showed
  the problem was not vision: a pipeline control should have come first. Single-factor
  controls (every step instead of every 4th: no; fp32: no; 4x training: yes) pinned it on
  **training length**. Offline validation error never separated the variants.
- v4's checkpoint takes 25 robot numbers + 2 images, nothing privileged. Encoder,
  augmentation and DART were held constant from v2 to v4; only training length (+EMA)
  changed -- so only that is established as the cause.
- Jitter seen in the videos: chunk-to-chunk inconsistency (consecutive chunks disagree on
  their overlap 0.91 deg for v2 vs 0.35 for the state policy); inference fixes (fixed
  noise, ACT temporal ensembling) implemented, not yet evaluated on v4.

**A new background breaks it.** The scanned kitchen as a visual-only backdrop, physics
unchanged: v4 drops from 97 % to **0 %** (door 62 %, lifted 27 %). It learned the scene.
DoorMan (arXiv 2512.01061) shows the same failure and the fix by ablation: no
randomisation 5-20 %, solid colours 66-70 %, + real dome-light environments +15-30 %.
v6 (running): DoorMan-style -- real indoor HDRI environments (40 from Poly Haven) as
background and light, random floors/walls, random lights, object tints, camera jitter,
two random scenes per demo, colour/blur/erasing augmentation; the kitchen held out.

## 8. Human-guided robot demonstrations (the human video now sets *how*, not only *what*)

Until here the video set only the task: pick/place points, door angle, order; every robot
motion was the solver's. Direct retargeting fails here (hand tracked 100 % on the approach,
62-94 % on the carry, 0-29 % inside the cavity; 1-2 cm wrist error vs ~12 mm grasp margin;
a wrap is not a pinch). **Partial retargeting** (`sim/human_paths.py`, as EgoEngine /
X-Sim): the solver follows the wrist's tracked positions for the approach and the carry,
keeps its own hand orientation, aligns the last 10 cm to the pot's true pose and plans the
unseen insertion. On take_000: carry 0.76 m in 5.7 s at <= 16.7 cm, vs the solver's 1.01 m
in 6.8 s up to 25 cm. All three takes succeed. 86 human-guided demos generated with the
same samples as the 96 solver-only ones. **A/B running**: the v4 recipe on each, gradient
steps matched; success and motion (steps, path, height, stops).

## 9. Another body: Unitree G1 (scripted)

ManiSkill's G1 upper body solves the task from take_000's keyframes (left hand opens and
shuts the door, right hand carries the pot; success), but narrowly: door 60-75 deg, pot
yaw 11-21 deg; arms have 5 DoF (no wrist); no collision checking. The object-centric task
representation transfers; the solver needs work (est. 1-2 weeks) before a learned G1 policy.

## 10. NVIDIA Newton (time-boxed check; not adopted)

Replaying take_000's solver trajectory open-loop in Newton 1.0 (MuJoCo-Warp): fails at
ManiSkill's 0.01 s timestep (fingers pass through the 9 mm handle; door 25 deg). At 0.002 s
it succeeds, tracking ManiSkill until the insertion (pot error jumps to 7.7 cm); 30/32
identical replicas succeed (GPU non-determinism). At that timestep ~10 k steps/s on the
5090, about the CPU pool's rate. Port traps: dropped joint friction, extra mimic
constraint, convex decomposition, friction combine rule.

---

## Open questions worth an outside look

1. Why does the camera policy beat the privileged-state policy through the same pipeline
   (97 vs 38 %)? Ruled out as the main cause: missing relative geometry (+ tcp - pot,
   tcp - handle: 42 %, p = 0.61), near-constant dimensions (48 %, p = 0.16). Precedent:
   arXiv 2509.18644 (state input -> overfitting to training trajectories; images only
   generalise far better), causal confusion (de Haan 2019), diffusion policy as a lookup
   table with few demos (2505.05787). Testing: robot state only (H4), no door angle (H5).
2. Randomisation (v6, DoorMan-style, hq cameras): kitchen 0 -> 70 %, grey 97 -> 65 % at the
   same 49.5 k steps. Does a ViT with all patch tokens (v3 long) do better?
3. Human-guided data: B 59 % vs A 97 % (carry 8.6 vs 9.2 m over the episode). The human
   sets only 2-9 % of B's steps, the solver's clamps reshape the approach by 1-5 cm, and
   the two-handed take_002 doubles the carry. Control at B's demo count running.
4. Which of v4's ingredients matter at full training (encoder, DART, EMA, augmentation)?
5. Why did intervention-style DAgger hurt the state policy, even with local corrections?

## What is solid enough for the submission

human video -> object keyframes (+ the hand's paths) -> measured twin -> solver (97 %) ->
DART data -> camera policy at **97 %** in simulation with no privileged input, with the
diagnosis that got it there (training length, found by controls); the CPU/GPU physics gap;
a diagnosed negative RL result; a calibrated real-camera rendering pipeline; the honest
limits: one clean scene (0 % in the kitchen without randomisation), simulation only.
