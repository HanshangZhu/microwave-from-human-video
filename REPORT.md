# Report: results, design choices, what did not work

The long form behind the [README](README.md). Every number is 128 closed-loop episodes from
the same 128 start states unless marked; paired McNemar tests and every command are in
[`sim/EXPERIMENTS.md`](sim/EXPERIMENTS.md); how to run anything is in
[`REPRODUCE.md`](REPRODUCE.md).

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

## Pipeline stage outcomes

| stage | what | result |
|---|---|---|
| capture | 3 takes x 40 s, RealSense D435i, fixed camera, no markers | 0 depth frames dropped |
| twin | microwave: tape-measured URDF + scanned colours; pot: measured mesh | per-frame registration 2.0-2.2 mm rmse, 79 / 80 frames kept |
| extraction | per take: door angle per frame, pot pick / place, hand | all 3 takes (`recon/DEMOS.md`) |
| solver | each take reproduced by the Panda in physics | 3 / 3 |
| data | new pot / place positions in the region the takes span | 58 / 60 clean; 34 / 80 at DART sigma 0.02 rad; 4 / 40 at 0.04 = 96 demos |

## What did not work, or only partly

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

## The humanoid run

`sim/g1_spike/`, `EXPERIMENTS.md` 9: left hand hooks the handle and opens the door to 76 deg,
right hand carries the pot onto the turntable, left hand shuts the door. ManiSkill's
upper-body G1 has no legs and its pelvis sits at counter height, so re-filmed in the splat
(`sim/g1_spike/replay_splat.py`: recorded states replayed, the part below the counter hidden
by the table) it looks mounted at the table. Its waist does what legs would do: with the
pelvis fixed and 5-DoF arms, the four hand tasks (hook at 0 and 60 deg, pick, place) are all
reachable only if the waist may swing +-45 deg (`stance_scan.py`); at +-7, 15, 25 or 35 deg
no stance in the grid reaches them all, and the run above used -107 to +38 deg.

## Limitations

- Simulation only. One task, one microwave, one pot, one training seed per variant.
- The pot's start region is the one the takes span (10 x 14 cm); v4 falls to 47 % within
  2.5 cm beyond its edge and 0 % beyond 5 cm; with the pot on top of the microwave the door
  is never opened. v6-long was not tested outside it.
- The splatted kitchen is the room's look, not a real image: the robot, microwave and pot
  are simulator renders over it, without shadows on the real table; the splat is blurry
  close up and where the sweep did not look; the ceiling is a flat grey.
- The object's appearance was never randomised (README, conclusion 4).
- Hinge friction / damping and the pot's mass are set, not measured.
- Rollouts are not bit-identical across machines (x86 vs aarch64 PhysX); every comparison is
  paired on one machine. Offline validation error does not predict closed-loop success.
