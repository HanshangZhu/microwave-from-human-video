# Simulation experiments: log (last update 2026-10-16 18:30)

Everything after the human takes: robot demonstrations, policies, RL. One section per
experiment, newest results last. RL is **not finished**; its sections record runs and
their numbers, not conclusions about what RL can do here.

Outputs are under `Gitignore-but-important/sim/<run>/` (not in the repo). Environment
`maniskill` (mani_skill 3.0.1, sapien 3.0.3, torch 2.11), RTX 5090, `PYTHONNOUSERSITE=1`.

## Task and evaluation

`MicrowavePot-v0` (`microwave_env.py`): Panda on the counter, measured microwave
(`assets/microwave/microwave_textured.urdf`), Snap Pot (`assets/heinz_snap_pot`). The pot
starts anywhere in the pick region the three takes span, `POT_REGION = (0.421, -0.299)
.. (0.524, -0.153)` in the microwave frame. 3000 control steps per episode.

Stages, each counted once per episode: **door** opened > 60 deg; **lifted** pot > 5 cm
above the counter; **inside** pot in the cavity on the turntable; **success** inside and
door < 5 deg.

Over the successful episodes only (`rl_residual.summarize`):

| metric | definition |
|---|---|
| steps | control steps until success |
| path | tcp path length, m |
| stops | times the tcp stood still (< 1 mm/step for 10 steps) after having moved (> 3 mm/step) |
| acc | mean tcp acceleration, mm/step^2 (second difference of the tcp position) |

Two evaluators exist and differ slightly: `dit.py eval` (its own loop, seeds from 1000)
and `rl_cpu.py eval` (the RL loop, residual optional; `--seed` picks the 128 start
states). **Only rows with the same evaluator and seed are directly comparable.** The
same DiT on 128 starts: 69.5 % (`dit.py eval`), 63.3 % (`rl_cpu.py`, seed 0), 68.0 %
(`rl_cpu.py`, seed 5) -- that spread is the noise floor for n = 128 (about +-4 points,
one standard error).

## 1. Robot demonstrations (`solve_microwave.py`, `generate.py`)

| run | start states | DART sigma (rad) | kept / tried | yield |
|---|---|---|---|---|
| `take_00{0,1,2}` | the takes' own pick / place | 0 | 3 / 3 | |
| `gen` | sampled in the takes' region | 0 | 58 / 60 | 97 % |
| `gen_dart02` | sampled, seed 2 | 0.02 | 34 / 80 | 43 % |
| `gen_dart04` | sampled, seed 3 | 0.04 | 4 / 40 | 10 % |

DART: the executed joint targets get a smooth random offset (halved while the gripper
holds something), the recorded action is the clean planned target, and only runs that
still succeed are kept. So the data contains the way back onto the plan.

```bash
python sim/generate.py --n 60 --out .../gen --jobs 10
python sim/generate.py --n 80 --seed 2 --dart-noise 0.02 --jobs 14 --videos 0 --out .../gen_dart02
python sim/generate.py --n 40 --seed 3 --dart-noise 0.04 --jobs 14 --videos 0 --out .../gen_dart04
```

## 2. Imitation policies (`bc.py`, `dit.py`), CPU physics, `dit.py eval`, n = 20

| run | model | data | success | notes |
|---|---|---|---|---|
| `bc_v1` | MLP, chunk H 8 / exec 4 | gen | 0 % | leaves the demonstrations within ~350 steps (drift in the redundant DoF while holding the handle) |
| `dit_v1` | DiT, H 16 / E 8, 2 past states, DDIM 10 | gen | 0 % | 15 % grasp |
| `dit_dart_time` | DiT + elapsed-time input | gen + dart02 + dart04, `--squeeze` | 55 % | |
| `dit_dart_notime` | DiT, no time input | same | **65 %** | 80 % grasp; base for everything below |

`--squeeze` drops waiting steps (action unchanged and all joints still), keeping at most
2 of each run. time vs no-time (55 vs 65 %, n = 20) is within noise; no time input was
kept because it is the simpler model.

```bash
python sim/dit.py train --data .../gen,.../gen_dart02,.../gen_dart04 --squeeze --no-time --out .../dit_dart_notime --epochs 100
python sim/dit.py eval --ckpt .../dit_dart_notime/dit.pt --out .../dit_dart_notime/eval --n 20
```

## 3. CPU vs GPU physics, same checkpoint (`dit_dart_notime`), n = 128

| physics | evaluator | success | lifted | inside |
|---|---|---|---|---|
| CPU | `dit.py eval` | 69.5 % | 79.7 % | 71.9 % |
| CPU | `rl_cpu.py`, seed 0 | 63.3 % | 77.3 % | 68.0 % |
| GPU | `rl_residual.py`, seed 0 (`rl_base`) | **35.2 %** | 64.8 % | 60.2 % |

All demonstrations were generated in CPU physics. GPU physics (PhysX GPU, batched) loses
in the grasp and above all in the door shut (inside 60 % -> success 35 %). Not
investigated further: which contact parameters differ.

## 4. Residual RL in GPU physics (`rl_residual.py`) -- stopped

Frozen DiT + Gaussian residual (<= 0.03 rad per joint target), PPO, 128 GPU envs.
Success below is the running training-time rate (with exploration noise).

| run | init std | steps | success, early -> last |
|---|---|---|---|
| `rl_v1` | 0.36 | 0.5 M | 7 % in the first 128 episodes, against 35 % deterministic: the exploration noise alone breaks the task |
| `rl_v2` | 0.11 | 2.0 M | 40 % -> 23 % (the style terms outweighed the task terms in the reward) |

Stopped because RL here first has to make up the 35 vs 70 % physics gap.

## 5. Residual RL in CPU physics (`rl_cpu.py`)

24 CPU envs in worker processes, DiT + residual batched on the GPU, ~4.5 k steps/s.
Reward: bonus per stage reached + `w_prog` x progress towards the current stage's goal
- style terms (time, acceleration, stops) - `w_res` |residual|^2. gamma 0.999,
init log std -2.2. Checkpoints every 0.5 M steps.

Evaluation: `rl_cpu.py eval`, deterministic (residual mean, no noise), n = 128.

### 5a. `rl_cpu_v1` -- defaults (w_time 2e-4, w_jerk 5e-4, bonus 1 2 3 5), stopped at 2 M

| ckpt | seed | success | stops | acc |
|---|---|---|---|---|
| base | 0 | 63.3 % | 8.0 | 0.59 |
| 1.1 M | 0 | 60.4 % (n = 48) | 9.0 | 0.67 |
| 2.0 M | 0 | 60.9 % | 5.4 | 0.82 |

### 5b. Two stages, all evaluated on seed 5

Stage 1 (`rl_stage1`): task terms only (`--w-time 0 --w-jerk 0 --w-stop 0 --bonus 1 2 3 10
--lr 1e-4`, 3 M steps). Stage 2 (`rl_stage2`): from stage 1's 1013k checkpoint, with
`--w-time 5e-4 --w-jerk 2e-3 --w-stop 2e-3` (2 M steps).

| ckpt | success | lifted | inside | steps | path m | stops | acc |
|---|---|---|---|---|---|---|---|
| base (no residual) | 68.0 % | 82.8 % | 75.0 % | 2164 | 9.25 | 7.2 | 0.60 |
| stage 1, 1013k | **73.4 %** | 85.2 % | 80.5 % | 2221 | 9.34 | 8.1 | 0.69 |
| stage 1, 2027k | 70.3 % | 82.0 % | 74.2 % | 2188 | 9.56 | 6.5 | 0.74 |
| stage 1, 2519k | 65.6 % | 76.6 % | 70.3 % | 2115 | 9.52 | 5.4 | 0.78 |
| stage 1, 3004k | 64.1 % | 80.5 % | 69.5 % | 2122 | 9.69 | 4.6 | 0.87 |
| stage 2, 522k | 70.3 % | 83.6 % | 78.1 % | 2207 | 9.42 | 7.0 | 0.75 |
| stage 2, 1013k | 64.1 % | 78.1 % | 70.3 % | 2213 | 9.46 | 7.6 | 0.75 |
| stage 2, 1505k | 61.7 % | 79.7 % | 71.9 % | 2101 | 9.41 | 4.4 | 0.82 |
| stage 2, 2002k | 48.4 % | 68.8 % | 60.2 % | 2145 | 9.44 | 4.8 | 0.86 |

What these numbers do and do not show so far:

- Stage 1 at 1 M: +5.4 points over the base on the same 128 starts. One standard error
  is ~4 points; z ~ 0.95, **not significant**. The gain is in the grasp (lifted +2.4,
  inside +5.5).
- After ~1 M steps every run trades success for fewer stops -- stage 1 too, with no
  style reward at all. So stage 2 does not isolate the effect of the style terms.
- The style objective is not reached: stops fall, but acceleration rises and path length
  and steps barely move. The motion is not more direct.
- Open explanations, none tested yet: (a) the residual's authority (0.03 rad) is too
  small to change the path the DiT decides; (b) PPO drifts (value error, learning rate)
  once the success bonus is mostly collected; (c) the residual is chosen per step with nothing
  tying consecutive steps together, so a residual that helps the task can still add
  acceleration that the reward term is too weak to remove.

```bash
python sim/rl_cpu.py train --ckpt .../dit_dart_notime/dit.pt --out .../rl_stage1 --n-envs 24 --steps 3e6 \
    --lr 1e-4 --w-time 0 --w-jerk 0 --w-stop 0 --bonus 1 2 3 10
python sim/rl_cpu.py train --ckpt .../dit_dart_notime/dit.pt --res .../rl_stage1/residual_01013k.pt \
    --out .../rl_stage2 --n-envs 24 --steps 2e6 --lr 1e-4 --w-time 0.0005 --w-jerk 0.002 --w-stop 0.002 --bonus 1 2 3 10
python sim/rl_cpu.py eval --ckpt .../dit_dart_notime/dit.pt --res .../rl_stage1/residual_01013k.pt \
    --out .../eval --n 128 --n-envs 24 --seed 5
```

### 5c. Reward audit: does a worse checkpoint score a better return?

`rl_cpu.py eval` now also logs each episode's reward components, raw and discounted
(gamma^t from the episode start), so every checkpoint's objective can be recomputed under
either stage's weights. Re-running the evals reproduced every success rate above exactly
(the CPU rollouts are deterministic). n = 128, seed 5; J = discounted return from t = 0,
the quantity PPO maximises; SE of J ~0.12.

| ckpt | success | J (stage-1 reward) | J (stage-2 reward) | disc. success bonus | disc. door+lift+inside bonus | disc. progress | disc. style penalty |
|---|---|---|---|---|---|---|---|
| base | 68.0 % | 3.898 | 1.973 | 0.798 | 1.422 | 0.839 | 1.93 |
| stage 1, 1013k | 73.4 % | 3.888 | 1.826 | 0.815 | 1.423 | 0.833 | 2.06 |
| stage 1, 2027k | 70.3 % | 3.929 | 1.802 | 0.808 | 1.446 | 0.854 | 2.13 |
| stage 1, 3004k | 64.1 % | 3.965 | 1.617 | 0.777 | 1.500 | 0.876 | 2.35 |
| stage 2, 522k | 70.3 % | 3.955 | 1.778 | 0.790 | 1.483 | 0.854 | 2.18 |
| stage 2, 2002k | 48.4 % | 3.435 | 0.954 | 0.578 | 1.311 | 0.814 | 2.48 |

- **Stage 1 optimised its objective while success fell**: J rises 3.888 -> 3.929 -> 3.965
  as success falls 73 -> 70 -> 64 %. The rise comes from the early-stage bonuses (1.423 ->
  1.500) and progress (0.833 -> 0.876); the discounted success bonus falls (0.815 -> 0.777).
  The explanation proposed at the time (**refuted by 5d**): the discount, not farming of
  the progress term (which telescopes within a
  stage): with gamma = 0.999 the +10 success bonus at step ~2160 is worth 10 x 0.115, about
  a fifth of the return, while the door / lift / inside bonuses come earlier and keep
  ~0.3 of their value on average (1.42 of 4.9 undiscounted). Reaching the early stages sooner pays more than finishing more often.
  Caveat: each step in J (~0.04) is below one SE (0.12); the evidence is the monotone trend
  over three checkpoints with consistent components, not a single significant difference.
- **Stage 2 did not optimise its objective**: J falls 1.826 -> 1.778 -> 0.954, and the
  style penalty it was trained to reduce grows (2.06 -> 2.48). That is an optimisation
  failure, not reward hacking. One candidate: training runs with exploration noise, under
  which the tcp acceleration is ~1.5 mm/step^2 regardless of the mean (training logs), so
  the gradient on the mean's smoothness is buried in noise; the deterministic eval sees
  0.7-0.9.

### 5d. Stage 1 with gamma 0.9998 (`rl_stage1_g9998`) -- the discount explanation fails

Only change from `rl_stage1`: `--gamma 0.9998` (the success bonus at step ~2160 keeps
0.65 of its value instead of 0.115). Deterministic, n = 128, seed 5:

| ckpt | 522k | 1013k | 1505k | 2027k | 2519k | 3004k |
|---|---|---|---|---|---|---|
| success | 62.5 % | 60.2 % | 59.4 % | 58.6 % | 63.3 % | 53.1 % |
| lifted | 75.0 % | 78.1 % | 73.4 % | 72.7 % | 77.3 % | 69.5 % |

Every checkpoint is below the base (68.0 %). If the short horizon had been the main cause
of 5c's decline, a longer one should have helped; it did not. Across all CPU RL
checkpoints evaluated on seed 5 (stage 1, stage 2, this run: 14), 3 are above the base,
all within 1.5 SE -- and the best of them (73.4 %) was picked as the maximum of several,
so it is biased upwards.

Training-time success (with exploration noise, last 72 episodes per log line) seemed to
tell a different story -- 52 % at the start, 65 % in the last 0.5 M -- suggesting that PPO
improves success *under its own noise* while the noise-free policy gets worse. Tested
directly (5e): it does not.

### 5e. The same checkpoints evaluated with the exploration noise (`rl_cpu.py eval --stochastic`)

n = 128, seed 5, residual sampled as in training (base = a zero residual plus the noise).

| policy | without noise | with noise |
|---|---|---|
| base | 68.0 % | 58.6 % |
| stage 1, 1013k | 73.4 % | 60.2 % |
| stage 1, 3004k | 64.1 % | 56.2 % |
| gamma 0.9998, 3004k | 53.1 % | 56.2 % |

The noise costs ~10 points. Under noise, no checkpoint is better than the base either; the
65 % training-time figure was a lucky window (72 episodes: one SE is ~6 points).

**Where this leaves the RL so far.** None of the three explanations tested holds up as the
main cause: reward hacking of a dense term (5c: the progress term telescopes; stage 2 did
not even raise its own objective), the short discount (5d), adaptation to the exploration
noise (5e). What the data do show consistently is that residual PPO here has not improved
success measurably by any measure, and that success slowly declines with training. The
next suspect is the sample budget against the horizon, which was never checked: episodes
are ~2200 control steps, so 3 M steps are ~1400 episodes, and one PPO update (24 envs x 256
steps) contains ~3 finished episodes -- i.e. ~2 success bonuses per gradient step, with a
per-step residual whose effect on any outcome is tiny next to that outcome's variance.

### 5f. Throughput: more environments

One CPU environment step takes ~0.5 ms, but a worker process with torch + SAPIEN loaded
takes ~0.8-1.1 GB, so one process per environment ran the 46 GB machine into swap at 24-64
environments while the cores stayed half idle. `rl_cpu.py --envs-per-proc k` now runs k
environments per process (and no renderer, `render_backend="none"`); the base eval with
k = 4 reproduces 68.0 % exactly (same start states).

| envs | per process | steps/s | memory used |
|---|---|---|---|
| 24 | 1 | 4.8 k | swapping |
| 64 | 1 | 7.6 k | swapping |
| 64 | 2 | 7.7 k | 29.6 GB |
| 128 | 4 | 9.9 k | 29.5 GB |
| 192 | 6 | 11.2 k | 29.7 GB |
| 256 | 8 | 12.0 k | 29.8 GB |
| 256 | 16 | 10.0 k | 23.4 GB |

At 256 / 8, half the loop time is the lockstep wait for the workers; the DiT takes 7 %.
What more environments change: one PPO update (256 x 256 steps) now contains ~30 finished
episodes instead of ~3, and 3 M steps take ~4 min instead of ~11. What they do not change:
episodes per million steps (~450, fixed by the ~2200-step episode).

## 6. DAgger with the solver as the privileged expert (`dagger.py`)

Why: the residual RL never moved success measurably (5c-5e), and the base policy's failures
are contact failures with the state fully known -- of 26 not-lifted episodes in 128, ~17
pushed the pot 3-17 cm along the counter, ~7 knocked it over. Privileged state solves
perception, not imitation error: the solver (same physics, same information) succeeds in
97 %, the DiT learned from it in 70 %. The fix for imitation error is labels on the states
the student actually visits.

How (`sim/dagger.py`): the student runs from a start state; at K branch times between
turning to the pot and the lift (three of five in the last 15 % of that window), the
episode is handed to the solver (`Solver`, refactored out of `solve_microwave.py`; take_000
still reproduces bit for bit), which finishes the task from there. The solver's options
are tried most local first, each from a fresh deterministic replay (drift 0.00 mm):
hold (lift what the hand has closed on), continue (straight in from here), regrasp (open,
back off 8 cm, in again), restart (open, tuck, the whole pick). Only successful
continuations are kept. Start states use seed 7, apart from both eval sets.

Round 1 (student `dit_dart_notime`, 96 starts x 5 branches):

| outcome | n |
|---|---|
| kept: restart / continue / hold / regrasp | 254 / 81 / 54 / 14 (403 of 480) |
| pot knocked over before the hand-over | 26 |
| no option succeeded | 51 |

A first observation from the smoke test: one failed student episode had closed the hand
7 mm off-centre with the pot tilted 3.8 deg -- and "hold" from that state succeeded. The
grasp was liftable; the student did not lift it.

Round 1 result -- **worse**: the DiT retrained on gen + DART + all 403 hand-overs drops from
69.5 % to 15.6 % (same 128 starts, paired; lifted 80 -> 34 %; door 100 % both). Likely
cause, my expert design: 254 of the 403 kept continuations are "restart" (retreat to the
tuck, start the pick again), 165 of them from episodes the student would have finished.
Near the pot the data then says "approach" (original demos) and "retreat" (DAgger) for
similar states. Retrained with only the local corrections (hold / continue / regrasp, 149
runs): **57.0 %** -- most of the drop was the restart labels, but it is still below the
base 69.5 % (paired: 38 episodes only the base solves, 22 only the new one; McNemar
p ~ 0.04). So one round of this intervention-style DAgger did not help the state policy.
Untested suspects: the "continue" line from wherever the hand is approaches the pot at
angles and heights the original demos never use (a second mode for the same states), and
hand-over data (149 runs) now outweighs the original demos (96) after the grasp.

## 7. Camera views for a camera-based student (`views.py`, `render_views.py`)

Every demonstration stores the simulator state behind each observation (generator runs:
`solve.h5`; DAgger runs now: `demo.npz` `states`), so images are rendered afterwards by
setting each state -- cameras can change without regenerating data.

- **front**: the RealSense of the human take, pose from the take's registration and the
  real intrinsics (principal point included). Check: the microwave's front corners
  projected through the registration + real K and through the simulator camera agree to
  0.001 of the image (before adding the principal point they were 3.6 % apart vertically).
  224 x 126.
- **wrist**: where ManiSkill's panda_wristcam mounts its camera, looking along the fingers;
  128 x 128.

Rendered every 4th step for all 96 generator demonstrations (1.7 GB, ~7 s per run).
Known gap to real images: no walls or furniture (black/grey background), ground plane for
the counter, minimal shader.

### 7a. Camera-based student v1 (`dit_vis.py`): small CNN from scratch, minimal shader

Inputs: front + wrist images and the robot's joints / tcp; no pot pose, no door angle.
Same 96 demonstrations as the state DiT, rendered every 4th step (56 k samples), 60 epochs.
Offline: val first-step joint error 0.95 deg, gripper command wrong 1.2 % (state DiT: ~0.9
deg / 1.6 %). Closed loop, same 128 starts as the state DiT:

| policy | door | lifted | inside | success |
|---|---|---|---|---|
| state DiT (privileged) | 100 % | 80 % | 72 % | 70 % |
| camera DiT v1 | 84 % | 25 % | 4 % | **1 %** |

The losses are where vision is needed (the pot's position varies; the insertion needs
~1 cm), not at the door (fixed in space). Offline error says nothing about closed loop,
again. At the handle the arm hides the bar from the front camera and the wrist camera
looks at the sky until the last centimetres; the 9 mm bar is 1.4 px in the front view.

v2: the default shader rendered at 2x and area-downsampled (no aliasing), and an
ImageNet-pretrained ResNet-18 per camera, fine-tuned at 0.1x the learning rate (tokens still
pooled to 4x7 + 4x4). Same data, same 128 starts:

| policy | door | lifted | inside | success |
|---|---|---|---|---|
| v1 | 84 % | 25 % | 4 % | 0.8 % (1) |
| v2 | 87 % | 21 % | 5 % | 4.7 % (6) |

No change where it matters: the grasp (paired: 24 lifted only by v1, 19 only by v2); the
success difference is 6 vs 1 discordant, p ~ 0.13. Aliasing and an untrained encoder are not
why the grasp fails. v3 tests the pooling: frozen DINOv2 ViT-S/14, every patch a token
(144 front + 81 wrist).

### 7b. v3, and the control that should have come first

v3: frozen DINOv2 ViT-S/14, all 225 patch tokens, hq views: door 88 %, lifted 10 %, success
0 %. Three different encoders, the grasp stuck at 10-25 %.

Control 1 -- `dit_vis.py`'s own pipeline with the **privileged state** (pot pose + door
angle) and **no images**: door 49 %, lifted 6 %, success **0 %**, against 70 % for
`dit.py` with the same state. So the camera policy's failure was mostly not vision.
Checked first: on the shared samples both pipelines build bit-identical inputs and action
chunks (gen/run_000: 2163 vs 540 samples, max difference 0) -- dit_vis.py only takes
every 4th step. Then one factor at a time, privileged state, 128 starts:

| variant | samples | gradient steps | door | lifted | inside | success |
|---|---|---|---|---|---|---|
| control 1 | every 4th step | 12 k | 49 % | 6 % | 0 % | 0 % |
| (a) every step | every step | 11 k | 64 % | 1 % | 0 % | 0 % |
| (c) fp32 instead of bf16 | every 4th | 12 k | 91 % | 16 % | 5 % | 2 % |
| **(b) 4x longer** | every 4th | **49 k** | **100 %** | **79 %** | 40 % | **38 %** |
| `dit.py` (reference) | every step | 37 k | 100 % | 80 % | 72 % | 70 % |

**Training length is the cause.** Validation error barely separates these (~0.9-1.3 deg)
-- closed loop does. v1-v3 were all trained for ~12 k steps, so their comparison of
encoders says little. Diffusion Policy's reference configs train 5000 epochs with an EMA
of the weights (decay up to 0.9999) and select checkpoints by closed-loop success; we had
neither EMA nor selection. Added: `--ema`, `--save-every`.

Jitter (user's observation in the videos): over the first 800 steps of one start, the
commanded joint targets change 0.58 deg/step inside a chunk for both the state DiT and v2,
but at chunk switches 0.65 (state) vs 0.98 deg (v2); consecutive chunks disagree on their
8-step overlap by 0.35 (state) vs 0.91 deg (v2). The jitter is chunk-to-chunk
inconsistency. Inference-only fixes added (`eval --infer fixed_noise | ensemble`), to be
tested on a properly trained checkpoint.

Running: v4 = v2's setup + EMA, 240 epochs (49 k steps), checkpoints every 40 epochs; on
the Spark, control 1d = privileged + EMA, 240 epochs (does EMA close 38 -> 70 %?).

### 7c. Trained long enough: the camera policy works

Same 128 starts as every row above. v4 = v2's setup (ImageNet ResNet-18 per camera, hq
renders, robot joints + tcp only) + EMA, 240 epochs (49 k steps):

| policy | input | steps | door | lifted | inside | success |
|---|---|---|---|---|---|---|
| state DiT (`dit.py`) | privileged state | 37 k | 100 % | 80 % | 72 % | 70 % |
| v2 | images + robot | 12 k | 87 % | 21 % | 5 % | 5 % |
| (b) | privileged state (dit_vis) | 49 k | 100 % | 79 % | 40 % | 38 % |
| 1d = (b) + EMA | privileged state (dit_vis) | 49 k | 100 % | 49 % | 27 % | 25 % |
| **v4** | **images + robot** | 49 k | **100 %** | **100 %** | **97 %** | **97 %** |
| v3-long | images + robot, frozen DINOv2 ViT-S/14 (225 patch tokens) + EMA | 49 k | 100 % | 63 % | 34 % | 33 % |

v3-long (trained on the Spark, same 96 demos, hq views, 240 epochs) answers whether DINOv2 was
only undertrained: no -- at v4's training length it is 33 % against 97 %, losing at the grasp
(lifted 63 %). Not a fair encoder comparison yet: DINOv2 is frozen here while the ResNet-18 is
fine-tuned (lr x 0.1). In the kitchen backdrop (Thor) it was 0 / 94 (lifted 30 %) before a
Vulkan crash. A fine-tuned DINOv2 is untested.

Checked: v4's checkpoint is `prop: robot` and its input layer takes 25 numbers (joints,
velocities, tcp pose) -- no pot pose, no door angle. At equal training the camera policy
far outperforms the privileged-state policy through the same pipeline. Unexplained so far;
hypothesis (untested): with 96 demonstrations a state input lets the network key on exact
numbers and overfit, while images with random-shift augmentation regularise. EMA did not
help the state policy (38 -> 25 %, paired desktop evaluation; 23 % on the Spark).

Why do images beat the privileged state? Three single changes to (b), each trained 49.4 k
steps and evaluated on the same 128 starts on the Spark (paired McNemar vs (b)):

| (b) + | success | wins / losses vs (b) | p |
|---|---|---|---|
| nothing (Spark re-eval of (b)) | 38 % | -- | -- |
| H1 relative geometry (tcp - pot, tcp - handle) | 42 % | 33 / 28 | 0.61 |
| H3 without the near-constant pot z and quaternion | 48 % | 36 / 24 | 0.16 |
| H2 Gaussian noise 0.3 on the normalised state | 0 % (door 11 %) | -- | -- |

None explains 38 -> 97 %. H2 is a broken recipe rather than a test (noise of 0.3 std on
every normalised dimension swamps the door-phase cues), so overfitting to exact values is
untested, not refuted. Still unexplained.

Queued on the desktop (same recipe, 49,440 steps; compare with (b)'s desktop 38 %):
H4 the robot's own 25-d state only (no images, no pot, no door) -- if >= 38 %, the
privileged object state is actively harmful; H5 privileged without the door angle -- the
door angle tracks task progress and is the most likely nuisance cue.

Literature (searched 15 Oct, keywords privileged teacher / teacher-student / imitation gap):

- Teacher -> student distillation (Learning by Cheating, arXiv 1912.12294; WMP,
  2409.16784; visual whole-body control, 2403.16967; ADVISOR's "imitation gap",
  2007.12173; SITT, 2412.09149): the student with less information usually trails the
  teacher, more so when the information gap is large. Opposite to what we see, and not our
  setting: here both policies are students of the same solver on the same demos.
- More/precise state hurts imitation:
  - "Do You Need Proprioceptive States in Visuomotor Policies?" (2509.18644): with images +
    proprioceptive state the policy over-relies on the state and overfits to the training
    trajectories; dropping the state (images only) lifts real-robot spatial
    generalisation 0 -> 85 % (height) and 6 -> 64 % (horizontal). Closest to our case.
  - Causal confusion (de Haan et al., NeurIPS 2019, 1905.11979): more information can
    yield worse imitation by latching onto non-causal cues; distribution shift amplifies it.
  - Diffusion policies as action lookup tables in the sparse-data regime (2505.05787):
    with ~100 demos, an exact low-dim state nearly indexes "demo k, step t" -- precise
    on the training trajectories, wrong off them; images + random shift smooth the lookup.
  - Shortcut learning in robot policies (2508.06426): reliance on task-irrelevant
    features, reduced by augmentation (our camera policy is augmented, the state one not).
  - Copycat (Wen et al., NeurIPS 2020) does not separate the two: both have the same
    2-frame robot history.

Results (desktop, paired vs (b) 38 %): H4 robot state only **0 %** (lifted 9 %) -- without
images or the pot pose it cannot find the pot, so the privileged object state is needed,
not harmful; H5 without the door angle 42 % (32 / 27 discordant, p = 0.60) -- the door angle
is not the nuisance cue. Causal confusion through the door angle: not supported.

Read: our observation has precedent (2509.18644 most directly), but none of these is
established as the cause here. A further test in that paper's direction: v4 without the
robot state (images only).

### 7d. A new background breaks it

Zero-shot test of a changed scene, physics unchanged: the scanned kitchen
(`recon/kitchen_backdrop.py`: the sweep's fused mesh in the microwave frame via its
registration, cropped, decimated to 300 k faces, the scanned microwave and anything on the
counter's workspace removed) replaces the grey grid floor as a visual-only backdrop
(`MicrowavePotEnv(backdrop=...)`, `dit_vis.py eval --backdrop`). The sweep saw little of
what the front camera faces, so the backdrop is mostly a speckled white counter -- a
milder shift than the real room, and still:

Found 16 Oct: every kitchen evaluation below rendered the backdrop **without its colours**.
The GLB carries the scan's vertex colours, but SAPIEN's `add_visual_from_file` ignores vertex
colours and uses its default white material. Restoring them changes little in the policy's
view, because the reconstruction itself is the problem: the real table is a smooth grey
board with yellow wooden chairs (take_000's real frame vs the backdrop from the same camera:
`docs/media/real_vs_backdrop.png`); the fused scan makes it a speckled near-white surface,
the chairs scattered orange points, and has nothing beyond. (Later the same day: the speckles
are holes from `kitchen_backdrop.py`'s workspace cut at 1.5 cm through a counter scanned +-1-2 cm
rough, showing the renderer's clear colour; `--cut-z 0.04 --flatten 0.03` removes them. Not
re-evaluated.) A cruder check of the real room, front camera only (`dit_vis.py eval --real-bg
recon/takes/take_000/rest_frame.png`: every ground / empty pixel of the front image replaced
by take_000's real frame at rest, same pose; robot, microwave and pot still rendered, wrist
camera unchanged): v6-long 94 / 128 (73 %) vs 106 / 128 grey, 23 / 11 discordant, p = 0.058;
never lifted 28 vs 15 -- the loss is at the grasp. A static photo, no 3D, so not reported in
the README. So "kitchen" below means an
unseen noisy background, not the real room. Coloured vs uncoloured, v6-long, Thor, same 128 starts:
uncoloured 110 / 128 (86 %), coloured **95 / 128 (74 %)**, 19 / 4 discordant, p = 0.003 (the
first complete Thor evaluations, with `--tasks-per-child 4`). `recon/backdrop_colours.py` clusters the face colours
into 64 PBR materials (`backdrop_mw_color.glb`, mean colour error 2 / 255); v6-long is
re-evaluated in it on the Thor, paired with the uncoloured backdrop on the same machine.

| v4 | door | lifted | inside | success |
|---|---|---|---|---|
| grey floor (trained) | 100 % | 100 % | 97 % | **97 %** |
| kitchen backdrop | 62 % | 27 % | 19 % | **0 %** |

v4 learned the scene, not only the task: no visual variation in training. DoorMan
(arXiv 2512.01061, Table 1) shows the same and how much each kind of randomisation buys:
none 5-20 %, solid colours only 66-70 %, + real dome-light environments another 15-30 %,
textures 10 % -> 100 % only 4-8 %. Two randomised renders (`sim/dr.py`), same demos:

- *procedural* (v5): random floor/walls (procedural textures) + random lights -- rendering
  crashed once (GPU memory); deprioritised, it is DoorMan's "solid colours" row.
- *doorman* (v6): a real indoor environment map per scene (40 Poly Haven HDRIs, as
  background and light), a random floor (walls in 40 % of scenes), random lights, the
  microwave / pot / robot tinted, the front camera jittered (~8 mm, ~0.8 deg) and the wrist
  camera's mount (~3 mm, ~1.2 deg); two scenes per demonstration; training adds colour /
  white-balance jitter, blur (p 0.3) and random erasing (p 0.3). The kitchen is never a
  training scene. 120 epochs over the two scene copies = 49.5 k steps, as v4.

First evaluation, later found to be mismatched: `load_model` chose hq cameras only for
`views == "views_hq"`, so v6 (trained on `views_drm0,views_drm1`, rendered hq) was evaluated
with the minimal shader at 1x -- a render it never saw. Under that mismatch: grey 75 %
(lifted 81 %), kitchen **38 %** (lifted 45 %) vs v4's 0 %. Fixed (any `views_hq` /
`views_dr*` -> hq). Re-evaluated with the training cameras:

| | grey floor | kitchen backdrop |
|---|---|---|
| v4 (one clean scene) | 97 % | 0 % |
| v6, cameras mismatched (minimal shader, 1x) | 75 % | 38 % |
| **v6, training cameras (hq)** | **65 %** (lifted 76 %) | **70 %** (lifted 75 %) |

Randomisation bought the unseen kitchen (0 -> 70 %) at the cost of the clean scene (97 ->
65 %), at the same 49.5 k steps. v6 never saw the plain grey scene: every training scene
had an environment map and a textured floor, so grey is out of distribution for it too
(kitchen > grey fits that). Untested alternatives: twice the scene diversity at the same
step count (underfitting: v6 val error 0.82 vs v4 0.33 deg), and the tint / camera jitter.
Why the mismatched shader scored higher on grey (75 vs 65 %) is not understood.

v6-long: the same data and recipe, 240 epochs over the two scene copies (102 k steps, twice
v6), desktop, paired with v6 on the same 128 starts:

| | grey floor | kitchen backdrop |
|---|---|---|
| v6, 51 k steps | 65 % | 70 % |
| **v6-long, 102 k steps** | **83 %** (35 / 12, p = 0.001) | **81 %** (26 / 12, p = 0.03) |

Underfitting was most of v6's loss on grey (val error 0.82 -> 0.75 deg); the gap to v4's 97 %
on grey shrinks from 32 to 14 points, and the kitchen holds at 81 % (v4: 0 %). Not
checked: whether 3x continues the trend.

v6-x3 (360 epochs, 153 k steps, from scratch; val error 0.69 deg): grey **80 %** (vs v6-long
15 / 19, p = 0.61), kitchen **72 %** (12 / 24, p = 0.065). The trend stops at 102 k. The
failures move: never lifted 15 -> 12 (grey) and 22 -> 15 (kitchen), but lifted and not
put inside 2 -> 11 and 2 -> 16 -- the grasp improves, the carry / insertion gets worse.
Training length is no longer the limit; ~80 % is what this data and model give.

v7-cap: v6-long's data, recipe and 102 k steps with a bigger action transformer (width 512,
8 layers, 8 heads: 26 M parameters outside the backbone instead of 3.5 M; `--width --layers
--heads`). Val error 0.65 deg (v6-long 0.75). Grey **80 %** (vs v6-long 15 / 18, p = 0.73),
kitchen **77 %** (15 / 21, p = 0.41). Never lifted 15 -> 21 (grey) and 22 -> 28 (kitchen).
Capacity is not the limit either. Twice now (x3, cap) a lower val error has come with no
closed-loop gain: the offline metric no longer tracks success at this level.

More demonstrations: 109 new solver demos with new seeds (gen2 55 / 60, gen2_dart02 49 / 120,
gen2_dart04 5 / 40; seeds 10 / 12 / 13), rendered in two DR scenes like the old ones -> 205
demos. The doubled image set (~29 GB resident) does not fit the 5090 or the desktop's RAM, so:

- v7-data1s (desktop): old + new, ONE scene per demo (`--views views_drm0`; 205 copies, about
  v6-long's 192), 102 k steps. Val error 0.50 deg (lowest of all). Grey **38 %** (vs v6-long
  9 / 67, p < 0.001), kitchen **45 %** (9 / 56, p < 0.001); never lifted 64 and 54.
- The new demos match the old on every trajectory statistic checked (steps 2428 vs 2408,
  8 gripper flips in 92 / 96 vs 98 / 109, pot max z 0.204 m both, same start and end pot
  positions, same camera take, similar image brightness; `human` unused).
- Suspect: with one scene per demo the background identifies the demo, so the policy can
  key on it (a shortcut); v4 (one scene for all demos) and v6-long (two per demo) both break
  that one-to-one link. Untested. Controls: v6s1 = old demos, one scene each, 102 k steps
  (desktop); v7-data = old + new, two scenes each, 120 k steps (Spark, evaluated on the
  desktop).

Results (desktop, paired vs v6-long 83 / 81 %):

| run | demos x scenes | steps | val err | grey | kitchen (uncoloured) | never lifted / lifted not inside (grey) |
|---|---|---|---|---|---|---|
| v6-long | 96 x 2 | 102 k | 0.75 | 83 % | 81 % | 15 / 2 |
| v7-data (Spark) | 205 x 2 | 120 k | 0.55 | 82 % (15 / 14, p = 1.0) | 80 % (18 / 17, p = 1.0) | 10 / 12 |
| v6s1 | 96 x 1 | 102 k | 0.46 | 64 % (38 / 14, p = 0.001) | 58 % (43 / 13, p = 0.0001) | 26 / 6 |
| v7-data1s | 205 x 1 | 102 k | 0.50 | 38 % | 45 % | 64 / 15 |

- Twice the demos (with two scenes each) changes nothing: data joins steps and capacity as a
  non-limit at ~80 %. As with x3, the grasp gets better (never lifted 15 -> 10) and the
  insertion worse (2 -> 12).
- One scene per demo costs ~20 points on the old demos (v6s1): consistent with the shortcut
  suspicion, not a proof of the mechanism (v6s1 also has half v6-long's images).
- Unexplained: 205 x 1 (38 %) is far below 96 x 1 (64 %) though the new demos look like the
  old and work with two scenes. Run-to-run (training seed) variance has never been measured
  here; every arm is one training run.

### 7e. The kitchen as a Gaussian splat (16 Oct, afternoon)

Goal: a background that looks like the room, for the evaluation and the demo video. Code:
`recon/sweep_dense.py`, `gs_inpaint.py`, `gs_kitchen.py`, `gs_server.py`; `sim/gs_cams.py`,
`gs_client.py`, `gs_demo.py`; `dit_vis.py eval --real-bg <render>.npz [--splat-wrist <ckpt>]`.

Reconstruction (gsplat 1.5.3, colour L1 + SSIM + 0.2 x RealSense depth L1, init from
Pi3X's dense cloud of the sweep, every view in the microwave frame via the sweep's
registration; take_000's rest frame from the front camera is one training view, its pose
fixed). Held-out PSNR on 8 of the 60 sparse frames (every 8th), each view's pose aligned to
the frozen splat first (300 steps; without it a pose-refined splat is scored against poses
it moved away from):

| splat | train PSNR | held-out, as posed | held-out, pose-aligned |
|---|---|---|---|
| 60 frames, 10 k steps | 29.3 | 23.8 | -- |
| 60 frames, 30 k steps | 32.2 | 24.0 | 24.7 |
| 60 frames + pose refinement, 20 k | 32.2 | 22.0 | 25.6 |
| 300 frames + pose refinement, 20 k | 28.9 | 21.6 | **26.5** |

More steps fit the training views and not the held-out ones (10 k -> 30 k: +3.0 / +0.2 dB):
coverage-limited, not undertrained. The 240 extra frames were posed in batches of 16 with
the 24 sparse frames nearest in time as anchors (residual at the anchors 0.4-2.4 cm median,
0.4-4.5 deg); Pi3X's chunked long-sequence mode over all 300 drifted (18 cm / 27 deg
median at the 60 shared cameras) and was not used.

The microwave: the simulator draws its own (its door opens), so the splat must not have one.
Deleting the Gaussians in its box after training left a hole with floaters around it; it is
painted out of every training image instead (LaMa; mask = its box projected at each frame's
pose, dilated 12-18 px). LaMa rather than a diffusion inpainter: fills that agree across views.

Evaluation, v6-long, same 128 starts, paired with the grey floor (106 / 128):

| front background | wrist | success | vs grey (discordant, p) | never lifted |
|---|---|---|---|---|
| grey floor | sim | 106 (83 %) | -- | 15 |
| real photo (take_000 rest frame), front only | sim | 94 (73 %) | 23 / 11, 0.058 | 28 |
| **splat, 300 frames, nothing deleted** | sim | **80 (62 %)** | 34 / 8, 0.0001 | 40 |
| **splat, front and wrist** (wrist rendered every step, `--splat-wrist`) | splat | **98 (77 %)** | 21 / 13, 0.23 | 22 |

Three earlier splat evaluations are void: the render deleted the Gaussians inside the
microwave's box and above the counter's workspace, but the front view is seen by one camera
without depth, so part of what explains it floats inside those boxes -- deleting them
uncovered junk right where the pot is picked (60 frames, deleted after training: 84 / 128;
60 frames LaMa: 76; 300 frames LaMa: 51). With nothing deleted the front render matches its
(painted-out) training frame to 3 / 255 mean.

Front + wrist vs front only: 26 / 8 discordant, p = 0.003. Reading: the room's look costs
little (77 vs 83 %, n.s.); what cost 21 points was the two cameras disagreeing -- a real-
looking room in front, the grey grid floor at the wrist -- where training always showed
both cameras the same randomised scene. (The real-photo front-only run, 73 %, has the same
mismatch.) Before the wrist run I read the front-only 62 % as "the more realistic the
background, the worse"; that was wrong. The wrist renders' empty directions (the ceiling)
are filled light grey (`gs_kitchen.EMPTY`); black fill reached 13 % of wrist pixels and was
replaced before this run. One training seed, one splat, one evaluation.

## 8. Human-guided robot demonstrations: the hand where it was seen (`human_paths.py`)

Why: so far the human video set only *what* to do (pick and place points, door angle,
order); every robot motion came from the solver's rules. Directly retargeting the hand is
not possible here: tracked in 100 % of the approach to the pot but 62-94 % of the carry and
0-29 % of the last 0.7 s before release (inside the cavity); ~1-2 cm wrist error where the
grasp has ~12 mm of margin; a whole-hand wrap is not a pinch. Partial retargeting instead
(as EgoEngine / X-Sim do): the solver follows the wrist's tracked *positions* for the
approach (moved with the pot) and the carry (warped from the take's pick/place to the
episode's), with its own hand orientation; it aligns the last 10 cm to the pot's true pose
and plans the unseen insertion. If the arm cannot follow the whole human path it falls back
to the human carry only, then to the plain solver (planning moves nothing). The plain
solver still reproduces take_000 bit for bit.

On take_000's own configuration, grasp to release:

| | time | tcp path | max height |
|---|---|---|---|
| solver only | 6.8 s | 1.01 m | 25 cm (up, across, down, in) |
| human-guided | 5.7 s | 0.76 m | 16.7 cm (low, diagonal, the hand's path) |

All three takes succeed human-guided (take_001 with its carry only: its approach comes from
the upper left, which the level hand cannot follow). Generated with the same start/place
samples as the solver-only data (same seeds; on the DGX Spark): 55/60 clean, 29/80 at DART
0.02, 2/40 at 0.04 = 86 demonstrations (A: 96); 69 use the hand's approach and carry, 13
the carry only, 4 neither.

A/B, everything else equal (v4 recipe, gradient steps matched: 49.3 k vs 49.4 k): A = v4
on solver-only demos, B = on human-guided demos. Measured: success and, over successful
episodes, steps, tcp path, max height, stops. A: 97 %, 2052 steps, 9.22 m, 0.46 m, 5.0
stops.

| arm | demos (plain / 0.02 / 0.04) | door | lifted | inside | success | steps | tcp path | max z | stops |
|---|---|---|---|---|---|---|---|---|---|
| A (v4) | 96 (58 / 34 / 4) | 100 % | 100 % | 97 % | **97 %** | 2052 | 9.22 m | 0.46 | 5.0 |
| B human-guided | 86 (55 / 29 / 2) | 100 % | 80 % | 59 % | **59 %** | 1995 | 8.63 m | 0.46 | 4.9 |

B fits its data as well as A offline (val first-step error 0.36 vs 0.33 deg) but loses 20
points at the grasp and 38 overall in closed loop; its failures are not tied to the pot
position (mean pot of failed vs successful starts equal to 1 cm). Not yet a verdict on the
human paths: B has 10 fewer demonstrations, 7 fewer of them DART-perturbed (the recovery
data). Control running: A subsampled to B's exact composition (`vis_a86`, `gen:55,
gen_dart02:29, gen_dart04:2`, 48.8 k steps). Second candidate: B mixes three strategies
(69 human approach + carry, 13 carry only, 4 plain) with nothing in the image saying which.

Step by step (`docs/viz`, scene 4): ten configurations re-solved on the desktop as A and
B with the phase boundaries and the commanded human points recorded
(`solve_microwave.py` now logs each phase's end step and `human_target`; physics
unchanged), every step labelled solver / human approach / human carry:

- The human sets only **2-9 % of B's steps** (approach + carry); the other 91-98 % --
  door, transits, the last 10 cm, lift, insertion, retreat, shutting -- are the solver's,
  identical in kind to A.
- The solver's clamps (stay 9 cm in front of the pot, not below grasp height) move the
  human approach points by 1-5 cm on average; take_000's approach collapses to almost a
  vertical line.
- take_002 (the two-handed take) gives carries of 2.2 m against A's 1.0 m (take_000:
  0.75 m); about a third of B's demos follow take_002.
- The arm follows the commanded human points to 0.2-1.2 cm (median), so the difference
  is not tracking.
- Re-solving is not reproducible across machines: run_005 now fails in both arms (grasp
  missed, pot pushed) though both succeeded in the training data; run_000 and run_017's B
  succeed here but failed on the Spark.

Control at B's demo count (`vis_a86`: A subsampled to 55 / 29 / 2, 48.8 k steps), paired
on the same 128 starts:

| | success | paired |
|---|---|---|
| A, 96 demos | 97 % | vs a86: 15 / 3 discordant, p = 0.008 |
| A, 86 demos (a86) | 88 % | vs B: 44 / 7 discordant, p < 0.0001 |
| B, 86 human-guided demos | 59 % | |

Ten fewer demos cost 9 points; the human paths cost another 29. Smoothness (jerk = third
difference / dt^3, dt = 0.05 s): the solver's joint-target labels are piecewise linear
(median jerk ~0); in B's human-guided steps the labels' median jerk is 13-127 rad/s^3, and
over the carry window B's training labels are 8x A's (7.4 vs 0.9). The tcp jerk at the
junctions between human and solver segments is ~100x the solver's own (5-8 vs ~0.05
m/s^3 median). Mechanism: the hand path is resampled every 1 cm and executed one point per
control step, while the hand tracking is good to ~1-2 cm, so the direction jitters from
step to step; the solver's segments join it without blending. The demonstrator's motion
itself is not shown to be jerky -- this is the processing (smoothing, time
parametrisation, blending), which is fixable. Not shown either: that the jerk, rather than
the strategy mix or take_002's long carries, is what costs the 29 points.

Smoothed human paths (`--human-smooth`; `human_paths.smooth_path`): a smoothing spline
with an RMS residual of the tracking noise (1.2 cm), resampled every 1 cm; the carry is one
curve from the top of the lift through the hand's path to the point in front of the
opening (end points pinned, no corners at the joins); the approach runs smoothly into the
pre-grasp pose. The plain path is unchanged (B run_003 re-solved without the flag:
bit-identical). On eight of the pair configurations (desktop), in the carry phase only:

| | label jerk, median (rad/s^3) | tcp jerk, median (m/s^3) | success |
|---|---|---|---|
| A (solver's straight lines) | 0.9-1.9 | ~0.02 | 8/8 |
| B (tracked points as they are) | 10-93 | 1.7-7.0 | 7/8 |
| Bs (smoothed) | 1.8-7.9 | 0.6-2.3 | 7/8 |

The human shape is kept (carry 0.65-0.92 m vs A's ~1.0 m, max height 16-28 cm vs A's 25
cm); run_004's approach is now followable (it fell back to carry-only). Full Bs data set
(same seeds as gen*/genh*, desktop) -> render -> train (49.4 k steps) -> 128-start eval
running: if Bs recovers towards a86's 88 %, the 29 points were the jerk.

Result: Bs (56 / 24 / 4 = 84 demos, 49.4 k steps) **59 %** -- the same 75/128 as B (32 / 32
discordant, p = 1.0); vs a86 7 / 44, p < 0.0001. **The jerk is not what costs the 29
points.** Failures: a86 16 (1 at the grasp, 14 after the lift); B 53 (26 at the grasp, 26
after the lift); Bs 53 (24 / 27). The human-guided policies lose the grasp, which a86
almost never does. Next suspect: the mix of three takes (three approach and carry shapes,
picked by run index, with nothing in the image saying which) -- Bs0: as Bs, every run
following take_000.

Bs0 (take_000 only, smoothed; 57 / 30 / 6 = 93 demos, all approach + carry): **65 %** -- vs
Bs 34 / 26, p = 0.37; vs a86 10 / 39, p < 0.0001. The mix of takes is not the main cause
either. Failures move: 34 of 45 now at the grasp (Bs 24), 11 after the lift (Bs 27) -- one
carry shape helps the carry; the grasp stays bad. Suspect now: the human approach itself
(take_000's, clamped to a near-vertical descent 9 cm in front of the pot, where the
solver's own approach is a joint-space move to the pre-grasp pose). Bc0: as Bs0 but with
the solver's approach and the hand's carry only (`--human-carry-only`).

| run (desktop, 128 starts) | demos | data before the grasp | after the grasp | success | lifted |
|---|---|---|---|---|---|
| A2: the current solver, no human | 94 (57/29/8) | solver | solver | **90 %** | 95 % |
| a86 | 86 | solver (older version) | solver | 88 % | 99 % |
| Bc0: hand carry only, take_000, smoothed | 81 (58/20/3) | solver | hand | **66 %** | 83 % |
| Bs0 | 93 | hand approach | hand | 65 % | 73 % |

A2 vs a86 14 / 11, p = 0.69: the solver's code drift (identical door; reach within 1 mm and
1 deg of null-space joint angle from step 921) is not it. Bc0 vs A2 8 / 38, p < 0.0001: with
the data before the grasp of the same kind as A's, the hand's carry still costs the grasp.

Checked and identical across data sets: the tcp-pot offset when the gripper closes (-0.03,
-0.96, 4.99 cm, sd ~0.5), the 16-label chunk at the close (closing only, no lift in it), the
action normalisation (a_sd within 0.5 deg per joint, gripper 0.73-0.76).

Instrumented rollouts (`closes` in eval.json: tcp - pot at every close command within 15 cm
of the pot), Bc0: successful episodes close at y = -1.02 cm (training -0.96); the 22
never-lifted ones close at **y = -3.62 cm** (90 % more than 1 cm short), i.e. ~2.7 cm before
the hand reaches the grasp depth along the approach, then re-close 6 times (median, up to 16)
while drifting, until the time limit. The failure is a premature close. Candidate: the close
is decided at one query and executed open-loop over 8 steps while the arm lags its targets --
testing the execution horizon (4, 2 steps; temporal ensembling) and a86's closes.

Execution horizon (`dit_vis.py eval --exec k`: actions executed per query; trained H = 16,
default 8), Bc0, desktop: **8 -> 66 %, 4 -> 34 %, 2 -> ~6 %**. Re-planning more often makes it
far worse, so the open-loop length is not the cause; the opposite -- successive predictions
disagree, and re-planning more often means more switching between "keep pushing" and
"close" in front of the pot. Longer: Bc0 12 -> 58 % (vs 8: 21 / 32 discordant, p = 0.17),
16 -> 36 %. A2: 4 -> 90 % (10 / 10, p = 1), 16 -> 59 % (6 / 46, p = 1e-8). So 8 is the
sweet spot for both, but only the human-carry data is fragile to *shorter* horizons: A2
re-plans every 4 steps at no cost, Bc0 loses 32 points. ACT temporal ensembling (query
every step, exponentially weighted mean of all chunks' predictions for that step), Bc0:
77 % (vs 66 %: 32 / 19, p = 0.09) -- averaging the disagreeing predictions helps, not
significant at n = 128.

Privileged-state policies (no images; Spark; 49.4 k steps; paired vs the A-86 state policy):
A-86 23 % (lifted 48 %); Bc0 23 % (lifted 35 %; success 25 / 24 discordant, p = 1; lifted
27 / 44, p = 0.057); Bs0 5 % (lifted 21 %; 5 / 28, p = 0.0001). Without images the hand's
carry costs no success (the hand's approach still does). So the 24 points the hand's carry
costs the camera policy are mostly visual. Candidate: the hand carries low (~16 cm), and
the wrist view of the pot held just above the counter resembles the moment before the
grasp (pot between the open fingers); A carries at 25 cm. Test without any human data:
A-low = the solver's own carry at tcp z 0.15 (`--carry-z 0.15`; default 0.25 bit-identical),
generated on the Spark, rendered and trained on the desktop (49.4 k steps, resnet18, hq).

A-low result (desktop, 128 starts): **31 %** (A2 90 %: 5 / 80 discordant, p = 2e-18; Bc0
66 %: 14 / 59, p = 1e-7). Failure breakdown, success / never lifted / lifted-not-inside:
A2 115 / 6 / 7, Bc0 85 / 22 / 18, A-low 40 / **77** / 11. A-low's never-lifted episodes
show Bc0's signature: first close at y = -3.29 cm (successes -1.12), 3 re-closes (median),
pot moved 1.4 cm, time limit. So a low carry alone, with no human data at all, reproduces
the premature close -- and worse than the human's carry. Lifted pots peak at 0.114 m for
both A-low and Bc0 (A2 0.207).

Confound: the low carry also collapsed the DART yield on the Spark (dart02 12 / 80, dart04
1 / 40, against A2's 29 / 80, 8 / 40), so A-low trained on 71 demos (58 / 12 / 1) against
A2's 94 (57 / 29 / 8) -- fewer, and almost no recovery demos. a86 (86 demos) lost 9 points
to A96. Control: A2m = A2 capped to 57 / 13 / 1 = 71 demos, same 49.4 k steps, carry 25 cm,
no human data: **48 %** (vs A2 90 %: 8 / 61, p = 3e-11; vs A-low 31 %: A-low 17 / 39,
p = 0.005). A2m's 31 never-lifted episodes close first at y = **-3.19 cm** -- the same
premature close. So:

- The premature close is what this policy does when it has too few demos near the grasp,
  with or without human data, at either carry height. It is not a signature of ego data.
- Demo count / composition explains most of A-low's loss (90 -> 48), the low carry the
  rest at matched count (48 -> 31, p = 0.005).
- Temporal ensembling on A-low: 27 % (vs 31 %: 19 / 25, p = 0.45). No help.
- **Bc0 was never compared at matched composition**: Bc0 81 demos (58 / 20 / 3) vs A2 94
  (57 / 29 / 8). Its 24-point gap to A2 is confounded with the demo count. Control running:
  A2c = A2 capped to 57 / 21 / 3 = 81 demos. Bc0 66 % already *beats* the smaller A2m (38 / 15,
  p = 0.002), as a count effect predicts.
- A real, indirect cost of the human's carry: the perturbed (DART) solver runs succeed less
  often on the low carries (Bc0 20 / 80 and 3 / 40, A-low 12 / 80 and 1 / 40, A2 29 / 80 and
  8 / 40), so the same generation budget yields fewer recovery demos. Why is not checked.

A2c (A2 capped to Bc0's 57 / 21 / 3 = 81 demos): **78 %**. Bc0 66 % vs A2c: 14 / 29, p = 0.03.
So at matched composition the human's carry costs ~12 points (not 24: the other half was
the demo count). Count curve of the same solver data: 71 -> 48 %, 81 -> 78 %, 94 -> 90 %.

State-only (privileged, Spark), A-low vs A2m data, 71 demos each: s_low **0 %** (lifted 4 %)
vs s_a2m 19 % (lifted 47 %): 0 / 24, p = 1e-7; lifted 2 / 57, p = 6e-15. The low carry
hurts without images too, and at the grasp -- so it is not (only) visual aliasing; the
joint-space reading (carry states near grasp height overlap the approach, with the opposite
gripper command) fits. Not consistent with it: the human's low carry cost the state policy
nothing (s_c0 23 % vs s_a86 23 %); the solver's carry is a straight line at constant
height, the hand's rises and varies -- unexplained.

Correction to the joint-space reading: the 25-d robot input contains both finger joints, so
"carrying" (fingers closed on the pot) and "about to grasp" (fingers open) differ in the
current observation; the premature close also happens with the fingers open. Overlap of
states is not established. Two offline probes (`sim/probe_close.py`):

- *Open loop, on recorded approaches* (`openloop`): A-low, A2m, A2c, A2, Bc0 queried on the
  same windows of the last ~3 s before the solver's close (held-out gena2 DART runs, and
  training runs of genlow / gena2; 8 noise draws). P(close among the 8 executed actions):
  0 for every policy up to 16 steps before the true close (tcp 6-13 cm short), 0-19 % at
  8-16 steps before, 74-100 % in the last 8. The failing policies (A-low, A2m) are no earlier
  than A2 / A2c. **On the demonstrations' own states nobody closes early**: the premature
  close is not in the learned function there; it appears only after closed-loop drift.
- *Nearest training samples to the closed-loop early closes* (`nn`): 12 failed episodes
  re-run with every query's inputs kept; each close within 15 cm of the pot and > 2 cm short
  of the grasp depth embedded with the policy's own conditioning (mean of its prop + image
  tokens, cosine) and matched to its 20 nearest training samples, labelled by phase
  relative to that demo's close (approach: > 16 steps before; boundary: the chunk contains
  the close; carry: after it):

| policy | early closes (episodes) | approach | boundary | carry | median steps from their demo's close |
|---|---|---|---|---|---|
| A-low (carry 15 cm) | 12 (5) | 3 % | 26 % | **71 %** | +22 |
| A2m (carry 25 cm) | 46 (8) | 5 % | **60 %** | 34 % | -4 |

  A-low's early closes look most like its own low-carry samples just after a grasp (3 of 5
  episodes by majority); A2m's look like the moments just before a demo's close (6 of 8).
  So with few demos the drifted state lands near the grasp boundary (A2m), and with a low
  carry it lands nearer the post-grasp carry, whose labels are "keep closed" (A-low). Weak
  evidence: 5 and 8 episodes, a mean-pooled embedding (the 2 finger joints are 2 of 25 prop
  dimensions in one of many tokens), and the re-runs are not the original episodes (7 of 12
  A-low re-runs never closed near the pot at all -- the other failure mode, stalling).

## 8a. How far outside the training region does the pot still get picked? (v4, Thor)

`dit_vis.py eval --pot-shift dx,0`: the pot's start region (x 0.422-0.518, y -0.296..-0.153,
mw frame) moved +x -- along the microwave front, away from the robot -- by 0 / 5 / 10 / 15 cm,
64 starts each, v4 (97 % on the desktop). The solver itself succeeds at x 0.57-0.67, so
reach is not the limit. Pooled by distance beyond the region's edge (x 0.518):

| beyond the edge | n | success | lifted |
|---|---|---|---|
| inside | 89 | 97 % | 99 % |
| 0-2.5 cm | 34 | 47 % | 56 % |
| 2.5-5 cm | 30 | 23 % | 33 % |
| 5-7.5 cm | 34 | 0 % | 3 % |
| 7.5-10 cm | 30 | 0 % | 0 % (24 never close) |
| > 10 cm | 39 | 0 % | 0 % (none closes) |

The door is always opened (100 % at every shift). The arm does head for the real pot -- it
is not replaying the region's centre -- but ends beside it and stops without closing; the
closes it does make land on the robot's side of the pot (x -1.6 to -3.8 cm) and short
(y -4 to -6 cm). The grasp generalises over a few centimetres, not a decimetre: 96 demos
spanning 10 x 14 cm cover the pot's position and nothing more.

Pot on top of the microwave (`--pot-z 0.277`: the top surface is at z 0.278; new env /
solver option `pot_z`, default 0 -- a floor solve is bit-identical, solve.h5 compared). The
solver does the task from the top at x 0.40 (y 0.05) and finds no plan at x 0.20 / 0.30.
v4, 32 starts at x 0.25-0.35 and 16 at x 0.37-0.47 (solver-feasible), y 0.04-0.18: **0 / 48,
the door never opened** (max 1.4 deg), no close near the pot. The arm skips the door, reaches
across the top toward the pot and stalls beside it. Moving the pot on the floor kept the
door stage intact (100 % opened at every shift); putting it where the handle approach
normally happens breaks the task's order -- the stage the policy is in is read off where the
pot appears, not off the door.

## 8b. LingBot-VA zero-shot (`va_zeroshot.py`, Thor)

`lerobot/lingbot_va_libero_long` (LIBERO-long post-train), bf16 on the Thor GPU, separate
venv `.venv-va`. Pipeline check on its own benchmark: libero_10 task 0, 2 / 2. Mapping to our
env: `pd_ee_delta_pose`, front camera re-rendered square from the RealSense pose (same
horizontal FOV), wrist 128 x 128, images mirrored as LIBERO's processor does (the imagined
video confirms the orientation), robosuite action scales, rotation sign checked.
Microwave, 3 episodes x 1500 steps (2 long prompts, 1 short): door max 0.09 deg, pot never
moved, gripper never closer to the handle than at the start. Actions are dominated by a
constant forward-left drift (a LIBERO prior), toward the hinge side. The imagined video is
a stable reproduction of our scene; with the long prompt it imagines the door swinging open
at its hinge edge (physically wrong). Imagined-vs-real frame error 6.2 vs 1.0 for repeating
the frame 16 steps back: no better than predicting no change. Speed: 11.6 s per 16-action
chunk steady state, 0.76 s per env step, peak 13.7 GB.
Not a test of whether VA can learn the task: the camera faces the robot from behind (LIBERO's
faces it), no table, action scales not measured against LIBERO's controller. Only: this
checkpoint does not transfer out of the box.

## 9. Same pipeline, another body: Unitree G1 (`sim/g1_spike/`, scripted only)

ManiSkill's G1 upper body (fixed torso, two 5-DoF arms -- no wrist -- three-finger hands) in
the same scene, solved from take_000's keyframes: left hand hooks the handle and opens the
door to 76 deg, right hand carries the pot onto the turntable at (0.182, 0.095) (take:
(0.183, 0.095)), left hand shuts the door: success. Narrow: 9/9 random pick points at
door 75 deg, door 60-75 deg only, pot yaw 11-21 deg only (outside it the pot turns in the
three-finger grasp); takes 001/002 only with changed door angle or pot yaw; the arm brushes
the open door (no collision checking). Estimated 1-2 weeks to a data -> policy pipeline.

## 10. NVIDIA Newton instead of ManiSkill's PhysX? (time-boxed check, `sim/newton_spike/`)

Question: does Newton 1.0 (MuJoCo-Warp solver) reproduce the PhysX CPU physics well enough
that take_000's solver trajectory still completes the task when replayed open-loop?

| Newton timestep | door > 60 | lifted | inside | door < 5 | success |
|---|---|---|---|---|---|
| 0.01 s (ManiSkill's) | no, 25.6 deg | no | no | -- | no |
| 0.01 s + finger armature | no | no | no | -- | no |
| 0.002 s | yes | yes | yes | yes | yes |
| 0.0025 s | yes | yes | no | no | no |

At 0.01 s the fingers close through the 9 mm handle bar and through the pot. At 0.002 s the
replay tracks ManiSkill closely (door within 0.3 deg while opening, pot within ~1 cm while
carried) until the insertion, where the pot error jumps to 7.7 cm; 30 of 32 identical
replicas succeed (GPU non-determinism alone flips outcomes; ManiSkill: 9 / 9 under start
perturbations). Throughput at 0.002 s: ~10 k control steps/s at 1024-4096 replicas on the
5090 (shared with training) -- about what the CPU pool gives (12 k at 256 environments).
Port traps found: the URDF importer drops the door's joint friction; it adds a finger
mimic constraint SAPIEN does not have; the pot needs its 35 convex parts, not one hull;
MuJoCo takes the max of two frictions where PhysX averages.

Verdict: not ported. Revisit for large-scale RL, after making the insertion robust.

## Open, as of 08:40

- A at B's composition: done (88 %; the human paths cost 29 points). Next: smooth the
  human paths (spline / minimum-jerk fit, solver-speed time parametrisation, blended
  junctions) and regenerate B; B restricted to take_000/take_001.
- H4 / H5: done (0 %, 42 %); why images beat the privileged state is still open.
- v3 long (DINOv2, all patches) on the Spark: epoch 130/240.

## Open, as of 05:55 (superseded where noted above)

- B arm of the human-guided A/B (section 8); v6 (DoorMan-style randomisation) on the grey
  floor and the kitchen (7d); v3's recipe (frozen DINOv2, all patch tokens) trained as
  long as v4 (on the Spark) on both scenes -- does attention over patches resist the
  background change better than pooled CNN features?
- Which of v4's ingredients matter, at full training: encoder (CNN from scratch vs
  ResNet-18 vs DINOv2), DART data, EMA, random-shift augmentation -- only training length
  is established.
- Why the camera policy beats the privileged-state policy through the same pipeline (97 vs
  38 %): overfitting to exact state values with 96 demos? Untested.
- DAgger for the camera student (the solver as expert, local corrections only).
- Residual RL (section 5) and state DAgger (section 6) are closed as negative results.
