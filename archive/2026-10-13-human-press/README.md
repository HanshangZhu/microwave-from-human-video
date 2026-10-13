# Archived, 2026-10-13: human press capture -> RoboTwin -> LingBot-VLA-2

Built in one day, then set aside. Kept because the measurements are real and
some of them are worth not re-deriving.

Data and checkpoints are under `Gitignore-but-important/archive/`:
`data/` (recordings, LeRobot sets, specs), `train_out/` (two post-trained
checkpoints, 115 GB), `eval/` (the closed-loop matrix), `icl/` (demo bank).

## What it did

A finger presses a bottle cap on a desk; a RealSense D435i records it; the cap's
vertical displacement becomes a metric trajectory; the press location and
descent shape become a `click_bell` episode in RoboTwin; the rendered episodes
post-train the action expert of LingBot-VLA-2.

`capture/` -> `retarget/` -> `train/` -> `eval/`, each runnable on its own.

## What was measured

**Capture.** Rest noise 0.03-0.46 mm against a 10-27 mm press; 26 usable press
events from 41 takes (63 %); every event lost 0 frames.

The one thing worth carrying forward: **put the vertical in the image plane, not
the depth axis.** Looking down at the desk, "down" lies along the camera's depth
axis where per-pixel noise is ~3 mm, and no amount of threshold tuning fixes it
— colour band, object shape, robust statistics and erosion radius were all tried
and none moved it. Turned side-on at 0.32 m, "down" lies in the image plane
where a blob centroid is good to a fraction of a pixel:

| camera | rest noise | object | usable events |
|---|---|---|---|
| 50° down, 0.76 m | 19.7 mm | 393 px | 0 / 10 |
| near-horizontal, 0.32 m | 0.04 mm | 1246 px | 2 / 2 |

Track the object, not the fingertip: a press is then a displacement you can
measure rather than a sub-millimetre contact threshold you cannot.

**Retarget.** 52/52 episodes planned, no IK failures, including the 19 whose
bell lands where `click_bell` never samples. The task's own exclusion of
`|x| < 0.05` is not about reachability.

**Inference (closed loop, 60 paired seeds, McNemar).** On the released
checkpoint:

| comparison | effect | p |
|---|---|---|
| time grid BF16 -> FP32, upstream path | **+18.3 pp** | 0.013 |
| time grid BF16 -> FP32, fast path | +13.3 pp | 0.096 |
| 126 ms -> 79 ms, grids fixed | −5.0 pp | 0.61 |
| 126 ms -> 79 ms, grids buggy | 0.0 pp | 1.00 |

The 1.57x costs nothing measurable, in four independent comparisons. The time
grid is what the points were. Reported upstream as
[Robbyant/lingbot-vla-v2#59](https://github.com/Robbyant/lingbot-vla-v2/pull/59).

**Post-training, the negative result.** 26 demos, expert-only, 5 epochs:
click_bell 75.0 % -> 50.0 % (discordant 16 v 1, p = 0.0003). One epoch was worse
still (21.7 %), so it is not an epoch-count problem.

Why, as far as it was traced: the demos replace one 45 mm descent with eight
waypoints following the human profile, so only 27 % of the descent is spent near
full depth against the scripted expert's 18-22 % of a much shorter episode, and
the episodes run 119 frames against 76-81. The model was pushed toward hovering
— which is the base policy's own failure mode.

**What survives the human-to-robot transfer, item by item.** Press location:
survives, and is replaceable by a uniform sampler, so worth nothing here.
Descent shape: survives and actively hurts. Timing: does not survive, the
planner re-times every waypoint. Approach: never captured, because the object
does not move before contact.

The honest summary is that `click_bell` scores a binary contact and therefore
cannot see anything a human press contains beyond its location. Four separate
attempts to make the data pay — out-of-distribution bell placements, fewer
epochs, targeted collection, in-context learning — all failed against that same
wall. The out-of-distribution axis was dropped when the base policy turned out
to score 100 % there: never-sampled and hard are independent.

**FP8 training on sm_120.** Transformer Engine builds and runs
(`train/fp8_env.sh` has the three undocumented steps), but FP8 is slower than
BF16 at the shapes being trained: 0.42-0.60x at the action expert's hidden 768,
crossing over around 2560 and reaching 2.98x at 4096 with a large batch. The
loss is entirely in the backward, and `DelayedScaling` does not recover it, so
it is the cast and transpose traffic rather than the amax reduction. The module
that would benefit is the frozen 4B VLM; unfreezing it does not fit in 32 GB.

**In-context learning.** A VLM given k=8 demonstrations localises the press to a
median 109 mm in table coordinates, against the 25 mm the task allows.

## Mistakes worth not repeating

- Two policy servers plus two renderers do not fit in 32 GB. Every episode came
  back `success: null, error: 'cannot create buffer'`, which reads as 1.7 %
  success. `success: null` and `success: false` are different things.
- `np.asanyarray(frame.get_data())` does not copy; holding the array holds the
  frame, and librealsense then drops frames silently, mid-clip.
- A `min_area_px` floor meant as "is this really the object" reported the object
  LOST for 80 % of a clip in which it was visible in every frame.
- The whole day was spent satisfying a constraint the brief never stated. It
  asks you to *drive* a manipulator with your own data, and names retargeting
  and inference optimisation as acceptable approaches. It does not ask for a
  policy improvement.
