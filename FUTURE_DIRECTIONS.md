# Future directions (notes, 2026-10-14)

Ideas that came up while building this, with the reasoning and rough numbers. Not done.

## 1. When is human demonstration data needed at all?

For a task whose geometry and articulation can be scanned and measured (this one: a
microwave and a pot), the human video turned out to carry the **task intent** -- which
door, how far it opens, where the pot goes, in what order, how the start states are
distributed -- and almost no usable **motion**: a human hand is not a parallel gripper,
and in the dark cavity the hand is not even visible. The robot's motions can be generated
in simulation from the scanned twin.

Testable as an A/B: train the same observation-only policy on simulation data whose task
specification comes (A) from hand-written / randomly sampled goals, (B) from the human
videos. Equal success means human video is a zero-engineering substitute for a written
spec; a gap means the video carries something the spec misses (e.g. people place the pot
near the front of the turntable, and push it before lifting it). Expected answer: human
data becomes irreplaceable when objects cannot be modelled (deformable, unknown), contact
skills matter, or tasks are too many to specify by hand.

## 2. A wrist IMU to bridge visual occlusion

When the hand enters the cavity it disappears from the camera for 1-2 s. A phone strapped
to the wrist (100-500 Hz accelerometer + gyroscope, synchronised to the video with a sharp
wrist flick at the start) fills the gap:

- **Orientation** from the gyro is accurate over seconds -- exactly what the gripper needs
  and what vision loses.
- **Position** from double integration alone drifts quadratically, and a 1 deg tilt error
  leaks ~0.17 m/s^2 of gravity: >10 cm over 1.5 s open-loop.
- But the gap has **vision at both ends**: a two-point boundary problem (smoother / spline
  through the IMU accelerations, bias estimated from the end constraints), not open-loop
  dead reckoning. Estimate (unverified): 1-2 cm over a 1-2 s gap.

## 3. "UMI-lite": demonstrate with a gripper-like tool

UMI works because the demonstrator holds the robot's gripper, so the recorded pose and
width are robot actions. Kitchen tongs held in the hand would make the demonstrations
embodiment-matched for free: a two-finger opening to track instead of a hand, and the
tongs' tips stay visible longer than fingers inside the cavity.

## 4. Data-collection loop

Every take is processed without manual steps (camera anchoring with a seed box, door angle,
pot start/pick/place, replay video). The yield worth reporting is *successful robot
demonstrations in simulation per minute of human video*, and how policy success scales
with the number of human takes.

## 5. GPU simulation (Newton / GPU PhysX)

The same state policy succeeds 70 % in PhysX CPU and 35 % in PhysX GPU; in Newton
(MuJoCo-Warp) the recorded solver trajectory only completes the task at a 5x finer
timestep, where the GPU's throughput advantage is gone (sim/EXPERIMENTS.md, section 8).
The contact-rich parts -- a 9 mm handle bar, a 5.5 cm pot, an insertion with ~1 cm
clearance -- are what each engine treats differently. Thousands of parallel environments
pay off for RL, not for DAgger; worth it once the insertion is robust to the engine.

## 6. From one task to one student

Diffusion Policy trains one checkpoint per task. Here the structure is already
expert -> student: the solver reads the simulator's state, the camera policy learns from its
demonstrations. More kitchen tasks (cabinet, bowl, kettle) would each get a solver (or an
RL expert, as X-Sim does with object-trajectory rewards from the human video), and their
demonstrations, with a task condition (language or goal image), would train one
multi-task camera student. DoorMan's teacher -> DAgger student -> RL fine-tune is the same
pattern at scale.

## 7. Toward the real robot

The camera policy works in the scene it was trained in and fails on a new background
(97 % -> 0 %). In order: DoorMan-style randomisation (running), DAgger for the camera
student (the solver as expert, local corrections only -- section 6 of sim/EXPERIMENTS.md
shows why restarts poison the data), then the real take frames as a test set: the front
camera already sits where the real RealSense stood, with its real intrinsics.

## 8. Ego video as a video target, not as action labels (2026-10-15)

Hypothesis (Hanshang): a world-action model can absorb human ego video where a VLA trained
on the same data as action labels cannot. In this project the human data hurt only through
its action labels: retargeted, then imitated (sim/EXPERIMENTS.md 8: Bc0 66 % vs A2 90 %,
premature close). A model whose video branch learns from action-free frames, and whose
action branch learns only from robot data, never imitates a retargeted human action.

What the distinction really is: not WAM vs VLA but which loss the human frames enter.
LingBot-VA is itself a WAM, and VLAs also take action-free video through latent actions
(LAPA, GR00T N1). The variable to test is "human frames supervise video / latent
dynamics only" against "human frames supervise robot actions".

Strongest counter-case: if the damage comes from *what the human did* (carrying low, A-low
31 % with no human data, control A2m pending) rather than from retargeting, a WAM inherits
it -- its video branch imagines the low carry and its inverse dynamics executes it. Then
video-only use removes the embodiment gap but not the behaviour gap.

The LingBot-VA zero-shot (EXPERIMENTS.md 8b) does not bear on this: it saw no ego data. Its
imagined door opening is suggestive only (opened at the hinge; imagined-frame error equal to
repeating a 16-step-old frame).

Test, same task, 128 starts: (i) video+IDM policy (UniPi-style) on robot demos only;
(ii) + the human takes as video-only data; (iii) + the human takes as retargeted action
labels. (ii) > (i) >= (iii) supports the hypothesis. Needs a video model small enough to
train here and more than 3 takes; out of scope before the deadline.

## 9. A room the cameras can trust (2026-10-16)

The scanned-kitchen backdrop is a poor copy of the room (README): the smooth grey table
came out as speckled white, the chairs as scattered points. Its geometry is RealSense depth
fused along Pi3X poses (`recon/fuse_depth.py`); Pi3X's own dense point maps were not used,
a choice made on the pot orbit (where Pi3X's per-view depths disagreed and the cloud was
haloed) and carried over to the room without being tested there. A feed-forward model
predicts a flat textureless table from its prior, where noisy depth at 1 m and grazing
angles does not, so the first step is cheap: Pi3X dense cloud of the same 60 frames, scale
from the measured depth, rendered from the front camera next to the real frame.

Beyond that: a denser sweep (hundreds of frames around the workspace) into a Gaussian
splat (initialised from the Pi3X cloud), rendered as the background and depth-composited
with SAPIEN's foreground (robot, microwave, pot), as SplatSim / Re3Sim do. What it adds over
the single-photo composite (`dit_vis.py --real-bg`) is only what a photo cannot give: the
wrist camera's view, a moving or jittered camera, and real backgrounds at training time.
It does not touch the larger gap for a real robot -- the foreground (robot, pot, lighting,
shadows) is still rendered.

Also unmeasured so far: training-seed variance. Every arm in EXPERIMENTS.md 7d is one run;
two seeds of v6-long would say how much of the 205x1 vs 96x1 gap is noise.
