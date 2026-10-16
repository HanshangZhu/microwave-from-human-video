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

Done on the last day (sim/EXPERIMENTS.md 7e): the kitchen as a Gaussian splat behind the
simulator's robot, microwave and pot, front and wrist cameras. v6-long: 83 % on the grey
floor, 77 % there (n.s.), but 62 % when only the front camera sees the room -- the cameras
disagreeing hurt more than the room's look. Next, in order of what it would tell:

- **Train with it.** Render the demonstrations' images with splat backgrounds (other rooms,
  or this one from jittered cameras) as part of the domain randomisation, and test on a
  splat of a room never trained on; and randomise the two cameras' scenes independently
  sometimes, so a mismatch between them is not out of distribution.
- **Contact shadows and lighting.** The simulated pot casts no shadow on the splatted table;
  relighting the foreground to the splat's light is the larger visual gap left.
- **Randomise the object's own appearance.** README, "A 7 mm finding": the lid's rim colour
  alone moves v6-long by 9 points. Render each demo under several lid textures (and pot
  colours), as the scenes are already randomised; test on the real print held out.
- **A denser, closer sweep** of the workspace: the wrist camera sees the table from
  centimetres away, where a 1-3 m sweep gives the splat little to go on.
- **A whole-body G1, not the legless upper body.** The pieces exist on this machine already:
  NVIDIA's GR00T-WholeBodyControl (SONIC: a G1 whole-body RL controller with a decoupled
  upper-body command interface, trained checkpoint included), ProtoMotions' G1 motion
  trackers, BeyondMimic. The plan: move the task into Isaac Lab (Isaac Sim 5 renders Gaussian
  splats natively; the `3dgrut` env exports ours), let SONIC hold the stance and balance, and
  feed it the hand trajectories the scripted solver (`sim/g1_spike/`) already produces from
  the take's keyframes; then generate G1 demonstrations across start states and train the same
  camera policy recipe. The renderer is the easy part -- it is what the compositing here
  already does. The costs: the physics port (the Newton check, EXPERIMENTS 10, found five
  importer traps and an open-loop replay that fails at the simulator's own timestep; GPU vs
  CPU PhysX already changes outcomes in ManiSkill, EXPERIMENTS 3), and matching SONIC's G1
  (joint order, gains, observation format) to this task's. Budget: 1-2 days to a first
  whole-body scripted run in Isaac Lab, another for a paired 128-start evaluation of the Panda
  policy there, then 1-2 weeks to a learned G1 policy.

Also unmeasured so far: training-seed variance. Every arm in EXPERIMENTS.md 7d is one run;
two seeds of v6-long would say how much of the 205x1 vs 96x1 gap is noise.

## 10. The design this should have had: human video trains the model, the twin only examines it (2026-10-16)

The chain, from this repo's own results. (i) With a scripted oracle available in simulation,
human data cannot show its value: the oracle is a free, perfect demonstrator for exactly the
evaluated task and region, so anything the human adds to a trajectory is noise relative to
it (EXPERIMENTS 8: human-guided paths 59-66 % against solver paths 78-90 % at matched demo
counts). (ii) So the human video should not compete with the simulator as a *source of
demonstrations*; it should be the *training data of a model* that the simulator cannot
produce, and the simulator should only be the exam. (iii) The model that fits three takes
is not a pixel world model fine-tuned from V-JEPA: two minutes of video, with the hand lost
inside the cavity, cannot support an action-conditioned video predictor with measurable
quality. It is an object-centric, action-conditioned predictor: state = pot pose + door
angle (both already extracted per frame, `recon/`), action = wrist pose as the proxy for
"what the manipulator did" (tracked outside the cavity, interpolated inside); pretrained
video features (V-JEPA 2, DINOv2) can enter as an encoder for the state, but the dynamics
are learned in the object space, where three takes are hundreds of transitions, not a few
thousand pixels-frames. (iv) The exam: in the twin, the robot's end-effector pose stands in
for the wrist, the predictor's rollouts are scored against the simulated object states on
starts the takes never showed; the splat room (section 9) closes the background half of the
visual gap, the object-centric state closes the robot half (the model never sees the arm).

This matches the brief's own weighting -- world modelling "less focusing on policy
performance", and the collected data irreplaceable in the method -- better than the
submitted policy does. It needs the twin, the registration and the per-frame object
tracking first, which is what these four days built; from here it is about three days of
work, and the next project.
