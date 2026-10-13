# capture/ — human press demonstrations from a RealSense D435i

A human finger presses a small object; we recover the object's **vertical
displacement** as a metric trajectory with a contact signal. The RGB never
becomes training data — only the trajectory crosses into simulation, because the
policy's observations have to come from the robot's own cameras.

## Why the camera is near-horizontal

The single change that mattered. With the camera looking down at the desk,
"vertical" lies along the camera's **depth** axis, where the per-pixel noise is
~3 mm. Turned side-on, vertical lies in the **image plane**, where a blob
centroid is good to a fraction of a pixel (0.52 mm/px at 0.32 m).

Measured on the same object and the same press:

| camera | rest noise | object | usable events |
|---|---|---|---|
| 50° down, 0.76 m | 19.7 mm | 393 px | 0 / 10 |
| near-horizontal, 0.32 m | **0.04 mm** | **1246 px** | **2 / 2** |

Everything else — colour band, object shape, statistics, thresholds — was tuned
at length against the first row and none of it helped. The geometry was the bug.

Two consequences worth keeping:

- **Aim the camera AT the object.** Depth noise leaks into the height reading
  through the lever arm `|v - ppy| / fy`, which is zero on the optical axis.
  At `lever = 0.024` the depth contribution is ~0.07 mm.
- **Get close.** Apparent area goes as `1/z²` and depth noise as `z²`.

`aim.py` reports both live; move the camera until everything is green.

## Pipeline

```
aim.py                 live readout: distance / area / lever / gravity
calibrate_session.py   once per session: gravity vertical, origin, colour band
record.py              one .db3 per demo, with the live mask overlaid
extract.py             .db3 -> trajectory + press events + QA video and plot
```

```bash
E=PYTHONNOUSERSITE=1 ~/miniforge3/envs/rscap/bin/python     # env note below
S=<session dir>

$E capture/aim.py --colour yellow
$E capture/calibrate_session.py --out $S --colour yellow --pick <u> <v>
$E capture/record.py --out $S --session $S/session.json --count 40
$E capture/extract.py $S/press_*.db3 --session $S/session.json --out <proc dir>
```

## Recording protocol

Each demo is **3 s RESET + 3 s GET READY + 10 s REC**, shown on the live preview
and advancing on its own. SPACE starts the first demo, `q` quits at any point.
You never touch the keyboard mid-session — a trip to the keyboard lands in the
recording.

| phase | colour | do this |
|---|---|---|
| RESET | orange | Move the object to a **new position**, then take your hand out of frame. Coverage is the point: forty presses in one spot is a scatter plot with one dot. |
| GET READY | yellow | **Stay out of frame for the first second** — the tracker anchors there. An orange circle appears when it locks; only then bring your hand in and hover. |
| REC | red | Hold still ~1 s, press 2–3 times, then **take your hand away for the last 2 s**. |

Both end margins matter. The opening still frames set the rest height; the
closing ones are the only way to measure whether the object slid, by comparing
its position before and after the press.

If the overlay says **NOT VISIBLE**, that is the truth, not a glitch: `q` and
redo that demo.

## What the tracker does

Detecting the largest coloured blob every frame fails the moment a hand enters —
a forearm is larger than the object and its hue overlaps both the red and the
yellow band, so the pick jumps to the arm (measured: 316 mm of "drift" in a clip
where the object never moved). Instead:

- the anchor (position + area) is locked during GET READY, hand clear;
- each frame accepts only blobs **within 60 px of the anchor** and **≤2.5× its
  area**, and at a depth inside the session's band;
- anything else is reported as *not visible*, which is the honest answer.

## Event criteria

A clip is not the unit of judgement — a press is. If the object slides after the
second of three presses, the first two are still good data, and retargeting
consumes single presses anyway.

Contact threshold `thr = max(2.0 mm, 3 × rest_noise)`.

An **event** is a run with `press > thr` lasting ≥0.20 s, with runs less than
0.35 s apart merged, extended outwards while `press > 0.3 × thr`.

| check | limit | why |
|---|---|---|
| lateral drift | ≤ 8 mm | position before vs after, both from unoccluded frames; a sliding object is not a stationary displacement sensor |
| frames lost | ≤ 25 % of the event | |
| peak depth | ≥ 1.5 × thr | |
| view after release | required, **unless the press ran into the clip end** | needed to measure drift |

"Unoccluded" means visible area ≥ 0.7 × the anchor area. A fingertip takes the
object from ~1000 px to ~250, so statistics that mix the two are meaningless:
press depth is measured on every frame, but **position only on unoccluded ones**.

## Outputs per demo

| file | contents |
|---|---|
| `press_NNN.db3` | raw colour + depth (rosbag2) — kept so extraction can be re-run |
| `press_NNN.meta.json` | recorder-side: the locked anchor, per-frame areas, gravity |
| `press_NNN.traj.npz` | `t_s`, `press_m`, `h_tab`, `x_tab`, `y_tab`, `area_px`, `pressed`, event bounds |
| `press_NNN.extract.json` | plane, rest height and noise, per-event verdicts |
| `press_NNN.qa.mp4` / `.qa.png` | overlay video; press-depth and area plot with events shaded |

## Environment

conda `rscap` (python 3.12, pyrealsense2 2.59.1, opencv 5, numpy 2.5).
Always run with `PYTHONNOUSERSITE=1` — `~/.local/lib/python3.10/site-packages`
shadows conda envs on this machine.

librealsense 2.59 writes recordings as `.db3` (rosbag2), not `.bag`. Depth+colour
stream without udev rules; the **IMU does not**, and gravity is the vertical
reference, so install `99-realsense-libusb.rules` before a session.

One gotcha worth repeating: `np.asanyarray(frame.get_data())` does not copy. Hold
the array and you hold the frame; exhaust the pool and librealsense drops frames
silently, mid-clip. Copy on the way out of the read loop.
