# capture/ -- laptop side (RealSense D435i)

Copied from the laptop it ran on (`~/Desktop/humanoid-realsense-capture`). It was written
for the first version of the task (a kitchen cabinet and a beans pot) and then used
**unchanged** for the microwave: the spoken prompts and the text below still say
"cabinet" / "beans pot". What was actually recorded:

```bash
conda activate realsense          # pyrealsense2 2.58.1, opencv 4.11
python scan.py env    --out scans/microwave      # microwave: closed, open_45, open_90
python scan.py env    --out scans/kitchen        # wider sweep of the counter
python scan.py target --out scans/beans          # Heinz Snap Pot orbit
python record.py --out captures/microwave01 --count 3 --duration 40 --reset 15
```

This machine only records; everything else runs on the desktop (`recon/`, `sim/`).

## 1. Scans — environment and target (`scan.py`)

Input for SfM -> dense reconstruction (MVS / TSDF / 3DGS) on the desktop. Every shot
is a slow continuous moving scan; planes, gravity, shelf dimensions, the hinge axis and
the beans pot footprint are all measured in the reconstruction, with metric scale from
the recorded depth. Depth runs the High Density preset (`--preset` to change).

```bash
python scan.py env    --out scans/cabinet     # ~2 min
python scan.py target --out scans/beans       # ~1.5 min
```

| kind | shot | seconds | what it gives |
|---|---|---|---|
| env | closed | 40 | static reference cloud (cabinet, counter, floor) the demonstrations register to |
| env | open_45, open_90 | 15, 25 | door in two more states, reconstructed separately; with `closed`, three door planes whose intersection is the **hinge axis**; `open_90` also covers the shelf interior |
| target | orbit | 60 | full shape; footprint check: **is the short side ≤ 70 mm** (Panda opens to 80) |

Move slowly — the colour camera has a rolling shutter. Each shot is announced, then you
get 8 s to set it up, then it records. SPACE before the first shot; `--manual` waits for
SPACE every time; `--only open_45` redoes one.

## 2. Demonstrations (`record.py`)

```bash
python record.py --out captures/session01 --count 30
```

| phase | spoken | default | do |
|---|---|---|---|
| RESET | "reset" | 10 s | close the door, beans pot back on the shelf at a new spot |
| GET READY | "get ready" | 4 s | start pose, cabinet front in view |
| REC | "go" ... "stop" | 25 s | open, take the beans pot out, put it on the counter, step back |

SPACE starts the first take, then it loops on its own. `q` quits; re-running resumes.

There are no markers, so camera pose, door angle and beans pot pose all come from depth:

- **keep the cabinet front or the counter edge in view**, including while reaching in;
- **clear the shelf except for the beans pot**;
- **stay further than ~0.2 m from surfaces** — the preview turns red under 60% valid depth;
- vary across takes: beans pot position and orientation, door closed vs ajar.

## Output, per take or shot

```
<tag>.avi        colour, 1280x720, MJPG q95
<tag>_depth/     depth, 848x480, 16-bit lossless PNG, one per colour frame
<tag>.json       both intrinsics, depth scale, depth->colour extrinsics, depth preset,
                 per-frame colour/depth timestamps, depth-valid fraction
```

~12 MB/s, about 300 MB per 25 s take. Both scripts stop below 3 GB free.
