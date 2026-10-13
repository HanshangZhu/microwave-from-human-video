#!/usr/bin/env python
"""Turn extracted human press events into RoboTwin episode specifications.

What actually crosses from the real world into simulation is narrow, and worth
being explicit about:

  from the human data   where the press happens (bell pose), which arm that
                        implies, and the shape and timing of the descent --
                        how fast it goes down, how long it is held, how it is
                        released
  from the simulator    the approach to a point above the bell, because the
                        object does not move before contact and so the capture
                        contains no approach trajectory at all
  from the task         the absolute descent distance. RoboTwin's bell is
                        `is_static=True`, so "press depth" has no physical
                        counterpart there; the human profile is normalised onto
                        the task's own descent and contributes its shape, not
                        its millimetres.

Coordinates. Capture frame: +z along gravity (up), x/y in the table plane with
+x the camera's right. RoboTwin's click_bell samples the bell in
`x in [-0.25, 0.25]` (rejecting `|x| < 0.05`) and `y in [-0.2, 0.0]`, with the
arm chosen by `sign(x)`. The map below is a fixed affine transform, stated once
and applied to every episode -- not fitted per episode, which would make the
coverage claim circular.

  python retarget/events_to_specs.py <processed dir> -o specs.json
"""
import argparse
import glob
import json
import os

import numpy as np

# RoboTwin click_bell sampling, metres (envs/click_bell.py)
BELL_XLIM = (-0.25, 0.25)
BELL_XMIN_ABS = 0.05
BELL_YLIM = (-0.20, 0.00)
SCRIPTED_DESCENT_M = 0.045      # move_by_displacement(z=-0.045)


def load_events(proc_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(proc_dir, "press_*.extract.json"))):
        d = json.load(open(f))
        traj = np.load(f.replace(".extract.json", ".traj.npz"))
        for k, e in enumerate(d["events"]):
            if not e["ok"]:
                continue
            sl = slice(e["i0"], e["i1"] + 1)
            out.append(dict(take=d["tag"], index=k,
                            t=traj["t_s"][sl] - traj["t_s"][e["i0"]],
                            press=traj["press_m"][sl],
                            x=e["x_m"], y=e["y_m"], peak_mm=e["peak_mm"]))
    return out


def descent_profile(t, press, n=24):
    """Resample the measured displacement onto `n` waypoints, normalised so its
    peak is 1.0. Shape and timing survive; absolute depth does not."""
    good = np.isfinite(press)
    t, press = t[good], press[good]
    if len(t) < 4:
        return None
    press = np.clip(press, 0, None)
    pk = press.max()
    if pk <= 0:
        return None
    ts = np.linspace(t[0], t[-1], n)
    return ts - ts[0], np.interp(ts, t, press / pk)


def fit_map(xs, ys, mode):
    """Capture-frame metres -> RoboTwin table metres.

    `direct` keeps real millimetres, so the recorded spread is what the robot
    sees -- including presses outside the region the task ever samples, which is
    the whole point. `fit` rescales the cloud into the task's own ranges, which
    makes every episode reachable but throws the out-of-distribution claim away.
    """
    xs, ys = np.asarray(xs), np.asarray(ys)
    if mode == "direct":
        return (lambda x, y: (float(x), float(y - 0.05)),
                "direct: capture metres, y shifted -50 mm to sit over the task's band")
    sx = (BELL_XLIM[1] - BELL_XLIM[0]) / max(np.ptp(xs), 1e-6) * 0.9
    sy = (BELL_YLIM[1] - BELL_YLIM[0]) / max(np.ptp(ys), 1e-6) * 0.9
    cx, cy = xs.mean(), ys.mean()
    tx, ty = 0.0, (BELL_YLIM[0] + BELL_YLIM[1]) / 2
    return (lambda x, y: (float((x - cx) * sx + tx), float((y - cy) * sy + ty)),
            f"fit: scaled x by {sx:.2f}, y by {sy:.2f} into the task's sampling ranges")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proc_dir")
    ap.add_argument("-o", "--out", default="specs.json")
    ap.add_argument("--map", choices=("direct", "fit"), default="direct")
    ap.add_argument("--waypoints", type=int, default=24)
    a = ap.parse_args()

    ev = load_events(a.proc_dir)
    if not ev:
        raise SystemExit(f"no usable events under {a.proc_dir}")
    to_bell, map_note = fit_map([e["x"] for e in ev], [e["y"] for e in ev], a.map)

    specs, skipped = [], []
    for e in ev:
        prof = descent_profile(e["t"], e["press"], a.waypoints)
        if prof is None:
            skipped.append((e["take"], "flat or too short"))
            continue
        ts, frac = prof
        bx, by = to_bell(e["x"], e["y"])
        why = []
        if not (BELL_XLIM[0] <= bx <= BELL_XLIM[1]):
            why.append(f"x {bx*1000:.0f} mm outside the task's x range")
        if abs(bx) < BELL_XMIN_ABS:
            why.append(f"|x| {abs(bx)*1000:.0f} mm inside the band the task excludes")
        if not (BELL_YLIM[0] <= by <= BELL_YLIM[1]):
            why.append(f"y {by*1000:.0f} mm outside the task's y range (this is the OOD axis)")
        specs.append(dict(
            episode=len(specs), take=e["take"], event=e["index"],
            bell_xy=[bx, by], arm="right" if bx > 0 else "left",
            duration_s=float(ts[-1]),
            descent_t=[float(v) for v in ts],
            descent_frac=[float(v) for v in frac],
            descent_m=[float(v * SCRIPTED_DESCENT_M) for v in frac],
            human_peak_mm=e["peak_mm"],
            capture_xy_mm=[e["x"] * 1000, e["y"] * 1000],
            outside_task_distribution=why))

    json.dump(dict(map=a.map, map_note=map_note,
                   scripted_descent_m=SCRIPTED_DESCENT_M,
                   n_specs=len(specs), specs=specs), open(a.out, "w"), indent=1)

    nx = sum(1 for s in specs if any("x range" in w or "excludes" in w
                                     for w in s["outside_task_distribution"]))
    ny = sum(1 for s in specs if any("y range" in w for w in s["outside_task_distribution"]))
    inside = sum(1 for s in specs if not s["outside_task_distribution"])
    arms = [s["arm"] for s in specs]
    pk = [s["human_peak_mm"] for s in specs]
    print(f"{len(specs)} episode specs  ({len(skipped)} skipped)")
    print(f"  mapping: {map_note}")
    print(f"  inside the task's sampling region : {inside}")
    print(f"  outside on y (never sampled)      : {ny}")
    print(f"  in/outside x (incl. excluded band): {nx}")
    print(f"  arms: left {arms.count('left')}  right {arms.count('right')}")
    print(f"  human peak depth {min(pk):.1f}-{max(pk):.1f} mm "
          f"(median {np.median(pk):.1f}) vs scripted {SCRIPTED_DESCENT_M*1000:.0f} mm")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
