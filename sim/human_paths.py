"""The demonstrator's hand paths from a take, as robot path constraints.

recon/hand_track.py reads a "gripper" off the hand in every tracked frame (midpoint of the
thumb and index tips, in the microwave frame). Its position is good to ~1-2 cm where the
hand is visible -- reaching for the pot and carrying it to the microwave -- and missing
where it is not: inside the dark cavity. Its orientation is too noisy to use (and a whole-
hand wrap is not a parallel-gripper pinch), so only the positions are used:

  approach  the last APPROACH_S seconds before the pick, relative to where the pot was
  carry     from just after the pick to the last tracked point before the hand goes into
            the cavity, relative to the pick (start) and to the place (end)

Gaps of a few frames are interpolated, the path is smoothed and resampled every
--step metres. sim/solve_microwave.py's Solver.pick_and_place(human=...) follows these
positions with its own hand orientation, aligns the last centimetres to the pot's true
pose itself, and plans the insertion the hand was not seen doing.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAKES = os.path.join(ROOT, "Gitignore-but-important/recon/takes")
APPROACH_S = 1.8          # seconds of reach before the pick
PICK_GAP_S = 0.15         # leave out the grasp itself (the hand closes, the keypoints jump)
CAVITY_Y = -0.06          # beyond this (towards the cavity) the hand is no longer trusted


def _smooth_fill(t, x, valid, t0, t1, win=7):
    """Positions on the tracked frames in [t0, t1], gaps linearly filled, moving-average smoothed."""
    sel = (t >= t0) & (t <= t1)
    tv, xv = t[sel & valid], x[sel & valid]
    if len(tv) < 4:
        return None
    ts = t[sel]
    xs = np.stack([np.interp(ts, tv, xv[:, k]) for k in range(3)], 1)
    k = np.ones(win) / win
    pad = np.pad(xs, ((win // 2, win // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(3)], 1)


def resample(path, step):
    """Points every `step` metres along the path (first and last kept)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < step:
        return path[[0, -1]]
    q = np.r_[np.arange(0, s[-1], step), s[-1]]
    return np.stack([np.interp(q, s, path[:, k]) for k in range(3)], 1)


def load(take, step=0.01):
    """{'approach': (N,3) relative to the pick xy (z absolute), 'carry': (M,3) absolute,
    'pick': (2,), 'place': (2,)} in the microwave frame."""
    d = os.path.join(TAKES, take)
    h = np.load(os.path.join(d, "hand.npz"))
    p = json.load(open(os.path.join(d, "pot.json")))
    t, v, ee = h["t"], h["valid"].astype(bool), h["ee"]
    pick = np.array([p["pick"]["x"], p["pick"]["y"]])
    place = np.array([p["end"]["x"], p["end"]["y"]])
    tp, tl = p["t_pick"], p["t_place"]
    app = _smooth_fill(t, ee, v, tp - APPROACH_S, tp - PICK_GAP_S)
    # carry: up to the last tracked point before the hand is in the cavity
    in_front = v & (t > tp + PICK_GAP_S) & (t < tl) & (ee[:, 1] < CAVITY_Y)
    t_end = t[in_front].max() if in_front.any() else tp + PICK_GAP_S
    car = _smooth_fill(t, ee, v, tp + PICK_GAP_S, t_end)
    out = dict(pick=pick, place=place)
    out["approach"] = None if app is None else resample(app, step) - np.r_[pick, 0]
    out["carry"] = None if car is None else resample(car, step)
    return out


def smooth_path(pts, sigma=0.012, step=0.01, fix_start=False, fix_end=False):
    """A smoothing spline through the points, resampled every `step` metres.

    The hand is tracked to ~1-2 cm and the path is executed one 1 cm point per control
    step, so unsmoothed the direction jitters from step to step (median label jerk 13-127
    rad/s^3 against ~0 for the solver's lines). The spline is allowed an RMS residual of
    `sigma` (the tracking noise), keeping the path's shape at the scale of centimetres to
    decimetres and dropping the jitter. fix_start / fix_end pin the end points (weight
    100), so the curve meets the solver's own segments where they join."""
    from scipy.interpolate import splev, splprep
    pts = np.asarray(pts, float)
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-4]
    pts = pts[keep]
    if len(pts) < 4:
        return resample(pts, step)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    w = np.ones(len(pts))
    w[0] = 100.0 if fix_start else 1.0
    w[-1] = 100.0 if fix_end else 1.0
    tck, _ = splprep(pts.T, u=s / s[-1], w=w, s=len(pts) * sigma ** 2, k=3)
    dense = np.stack(splev(np.linspace(0, 1, max(200, 20 * len(pts))), tck), 1)
    out = resample(dense, step)
    if fix_start:
        out[0] = pts[0]
    if fix_end:
        out[-1] = pts[-1]
    return out


def warp_carry(carry, pick_take, place_take, pick_new, place_new):
    """Carry path for a new episode: each point is shifted by a blend of the pick offset
    (at the start) and the place offset (at the end), by arc-length fraction."""
    seg = np.linalg.norm(np.diff(carry, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    w = (s / max(s[-1], 1e-9))[:, None]
    d0 = np.r_[np.asarray(pick_new) - pick_take, 0]
    d1 = np.r_[np.asarray(place_new) - place_take, 0]
    return carry + (1 - w) * d0 + w * d1


if __name__ == "__main__":
    for tk in ("take_000", "take_001", "take_002"):
        H = load(tk)
        a, c = H["approach"], H["carry"]
        print(tk, "approach", None if a is None else (len(a), np.round(a[[0, -1]], 3).tolist()),
              "carry", None if c is None else (len(c), np.round(c[[0, -1]], 3).tolist()))
