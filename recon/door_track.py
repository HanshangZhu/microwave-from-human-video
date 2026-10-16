#!/usr/bin/env python
"""Door angle in every frame of a take, from the measured depth.

The camera pose per frame is known (take_anchor.py + register_model.py), so each
frame's depth goes into the microwave's frame. What lies in front of the body and
within the door's reach of the hinge, and is not the counter, is scored against
the door panel -- a 35.6 x 27 x 3.6 cm box turning about the hinge (-z axis through
the origin as in the URDF, q > 0 swings it out towards the front-left) -- for every angle in 1 deg steps:
the angle with the most points within --tol of the panel wins. Box distances are
analytic, so the whole take takes about a minute. Frames with too little door in
view are left as NaN and filled by interpolation; the result is median-filtered.

Writes <anchor>/door.npz (t, q_raw, q, support). Runs in any env with numpy + cv2:
  python recon/door_track.py <capture dir> <tag> --anchor <dir> --urdf <urdf>
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_to_r2s import align_depth  # noqa: E402


def door_box(urdf):
    """The door panel box (min, max corner) in the door link frame."""
    root = ET.parse(urdf).getroot()
    for v in root.iter("visual"):
        if v.get("name") == "door_panel":
            c = np.array([float(x) for x in v.find("origin").get("xyz").split()])
            s = np.array([float(x) for x in v.find("geometry/box").get("size").split()])
            return c - s / 2, c + s / 2
    raise SystemExit("no door_panel visual in " + urdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("tag")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--urdf", default="assets/microwave/microwave.urdf")
    ap.add_argument("--tol", type=float, default=0.012)
    ap.add_argument("--min-support", type=int, default=150)
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.capture, a.tag + ".json")))
    ci, d = meta["intrinsics"], meta["depth"]
    R = np.array(d["depth_to_color"]["rotation_colmajor"]).reshape(3, 3).T
    tr = np.array(d["depth_to_color"]["translation_m"])
    fps = meta.get("fps", 30)
    ddir = os.path.join(a.capture, d["folder"])
    reg = json.load(open(os.path.join(a.anchor, "run", "registration.json")))
    T_last_model = np.linalg.inv(np.array(reg["T_model_to_world"]))
    segs = json.load(open(os.path.join(a.anchor, "anchor.json")))["segments"]
    lo, hi = door_box(a.urdf)
    reach = np.hypot(hi[0], lo[1]) + 0.03
    qs = np.deg2rad(np.arange(0, 101, 1.0))
    cq, sq = np.cos(qs), np.sin(qs)

    H, W = ci["height"], ci["width"]
    vv, uu = np.mgrid[0:H:a.stride, 0:W:a.stride]
    n = len(os.listdir(ddir))
    q_raw, support = np.full(n, np.nan), np.zeros(n, int)
    for i in range(n):
        sg = next((s for s in segs if s["start"] <= i < s["end"]), segs[-1])
        C = T_last_model @ np.array(sg["T_last_from_cam"])
        z16 = cv2.imread(os.path.join(ddir, f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        z = align_depth(z16, d["depth_scale_m"], d["intrinsics"], ci, R, tr)[::a.stride, ::a.stride]
        m = z > 0.1
        Xc = np.stack([(uu[m] - ci["ppx"]) / ci["fx"] * z[m], (vv[m] - ci["ppy"]) / ci["fy"] * z[m], z[m]], 1)
        X = Xc @ C[:3, :3].T + C[:3, 3]
        keep = (np.hypot(X[:, 0], X[:, 1]) < reach) & (X[:, 1] < -0.01) & (X[:, 2] > lo[2] + 0.01) \
            & (X[:, 2] < hi[2] + 0.02) & ~((X[:, 0] > hi[0] - 0.005) & (X[:, 1] > lo[1] - 0.02))
        X = X[keep]
        if len(X) < a.min_support:
            continue
        # points in the door frame for every angle. The hinge axis is -z (the URDF's
        # door_hinge): the door at q is the closed door turned by -q about +z, so a
        # point goes into the door frame by turning it by +q about +z.
        px = X[None, :, 0] * cq[:, None] - X[None, :, 1] * sq[:, None]
        py = X[None, :, 0] * sq[:, None] + X[None, :, 1] * cq[:, None]
        dx = np.maximum(np.maximum(lo[0] - px, px - hi[0]), 0)
        dy = np.maximum(np.maximum(lo[1] - py, py - hi[1]), 0)
        dz = np.maximum(np.maximum(lo[2] - X[:, 2], X[:, 2] - hi[2]), 0)[None]
        inside = (dx == 0) & (dy == 0) & (dz == 0)
        dist = np.where(inside, 0, np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        # on the surface, not buried in it: a point deep inside the panel is wrong too
        depth_in = np.minimum(np.minimum(px - lo[0], hi[0] - px), np.minimum(py - lo[1], hi[1] - py))
        on = (dist < a.tol) & ~(inside & (depth_in > a.tol))
        cnt = on.sum(1)
        k = int(np.argmax(cnt))
        if cnt[k] >= a.min_support:
            q_raw[i], support[i] = qs[k], cnt[k]
    t = np.arange(n) / fps
    ok = np.isfinite(q_raw)
    q = np.interp(t, t[ok], q_raw[ok]) if ok.any() else np.zeros(n)
    k = 7                                                   # 1/4 s median filter
    q = np.array([np.median(q[max(0, j - k):j + k + 1]) for j in range(n)])
    np.savez(os.path.join(a.anchor, "door.npz"), t=t, q_raw=q_raw, q=q, support=support)
    print(f"{a.tag}: door seen in {ok.sum()}/{n} frames; max {np.degrees(q.max()):.0f} deg at "
          f"{t[np.argmax(q)]:.1f}s; open (>10 deg) {t[q > np.deg2rad(10)].min() if (q > np.deg2rad(10)).any() else float('nan'):.1f}"
          f"-{t[q > np.deg2rad(10)].max() if (q > np.deg2rad(10)).any() else float('nan'):.1f}s", flush=True)


if __name__ == "__main__":
    main()
