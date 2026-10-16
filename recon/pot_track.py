#!/usr/bin/env python
"""Where the Snap Pot starts and where it ends up in a take, in the microwave's frame.

Start, on the counter: the lid is the one saturated teal thing in view. A pot at rest
stands upright, so its lid lies in the plane z = pot height; the ray through the lid's
pixel centroid meets that plane at the pot's axis (more stable than the depth, which
is checked against it). The lid outline in the plane gives the yaw (long side).

End, in the cavity: the camera looks into the cavity almost level, so the lid is not
visible there -- the pot's white side is. With the door open, unsaturated pixels inside
the cavity opening clearly brighter than the (dark) cavity whose measured depth puts them inside the cavity, above the
turntable and below the pot's top, are the pot's front surface (reflections off the
steel land outside that volume); the pot's axis is half a pot further along the view.

Pick: the first frame the lid is more than 1.5 cm above its resting height (by depth);
pushing the pot along the counter does not count. The pick pose is where it last rested
before that (it may have been pushed since the start). Place: the start of the
last steady run of in-cavity detections before the door closes again.

Writes <anchor>/pot.json, pot_track.npz (per frame: lid x, y, yaw on the counter,
lifted, in-cavity x, y) and pot_check.png. Runs with numpy + cv2 (rscap):
  python recon/pot_track.py <capture dir> <tag> --anchor <dir>
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_to_r2s import align_depth  # noqa: E402

POT_H, POT_HALF_DEPTH = 0.0635, 0.026        # assets/heinz_snap_pot: height; ~half of 48-55 mm
TEAL = ((80, 90, 40), (105, 255, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("tag")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--params", default="assets/microwave/parameters.json")
    ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.capture, a.tag + ".json")))
    ci, d = meta["intrinsics"], meta["depth"]
    R = np.array(d["depth_to_color"]["rotation_colmajor"]).reshape(3, 3).T
    tr = np.array(d["depth_to_color"]["translation_m"])
    fps = meta.get("fps", 30)
    ddir = os.path.join(a.capture, d["folder"])
    K = np.array([[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    der = json.load(open(a.params))["derived"]
    (x0, x1), (cy0, cy1), (z0, z1) = der["cavity_x_range"], der["cavity_y_range"], der["cavity_z_range"]
    turn = der["turntable_top_z"]
    reg = json.load(open(os.path.join(a.anchor, "run", "registration.json")))
    T_last_model = np.linalg.inv(np.array(reg["T_model_to_world"]))
    segs = json.load(open(os.path.join(a.anchor, "anchor.json")))["segments"]
    door_q = np.load(os.path.join(a.anchor, "door.npz"))["q"]

    def cam(i):
        sg = next((s for s in segs if s["start"] <= i < s["end"]), segs[-1])
        return T_last_model @ np.array(sg["T_last_from_cam"])

    def on_plane(C, uv, zp):
        r = (np.c_[uv, np.ones(len(uv))] @ Ki.T) @ C[:3, :3].T     # rays in the model frame
        s = (zp - C[2, 3]) / r[:, 2]
        return C[:3, 3] + s[:, None] * r

    cap = cv2.VideoCapture(os.path.join(a.capture, a.tag + ".avi"))
    n = len(door_q)
    lid = np.full((n, 3), np.nan)                  # x, y, yaw on the counter
    lid_depth_z = np.full(n, np.nan)
    inside = np.full((n, 2), np.nan)
    for i in range(0, n, a.step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            break
        C = cam(i)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.morphologyEx(cv2.inRange(hsv, *TEAL), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        za = None
        if cs:
            c = max(cs, key=cv2.contourArea)
            if cv2.contourArea(c) > 400:
                P = on_plane(C, c[:, 0, :].astype(float), POT_H)
                mo = cv2.moments(c)
                ctr = on_plane(C, np.array([[mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]]]), POT_H)[0]
                xy = P[:, :2] - ctr[:2]
                w, v = np.linalg.eigh(xy.T @ xy)
                lid[i] = [ctr[0], ctr[1], np.arctan2(v[1, 1], v[0, 1])]
                za = align_depth(cv2.imread(os.path.join(ddir, f"{i:06d}.png"), cv2.IMREAD_UNCHANGED),
                                 d["depth_scale_m"], d["intrinsics"], ci, R, tr)
                inn = np.zeros(m.shape, np.uint8)
                cv2.drawContours(inn, [c], -1, 255, -1)
                zz = za[(inn > 0) & (za > 0)]
                if len(zz) > 50:
                    vv, uu = np.nonzero((inn > 0) & (za > 0))
                    X = (np.c_[uu, vv, np.ones(len(uu))] @ Ki.T) * za[vv, uu][:, None]
                    lid_depth_z[i] = np.median((X @ C[:3, :3].T + C[:3, 3])[:, 2])
        if door_q[i] > np.deg2rad(30):
            w2c = np.linalg.inv(C)
            opening = np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]]) @ w2c[:3, :3].T + w2c[:3, 3]
            poly = ((opening @ K.T)[:, :2] / opening[:, 2:3]).astype(np.int32)
            pm = np.zeros(hsv.shape[:2], np.uint8)
            cv2.fillPoly(pm, [poly], 255)
            # the pot is in the cavity's shade: "white" relative to the dark cavity
            vmed = np.median(hsv[..., 2][pm > 0])
            white = (pm > 0) & (hsv[..., 1] < 70) & (hsv[..., 2] > max(90, vmed + 40))
            if white.sum() > 200:
                if za is None:
                    za = align_depth(cv2.imread(os.path.join(ddir, f"{i:06d}.png"), cv2.IMREAD_UNCHANGED),
                                     d["depth_scale_m"], d["intrinsics"], ci, R, tr)
                vv, uu = np.nonzero(white & (za > 0))
                X = ((np.c_[uu, vv, np.ones(len(uu))] @ Ki.T) * za[vv, uu][:, None]) @ C[:3, :3].T + C[:3, 3]
                ok = (X[:, 0] > x0) & (X[:, 0] < x1) & (X[:, 1] > cy0 - 0.02) & (X[:, 1] < cy1) \
                    & (X[:, 2] > turn + 0.005) & (X[:, 2] < turn + POT_H + 0.01)
                if ok.sum() > 150:
                    p = np.median(X[ok], 0)
                    view = p - C[:3, 3]
                    view[2] = 0
                    inside[i] = (p + POT_HALF_DEPTH * view / np.linalg.norm(view))[:2]

    t = np.arange(n) / fps
    have = np.isfinite(lid[:, 0])
    first = np.nonzero(have)[0][:10]
    start = np.median(lid[first], 0)
    # picked = the lid leaves the counter (its measured height rises); a push along the
    # counter only moves it sideways
    lifted = np.isfinite(lid_depth_z) & (lid_depth_z - POT_H > 0.015)
    i_pick = int(np.nonzero(lifted)[0][0]) if lifted.any() else None
    # where it is picked from: the last resting lid positions before the lift (it may
    # have been pushed since the start)
    pick = start
    if i_pick is not None:
        rest = np.nonzero(have & ~lifted & (np.arange(n) < i_pick))[0][-8:]
        if len(rest):
            pick = np.median(lid[rest], 0)
    hi = np.nonzero(np.isfinite(inside[:, 0]))[0]
    hi = hi[hi > (i_pick or 0)]
    end, i_place = None, None
    if len(hi):
        # the last steady run: detections within 1.5 cm of the final position
        fin = np.median(inside[hi[-6:]], 0)
        steady = hi[np.hypot(*(inside[hi] - fin).T) < 0.015]
        run = steady[np.nonzero(np.diff(np.r_[-99, steady]) > 3 * a.step)[0][-1]:]
        i_place, end = int(run[0]), np.median(inside[run], 0)
    res = dict(start=dict(x=float(start[0]), y=float(start[1]), z=0.0, yaw=float(start[2]),
                          lid_depth_z_minus_model=float(np.nanmedian(lid_depth_z[first]) - POT_H)),
               pick=dict(x=float(pick[0]), y=float(pick[1]), z=0.0, yaw=float(pick[2])),
               end=None if end is None else dict(x=float(end[0]), y=float(end[1]), z=turn, yaw=None),
               t_pick=None if i_pick is None else i_pick / fps,
               t_place=None if i_place is None else i_place / fps,
               lid_frames=int(have.sum()), inside_frames=int(np.isfinite(inside[:, 0]).sum()),
               frame="microwave model: z up, counter z = 0; x, y = pot axis; yaw of the long side")
    json.dump(res, open(os.path.join(a.anchor, "pot.json"), "w"), indent=1)
    np.savez(os.path.join(a.anchor, "pot_track.npz"), t=t, lid=lid, lifted=lifted, inside=inside)
    print(f"{a.tag}: start ({start[0]:.3f}, {start[1]:.3f}) yaw {np.degrees(start[2]):.0f} deg "
          f"[lid depth vs model {res['start']['lid_depth_z_minus_model']*1000:+.0f} mm]; "
          + f"picked from ({pick[0]:.3f}, {pick[1]:.3f}); "
          + (f"end ({end[0]:.3f}, {end[1]:.3f}) on the turntable" if end is not None else "end NOT FOUND")
          + f"; pick {res['t_pick']}s, place {res['t_place']}s", flush=True)
    check(a, cap, cam, K, res, i_pick, i_place, door_q, n, fps, turn)


def check(a, cap, cam, K, res, i_pick, i_place, door_q, n, fps, turn):
    def box(x, y, yaw, z):                     # the pot as its rim box, 95 x 74 x 63.5 mm
        c, s = np.cos(yaw or 0), np.sin(yaw or 0)
        pts = []
        for dz in (0, POT_H):
            for dx, dy in ((-0.0475, -0.037), (0.0475, -0.037), (0.0475, 0.037), (-0.0475, 0.037)):
                pts.append([x + c * dx - s * dy, y + s * dx + c * dy, z + dz])
        return np.array(pts)
    frames = [int(1.5 * fps), i_pick or 0, i_place or 0, min(n - 1, (i_place or 0) + 15)]
    tiles = []
    for i, what in zip(frames, ("start", "pick", "place", "after")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        w2c = np.linalg.inv(cam(i))
        for key, col in (("start", (0, 255, 255)), ("pick", (255, 160, 0)), ("end", (0, 0, 255))):
            p = res[key]
            if p is None:
                continue
            B = box(p["x"], p["y"], p["yaw"], p["z"]) @ w2c[:3, :3].T + w2c[:3, 3]
            uv = ((B @ K.T)[:, :2] / B[:, 2:3]).astype(int)
            for e in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
                cv2.line(f, tuple(uv[e[0]]), tuple(uv[e[1]]), col, 2)
        cv2.putText(f, f"{what} {i / fps:.1f}s  door {np.degrees(door_q[i]):.0f}", (10, 40), 0, 1.2, (0, 255, 0), 3)
        tiles.append(cv2.resize(f, (640, 360)))
    p = os.path.join(a.anchor, "pot_check.png")
    cv2.imwrite(p, np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    print("check ->", p)


if __name__ == "__main__":
    main()
