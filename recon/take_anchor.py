#!/usr/bin/env python
"""Anchor a demonstration take recorded from a fixed camera: one clean RGB-D view of
the scene without the hand, as a capture + run that register_model.py can take.

The take's last seconds are the scene at rest (hand gone, door closed). Their depth,
aligned to colour, is median-filtered per pixel over --still-s seconds, which removes
most of the D435 noise; the colour is the per-pixel median too. The view is written as
a one-frame capture (input/, depth/, intrinsics.json) and a run (fused.ply = that view
back-projected, poses.npz = the camera at the origin), so

  python recon/register_model.py <out> <out>/run --urdf ... --seed "0 x0 y0 x1 y1"

gives the model in the camera frame, i.e. the camera in the model (= simulation) frame.
The camera is fixed but can be knocked (take_000: opening the door wide moved it by
~1 deg). The background's image shift against the last frame is tracked every few
frames; a run of frames shifted by more than --jump-px is a separate camera segment.
Each earlier segment gets its own still view (its first quiet second) and is aligned
to the last view by point-to-plane ICP on the static scene. anchor.json lists the
segments with T_last_from_cam, the pose of that segment's camera in the last one.

  python recon/take_anchor.py <capture dir> <tag> --out <dir>
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_to_r2s import align_depth  # noqa: E402


def still(cap_path, meta, frames, ddir):
    ci, d = meta["intrinsics"], meta["depth"]
    R = np.array(d["depth_to_color"]["rotation_colmajor"]).reshape(3, 3).T
    t = np.array(d["depth_to_color"]["translation_m"])
    cap = cv2.VideoCapture(cap_path)
    cols, deps = [], []
    for i in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        z16 = cv2.imread(os.path.join(ddir, f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        cols.append(f)
        deps.append(align_depth(z16, d["depth_scale_m"], d["intrinsics"], ci, R, t))
    D = np.stack(deps)
    D[D == 0] = np.nan
    z = np.nanmedian(D, 0)
    valid = np.isfinite(D).mean(0) > 0.5                    # seen in most frames
    z = np.where(valid & np.isfinite(z), z, 0)
    return np.median(np.stack(cols), 0).astype(np.uint8), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("tag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--still-s", type=float, default=10.0, help="seconds at the end at rest")
    ap.add_argument("--n", type=int, default=60, help="frames used from that window")
    ap.add_argument("--jump-px", type=float, default=4.0)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.capture, a.tag + ".json")))
    ci = meta["intrinsics"]
    fps = meta.get("fps", 30)
    vid = os.path.join(a.capture, a.tag + ".avi")
    n = int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FRAME_COUNT))
    ddir = os.path.join(a.capture, meta["depth"]["folder"])
    nd = len(os.listdir(ddir))
    last = min(n, nd)
    late = np.linspace(last - a.still_s * fps, last - 1, a.n).astype(int)
    rgb, z = still(vid, meta, late, ddir)

    # camera segments: background shift against the last still view
    cap = cv2.VideoCapture(vid)
    ref = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi = (slice(0, 240), slice(1000, 1280))     # textured background, top right (the
                                                 # top left is bare wall: correlation unreliable)
    idx = np.arange(0, last, 5)
    sh = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sh.append(np.hypot(*cv2.phaseCorrelate(ref[roi], g[roi])[0]))
    sh = np.array(sh)
    sh = np.array([np.median(sh[max(0, k - 3):k + 4]) for k in range(len(sh))])   # arm passing by
    moved = sh > a.jump_px
    segs, k = [], 0
    while k < len(idx):
        j = k
        while j < len(idx) and moved[j] == moved[k]:
            j += 1
        segs.append(dict(start=int(idx[k]), end=int(idx[j]) if j < len(idx) else last,
                         moved=bool(moved[k]), shift_px=float(np.median(sh[k:j]))))
        k = j

    os.makedirs(os.path.join(a.out, "input"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "run"), exist_ok=True)
    cv2.imwrite(os.path.join(a.out, "input", "0000.png"), rgb)
    cv2.imwrite(os.path.join(a.out, "depth", "0000.png"), np.clip(z * 1000, 0, 65535).astype(np.uint16))
    json.dump(dict(width=ci["width"], height=ci["height"], fx=ci["fx"], fy=ci["fy"], cx=ci["ppx"],
                   cy=ci["ppy"], model=ci["model"], coeffs=ci["coeffs"], depth_unit="mm (uint16)",
                   aligned_to="color", source=f"{a.tag}: median of {len(late)} still frames"),
              open(os.path.join(a.out, "intrinsics.json"), "w"), indent=1)

    vv, uu = np.nonzero((z > 0.1) & (z < 2.5))
    zz = z[vv, uu]
    X = np.stack([(uu - ci["ppx"]) / ci["fx"] * zz, (vv - ci["ppy"]) / ci["fy"] * zz, zz], 1)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    pc.colors = o3d.utility.Vector3dVector(rgb[vv, uu, ::-1] / 255.0)
    o3d.io.write_point_cloud(os.path.join(a.out, "run", "fused.ply"), pc.voxel_down_sample(0.003))
    np.savez(os.path.join(a.out, "run", "poses.npz"), c2w=np.eye(4)[None], names=["0000.png"])
    def cloud(z, rgb):
        vv, uu = np.nonzero((z > 0.1) & (z < 2.5))
        zz = z[vv, uu]
        X = np.stack([(uu - ci["ppx"]) / ci["fx"] * zz, (vv - ci["ppy"]) / ci["fy"] * zz, zz], 1)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X)).voxel_down_sample(0.005)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.03, 30))
        return pc

    # a segment shorter than a second is the arm passing the background, not a move:
    # fold it into the one before (or after, at the start)
    merged = []
    for sg in segs:
        if merged and (sg["end"] - sg["start"] < fps or merged[-1]["moved"] == sg["moved"]):
            merged[-1]["end"] = sg["end"]
        else:
            merged.append(sg)
    if len(merged) > 1 and merged[0]["end"] - merged[0]["start"] < fps:
        merged[1]["start"] = merged[0]["start"]
        merged = merged[1:]
    segs = merged

    tgt = cloud(z, rgb)
    for sg in segs:
        sg["T_last_from_cam"] = np.eye(4).tolist()
        if not sg["moved"]:
            continue
        a0 = max(sg["start"], int(1.0 * fps))               # after auto-exposure settles
        e_rgb, e_z = still(vid, meta, np.arange(a0, min(a0 + int(fps), sg["end"]), 3), ddir)
        T = np.eye(4)
        for thr in (0.05, 0.02, 0.01):
            r = o3d.pipelines.registration.registration_icp(
                cloud(e_z, e_rgb), tgt, thr, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=0.01)))
            T = r.transformation
        sg["T_last_from_cam"] = T.tolist()
        sg["icp_fitness"], sg["icp_rmse_mm"] = float(r.fitness), float(r.inlier_rmse * 1000)
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        sg["moved_by"] = dict(mm=float(np.linalg.norm(T[:3, 3]) * 1000), deg=float(ang))
    json.dump(dict(frames=late.tolist(), segments=segs),
              open(os.path.join(a.out, "anchor.json"), "w"), indent=1)
    print(f"{a.tag}: still view from frames {late[0]}-{late[-1]}, depth valid {(z > 0).mean()*100:.0f}%")
    for sg in segs:
        print(f"  frames {sg['start']:4d}-{sg['end']:4d}: " + (
            f"camera moved {sg['shift_px']:.0f} px -> {sg['moved_by']['mm']:.0f} mm "
            f"{sg['moved_by']['deg']:.1f} deg (ICP fit {sg['icp_fitness']:.2f}, "
            f"{sg['icp_rmse_mm']:.1f} mm)" if sg["moved"] else "same view as the last"))


if __name__ == "__main__":
    main()
