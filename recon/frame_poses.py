#!/usr/bin/env python
"""Camera pose of every frame of a shot, registered frame by frame to the measured
model -- for baking colours without trusting a fused cloud.

A fused cloud is only as consistent as the poses it was fused with. On a scan that
walks all the way round a box-like object, Pi3X can mistake back views for front
views, and the fusion grows a door on the back. A single frame's depth has none of
that: it is metric and needs no other frame. So each frame's own depth is
registered to the known geometry -- the URDF at the shot's door angle, plus the
counter as a plane at z = 0 -- by robust (Tukey) point-to-plane ICP, starting from
the shot-level registration. Reflections (points the model cannot explain) are
down-weighted rather than dragging the pose. A frame is kept when enough of its
points land on the model.

Writes <run>/frames/ with poses.npz (c2w in the model frame, OpenCV) and a
registration.json with the identity transform, so bake_colors.py can take
--shot <cap> <run>/frames directly; frame_check.png overlays the model on frames.

  python recon/frame_poses.py <cap dir> <run dir> --urdf assets/microwave/microwave.urdf
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register_model import overlay  # noqa: E402
from urdf_model import Model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cap")
    ap.add_argument("run")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--stride", type=int, default=3, help="use every n-th depth pixel")
    ap.add_argument("--min-fitness", type=float, default=0.5)
    ap.add_argument("--max-rmse", type=float, default=0.006)
    a = ap.parse_args()

    reg = json.load(open(os.path.join(a.run, "registration.json")))
    T = np.array(reg["T_model_to_world"])
    q = {reg["joint"]: reg["joint_angle_rad"]}
    c2w0 = np.linalg.inv(T)[None] @ np.load(os.path.join(a.run, "poses.npz"))["c2w"]   # model frame
    intr = json.load(open(os.path.join(a.cap, "intrinsics.json")))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    names = sorted(os.listdir(os.path.join(a.cap, "input")))

    model = Model(a.urdf)
    counter = trimesh.creation.box(extents=[1.6, 1.6, 0.001])
    counter.apply_translation([0.23, 0.15, -0.0005])
    tm = trimesh.util.concatenate([model.mesh(q), counter])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices),
                                     o3d.utility.Vector3iVector(tm.faces))
    mesh.compute_triangle_normals()
    target = mesh.sample_points_uniformly(200000, use_triangle_normal=True)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    H, W = intr["height"], intr["width"]
    vv, uu = np.mgrid[0:H:a.stride, 0:W:a.stride]
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=0.01))
    out, stats = [], []
    for i, n in enumerate(names):
        z = cv2.imread(os.path.join(a.cap, "depth", n), cv2.IMREAD_UNCHANGED)[::a.stride, ::a.stride] / 1000.0
        m = (z > 0.1) & (z < 1.2)
        Xc = np.stack([(uu[m] - K[0, 2]) / K[0, 0] * z[m], (vv[m] - K[1, 2]) / K[1, 1] * z[m], z[m]], 1)
        P = c2w0[i].copy()
        for thr in (0.04, 0.02, 0.01, 0.006):
            Xm = Xc @ P[:3, :3].T + P[:3, 3]
            near = rc.compute_distance(o3d.core.Tensor(Xm.astype(np.float32))).numpy() < 2 * thr + 0.02
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xm[near]))
            r = o3d.pipelines.registration.registration_icp(
                src, target, thr, np.eye(4), est,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            P = r.transformation @ P
        Xm = Xc @ P[:3, :3].T + P[:3, 3]
        d = rc.compute_distance(o3d.core.Tensor(Xm.astype(np.float32))).numpy()
        on_mw = d < 0.006
        mw_region = (Xm[:, 2] > 0.01) & (np.abs(Xm[:, 0] - 0.23) < 0.4) & (np.abs(Xm[:, 1] - 0.1) < 0.4)
        fit = float(on_mw[mw_region].mean()) if mw_region.sum() > 500 else 0.0
        rmse = float(np.sqrt((d[on_mw] ** 2).mean())) if on_mw.any() else 1.0
        dT = np.linalg.inv(c2w0[i]) @ P
        moved = (np.linalg.norm(dT[:3, 3]) * 1000,
                 np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1))))
        ok = fit >= a.min_fitness and rmse <= a.max_rmse
        stats.append(dict(frame=n, fitness=fit, rmse_mm=rmse * 1000, moved_mm=moved[0],
                          moved_deg=moved[1], kept=ok))
        out.append(P)
    out = np.array(out)
    keep = np.array([s["kept"] for s in stats])

    fdir = os.path.join(a.run, "frames")
    os.makedirs(fdir, exist_ok=True)
    np.savez(os.path.join(fdir, "poses.npz"), c2w=out, kept=keep, names=names)
    json.dump(dict(reg, T_model_to_world=np.eye(4).tolist(), frames_kept=int(keep.sum()),
                   per_frame=stats), open(os.path.join(fdir, "registration.json"), "w"), indent=1)
    for s in stats:
        print(f"{s['frame']}: fit {s['fitness']:.2f} rmse {s['rmse_mm']:.1f} mm, moved "
              f"{s['moved_mm']:5.1f} mm {s['moved_deg']:5.1f} deg {'' if s['kept'] else '  DROPPED'}")
    print(f"{keep.sum()}/{len(keep)} frames kept -> {fdir}", flush=True)
    overlay(a.cap, out, model, q, np.eye(4), os.path.join(fdir, "frame_check.png"))


if __name__ == "__main__":
    main()
