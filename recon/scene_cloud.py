#!/usr/bin/env python
"""The scanned scene as point clouds in the simulation frame.

Every shot was registered to the same measured URDF (register_model.py), so
T_model_to_world^-1 puts each shot's fused cloud into the model's frame -- the frame
the URDF is simulated in (z up, the counter at z = 0, origin under the front-left
corner of the microwave). Writes into --out:

  scene_<shot>.ply   each shot's whole fused cloud, in that frame
  scene_static.ply   all shots merged, without the points on the door (the door
                     is a joint in simulation and stands at a different angle in
                     each shot), voxel-downsampled
  scene_check.json   how well the shots agree: distances between the shots' static
                     points where they overlap, and each shot's door angle

  python recon/scene_cloud.py --urdf assets/microwave/microwave.urdf --out <dir> \\
      --shot closed <run dir> --shot open_90 <run dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register_model import Scorer, to_model  # noqa: E402
from urdf_model import Model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shot", nargs=2, action="append", required=True, metavar=("NAME", "RUN"))
    ap.add_argument("--door-link", default="door")
    ap.add_argument("--voxel", type=float, default=0.003)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = Model(a.urdf)

    static, check = [], {"shots": {}}
    for name, run in a.shot:
        reg = json.load(open(os.path.join(run, "registration.json")))
        T = np.array(reg["T_model_to_world"])
        pc = o3d.io.read_point_cloud(os.path.join(run, "fused.ply"))
        X = to_model(T, np.asarray(pc.points))
        C = np.asarray(pc.colors)
        q = reg["joint_angle_rad"]
        sc = Scorer(model, reg["joint"])
        on_door = sc.dist(X, q, links=[a.door_link]) < 0.01
        on_body = sc.dist(X, q, links=[model.root]) < 0.01
        out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
        out.colors = o3d.utility.Vector3dVector(C)
        o3d.io.write_point_cloud(os.path.join(a.out, f"scene_{name}.ply"), out)
        keep = ~on_door
        static.append((name, X[keep], C[keep], on_body[keep]))
        check["shots"][name] = dict(points=int(len(X)), door_deg=reg["joint_angle_deg"],
                                    door_points_removed=int(on_door.sum()),
                                    registration_inlier_rms_mm=reg["inlier_rms_mm"])
        print(f"{name}: {len(X):,} points, door {reg['joint_angle_deg']:.0f} deg, "
              f"{on_door.sum():,} on the door", flush=True)

    # do the shots agree? nearest-neighbour distance between shots, on the static
    # points both see: the microwave body, and the counter within 30 cm of it
    def near(X, body):
        return body | ((np.abs(X[:, 2]) < 0.01) & (X[:, 0] > -0.3) & (X[:, 0] < 0.76)
                       & (X[:, 1] > -0.3) & (X[:, 1] < 0.6))
    for i in range(len(static)):
        for j in range(i + 1, len(static)):
            (ni, Xi, _, bi), (nj, Xj, _, bj) = static[i], static[j]
            A = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xi[near(Xi, bi)]))
            B = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xj[near(Xj, bj)]))
            d = np.asarray(A.compute_point_cloud_distance(B))
            d = d[d < 0.03]                          # where both shots saw the surface
            check[f"{ni}_vs_{nj}"] = dict(overlap_points=int(len(d)),
                                          median_mm=float(np.median(d) * 1000),
                                          p90_mm=float(np.percentile(d, 90) * 1000))
            print(f"{ni} vs {nj}: overlap {len(d):,} points, median {np.median(d)*1000:.1f} mm, "
                  f"p90 {np.percentile(d, 90)*1000:.1f} mm", flush=True)

    m = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([s[1] for s in static])))
    m.colors = o3d.utility.Vector3dVector(np.concatenate([s[2] for s in static]))
    m = m.voxel_down_sample(a.voxel)
    o3d.io.write_point_cloud(os.path.join(a.out, "scene_static.ply"), m)
    check["static_points"] = len(m.points)
    check["frame"] = "URDF root link: z up, counter z = 0, origin below the front-left corner"
    json.dump(check, open(os.path.join(a.out, "scene_check.json"), "w"), indent=1)
    print(f"scene_static.ply: {len(m.points):,} points -> {a.out}")


if __name__ == "__main__":
    main()
