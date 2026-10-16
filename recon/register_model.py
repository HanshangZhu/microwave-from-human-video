#!/usr/bin/env python
"""Register a measured URDF model (e.g. assets/microwave) to a fused scan of it, and
find the door angle the scan was taken at.

The geometry is known exactly; the scan supplies only where it is. So:

  1. counter: the largest roughly-horizontal plane below where the cameras look.
     Its normal is "up", the model's base sits on it. Up is seeded from the
     cameras' own up vectors (a hand-held camera is held roughly upright).
  2. facing: seeded by the mean camera position (the front faces it when the scan
     is an arc in front), but every yaw in 15 deg steps is tried, because a scan
     that walks all the way round puts the mean camera nowhere in particular. The
     front is not the back -- the door stands 3.6 cm proud, the handle and knob
     further -- so the scores tell them apart.
  3. a grid of starts (yaw, along-counter shifts, door closed or open) is scored,
     the best few are refined by point-to-plane ICP against the model surface
     (counter points removed), each ICP stage projected back to the model standing
     on the counter (x, y, yaw only; --free lifts this), and after each stage the door angle is
     re-searched in 1 deg steps (most scan points on the door surface wins).
  4. the winner is the start that explains the most scan points (within 8 mm).

Writes into the run dir: registration.json (T_model_to_world, joint angle, inlier
statistics) and registration_overlay.png (model edges drawn over three frames --
look at it).

  python recon/register_model.py <cap dir> <run dir> --urdf assets/microwave/microwave.urdf
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_model import Model  # noqa: E402

INLIER = 0.008


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Scorer:
    """Distances from scan points to the model surface (exact, by ray-casting)."""

    def __init__(self, model, joint):
        self.model, self.joint, self.cache = model, joint, {}

    def scene(self, q, links=None):
        key = (round(q, 4), tuple(links) if links else None)
        if key not in self.cache:
            m = self.model.mesh({self.joint: q}, links=links)
            s = o3d.t.geometry.RaycastingScene()
            s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
                o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                          o3d.utility.Vector3iVector(m.faces))))
            self.cache[key] = s
        return self.cache[key]

    def dist(self, X_model, q, links=None):
        return self.scene(q, links).compute_distance(
            o3d.core.Tensor(X_model.astype(np.float32))).numpy()


def to_model(T, X):
    Ti = np.linalg.inv(T)
    return X @ Ti[:3, :3].T + Ti[:3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cap")
    ap.add_argument("run")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--joint", default="door_hinge")
    ap.add_argument("--door-link", default="door")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--seed", help="'frame x0 y0 x1 y1': a box around the object in one frame, for "
                                   "scans that do not keep looking at it (a room sweep): its "
                                   "depth places the object, that camera sets the front")
    ap.add_argument("--free", action="store_true",
                    help="let ICP tilt and lift the model off the counter (6 dof)")
    a = ap.parse_args()

    model = Model(a.urdf)
    lo, hi = model.mesh(links=[model.root]).bounds
    r_door = np.linalg.norm(model.mesh(links=[a.door_link]).vertices[:, :2], axis=1).max()
    qmax = 1.75
    sc = Scorer(model, a.joint)

    pcd = o3d.io.read_point_cloud(os.path.join(a.run, "fused.ply")).voxel_down_sample(a.voxel)
    X = np.asarray(pcd.points)
    c2w = np.load(os.path.join(a.run, "poses.npz"))["c2w"]
    C, Z, Yd = c2w[:, :3, 3], c2w[:, :3, 2], c2w[:, :3, 1]

    # where the cameras look: least-squares intersection of the optical axes
    A = sum(np.eye(3) - np.outer(z, z) for z in Z)
    b = sum((np.eye(3) - np.outer(z, z)) @ c for z, c in zip(Z, C))
    t0 = np.linalg.lstsq(A, b, rcond=None)[0]       # one camera (a still view): --seed sets it
    up = -Yd.mean(0)
    up /= np.linalg.norm(up)
    front_cam = C.mean(0)
    if a.seed:
        fi, x0, y0, x1, y1 = [int(v) for v in a.seed.split()]
        intr = json.load(open(os.path.join(a.cap, "intrinsics.json")))
        name = sorted(os.listdir(os.path.join(a.cap, "depth")))[fi]
        z = cv2.imread(os.path.join(a.cap, "depth", name), cv2.IMREAD_UNCHANGED)[y0:y1, x0:x1] / 1000.0
        vv, uu = np.nonzero(z > 0.1)
        zz = z[vv, uu]
        P = np.stack([(uu + x0 - intr["cx"]) / intr["fx"] * zz,
                      (vv + y0 - intr["cy"]) / intr["fy"] * zz, zz], 1)
        t0 = c2w[fi][:3, :3] @ np.median(P, 0) + c2w[fi][:3, 3]
        up = -Yd[fi] / np.linalg.norm(Yd[fi])
        front_cam = C[fi]
        print(f"seed: frame {fi}, {len(zz):,} depth pixels in the box, median depth {np.median(zz):.2f} m",
              flush=True)

    # 1. counter plane
    rest, best = o3d.geometry.PointCloud(pcd), None
    for _ in range(8):
        if len(rest.points) < 1000:
            break
        (n0, n1, n2, d), idx = rest.segment_plane(0.006, 3, 1000)
        n = np.array([n0, n1, n2])
        if n @ up < 0:
            n, d = -n, -d
        below = n @ t0 + d                         # height of the look-at point above it
        if n @ up > np.cos(np.deg2rad(30)) and 0.0 < below < 0.6:
            if best is None or len(idx) > best[2]:
                best = (n, d, len(idx))
        rest = rest.select_by_index(idx, invert=True)
    if best is None:
        sys.exit("no counter plane found below the look-at point")
    up, d, n_counter = best
    h = X @ up + d
    X = X[np.abs(h) > 0.01]                        # drop the counter itself
    print(f"counter: {n_counter:,} points; up vs camera-up {np.degrees(np.arccos(np.clip(up @ (-Yd.mean(0) / np.linalg.norm(Yd.mean(0))), -1, 1))):.1f} deg",
          flush=True)

    # 2. facing: model +y points away from the cameras
    t0p = t0 - (up @ t0 + d) * up                  # look-at point dropped onto the counter
    f = front_cam - t0p
    f -= (f @ up) * up
    f /= np.linalg.norm(f)
    y_m = -f
    x_m = np.cross(y_m, up)
    R0 = np.stack([x_m, y_m, up], 1)
    front_ctr = np.array([(lo[0] + hi[0]) / 2, 0.0, 0.0])

    def pose(yaw, dx, dy):
        R = R0 @ rot_z(yaw)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t0p + R0 @ np.array([dx, dy, 0]) - R @ front_ctr
        return T

    def on_counter(T):
        """Stand the model on the counter: its z axis is the counter normal and its
        base lies in the counter plane; only the in-plane pose (x, y, yaw) is kept.
        The counter is fitted to ~1e5 points; glossy steel lets free 6-dof ICP tip
        the model by a degree or two and lift it by a centimetre."""
        if a.free:
            return T
        x = T[:3, 0] - (T[:3, 0] @ up) * up
        x /= np.linalg.norm(x)
        S = np.eye(4)
        S[:3, :3] = np.stack([x, np.cross(up, x), up], 1)
        S[:3, 3] = T[:3, 3] - (up @ T[:3, 3] + d) * up
        return S

    def crop(T):
        Xm = to_model(T, X)
        m = np.all((Xm > lo - 0.05) & (Xm < hi + [0.05, 0.05, 0.05]), 1)
        m |= (np.linalg.norm(Xm[:, :2], axis=1) < r_door + 0.03) & (Xm[:, 1] < 0.02)  # door sweep
        return X[m & (Xm[:, 2] > -0.02) & (Xm[:, 2] < hi[2] + 0.05)]

    def score(T, q):
        Xc = crop(T)
        return int((sc.dist(to_model(T, Xc), q) < INLIER).sum()) if len(Xc) else 0

    def door_search(T):
        Xm = to_model(T, crop(T))
        body = sc.dist(Xm, 0.0, links=[model.root]) < INLIER
        Xd = Xm[~body]                             # points the body does not explain
        qs = np.deg2rad(np.arange(0, np.degrees(qmax) + 0.5, 1.0))
        cnt = [int((sc.dist(Xd, q, links=[a.door_link]) < INLIER).sum()) for q in qs]
        return float(qs[int(np.argmax(cnt))])

    # 3. starts
    starts = [(yaw, dx, dy, q) for yaw in np.deg2rad(np.arange(0, 360, 15))
              for dx in (-0.1, 0, 0.1) for dy in (-0.1, 0, 0.1, 0.2) for q in (0.0, np.pi / 2)]
    ranked = sorted(starts, key=lambda s: -score(pose(*s[:3]), s[3]))[:8]
    results = []
    for yaw, dx, dy, q in ranked:
        T = pose(yaw, dx, dy)
        for thr in (0.04, 0.02, 0.01, 0.006):
            tm = model.mesh({a.joint: q})
            mm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices),
                                           o3d.utility.Vector3iVector(tm.faces))
            mm.compute_triangle_normals()
            tgt = mm.sample_points_uniformly(60000, use_triangle_normal=True)
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(to_model(T, crop(T))))
            r = o3d.pipelines.registration.registration_icp(
                src, tgt, thr, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
            T = on_counter(T @ np.linalg.inv(r.transformation))  # src moved into the model -> move model
            q = door_search(T)
        results.append((score(T, q), T, q))
    s_best, T, q = max(results, key=lambda r: r[0])

    Xc = crop(T)
    dd = sc.dist(to_model(T, Xc), q)
    inl = dd < INLIER
    out = dict(T_model_to_world=T.tolist(), joint=a.joint, joint_angle_rad=q,
               joint_angle_deg=float(np.degrees(q)), inliers=int(inl.sum()),
               inlier_fraction_of_crop=float(inl.mean()),
               inlier_rms_mm=float(np.sqrt((dd[inl] ** 2).mean()) * 1000),
               counter_normal=up.tolist(), urdf=os.path.abspath(a.urdf))
    json.dump(out, open(os.path.join(a.run, "registration.json"), "w"), indent=1)
    print(f"door {np.degrees(q):.0f} deg; {inl.sum():,} inliers ({inl.mean()*100:.0f}% of crop), "
          f"rms {out['inlier_rms_mm']:.1f} mm; starts' scores {sorted(r[0] for r in results)[::-1][:4]}",
          flush=True)
    overlay(a.cap, c2w, model, {a.joint: q}, T, os.path.join(a.run, "registration_overlay.png"))


def overlay(cap, c2w, model, q, T, out):
    intr = json.load(open(os.path.join(cap, "intrinsics.json")))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    names = sorted(os.listdir(os.path.join(cap, "input")))
    edges = []
    for link, vs in model.visuals.items():
        L = model.link_T(link, q)
        for v in vs:
            if v["kind"] != "box":
                continue
            M = T @ L @ v["T"]
            hs = np.array(v["size"]) / 2
            for i in range(8):                     # the 12 edges join corners differing in one axis
                ci = np.array([(i >> k) & 1 for k in range(3)]) * 2 - 1
                for k in range(3):
                    if ci[k] < 0:
                        cj = ci.copy()
                        cj[k] = 1
                        edges.append(tuple(M[:3, :3] @ (c * hs) + M[:3, 3] for c in (ci, cj)))
    tiles = []
    for i in (0, len(names) // 2, len(names) - 1):
        img = cv2.imread(os.path.join(cap, "input", names[i]))
        w2c = np.linalg.inv(c2w[i])
        for p, r in edges:
            P = np.linspace(p, r, 30) @ w2c[:3, :3].T + w2c[:3, 3]
            P = P[P[:, 2] > 0.05]
            if len(P) < 2:
                continue
            uv = (P @ K.T)[:, :2] / P[:, 2:3]
            cv2.polylines(img, [uv.astype(np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    cv2.imwrite(out, np.hstack(tiles))
    print("overlay ->", out)


if __name__ == "__main__":
    main()
