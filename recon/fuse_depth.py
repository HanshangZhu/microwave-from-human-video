#!/usr/bin/env python
"""Fuse the RealSense depth along Pi3X camera poses (TSDF), instead of Pi3X's own depth.

A Pi3X cloud is each view's *predicted* depth placed at each view's predicted pose.
On the Snap Pot orbit every view is sharp on its own but the views disagree about
where the lid is, so the cloud comes out haloed. The measured depth is consistent
from view to view and is in metres, so only the poses are taken from Pi3X:

  1. Pi3X on <cap>/input, optionally conditioned on the real intrinsics and depth
     (--cond K or K+depth; Pi3X accepts both as priors)
  2. metric scale: per view, median(measured depth / Pi3X depth) over confident
     pixels; the median over views scales the pose translations. The spread across
     views is printed: if it is wide, the views were predicted at different scales.
  3. every view's measured depth is back-projected with the real colour intrinsics
     and fused into a TSDF volume (Open3D, metres)

Writes into --out:
  poses.npz     metric camera-to-world poses, per-view scale ratios
  views.npz     a subsample of every view's back-projected points, with view ids
  fused.ply     coloured points from the TSDF surface
  mesh.ply      the TSDF mesh

<cap> is the layout written by capture_to_r2s.py (input/, depth/ in mm aligned to
colour, intrinsics.json). Runs in the env with torch + pi3 + open3d (3dgrut):

  python recon/fuse_depth.py <cap> --out <dir> [--cond none|K|K+depth] [--voxel 0.003]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.environ.get("PI3_ROOT", os.path.expanduser("~/Research/R2S/external/Pi3")))
from pi3.models.pi3x import Pi3X  # noqa: E402
from pi3.utils.basic import load_multimodal_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", choices=["none", "K", "K+depth"], default="none")
    ap.add_argument("--voxel", type=float, default=0.003, help="TSDF voxel, metres")
    ap.add_argument("--trunc", type=float, default=0.012, help="TSDF truncation, metres")
    ap.add_argument("--max-depth", type=float, default=1.5, help="ignore measured depth beyond, m")
    ap.add_argument("--conf", type=float, default=0.3, help="Pi3X confidence for the scale fit")
    ap.add_argument("--depth", choices=["measured", "pi3x"], default="measured",
                    help="what is fused: the RealSense depth, or Pi3X's own predicted depth at the "
                         "measured scale (confident pixels only; for surfaces the sensor gets wrong)")
    ap.add_argument("--poses", help="npz with metric camera-to-world 'c2w' (OpenCV); skips "
                                    "Pi3X -- used to test the fusion on renders with known poses")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    names = sorted(f for f in os.listdir(os.path.join(a.cap, "input")) if f.endswith(".png"))
    intr = json.load(open(os.path.join(a.cap, "intrinsics.json")))
    W, H = intr["width"], intr["height"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    D = np.stack([cv2.imread(os.path.join(a.cap, "depth", n), cv2.IMREAD_UNCHANGED)
                  .astype(np.float32) / 1000.0 for n in names])
    D[D > a.max_depth] = 0
    N = len(names)

    if a.poses:
        P = np.load(a.poses)["c2w"].astype(np.float64)
        assert len(P) == N, f"{len(P)} poses for {N} frames"
        np.savez(os.path.join(a.out, "poses.npz"), c2w=P, scale=1.0, view_ratio=np.ones(N),
                 names=names, cond="given")
        return fuse(a, names, K, W, H, D, P)

    # 1. Pi3X poses
    dev = torch.device("cuda")
    cond = None
    if a.cond != "none":
        cond = dict(intrinsics=np.repeat(K[None].astype(np.float32), N, 0),
                    depths=D.copy() if a.cond == "K+depth" else None)
    imgs, c = load_multimodal_data(os.path.join(a.cap, "input"), conditions=cond, interval=1,
                                   device=dev, verbose=False)
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(dev).eval()
    kw = {}
    if cond is None:
        model.disable_multimodal()
    else:
        kw = {k: v for k, v in c.items() if k in ("intrinsics", "depths") and v is not None}
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        res = model(imgs, **kw)
    poses = res["camera_poses"][0].float().cpu().numpy()             # (N,4,4) c2w, Pi3 units
    lz = res["local_points"][0, ..., 2].float().cpu().numpy()         # (N,h,w) Pi3 depth
    conf = torch.sigmoid(res["conf"][0, ..., 0].float()).cpu().numpy()
    print(f"Pi3X ({a.cond}) on {N} views: {time.time()-t0:.1f}s", flush=True)
    del res, imgs
    torch.cuda.empty_cache()

    # 2. metric scale, per view and overall
    h, w = lz.shape[1:]
    ratios = []
    for n in range(N):
        d = cv2.resize(D[n], (w, h), interpolation=cv2.INTER_NEAREST)
        m = (d > 0) & (lz[n] > 0) & (conf[n] > a.conf)
        ratios.append(np.median(d[m] / lz[n][m]) if m.sum() > 500 else np.nan)
    ratios = np.array(ratios)
    s = float(np.nanmedian(ratios))
    p10, p90 = np.nanpercentile(ratios, [10, 90])
    print(f"scale measured/Pi3X: {s:.4f}  per-view p10-p90 {p10:.4f}-{p90:.4f} "
          f"({(p90-p10)/s*100:.1f}% of median)", flush=True)
    P = poses.copy()
    P[:, :3, 3] *= s
    np.savez(os.path.join(a.out, "poses.npz"), c2w=P, scale=s, view_ratio=ratios, names=names,
             cond=a.cond)
    if a.depth == "pi3x":
        D = np.stack([cv2.resize(np.where(conf[n] > a.conf, lz[n] * s, 0).astype(np.float32), (W, H),
                                 interpolation=cv2.INTER_NEAREST) for n in range(N)])
        D[D > a.max_depth] = 0
    fuse(a, names, K, W, H, D, P)


def fuse(a, names, K, W, H, D, P):
    """Back-project every view's measured depth with the c2w poses P; TSDF-fuse it."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=a.voxel, sdf_trunc=a.trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    pin = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rng = np.random.default_rng(0)
    vp, vc, vf = [], [], []
    for n, name in enumerate(names):
        bgr = cv2.imread(os.path.join(a.cap, "input", name))
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(D[n]), depth_scale=1.0,
            depth_trunc=a.max_depth, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, pin, np.linalg.inv(P[n]))
        m = D[n] > 0
        idx = np.nonzero(m.ravel())[0]
        idx = idx[rng.random(len(idx)) < 0.1]
        z = D[n].ravel()[idx]
        X = np.stack([(uu.ravel()[idx] - K[0, 2]) / K[0, 0] * z,
                      (vv.ravel()[idx] - K[1, 2]) / K[1, 1] * z, z], 1)
        vp.append(X @ P[n][:3, :3].T + P[n][:3, 3])
        vc.append(rgb.reshape(-1, 3)[idx])
        vf.append(np.full(len(idx), n, np.int16))
    np.savez_compressed(os.path.join(a.out, "views.npz"), P=np.concatenate(vp).astype(np.float32),
                        C=np.concatenate(vc), fid=np.concatenate(vf), cams=P[:, :3, 3])
    pc = vol.extract_point_cloud()
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_point_cloud(os.path.join(a.out, "fused.ply"), pc)
    o3d.io.write_triangle_mesh(os.path.join(a.out, "mesh.ply"), mesh)
    print(f"TSDF: {len(pc.points):,} points, {len(mesh.triangles):,} triangles -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
