#!/usr/bin/env python
"""Pi3X's dense cloud of a sweep, meshed directly (screened Poisson), as an alternative to
fusing depth maps into a TSDF (fuse_depth.py).

Every confident pixel of every view becomes a coloured point at Pi3X's predicted 3D position
(its global point map), scaled to metres with the scale fuse_depth.py fitted against the
measured depth (--ref <run>/poses.npz, so <run>/registration.json applies unchanged).
Poisson fills the holes a TSDF leaves where views are sparse; low-density vertices (surface
invented far from any point) are trimmed.

  python recon/pi3x_poisson.py <cap> --ref <run dir> --out <dir>      (env 3dgrut)
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.environ.get("PI3_ROOT", os.path.expanduser("~/Research/R2S/external/Pi3")))
from pi3.models.pi3x import Pi3X  # noqa: E402
from pi3.utils.basic import load_multimodal_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cap")
    ap.add_argument("--ref", required=True, help="fuse_depth.py run: its scale and poses define the frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--per-view", type=int, default=150000, help="points kept per view")
    ap.add_argument("--max-range", type=float, default=4.0, help="drop points farther from their camera, m")
    ap.add_argument("--depth", type=int, default=10, help="Poisson octree depth")
    ap.add_argument("--trim", type=float, default=0.05, help="drop this quantile of lowest-density vertices")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    intr = json.load(open(os.path.join(a.cap, "intrinsics.json")))
    names = sorted(f for f in os.listdir(os.path.join(a.cap, "input")) if f.endswith(".png"))
    N = len(names)
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float32)
    dev = torch.device("cuda")
    ref = np.load(os.path.join(a.ref, "poses.npz"))
    imgs, c = load_multimodal_data(os.path.join(a.cap, "input"), conditions=dict(
        intrinsics=np.repeat(K[None], N, 0), depths=None), interval=1, device=dev, verbose=False)
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(dev).eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        res = model(imgs, intrinsics=c["intrinsics"])
    pts = res["points"][0].float()                                   # (N,h,w,3) Pi3 world
    conf = torch.sigmoid(res["conf"][0, ..., 0].float())
    poses = res["camera_poses"][0].float().cpu().numpy()
    col = imgs[0].permute(0, 2, 3, 1).float()                        # (N,h,w,3) in [0,1]
    # the reference run's metric frame: same Pi3X frame up to scale (rotations agree), so the
    # scale is the camera-centre ratio
    s = float(np.median(np.linalg.norm(ref["c2w"][1:, :3, 3], axis=1) / np.linalg.norm(poses[1:, :3, 3], axis=1)))
    rng = np.random.default_rng(0)
    P, C = [], []
    for n in range(N):
        cam = torch.tensor(poses[n, :3, 3], device=dev)
        m = (conf[n] > a.conf) & ((pts[n] - cam).norm(dim=-1) * s < a.max_range)
        p, cc = pts[n][m].cpu().numpy(), col[n][m].cpu().numpy()
        k = rng.choice(len(p), min(a.per_view, len(p)), replace=False)
        P.append(p[k] * s)
        C.append(cc[k])
    P, C = np.concatenate(P), np.concatenate(C)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(C)
    pc = pc.voxel_down_sample(0.004)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    pc.orient_normals_towards_camera_location(ref["c2w"][:, :3, 3].mean(0))
    o3d.io.write_point_cloud(os.path.join(a.out, "cloud.ply"), pc)
    print(f"scale {s:.4f}; {len(pc.points):,} points after 4 mm voxel", flush=True)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=a.depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, a.trim))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(a.out, "mesh.ply"), mesh)
    print(f"Poisson depth {a.depth}: {len(mesh.triangles):,} triangles -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
