#!/usr/bin/env python
"""Pi3X point cloud from a folder of frames, long-sequence mode, without running out of RAM.

Pi3's example_vo.py copies every predicted point to the CPU before writing the PLY.
For 450 frames that is ~10^8 points and took the process to 29 GB on a 46 GB machine
before it was killed. Here the confidence mask and a per-frame subsample are applied
on the GPU, so only a few million points ever reach host memory. Camera poses are
kept separately, in full.

Runs in the env that has torch + pi3 (3dgrut):

  python recon/pi3x_cloud.py <frames dir> --out <dir> [--per-frame 20000]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

PI3 = os.environ.get("PI3_ROOT", os.path.expanduser("~/Research/R2S/external/Pi3"))
sys.path.insert(0, PI3)
from pi3.models.pi3x import Pi3X  # noqa: E402
from pi3.pipe.pi3x_vo import Pi3XVO  # noqa: E402
from pi3.utils.basic import load_multimodal_data, write_ply  # noqa: E402
from pi3.utils.geometry import depth_normal_edge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-frame", type=int, default=20000, help="points kept per frame")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--global", dest="glob", action="store_true",
                    help="one forward pass over all frames (<= ~40): every frame is aligned to "
                         "every other, so there is no chunk-to-chunk drift")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev = torch.device("cuda")
    t0 = time.time()
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(dev).eval()
    imgs, _ = load_multimodal_data(a.frames, conditions=None, interval=1, device=dev)
    print(f"loaded {imgs.shape[1]} frames {tuple(imgs.shape[-2:])} in {time.time()-t0:.0f}s",
          flush=True)

    t0 = time.time()
    with torch.no_grad():
        if a.glob:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                res = model(imgs)
            c = torch.sigmoid(res["conf"][..., 0])
            edge = depth_normal_edge(res["local_points"], rtol=0.03, mask=c > a.conf)
            res["conf"] = torch.where(edge, torch.zeros_like(c), c)
        else:
            res = Pi3XVO(model)(imgs, chunk_size=a.chunk, overlap=a.overlap, dtype=torch.bfloat16)
    print(f"inference {time.time()-t0:.0f}s; outputs: "
          + ", ".join(f"{k}{tuple(v.shape)}" for k, v in res.items() if torch.is_tensor(v)),
          flush=True)

    pts, conf = res["points"][0], res["conf"][0]           # (T,H,W,3), (T,H,W[,1])
    conf = conf.reshape(conf.shape[:3])
    rgb = imgs[0].permute(0, 2, 3, 1)                      # (T,H,W,3) in [0,1]
    g = torch.Generator(device=dev).manual_seed(0)
    keep_p, keep_c, kept = [], [], []
    for t in range(pts.shape[0]):
        m = (conf[t] > a.conf).reshape(-1)
        idx = torch.nonzero(m).squeeze(1)
        if idx.numel() > a.per_frame:
            idx = idx[torch.randperm(idx.numel(), generator=g, device=dev)[: a.per_frame]]
        keep_p.append(pts[t].reshape(-1, 3)[idx].float().cpu())
        keep_c.append(rgb[t].reshape(-1, 3)[idx].float().cpu())
        kept.append(float(m.float().mean()))
    P, C = torch.cat(keep_p), torch.cat(keep_c)
    write_ply(P, C, os.path.join(a.out, "points.ply"))

    poses = res.get("camera_poses")
    np.savez(os.path.join(a.out, "pi3x.npz"),
             camera_poses=poses[0].float().cpu().numpy() if poses is not None else None,
             conf_frac=np.array(kept), frames=sorted(os.listdir(a.frames)))
    print(f"wrote {P.shape[0]:,} points, conf>{a.conf} on median "
          f"{np.median(kept)*100:.0f}% of pixels -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
