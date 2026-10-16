#!/usr/bin/env python
"""Take the microwave out of the splat's training images before the splat is trained.

The simulator draws its own microwave (its door opens; the scanned one is shut), so the
splatted kitchen must not contain one. Deleting its Gaussians after training leaves a hole
and floaters around it. Here it is removed from every training image instead:

  mask     the microwave's box (gs_kitchen.MW_TIGHT, the same box the render deletes) ray-cast
           into every view at the view's pose -- the geometry and poses are known, no
           segmentation model -- dilated by --dilate px to take its shadow and reflections
  inpaint  LaMa (big-lama, TorchScript; --lama <file>). A diffusion inpainter invents
           different content in every view, which a splat then averages into blur; LaMa's
           smooth fills agree better across views.

gs_kitchen.py train --inpainted <out> then trains on these images, drops the measured
depth inside the masks and the initial points inside the box.

  python recon/gs_inpaint.py --lama big-lama.pt --out <dir> [--dilate 12]      (env sam3d)
"""
import argparse
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs_kitchen import KIT, MW_TIGHT, load_views  # noqa: E402


def box_mask(v, box):
    lo, hi = np.array(box[0]), np.array(box[1])
    mesh = o3d.geometry.TriangleMesh.create_box(*(hi - lo)).translate(lo)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    H, W = v["img"].shape[:2]
    rays = scene.create_rays_pinhole(o3d.core.Tensor(v["K"].astype(np.float64)),
                                     o3d.core.Tensor(np.linalg.inv(v["c2w"])), W, H)
    return np.isfinite(scene.cast_rays(rays)["t_hit"].numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lama", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", default=KIT, help="sweep frames (as gs_kitchen.py --sweep)")
    ap.add_argument("--dilate", type=int, default=12, help="px at 1280 wide")
    a = ap.parse_args()
    for d in ("images", "masks"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    dev = torch.device("cuda")
    lama = torch.jit.load(a.lama, map_location=dev).eval()
    views, _ = load_views(inpainted=None, sweep=a.sweep)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate + 1, 2 * a.dilate + 1))
    sheet = []
    for v in views:
        m = cv2.dilate(box_mask(v, MW_TIGHT).astype(np.uint8), k)
        img = v["img"]
        H, W = img.shape[:2]
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        x = torch.tensor(np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect"), device=dev).permute(2, 0, 1)[None].float() / 255
        mm = torch.tensor(np.pad(m, ((0, ph), (0, pw))), device=dev)[None, None].float()
        with torch.no_grad():
            y = lama(x, mm)[0].permute(1, 2, 0)[:H, :W].clamp(0, 1).cpu().numpy()
        out = np.where(m[..., None] > 0, (y * 255).round().astype(np.uint8), img)
        cv2.imwrite(os.path.join(a.out, "images", v["name"] if v["name"].endswith(".png") else v["name"] + ".png"),
                    out[..., ::-1])
        cv2.imwrite(os.path.join(a.out, "masks", v["name"] if v["name"].endswith(".png") else v["name"] + ".png"), m * 255)
        if len(sheet) < 8 and (m.mean() > 0.02):
            sheet.append(cv2.resize(np.hstack([img, out]), (960, 270)))
    cv2.imwrite(os.path.join(a.out, "sheet.jpg"), np.vstack(sheet)[..., ::-1])
    print(f"{len(views)} views -> {a.out}")


if __name__ == "__main__":
    main()
