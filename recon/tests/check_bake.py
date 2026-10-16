#!/usr/bin/env python
"""End-to-end check of register_model.py + bake_colors.py on a synthetic scan:
re-render the baked, textured URDF from the scan's own camera poses (at the
registered pose and door angle, same flat light) and compare it, pixel by pixel on
the object, with the frames it was baked from. Pixels where the frame's depth shows
something in front of the object are left out.

Runs in the env with sapien (maniskill):
  python recon/tests/check_bake.py <textured urdf> <cap dir> <run dir> --out <png>
"""
import argparse
import json
import os

import cv2
import numpy as np
import sapien


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("cap")
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    reg = json.load(open(os.path.join(a.run, "registration.json")))
    c2w = np.load(os.path.join(a.run, "poses.npz"))["c2w"]
    intr = json.load(open(os.path.join(a.cap, "intrinsics.json")))
    names = sorted(os.listdir(os.path.join(a.cap, "input")))

    scene = sapien.Scene()
    scene.set_ambient_light([1.0, 1.0, 1.0])
    ld = scene.create_urdf_loader()
    ld.fix_root_link = True
    mw = ld.load(a.urdf)
    mw.set_root_pose(sapien.Pose(np.array(reg["T_model_to_world"])))
    mw.set_qpos([reg["joint_angle_rad"]])
    scene.step()
    W, H = intr["width"], intr["height"]
    cam = scene.add_camera("cam", W, H, 2 * np.arctan(H / 2 / intr["fy"]), 0.05, 5)

    errs, tiles = [], []
    for i, name in enumerate(names):
        G = c2w[i][:3, :3] @ np.diag([1, -1, -1])         # OpenCV -> GL camera axes
        T = np.eye(4)
        T[:3, :3] = np.stack([-G[:, 2], -G[:, 0], G[:, 1]], 1)   # GL -> SAPIEN (x fwd, y left, z up)
        T[:3, 3] = c2w[i][:3, 3]
        cam.entity.set_pose(sapien.Pose(T))
        scene.update_render()
        cam.take_picture()
        ren = (cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
        on = cam.get_picture("Segmentation")[..., 1] > 0
        on = cv2.erode(on.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # skip edge pixels
        zr = -cam.get_picture("Position")[..., 2]
        zf = cv2.imread(os.path.join(a.cap, "depth", name), cv2.IMREAD_UNCHANGED) / 1000.0
        on &= ~((zf > 0) & (zf < zr - 0.02))      # the frame shows something in front of it
        src = cv2.imread(os.path.join(a.cap, "input", name))[..., ::-1]
        d = np.abs(ren.astype(int) - src.astype(int)).max(-1)[on]
        errs.append(d)
        if i in (0, len(names) // 2, len(names) - 1):
            show = ren.copy()
            show[~on] //= 3
            tiles.append(np.vstack([cv2.resize(src, (W // 3, H // 3)), cv2.resize(show, (W // 3, H // 3))]))
    e = np.concatenate(errs)
    print(f"{len(names)} views, {len(e):,} object pixels: |baked render - frame| max-channel "
          f"median {np.median(e):.0f}, p95 {np.percentile(e, 95):.0f}, "
          f"{(e > 20).mean()*100:.1f}% over 20/255")
    cv2.imwrite(a.out, np.hstack(tiles)[..., ::-1])
    print("side by side (top: scan frame, bottom: baked model) ->", a.out)


if __name__ == "__main__":
    main()
