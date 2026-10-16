#!/usr/bin/env python
"""Turn a laptop capture (avi + unaligned depth PNGs + json) into the R2S frame layout
that the Pi3X pipeline reads:

  <out>/input/0000.png       colour
  <out>/depth/0000.png       depth aligned to colour, uint16 millimetres
  <out>/intrinsics.json      colour intrinsics (R2S schema)

Pi3X takes at most ~40 frames at once on a 32 GB card, so the clip is cut into N
equal bins and the sharpest frame of each bin is kept (variance of the Laplacian)
-- a hand-held orbit always has some motion blur, and Pi3X is sensitive to it.

Depth was recorded at the depth camera's native 848x480 and is not aligned. It is
reprojected into the colour camera with the stored depth->colour extrinsics and
splatted 2x2 with a z-buffer, because the 1.5x upsample would otherwise leave a
regular grid of holes.

  python recon/capture_to_r2s.py <capture dir> <tag> --out <cap> [--n 40]
"""
import argparse
import json
import os

import cv2
import numpy as np


def sharpness(bgr):
    return cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def align_depth(z16, scale, di, ci, R, t):
    """Depth image (depth camera) -> metres in the colour camera's image."""
    h, w = z16.shape
    v, u = np.nonzero(z16)
    z = z16[v, u].astype(np.float64) * scale
    P = np.stack([(u - di["ppx"]) / di["fx"] * z, (v - di["ppy"]) / di["fy"] * z, z])
    Q = R @ P + t[:, None]
    ok = Q[2] > 0.05
    Q = Q[:, ok]
    uc = Q[0] / Q[2] * ci["fx"] + ci["ppx"]
    vc = Q[1] / Q[2] * ci["fy"] + ci["ppy"]
    W, H = ci["width"], ci["height"]
    out = np.full((H, W), np.inf)
    for du in (0, 1):
        for dv in (0, 1):
            x = np.floor(uc).astype(int) + du
            y = np.floor(vc).astype(int) + dv
            m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            np.minimum.at(out, (y[m], x[m]), Q[2][m])
    out[~np.isfinite(out)] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("tag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.capture, a.tag + ".json")))
    ci, d = meta["intrinsics"], meta["depth"]
    di, scale = d["intrinsics"], d["depth_scale_m"]
    R = np.array(d["depth_to_color"]["rotation_colmajor"]).reshape(3, 3).T
    t = np.array(d["depth_to_color"]["translation_m"])
    if any(abs(c) > 1e-9 for c in ci["coeffs"]):
        print("[warn] colour distortion coefficients are non-zero and are ignored")

    cap = cv2.VideoCapture(os.path.join(a.capture, a.tag + ".avi"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    n = len(frames)
    bins = np.array_split(np.arange(n), a.n)
    picks = [int(b[np.argmax([sharpness(frames[i]) for i in b])]) for b in bins if len(b)]

    os.makedirs(os.path.join(a.out, "input"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)
    ddir = os.path.join(a.capture, d["folder"])
    valid = []
    for k, i in enumerate(picks):
        name = f"{k:04d}"
        cv2.imwrite(os.path.join(a.out, "input", name + ".png"), frames[i])
        z16 = cv2.imread(os.path.join(ddir, f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        za = align_depth(z16, scale, di, ci, R, t)
        valid.append((za > 0).mean())
        cv2.imwrite(os.path.join(a.out, "depth", name + ".png"),
                    np.clip(za * 1000.0, 0, 65535).astype(np.uint16))

    json.dump(dict(width=ci["width"], height=ci["height"], fx=ci["fx"], fy=ci["fy"],
                   cx=ci["ppx"], cy=ci["ppy"],
                   fovy_deg=float(np.degrees(2 * np.arctan(ci["height"] / (2 * ci["fy"])))),
                   model=ci["model"], coeffs=ci["coeffs"], depth_unit="mm (uint16)",
                   depth_scale_m_per_unit=0.001, aligned_to="color",
                   source=f"{a.tag} from {os.path.abspath(a.capture)}",
                   source_frames=picks),
              open(os.path.join(a.out, "intrinsics.json"), "w"), indent=2)
    print(f"{len(picks)} of {n} frames -> {a.out}  "
          f"(aligned depth valid: median {np.median(valid)*100:.0f}%)")


if __name__ == "__main__":
    main()
