#!/usr/bin/env python
"""Pre-flight check for a capture session: is this scene good enough to record?

Grabs one live frame and reports the things that decide whether the demos will
be usable -- brick size and margin, how clean the desk ring around it is, the
viewing incidence, and the height noise you can actually expect. Run it, fix
what it complains about, run it again.

  python capture/check_scene.py
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MARGIN_PX = 60          # brick should sit this far from any frame edge
MIN_BRICK_PX = 300      # below this the height median gets noisy
MAX_INCIDENCE = 55      # degrees off the table normal
MAX_RING_RMS = 6.0      # mm; above this the ring is not clean desk


def grab(w, h, fps, warmup):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    try:
        for _ in range(warmup):
            pipe.wait_for_frames()
        f = align.process(pipe.wait_for_frames())
        c, d = f.get_color_frame(), f.get_depth_frame()
        return (np.asanyarray(c.get_data()).copy(),
                np.asanyarray(d.get_data()).astype(np.float32) * scale,
                C.intrinsics_dict(c.profile.as_video_stream_profile().intrinsics))
    finally:
        pipe.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=848)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--save", default=None, help="write an annotated frame here")
    a = ap.parse_args()

    bgr, dep, K = grab(a.w, a.h, a.fps, a.warmup)
    m, c, area = C.red_mask(bgr)
    problems = []

    if not area:
        print("FAIL: no red brick found. Check lighting, or widen the HSV band in common.py")
        return 1

    ys, xs = np.nonzero(m)
    edge = min(xs.min(), ys.min(), a.w - 1 - xs.max(), a.h - 1 - ys.max())
    core = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    bpts = C.pointcloud(dep, K, core if core.any() else m)
    dist = float(np.median(bpts[:, 2])) if len(bpts) else float("nan")

    print(f"brick      {area} px   centre ({c[0]:.0f},{c[1]:.0f})   "
          f"{dist:.3f} m   nearest frame edge {edge} px")
    if area < MIN_BRICK_PX:
        problems.append(f"brick only {area} px -- move the camera closer or use a bigger brick")
    if edge < MARGIN_PX:
        problems.append(f"brick {edge} px from the frame edge -- it will clip when your hand "
                        f"comes in, and the desk ring is cut off. Need >{MARGIN_PX} px")

    # how clean is the desk at each ring radius?
    print(f"{'ring (px)':>12} {'pts':>7} {'incidence':>10} {'rms (mm)':>9}  verdict")
    best = None
    for inner, outer in ((25, 70), (40, 110), (60, 160), (90, 220)):
        pts = C.pointcloud(dep, K, C.annulus(m, inner, outer))
        fit = C.fit_plane(pts)
        if fit is None:
            print(f"{inner:5d}-{outer:<6d} {len(pts):7d}  {'--':>10} {'--':>9}  no fit")
            continue
        n, p0, rms, inl, tot = fit
        inc, rms_mm = C.incidence_deg(n), rms * 1000
        ok = rms_mm <= MAX_RING_RMS
        print(f"{inner:5d}-{outer:<6d} {inl:7d}  {inc:9.1f}d {rms_mm:9.2f}  "
              f"{'clean desk' if ok else 'contaminated (keyboard/tissue/objects in the ring)'}")
        if ok and best is None:
            best = (inner, outer, inc, rms_mm, n, p0)

    if best is None:
        problems.append("no ring radius gives a clean plane -- clear ~15 cm of desk around "
                        "the tissue, away from the keyboard")
    else:
        inner, outer, inc, rms_mm, n, p0 = best
        print(f"\nuse --ring {inner} {outer}   (incidence {inc:.0f} deg, plane rms {rms_mm:.2f} mm)")
        if inc > MAX_INCIDENCE:
            problems.append(f"incidence {inc:.0f} deg -- tilt the camera more towards straight "
                            f"down, depth degrades at grazing angles")
        # expected height noise: per-pixel plane residual averaged over the brick
        npx = int((core > 0).sum())
        print(f"expected height noise ~{rms_mm / max(np.sqrt(npx), 1):.3f} mm "
              f"(plane rms {rms_mm:.2f} mm over {npx} brick pixels) -- a press into tissue "
              f"is several mm, so you want this well under 1 mm")

    if a.save:
        vis = bgr.copy()
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
        if best:
            cv2.circle(vis, (int(c[0]), int(c[1])), best[0], (255, 0, 0), 1)
            cv2.circle(vis, (int(c[0]), int(c[1])), best[1], (255, 0, 0), 1)
        cv2.rectangle(vis, (MARGIN_PX, MARGIN_PX), (a.w - MARGIN_PX, a.h - MARGIN_PX),
                      (0, 200, 0), 1)
        cv2.imwrite(a.save, vis)
        print(f"annotated frame -> {a.save}")

    print()
    if problems:
        print("NOT READY:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("READY to record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
