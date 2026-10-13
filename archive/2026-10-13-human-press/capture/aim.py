#!/usr/bin/env python
"""Live aiming helper: move the camera until all three readouts are green.

Running calibrate_session over and over to find out whether the camera is placed
well is a slow loop. The three numbers that decide the measurement are cheap to
compute every frame:

  distance   closer is better -- apparent area goes as 1/z^2 and depth noise as z^2
  area       pixels on the object; a fingertip will take most of them away
  lever      |v - ppy| / fy. Gravity lies almost along image-Y at this camera
             angle, so the height of the object is read off its pixel row. Depth
             noise leaks into that reading through this lever arm, and it goes to
             zero when the object sits on the optical axis. Aim AT the object.

  python capture/aim.py --colour yellow
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, AMBER = (0, 220, 0), (0, 0, 255), (0, 170, 255)

TARGET = dict(area=800, lever=0.05, dist=(0.30, 0.45))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colour", choices=sorted(C.PALETTE), default="yellow")
    ap.add_argument("--z-range", type=float, nargs=2, default=(0.25, 0.90))
    ap.add_argument("--w", type=int, default=848)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--display", default=":1")
    a = ap.parse_args()
    os.environ.setdefault("DISPLAY", a.display)
    band = C.PALETTE[a.colour]

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, a.w, a.h, rs.format.z16, a.fps)
    cfg.enable_stream(rs.stream.color, a.w, a.h, rs.format.bgr8, a.fps)
    cfg.enable_stream(rs.stream.accel)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    ex = prof.get_stream(rs.stream.accel).get_extrinsics_to(prof.get_stream(rs.stream.color))
    Racc = np.array(ex.rotation).reshape(3, 3).T

    win = "aim   [q] done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, a.w, a.h)
    g_s, K = None, None
    try:
        while True:
            fs = pipe.wait_for_frames()
            for fr in fs:
                m = fr.as_motion_frame()
                if m and fr.profile.stream_type() == rs.stream.accel:
                    d = m.get_motion_data()
                    v = -np.array([d.x, d.y, d.z])
                    g_s = v if g_s is None else 0.9 * g_s + 0.1 * v
            fs = align.process(fs)
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if not c or not d:
                continue
            if K is None:
                K = C.intrinsics_dict(c.profile.as_video_stream_profile().intrinsics)
            bgr = np.asanyarray(c.get_data()).copy()
            dep = np.asanyarray(d.get_data()).astype(np.float32) * scale

            centre = np.array([K["ppx"], K["ppy"]])
            cands = [x for x in C.red_candidates(bgr, dep, band)
                     if np.isfinite(x[3]) and a.z_range[0] <= x[3] <= a.z_range[1]]
            vis = bgr
            cv2.drawMarker(vis, (int(K["ppx"]), int(K["ppy"])), (255, 255, 255),
                           cv2.MARKER_CROSS, 26, 1)
            rows = []
            if cands:
                bm, uv, area, z = min(cands, key=lambda x: np.linalg.norm(
                    np.array(x[1]) - centre))
                lever = abs(uv[1] - K["ppy"]) / K["fy"]
                cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
                cv2.line(vis, (int(uv[0]), int(uv[1])),
                         (int(K["ppx"]), int(K["ppy"])), (255, 255, 0), 1)
                rows = [
                    ("distance", f"{z:5.3f} m", TARGET['dist'][0] <= z <= TARGET['dist'][1],
                     f"target {TARGET['dist'][0]:.2f}-{TARGET['dist'][1]:.2f}"),
                    ("area", f"{area:5d} px", area >= TARGET["area"],
                     f"target >={TARGET['area']}"),
                    ("lever", f"{lever:5.3f}", lever <= TARGET["lever"],
                     "aim AT the object"),
                ]
                if g_s is not None:
                    gn = Racc @ g_s
                    rows.append(("|g|", f"{np.linalg.norm(g_s):5.3f}",
                                 abs(np.linalg.norm(g_s) - 9.81) < 0.05, "hold still"))
                    rows.append(("gravity in image-Y",
                                 f"{abs(gn[1]) / np.linalg.norm(gn):5.3f}",
                                 abs(gn[1]) / np.linalg.norm(gn) > 0.80,   # the part of the vertical that
                                 "in-plane; 0.86 already leaves <0.3 mm from depth"))
            else:
                rows = [("object", "not found", False, "check colour / depth band")]

            for i, (name, val, ok, hint) in enumerate(rows):
                y = 34 + i * 30
                cv2.putText(vis, f"{name:>18}: {val:>10}   {hint}", (12, y), FONT, 0.6,
                            GREEN if ok else RED, 2)
            allok = rows and all(r[2] for r in rows)
            cv2.putText(vis, "READY -- press q, then calibrate" if allok else "ADJUST",
                        (12, a.h - 18), FONT, 0.9, GREEN if allok else AMBER, 2)
            cv2.imshow(win, vis)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    if cands:
        print(f"last reading: {area} px at {z:.3f} m, lever {lever:.3f}, "
              f"centroid ({uv[0]:.0f},{uv[1]:.0f})")
        print(f"calibrate with:  --pick {uv[0]:.0f} {uv[1]:.0f}")


if __name__ == "__main__":
    main()
