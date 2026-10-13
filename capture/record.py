#!/usr/bin/env python
"""Record human press demonstrations from a RealSense D435i.

One .db3 (librealsense 2.59 rosbag2) per demonstration, so extraction parameters
can be re-tuned later without re-recording. The live preview overlays the object
mask, so bad segmentation is caught while recording instead of afterwards.

Each demo runs RESET -> GET READY -> REC, all shown on the live preview:

  RESET      move the object to a NEW position -- spatial coverage is the point,
             forty presses at one spot make a scatter plot with one dot in it
  GET READY  get your hand off the keyboard and hovering over the object
  REC        hold still ~1 s, then press

SPACE starts the first demo; after that it advances on its own, so you never have
to reach for the keyboard mid-session. q or ESC quits during any phase.

  python capture/record.py --out <dir> --session <dir>/session.json --count 40
  python capture/record.py --out <dir> --count 3 --auto    # headless smoke test
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FONT = cv2.FONT_HERSHEY_SIMPLEX
QUIT_KEYS = (ord("q"), 27)
ALIGN = rs.align(rs.stream.color)
DEPTH_SCALE = [0.001]


def build_config(w, h, fps, bag=None):
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    if bag:
        cfg.enable_record_to_file(bag)
    return cfg


def overlay(bgr, pick, status, colour=(0, 255, 0), roi=None, big=None, gate=None):
    m, c, area = pick(bgr)
    vis = bgr.copy()
    if area:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
        if c:
            cv2.drawMarker(vis, (int(c[0]), int(c[1])), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    if roi:
        cv2.rectangle(vis, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (0, 200, 0), 1)
    if gate is not None:
        cv2.circle(vis, (int(gate[0][0]), int(gate[0][1])), int(gate[1]), (255, 120, 0), 1)
    cv2.putText(vis, f"{status}   object {area} px" + ("" if area else "   NOT VISIBLE"),
                (12, 28), FONT, 0.7, (0, 0, 255) if not area else colour, 2)
    if big:
        h, w = vis.shape[:2]
        (tw, th), _ = cv2.getTextSize(big, FONT, 5.0, 10)
        cv2.putText(vis, big, ((w - tw) // 2, (h + th) // 2), FONT, 5.0, colour, 10)
    return vis, area


def phase(pipe, win, seconds, pick, roi, label, colour, tracker=None, seed=False):
    """Run a timed phase on the live preview. False if the user quit."""
    t0 = time.time()
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            return True
        fs = ALIGN.process(pipe.wait_for_frames())
        f, dfr = fs.get_color_frame(), fs.get_depth_frame()
        if not f or not dfr:
            continue
        bgr = np.asanyarray(f.get_data())
        dep = np.asanyarray(dfr.get_data()).astype(np.float32) * DEPTH_SCALE[0]
        if seed and tracker is not None and not tracker.locked:
            tracker.seed(bgr, dep)
            if seconds - left > 0.7:        # must finish inside a short ready phase
                tracker.lock()
        gate = (tracker.uv, tracker.max_jump) if (tracker is not None and tracker.locked) else None
        vis, _ = overlay(bgr, lambda b: pick(b, dep) if callable(pick) and
                         getattr(pick, "locked", None) is not None else pick(b),
                         label, colour, roi, big=str(int(np.ceil(left))), gate=gate)
        cv2.imshow(win, vis)
        if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
            return False


def wait_for_space(pipe, win, pick, roi, label):
    while True:
        f = pipe.wait_for_frames().get_color_frame()
        if not f:
            continue
        vis, _ = overlay(np.asanyarray(f.get_data()), pick, label, (0, 255, 0), roi)
        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            return "go"
        if k in QUIT_KEYS:
            return "quit"


def load_band(session, colour):
    return C.band_from_session(session, colour)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for the .db3 recordings")
    ap.add_argument("--prefix", default="press")
    ap.add_argument("--count", type=int, default=40, help="demos to record this session")
    ap.add_argument("--duration", type=float, default=8.0, help="seconds per demo")
    ap.add_argument("--reset", type=float, default=5.0,
                    help="seconds to move the object to a new position before each demo")
    ap.add_argument("--countdown", type=float, default=5.0,
                    help="seconds to get your hand into position; the keyboard is in frame, "
                         "so this has to be watched on screen, not in the terminal")
    ap.add_argument("--manual", action="store_true",
                    help="require SPACE before every demo instead of advancing automatically")
    ap.add_argument("--colour", choices=sorted(C.PALETTE), default=None,
                    help="object colour band; default: from session.json, else red")
    ap.add_argument("--w", type=int, default=848)   # native D435 depth width
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--auto", action="store_true", help="no keyboard, back to back (smoke test)")
    ap.add_argument("--session", default=None, help="session.json from calibrate_session.py")
    ap.add_argument("--display", default=":1")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    sess = json.load(open(a.session)) if a.session else None
    roi = tuple(sess["roi"]) if (sess and sess.get("roi")) else None
    band = load_band(sess, a.colour)
    print(f"roi {roi}   band h1={band['h1']} h2={band['h2']} "
          f"s>{band['s_min']} v>{band['v_min']}", flush=True)

    preview = not a.no_preview and not a.auto
    if preview:
        os.environ.setdefault("DISPLAY", a.display)

    DEPTH_SCALE[0] = 0.001   # D435i z16 scale; confirmed by calibrate_session
    g = C.read_gravity()
    print(f"gravity (accel frame, sign-corrected): "
          f"{np.round(g, 3).tolist() if g else 'IMU unavailable'}", flush=True)

    # resume: never overwrite demos recorded earlier in the session
    done = len([f for f in os.listdir(a.out) if f.startswith(a.prefix) and f.endswith(".db3")])
    print(f"{done} demo(s) already in {a.out}; recording up to {a.count} more", flush=True)

    zr = tuple(sess["z_range"]) if (sess and sess.get("z_range")) else None
    tracker = C.Tracker(band, roi, zr)
    free_pick = lambda bgr: C.pick_brick(bgr, None, band, roi)   # unanchored: RESET phase

    win = "record   [SPACE] start   [q] quit"
    if preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, a.w, a.h)

    started = False
    for i in range(done, done + a.count):
        tag = f"{a.prefix}_{i:03d}"
        bag = os.path.join(a.out, tag + ".db3")
        n = i - done + 1

        if preview:
            pipe = rs.pipeline()
            pipe.start(build_config(a.w, a.h, a.fps))
            try:
                if not started or a.manual:
                    if wait_for_space(pipe, win, free_pick, roi,
                                      f"ready: {tag} ({n}/{a.count})   SPACE to start") == "quit":
                        print("quit", flush=True)
                        break
                    started = True
                elif a.reset > 0:
                    tracker.reset()
                    if not phase(pipe, win, a.reset, free_pick, roi,
                                 f"RESET {tag} ({n}/{a.count}) -- move the object to a NEW spot",
                                 (255, 160, 0)):
                        print("quit", flush=True)
                        break
                # Anchor on the first second of GET READY: the object is in its
                # new place and the hand has not arrived yet. After this the
                # tracker refuses anything far from the anchor, so a hand in
                # frame can no longer hijack the mask.
                tracker.reset()
                if not phase(pipe, win, a.countdown, tracker, roi,
                             f"GET READY   {tag}", (0, 200, 255), tracker, seed=True):
                    print("quit", flush=True)
                    break
            finally:
                pipe.stop()

        # ---- record ----------------------------------------------------------
        pipe = rs.pipeline()
        pipe.start(build_config(a.w, a.h, a.fps, bag))
        t0, nf, areas = time.time(), 0, []
        try:
            while time.time() - t0 < a.duration:
                f = pipe.wait_for_frames()
                nf += 1
                cf = f.get_color_frame()
                if cf:
                    # always segment: the per-frame object area is the cheapest
                    # signal that a demo is unusable, preview window or not
                    vis, area = overlay(np.asanyarray(cf.get_data()),
                                        tracker if tracker.locked else free_pick,
                                        f"REC {tag}   {a.duration - (time.time() - t0):.1f}s left",
                                        (0, 0, 255), roi,
                                        gate=(tracker.uv, tracker.max_jump)
                                        if tracker.locked else None)
                    areas.append(area)
                    if preview:
                        cv2.imshow(win, vis)
                        cv2.waitKey(1)
        finally:
            pipe.stop()

        C.save_json(os.path.join(a.out, tag + ".meta.json"), dict(
            tag=tag, bag=os.path.basename(bag), frames=nf, duration_s=a.duration,
            size=[a.w, a.h], fps=a.fps, gravity_accel_frame=g, band=band,
            anchor=dict(uv=tracker.uv.tolist() if tracker.locked else None,
                        area_px=tracker.area, max_jump_px=tracker.max_jump),
            object_area_px={"median": float(np.median(areas)), "min": float(np.min(areas)),
                            "lost_frames": int(sum(1 for x in areas if x == 0))}
            if areas else None,
            recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S")))
        mb = os.path.getsize(bag) / 1e6
        print(f"[{tag}] {nf} frames, {mb:.0f} MB"
              + (f", object median {np.median(areas):.0f} px, "
                 f"lost {sum(1 for x in areas if x == 0)}/{len(areas)}f"
                 if areas else ""), flush=True)

    if preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
