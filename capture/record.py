#!/usr/bin/env python
"""First-person demonstrations: open the cabinet, take the beans pot out, put it on the counter.

No markers. Camera pose, door angle and beans pot pose are all recovered from depth on the
desktop: the camera is registered against the cabinet front, countertop and floor
(see scan.py env), the door is its own plane, and the beans pot is the one object left on
an otherwise empty shelf. The preview shows what that depends on -- how much of the
image has valid depth, and how far away it is.

Each take runs RESET -> GET READY -> REC and advances on its own, speaking each phase,
because whoever wears the camera cannot see the screen. SPACE starts the first take,
q quits at any point. Re-running resumes the numbering.

  python record.py --out captures/session01 --count 30
"""
import argparse
import os
import sys

import cv2

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="take")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--duration", type=float, default=25.0, help="seconds per take")
    ap.add_argument("--reset", type=float, default=10.0, help="seconds to reset the scene")
    ap.add_argument("--countdown", type=float, default=4.0)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preset", choices=sorted(C.PRESETS), default="default",
                    help="depth preset; default fills more holes than high_accuracy")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--raw", action="store_true", help="record a raw .db3 instead (~60 MB/s)")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    speak = not a.quiet
    done = len([f for f in os.listdir(a.out) if f.startswith(a.prefix) and f.endswith(".json")])
    print(f"{done} take(s) already in {a.out}; recording up to {a.count} more", flush=True)

    win = "record   [SPACE] start   [q] quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, a.w, a.h)

    for i in range(done, done + a.count):
        if not C.disk_ok(a.out, a.min_free_gb, speak):
            break
        tag = f"{a.prefix}_{i:03d}"
        pipe, _ = C.start(a.w, a.h, a.fps, depth=True, preset=a.preset)
        try:
            if i == done:
                if not C.wait_space(pipe, win, f"ready: {tag}   SPACE to start"):
                    break
            else:
                C.say("reset", speak)
                if not C.timed(pipe, win, a.reset, f"RESET {tag}", (255, 160, 0),
                               note="close the door, beans pot back on the shelf at a new spot"):
                    break
            C.say("get ready", speak)
            if not C.timed(pipe, win, a.countdown, f"GET READY {tag}", (0, 200, 255),
                           note="start pose, cabinet front in view"):
                break
        finally:
            pipe.stop()
        _, quit_ = C.record_take(a.out, tag, a.duration, a.w, a.h, a.fps, win,
                                 depth=not a.no_depth, raw=a.raw, preset=a.preset, speak=speak)
        if quit_:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
