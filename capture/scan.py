#!/usr/bin/env python
"""Scans of the environment (the cabinet) and the target (the Heinz beans pot).

These are not demonstrations. They are the input to SfM -> dense reconstruction
(MVS / TSDF / 3DGS) on the desktop, which builds the simulation assets. Every shot
is a slow, continuous moving scan; nothing is measured at capture time. Planes,
gravity (from the floor/counter), shelf dimensions and the beans pot footprint are
all fitted in the reconstruction, and metric scale comes from the recorded depth.

  env      the cabinet
           closed    door closed. Wide sweep of cabinet, counter and floor from a
                     few heights: the static reference the demonstrations register to
           open_45   door at ~45 deg, short arc in front of it
           open_90   door fully open, short arc over the door, then into the shelf
                     The door is a different geometry in each state, so each state is
                     reconstructed on its own; the door planes meet in the hinge axis.
  target   the beans pot, alone on an empty patch of counter
           orbit     one continuous take: a low circle, a higher circle, then
                     tilt to straight down -- no gaps between viewpoints for SfM

Move slowly: the colour camera has a rolling shutter. Each shot is announced, then
counts down, then records. SPACE before the first shot; --manual waits for SPACE
before every shot.

  python scan.py env    --out scans/cabinet
  python scan.py target --out scans/beans
"""
import argparse
import json
import os
import sys
import time

import cv2

import common as C

SHOTS = {
    "env": [
        ("closed", 40, "door closed. slow wide sweep of cabinet, counter and floor, "
                       "at a few heights"),
        ("open_45", 15, "open the door about 45 degrees. slow short arc in front of it"),
        ("open_90", 25, "open the door fully. slow arc over the door, then look into the shelf"),
    ],
    "target": [
        ("orbit", 60, "beans pot alone on the counter. slow low circle, then a higher "
                      "circle, then look straight down"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=sorted(SHOTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--setup", type=float, default=8.0, help="seconds to set up each shot")
    ap.add_argument("--manual", action="store_true", help="SPACE before every shot")
    ap.add_argument("--only", nargs="*", help="record just these shots")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preset", choices=sorted(C.PRESETS), default="high_density",
                    help="high_density fills more holes for TSDF / depth supervision")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    speak = not a.quiet
    shots = [s for s in SHOTS[a.kind] if not a.only or s[0] in a.only]
    win = f"scan {a.kind}   [SPACE] start   [q] quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, a.w, a.h)

    done = []
    for i, (name, secs, instr) in enumerate(shots):
        if not C.disk_ok(a.out, a.min_free_gb, speak):
            break
        pipe, _ = C.start(a.w, a.h, a.fps, depth=True, preset=a.preset)
        try:
            if i == 0 or a.manual:
                if not C.wait_space(pipe, win, f"{name}: SPACE when ready", note=instr):
                    break
            say_line = instr.split(".")[0]
            C.say(say_line, speak)
            if not C.timed(pipe, win, a.setup, f"SET UP {name}", (255, 160, 0), note=instr):
                break
        finally:
            pipe.stop()
        meta, quit_ = C.record_take(a.out, name, secs, a.w, a.h, a.fps, win, depth=True,
                                    preset=a.preset, speak=speak,
                                    extra=dict(kind=a.kind, shot=name, instruction=instr))
        done.append(name)
        if quit_:
            break

    json.dump(dict(kind=a.kind, preset=a.preset, shots=done,
                   recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S")),
              open(os.path.join(a.out, "scan.json"), "w"), indent=1)
    cv2.destroyAllWindows()
    print(f"{a.kind}: {len(done)}/{len(shots)} shots -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
