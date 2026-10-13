#!/usr/bin/env python
"""Build the in-context demonstration bank from the rendered press episodes.

One entry per episode: the head-camera frame at t=0, and the press target in
table coordinates. That pair is the whole demonstration -- an image and where
the press went.

The instruction given to the model at evaluation time is deliberately empty of
task content ("follow the demonstrations"). Everything the model knows about
what to do has to come from these pairs, so removing them has to break the
policy. That is what makes the k-curve a measurement of the data rather than of
the prompt.

  python icl/build_bank.py --out <bank dir>
"""
import argparse
import json
import os

import cv2
import h5py
import numpy as np

ROBOTWIN = "/home/hanszhu/Research/WAM/WAM-OPD/RoboTwin"
WS = "/home/hanszhu/Research/VLA/Humanoid-Internship-Challenge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=f"{ROBOTWIN}/data/human_press_bell/hp_direct_all")
    ap.add_argument("--specs", default=f"{WS}/Gitignore-but-important/data/specs_direct.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="head_camera")
    a = ap.parse_args()

    os.makedirs(f"{a.out}/img", exist_ok=True)
    specs = json.load(open(a.specs))["specs"]
    manifest = {m["episode"]: m for m in
                json.load(open(f"{a.episodes}/manifest.json"))["manifest"]}

    bank = []
    for ep in sorted(manifest):
        s = specs[manifest[ep]["spec"]]
        h5 = f"{a.episodes}/data/episode{ep}.hdf5"
        if not os.path.exists(h5):
            continue
        with h5py.File(h5, "r") as f:
            buf = f[f"observation/{a.camera}/rgb"][0]
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        path = f"{a.out}/img/demo_{ep:02d}.png"
        cv2.imwrite(path, img)
        bank.append(dict(
            episode=ep, image=os.path.relpath(path, a.out),
            target_xy_mm=[round(s["bell_xy"][0] * 1000, 1), round(s["bell_xy"][1] * 1000, 1)],
            arm=s["arm"], source_take=s["take"]))

    json.dump(dict(camera=a.camera, n=len(bank), entries=bank),
              open(f"{a.out}/bank.json", "w"), indent=1)
    xs = [e["target_xy_mm"][0] for e in bank]
    ys = [e["target_xy_mm"][1] for e in bank]
    print(f"{len(bank)} demonstrations -> {a.out}")
    print(f"  image {img.shape[1]}x{img.shape[0]} from {a.camera}")
    print(f"  target x {min(xs):.0f}..{max(xs):.0f} mm   y {min(ys):.0f}..{max(ys):.0f} mm")
    print(f"  arms: left {sum(1 for e in bank if e['arm']=='left')}  "
          f"right {sum(1 for e in bank if e['arm']=='right')}")


if __name__ == "__main__":
    main()
