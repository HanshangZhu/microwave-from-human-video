#!/usr/bin/env python
"""Camera parameters, in the microwave frame, for rendering the splatted kitchen
(recon/gs_kitchen.py render) behind the simulator's objects.

  front   the policy's front camera at its render size (hq: 2x supersampled), static
  demo    display cameras (not policy inputs), each a look-at in the microwave frame

  python sim/gs_cams.py --out <npz> [--demo]          (env maniskill)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# display cameras: (eye, target) in the microwave frame, metres; 960 x 540, 60 deg vertical fov
DEMO = {
    "side": ((0.25, -1.25, 0.55), (0.30, -0.10, 0.10)),
    "high": ((-0.55, -0.95, 0.75), (0.30, -0.05, 0.05)),
}
DEMO_W, DEMO_H, DEMO_FOVY = 960, 540, np.deg2rad(60)


def look_at_cv(eye, target):
    """world(mw)-to-camera, OpenCV axes (x right, y down, z forward), z up in the world"""
    eye, target = np.array(eye, float), np.array(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ eye
    return w2c


def demo_K():
    f = DEMO_H / 2 / np.tan(DEMO_FOVY / 2)
    return np.array([[f, 0, DEMO_W / 2], [0, f, DEMO_H / 2], [0, 0, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo", action="store_true", help="the display cameras instead of the front camera")
    a = ap.parse_args()
    if a.demo:
        names = list(DEMO)
        np.savez(a.out, K=np.stack([demo_K()] * len(names)), w2c=np.stack([look_at_cv(*DEMO[n]) for n in names]),
                 width=DEMO_W, height=DEMO_H, names=names)
        print("->", a.out, names)
        return
    import gymnasium as gym
    import sim.microwave_env as M
    env = gym.make("MicrowavePot-v0", obs_mode="rgb", control_mode="pd_joint_pos", sim_backend="cpu", views="hq")
    obs, _ = env.reset(seed=0)
    p = obs["sensor_param"]["front"]
    E = np.eye(4)
    E[:3] = p["extrinsic_cv"][0].cpu().numpy()                   # world -> camera
    cfg = env.unwrapped._sensors["front"].config
    np.savez(a.out, K=p["intrinsic_cv"][0].cpu().numpy()[None], w2c=(E @ M.T_W_MW)[None],
             width=cfg.width, height=cfg.height, names=["front"])
    print("->", a.out, cfg.width, cfg.height)


if __name__ == "__main__":
    main()
