#!/usr/bin/env python
"""One episode of a camera policy in the splatted kitchen, filmed from display cameras.

The simulator draws the robot, the microwave and the pot; the splatted kitchen (recon/
gs_kitchen.py, rendered once per static camera: rgb + depth) is everything else. Per pixel,
the simulator's drawing is kept wherever it shows one of its objects (segmentation).
The policy's front image is composited the same way (dit_vis.real_background with the
front render); its wrist image too with --splat-wrist (rendered every step, sim/gs_client.py),
else it is the simulator's own (grey floor). The display cameras
(sim/gs_cams.py DEMO) are not policy inputs.

  python sim/gs_demo.py --ckpt <dit_vis.pt> --front <bg_front.npz> --demo <bg_demo.npz> \\
      --cams <cams_demo.npz> --out <dir> [--index 0]                         (env maniskill)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim.views as V  # noqa: E402
from mani_skill.sensors.camera import CameraConfig  # noqa: E402


def sapien_pose(w2c_mw):
    """OpenCV world(mw)-to-camera -> SAPIEN camera pose in the world (x forward, y left, z up)"""
    import sapien
    import sim.microwave_env as M
    C = np.linalg.inv(w2c_mw)
    G = C[:3, :3] @ np.diag([1, -1, -1])
    T = np.eye(4)
    T[:3, :3] = np.stack([-G[:, 2], -G[:, 0], G[:, 1]], 1)
    T[:3, 3] = C[:3, 3]
    return sapien.Pose(M.T_W_MW @ T)


def composite(sd, bg_rgb, bg_depth, ids):
    img = sd["rgb"][0].cpu().numpy().copy()
    seg = sd["segmentation"][0, ..., 0].cpu().numpy()
    d = sd["depth"][0, ..., 0].cpu().numpy().astype(np.float32) / 1000
    m = np.isin(seg, ids)                     # the simulator's objects always in front (see dit_vis.images)
    img[m] = bg_rgb[m]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--front", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", type=int, default=0, help="eval start index (dit_vis eval's start states, --seed 1)")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--splat-wrist", help="splat checkpoint rendered behind the wrist camera every step")
    a = ap.parse_args()
    cams = np.load(a.cams)
    names = [str(n) for n in cams["names"]]
    _cc = V.camera_configs

    def cc(env, take="take_000", hq=False):
        cfg = _cc(env, take, hq)
        for n, K, w2c in zip(names, cams["K"], cams["w2c"]):
            cfg.append(CameraConfig("demo_" + n, sapien_pose(w2c), int(cams["width"]), int(cams["height"]),
                                    intrinsic=K, near=0.01, far=20, shader_pack="default"))
        return cfg
    V.camera_configs = cc
    import gymnasium as gym
    import sim.microwave_env as M  # noqa: F401
    from sim import dit_vis as D
    gen = json.load(open(os.path.join(D.ROOT, "Gitignore-but-important/sim/gen/summary.json")))["runs"]
    pots = np.array([r["pot"] for r in gen])
    rng = np.random.default_rng(1)
    pot = [rng.uniform(pots.min(0), pots.max(0)).round(4).tolist() for _ in range(a.index + 1)][-1]
    seed = 1000 + a.index
    dev = torch.device("cuda")
    m, hq = D.load_model(a.ckpt, dev)
    env = gym.make("MicrowavePot-v0", obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
                   pot_mw=(*pot, np.pi / 2), views="hq" if hq else True, sim_backend="cpu", max_episode_steps=D.T_MAX)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    # the display cameras sit where the splat render assumed
    for n, w2c in zip(names, cams["w2c"]):
        E = np.eye(4)
        E[:3] = obs["sensor_param"]["demo_" + n]["extrinsic_cv"][0].cpu().numpy()
        err = np.abs(E @ M.T_W_MW - w2c).max()
        assert err < 1e-3, f"demo camera {n} off by {err}"
    bg = D.real_background(a.front, env)
    wsplat = None
    if a.splat_wrist:
        from sim.gs_client import SplatRenderer
        wsplat = SplatRenderer(a.splat_wrist)
    demo = np.load(a.demo)
    os.makedirs(a.out, exist_ok=True)
    hist = [D.prop_of(obs, m.prop)] * D.N_OBS
    torch.manual_seed(seed)
    t, done, info = 0, False, {}
    while t < D.T_MAX and not done:
        q = hist[-1][:7].numpy()
        fr_, wr_ = D.images(obs, dev, bg, wsplat)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not m.fp32):
            acts = m.sample(torch.stack(hist[-D.N_OBS:])[None].to(dev), fr_, wr_)[0].float().cpu().numpy()
        for act in acts[:D.E]:
            obs, _, _, _, info = env.step(np.r_[q + act[:7], float(np.clip(np.round(act[7]), -1, 1))])
            hist.append(D.prop_of(obs, m.prop))
            t += 1
            done = bool(info["success"][0])
            if t % a.every == 0 or done:
                sd = obs["sensor_data"]
                frames = {n: composite(sd["demo_" + n], demo["rgb"][i], demo["depth"][i], bg[1])
                          for i, n in enumerate(names)}
                fr_, wr_ = D.images(obs, dev, bg, wsplat)
                np.savez_compressed(os.path.join(a.out, f"f{t:05d}.npz"), front=fr_[0].cpu().numpy(),
                                    wrist=wr_[0].cpu().numpy(), door=float(info["door_deg"][0]), **frames)
            if done:
                break
    res = dict(pot=pot, seed=seed, steps=t, success=done, door_max=float(info["door_deg"][0]),
               pot_inside=bool(info["pot_inside"][0]))
    json.dump(res, open(os.path.join(a.out, "episode.json"), "w"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
