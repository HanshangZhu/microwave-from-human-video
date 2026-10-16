"""Control: replay take_000 actions open-loop in ManiSkill itself (CPU) -- does the reference reproduce, and how
fast is it? Optional pot jitter to compare fragility with Newton's robust.py.
(run in the `maniskill` env)  python ms_replay_check.py [--jitter-xy 0.005 --jitter-yaw-deg 3 --n 8]
"""
import argparse
import json
import os
import sys
import time

import gymnasium as gym
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import sim.microwave_env  # noqa: F401,E402  registers MicrowavePot-v0

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--jitter-xy", type=float, default=0.0)
ap.add_argument("--jitter-yaw-deg", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

take = os.path.join(ROOT, "Gitignore-but-important/sim/take_000")
meta = json.load(open(os.path.join(take, "solve.json")))
kw = meta["env_info"]["env_kwargs"]
acts = h5py.File(os.path.join(take, "solve.h5"))["traj_0/actions"][:]
rng = np.random.default_rng(args.seed)
out = []
for i in range(args.n):
    pot = np.array(kw["pot_mw"], float)
    if i > 0:
        pot[:2] += rng.uniform(-args.jitter_xy, args.jitter_xy, 2)
        pot[2] += np.deg2rad(rng.uniform(-args.jitter_yaw_deg, args.jitter_yaw_deg))
    env = gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", pot_mw=tuple(pot),
                   sim_backend="cpu", max_episode_steps=4000)
    env.reset(seed=0)
    t0 = time.time()
    for a in acts:
        _, _, _, _, info = env.step(a)
    wall = time.time() - t0
    r = dict(i=i, pot_mw=pot.round(4).tolist(), success=bool(info["success"][0]), door_deg=float(info["door_deg"][0]),
             pot_inside=bool(info["pot_inside"][0]), steps_per_s=len(acts) / wall)
    print(json.dumps(r), flush=True)
    out.append(r)
    env.close()
print("success_rate", np.mean([r["success"] for r in out]))
