#!/usr/bin/env python
"""Camera images for existing demonstrations, by re-rendering their simulator states.

Every demonstration stores the simulator state behind each recorded observation
(sim/generate.py runs: solve.h5 env_states; sim/dagger.py runs: demo.npz states), so
the images need no re-simulation and the cameras can change later without regenerating
any data. Each --stride-th step is set and rendered from sim/views.py's cameras: the real
front RealSense (pose and intrinsics from the human take) and the wrist camera.

  python sim/render_views.py --data <dir>[,<dir>...] [--stride 4] [--jobs 6]

Writes <run>/views.npz next to each successful run's demo: idx (the step of each image),
front (N, 126, 224, 3) and wrist (N, 128, 128, 3), uint8 RGB. --hq: the default shader,
rendered at 2x and area-downsampled (no aliasing) -> <run>/views_hq.npz.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def states_of(run_dir):
    """('dict', h5 env_states group arrays) or ('flat', (T, D) array), aligned with demo obs."""
    z = os.path.join(run_dir, "demo.npz")
    if os.path.exists(z):
        d = np.load(z)
        if "states" in d.files:
            return "flat", d["states"]
    import h5py
    with h5py.File(os.path.join(run_dir, "solve.h5"), "r") as f:
        g = f["traj_0/env_states"]
        out = {"actors": {k: g["actors"][k][:] for k in g["actors"]},
               "articulations": {k: g["articulations"][k][:] for k in g["articulations"]}}
    return "dict", out


def render_run(args):
    run_dir, stride, take, hq = args[:4]
    dr = args[4] if len(args) > 4 else None                 # None, "procedural" or "doorman"
    variant = args[5] if len(args) > 5 else 0
    name = {None: "views_hq" if hq else "views", "procedural": "views_dr", "doorman": f"views_drm{variant}"}[dr]
    out = os.path.join(run_dir, name + ".npz")
    if os.path.exists(out):
        return run_dir, "exists"
    import torch
    import gymnasium as gym
    import sim.microwave_env as M  # noqa: F401
    from sim.views import downsample
    torch.set_num_threads(1)
    import zlib
    env = gym.make("MicrowavePot-v0", obs_mode="rgb", control_mode="pd_joint_pos", sim_backend="cpu",
                   views="hq" if (hq or dr) else True, dr_style=dr or "procedural",
                   dr_seed=((zlib.crc32(os.path.abspath(run_dir).encode()) + 7777 * variant) & 0x7FFFFFFF) if dr else None)
    env.reset(seed=0)
    u = env.unwrapped
    kind, S = states_of(run_dir)
    n_obs = len(np.load(os.path.join(run_dir, "demo.npz"))["obs"]) if os.path.exists(os.path.join(run_dir, "demo.npz")) \
        else len(S["articulations"]["panda"]) - 1
    idx = np.arange(0, n_obs, stride)
    front, wrist = [], []
    for i in idx:
        if kind == "flat":
            u.set_state(torch.tensor(S[i:i + 1]))
        else:
            u.set_state_dict({"actors": {k: torch.tensor(v[i:i + 1]) for k, v in S["actors"].items()},
                              "articulations": {k: torch.tensor(v[i:i + 1]) for k, v in S["articulations"].items()}})
        sd = u.get_obs()["sensor_data"]
        front.append(downsample(sd["front"]["rgb"][0].cpu().numpy(), "front"))
        wrist.append(downsample(sd["wrist"]["rgb"][0].cpu().numpy(), "wrist"))
    np.savez_compressed(out, idx=idx, front=np.stack(front), wrist=np.stack(wrist), take=take)
    env.close()
    return run_dir, f"{len(idx)} frames"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma-separated generator / dagger output dirs")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--take", default="take_000", help="whose camera the front view is")
    ap.add_argument("--hq", action="store_true", help="default shader, 2x supersampled -> views_hq.npz")
    ap.add_argument("--dr", choices=["procedural", "doorman"], help="hq + a random visual scene per run (sim/dr.py)"
                    " -> views_dr.npz / views_drm<variant>.npz")
    ap.add_argument("--variant", type=int, default=0, help="--dr doorman: another random scene per run")
    a = ap.parse_args()
    runs = []
    for d in a.data.split(","):
        s = json.load(open(os.path.join(d, "summary.json")))
        runs += [os.path.join(d, f"run_{r['run']:03d}") for r in s["runs"] if r["success"]]
    print(f"{len(runs)} successful runs", flush=True)
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for k, (r, msg) in enumerate(ex.map(render_run, [(r, a.stride, a.take, a.hq, a.dr, a.variant) for r in runs])):
            print(f"  {k + 1:3d}/{len(runs)} {os.path.relpath(r, ROOT)}: {msg}", flush=True)


if __name__ == "__main__":
    main()
