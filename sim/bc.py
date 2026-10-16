#!/usr/bin/env python
"""Behaviour cloning on the generated MicrowavePot-v0 demonstrations, and its evaluation.

Policy: an MLP from the state (33-d: joint positions and velocities, tcp pose, pot pose,
door angle -- normalised) plus the elapsed fraction of the episode, to a chunk of the
next H joint targets as offsets from the current joint positions, and the gripper command
for each of them. At run time it is queried every E steps and the first E of its H
actions are executed (action chunking, as in ACT), targets = current joints + offsets.

  python sim/bc.py train --data Gitignore-but-important/sim/gen --out <dir>
  python sim/bc.py eval  --ckpt <dir>/bc.pt --n 20 --out <dir>/eval [--jobs 10]

Evaluation samples new pot positions in the region the generator used (seed differs from
the generator's) and reports, per stage: door opened (> 60 deg at some point), pot
grasped and lifted (> 5 cm off the counter), pot in the cavity, door shut at the end.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

H, E, T_MAX = 8, 4, 3000


def compress(obs, act, keep=2, still=0.05):
    """Drop waiting steps: the action unchanged and every joint (fingers too) still. At
    most `keep` of each such run are kept."""
    same = np.r_[False, np.all(np.abs(np.diff(act, axis=0)) < 1e-6, axis=1)]
    idle = same & np.all(np.abs(obs[:, 9:18]) < still, axis=1)
    run, keep_mask = 0, np.ones(len(act), bool)
    for i, d in enumerate(idle):
        run = run + 1 if d else 0
        keep_mask[i] = run <= keep
    return obs[keep_mask], act[keep_mask]


def load(data_dirs, squeeze=False):
    """Successful episodes from one or more generator output dirs (comma-separated).
    demo.npz (observation, clean planned action) is used when present -- with DART noise
    it is the only correct pairing; otherwise the recorded h5 (no noise: the same)."""
    import h5py
    eps = []
    for data_dir in data_dirs.split(","):
        s = json.load(open(os.path.join(data_dir, "summary.json")))
        for r in s["runs"]:
            if not r["success"]:
                continue
            d = os.path.join(data_dir, f"run_{r['run']:03d}")
            if os.path.exists(os.path.join(d, "demo.npz")):
                z = np.load(os.path.join(d, "demo.npz"))
                o, ac = z["obs"].astype(np.float32), z["actions"].astype(np.float32)
            else:
                with h5py.File(os.path.join(d, "solve.h5"), "r") as f:
                    o, ac = f["traj_0/obs"][:-1].astype(np.float32), f["traj_0/actions"][:].astype(np.float32)
            if squeeze:
                o, ac = compress(o, ac)
            eps.append((o, ac, r))
    return eps


def features(obs, t):
    return np.concatenate([obs, np.atleast_2d(t / T_MAX).T.astype(np.float32)], 1)


def chunks(obs, act):
    """Targets at every step: the next H joint targets minus the current joints, and the
    next H gripper commands (the episode's last action repeated past its end)."""
    T = len(act)
    idx = np.minimum(np.arange(T)[:, None] + np.arange(H)[None], T - 1)
    dq = act[idx, :7] - obs[:, None, :7]
    g = act[idx, 7]
    return np.concatenate([dq.reshape(T, -1), g], 1)


class Policy(nn.Module):
    def __init__(self, d_in, d_out, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, w), nn.GELU(), nn.Linear(w, w), nn.GELU(),
                                 nn.Linear(w, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, d_out))
        self.register_buffer("x_mu", torch.zeros(d_in))
        self.register_buffer("x_sd", torch.ones(d_in))
        self.register_buffer("y_mu", torch.zeros(d_out))
        self.register_buffer("y_sd", torch.ones(d_out))

    def forward(self, x):
        return self.net((x - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu


def train(a):
    eps = load(a.data)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(eps))
    val_ids = set(order[:a.n_val].tolist())
    X, Y, Xv, Yv = [], [], [], []
    for i, (o, ac, _) in enumerate(eps):
        x, y = features(o, np.arange(len(o))), chunks(o, ac)
        (Xv if i in val_ids else X).append(x)
        (Yv if i in val_ids else Y).append(y)
    X, Y, Xv, Yv = map(np.concatenate, (X, Y, Xv, Yv))
    print(f"{len(eps)} episodes: {len(X):,} train / {len(Xv):,} val samples, x {X.shape[1]}-d, y {Y.shape[1]}-d",
          flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pol = Policy(X.shape[1], Y.shape[1]).to(dev)
    for name, arr in (("x_mu", X.mean(0)), ("x_sd", X.std(0) + 1e-3), ("y_mu", Y.mean(0)), ("y_sd", Y.std(0) + 1e-3)):
        getattr(pol, name).copy_(torch.tensor(arr))
    Xt, Yt = torch.tensor(X, device=dev), torch.tensor(Y, device=dev)
    Xvt, Yvt = torch.tensor(Xv, device=dev), torch.tensor(Yv, device=dev)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(X) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ysd = pol.y_sd
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for k in range(0, len(Xt) - a.batch + 1, a.batch):
            b = perm[k:k + a.batch]
            loss = (((pol(Xt[b]) - Yt[b]) / ysd) ** 2).mean()   # in normalised units
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            with torch.no_grad():
                vl = (((pol(Xvt) - Yvt) / ysd) ** 2).mean().item()
                err = (pol(Xvt) - Yvt)[:, :7].abs().mean().item()
            print(f"epoch {ep + 1:3d}: train {loss.item():.4f}  val {vl:.4f}  "
                  f"(first joint-target offset error {np.degrees(err):.2f} deg)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(dict(state=pol.state_dict(), d_in=X.shape[1], d_out=Y.shape[1]), os.path.join(a.out, "bc.pt"))
    print("->", os.path.join(a.out, "bc.pt"))


def rollout(args):
    ckpt, pot, seed, video_dir = args
    import gymnasium as gym
    import sim.microwave_env as M  # noqa: F401
    torch.set_num_threads(1)
    c = torch.load(ckpt, map_location="cpu")
    pol = Policy(c["d_in"], c["d_out"])
    pol.load_state_dict(c["state"])
    pol.eval()
    env = gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", pot_mw=(*pot, np.pi / 2),
                   render_mode="rgb_array" if video_dir else None, sim_backend="cpu", max_episode_steps=T_MAX)
    if video_dir:
        from mani_skill.utils.wrappers.record import RecordEpisode
        env = RecordEpisode(env, video_dir, save_video=True, save_trajectory=False, video_fps=30,
                            max_steps_per_video=T_MAX)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    door_max, pot_max, info = 0.0, 0.0, {}
    t = 0
    while t < T_MAX:
        o = obs.cpu().numpy() if torch.is_tensor(obs) else obs
        o = o.reshape(-1)
        with torch.no_grad():
            y = pol(torch.tensor(features(o[None], np.array([t]))))[0].numpy()
        dq, g = y[:7 * H].reshape(H, 7), y[7 * H:]
        for k in range(E):
            act = np.r_[o[:7] + dq[k], float(np.clip(np.round(g[k]), -1, 1))]
            obs, _, term, trunc, info = env.step(act)
            t += 1
            door_max = max(door_max, float(info["door_deg"][0]))
            pot_max = max(pot_max, float(u.pot_in_mw()[0, 2]))
            if bool(info["success"][0]) and t > 200:
                break
        if bool(info["success"][0]) and t > 200:
            break
    p = u.pot_in_mw().cpu().numpy()[0]
    res = dict(pot=list(pot), seed=seed, steps=t, door_opened=door_max > 60, door_max_deg=door_max,
               grasped_lifted=pot_max > 0.05, pot_max_z=pot_max, pot_inside=bool(info["pot_inside"][0]),
               door_shut=bool(info["door_shut"][0]), success=bool(info["success"][0]), pot_final=p.round(3).tolist())
    env.close()
    return res


def evaluate(a):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    gen = json.load(open(os.path.join(a.data, "summary.json")))["runs"]
    pots = np.array([r["pot"] for r in gen])
    lo, hi = pots.min(0), pots.max(0)
    rng = np.random.default_rng(a.seed)
    jobs = [(a.ckpt, rng.uniform(lo, hi).round(4).tolist(), 1000 + i,
             os.path.join(a.out, f"video_{i:02d}") if i < a.videos else None) for i in range(a.n)]
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(rollout, jobs))
    keys = ["door_opened", "grasped_lifted", "pot_inside", "door_shut", "success"]
    summ = {k: float(np.mean([r[k] for r in res])) for k in keys}
    for i, r in enumerate(res):
        print(f"  {i:2d} pot {r['pot']}: door max {r['door_max_deg']:5.1f}  pot max z {r['pot_max_z']:.3f}  "
              + "  ".join(k for k in keys if r[k]), flush=True)
    print("rates:", {k: f"{v * 100:.0f}%" for k, v in summ.items()}, f"(n={len(res)})")
    json.dump(dict(rates=summ, episodes=res), open(os.path.join(a.out, "eval.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--data", default=os.path.join(ROOT, "Gitignore-but-important/sim/gen"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
