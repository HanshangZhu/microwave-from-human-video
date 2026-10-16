#!/usr/bin/env python
"""Residual RL on the DiT policy, in CPU physics (the physics the demonstrations and the
DiT were made in), with a stage-aware dense reward.

Why not sim/rl_residual.py's GPU physics: the same DiT succeeds 70 % in CPU physics and
35 % in GPU physics (128 episodes each), so RL there first has to make up the physics
difference; it went from 40 % to 23 % in 2 M steps. Here N CPU environments run in worker
processes and the DiT and the residual policy run batched on the GPU.

Reward per step (stages in order: door open, pot lifted, pot inside, door shut):
  bonus   +1 / +2 / +3 / +5 the first time each stage is reached
  progress  w_prog * decrease of the current stage's distance: tcp to the handle and the
          door's angle towards 90 deg, then tcp to the pot's grasp point, then the pot to
          the place target, then the door's angle towards 0
  style   -w_time per step, -w_jerk per mm/step^2 of tcp acceleration, -w_stop per step
          stopped (< 1 mm/step after having moved), -w_res |residual|^2
Two stages: first for success (task terms only), then from that residual with the
style terms on as well (--res to start from a checkpoint).

  python sim/rl_cpu.py train --ckpt <dit.pt> --out <dir> [--steps 4e6]
  python sim/rl_cpu.py eval  --ckpt <dit.pt> [--res <dir>/residual.pt] --out <dir> [--n 128]

Eval is deterministic (no exploration noise) and reports success per stage and, over the
successful episodes, steps, tcp path length, stops and mean tcp acceleration.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))
from rl_residual import POT_REGION, T_EP, Base, Residual, Tracker, summarize  # noqa: E402

HANDLE_DOOR = np.array([0.343, -0.062, 0.17])         # grasp point on the bar, door frame
PLACE = np.array([0.18, 0.09, 0.041])                 # demonstrations' place, on the turntable
POT_GRASP_Z = 0.05


def worker(conn, wid, seed, k):
    """k environments in one process, stepped one after the other (a CPU step takes ~0.5 ms;
    a process with torch + SAPIEN loaded takes ~0.8 GB, so one process per env runs out of
    memory long before it runs out of cores). Environment gid = wid * k + j keeps its own
    seed sequence, so the start states do not depend on k."""
    import gymnasium as gym
    import sim.microwave_env as M  # noqa: F401
    torch.set_num_threads(1)
    envs = [gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                     render_backend="none", pot_region=POT_REGION, max_episode_steps=T_EP) for _ in range(k)]
    us = [e.unwrapped for e in envs]
    eps = [0] * k

    def snap(u, info):
        return dict(tcp=u.tcp_in_mw()[0].numpy().copy(), pot=u.pot_in_mw()[0].numpy().copy(),
                    door=float(info["door_deg"][0]), inside=bool(info["pot_inside"][0]),
                    success=bool(info["success"][0]))

    def reset(j):
        obs, _ = envs[j].reset(seed=seed * 100000 + (wid * k + j) * 1000 + eps[j])
        eps[j] += 1
        return obs[0].numpy(), snap(us[j], us[j].evaluate())

    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            conn.send([reset(j) for j in range(k)])
        elif cmd == "step":
            out = []
            for j in range(k):
                obs, _, term, trunc, info = envs[j].step(data[j])
                s = snap(us[j], info)
                if bool(term) or bool(trunc):
                    o2, s2 = reset(j)
                    out.append((o2, s, True, s2))
                else:
                    out.append((obs[0].numpy(), s, False, None))
            conn.send(out)
        elif cmd == "close":
            for e in envs:
                e.close()
            conn.close()
            return


class Pool:
    def __init__(self, n, seed, k=1):
        assert n % k == 0, "--n-envs must be a multiple of --envs-per-proc"
        ctx = mp.get_context("spawn")
        self.k, self.conns, self.procs = k, [], []
        for i in range(n // k):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, i, seed, k), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        r = [x for c in self.conns for x in c.recv()]
        return np.stack([x[0] for x in r]), [x[1] for x in r]

    def step(self, acts):
        for i, c in enumerate(self.conns):
            c.send(("step", acts[i * self.k:(i + 1) * self.k]))
        return [x for c in self.conns for x in c.recv()]

    def close(self):
        for c in self.conns:
            c.send(("close", None))


def door_rot(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def stage_distance(stage, s):
    """The quantity the current stage should drive down (metres, or door fraction)."""
    q = np.deg2rad(s["door"])
    if stage == 0:
        return np.linalg.norm(s["tcp"] - door_rot(q) @ HANDLE_DOOR) + (1 - min(s["door"], 90) / 90)
    if stage == 1:
        return np.linalg.norm(s["tcp"] - (s["pot"] + [0, 0, POT_GRASP_Z]))
    if stage == 2:
        return np.linalg.norm(s["pot"] - PLACE)
    return s["door"] / 90


def run(a, train):
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    pool = Pool(a.n_envs, a.seed, a.envs_per_proc)
    obs_np, snaps = pool.reset()
    obs = torch.tensor(obs_np, device=dev)
    base = Base(a.ckpt, a.n_envs, dev)
    base.reset(obs)
    res = Residual(init_log_std=a.init_log_std).to(dev)
    if a.res:
        res.load_state_dict(torch.load(a.res, map_location=dev))
    else:
        res.mu.copy_(base.m.o_mu)
        res.sd.copy_(base.m.o_sd)
    sample = train or a.stochastic                  # eval --stochastic: with the exploration noise
    use_res = sample or a.res is not None
    opt = torch.optim.Adam(res.parameters(), lr=a.lr)
    tr = Tracker(a.n_envs, dev)
    tr.reset(torch.ones(a.n_envs, dtype=torch.bool, device=dev), torch.tensor(np.stack([s["tcp"] for s in snaps]), device=dev))
    stage = np.zeros(a.n_envs, int)
    t_env = torch.zeros(a.n_envs, dtype=torch.long, device=dev)
    # per-episode reward components, raw (unweighted) and discounted from the episode start,
    # so any checkpoint's return can be recomputed under any weights
    COMP = ("bonus", "prog", "acc_mm", "stopped", "res_sq")
    comp = {kk: torch.zeros(a.n_envs, device=dev) for kk in COMP + tuple("g_" + c for c in COMP + ("steps",))}
    total, it, t0, log = 0, 0, time.time(), []
    while total < a.steps:
        buf = {k: [] for k in ("x", "u", "logp", "v", "r", "done")}
        for _ in range(a.rollout):
            b, k = base.actions(obs, t_env)
            x = res.inp(obs, b, torch.full((a.n_envs,), k, device=dev, dtype=torch.long))
            with torch.no_grad():
                dist = res.dist(x)
                uu = dist.sample() if sample else dist.mean
                logp = dist.log_prob(uu).sum(1)
                v = res.v(x).squeeze(1)
            r_act = uu.clamp(-1, 1) * a.res_scale if use_res else torch.zeros_like(uu)
            act = torch.cat([b[:, :7] + r_act, b[:, 7:]], 1).cpu().numpy()
            d_before = np.array([stage_distance(stage[i], snaps[i]) for i in range(a.n_envs)])
            out = pool.step(act)
            new_obs = np.stack([o[0] for o in out])
            cur = [o[1] for o in out]
            done = np.array([o[2] for o in out])
            d_after = np.array([stage_distance(stage[i], cur[i]) for i in range(a.n_envs)])
            tcp = torch.tensor(np.stack([s["tcp"] for s in cur]), device=dev)
            new, acc = tr.step(tcp, torch.tensor([s["door"] for s in cur], device=dev),
                               torch.tensor([s["pot"][2] for s in cur], device=dev),
                               torch.tensor([s["inside"] for s in cur], device=dev),
                               torch.tensor([s["success"] for s in cur], device=dev))
            bonus = (new.float() * torch.tensor(a.bonus, device=dev)).sum(1)
            prog = torch.tensor(d_before - d_after, device=dev, dtype=torch.float32)
            # a stop: the tcp at rest (< 1 mm/step) after having moved -- the solver's waits
            stopped = (tr.still > 0) & tr.moved
            r = bonus + a.w_prog * prog - a.w_time - a.w_jerk * acc * 1000 - a.w_stop * stopped.float() \
                - a.w_res * (uu.clamp(-1, 1) ** 2).sum(1)
            gt = a.gamma ** t_env.float()
            for kk, val in (("bonus", bonus), ("prog", prog), ("acc_mm", acc * 1000), ("stopped", stopped.float()),
                            ("res_sq", (uu.clamp(-1, 1) ** 2).sum(1)), ("steps", torch.ones_like(bonus))):
                if kk != "steps":
                    comp[kk] += val
                comp["g_" + kk] += gt * val
            stage = tr.flags[:, :3].sum(1).cpu().numpy()            # stages are reached in order
            snaps = cur
            done_t = torch.tensor(done, device=dev)
            obs = torch.tensor(new_obs, device=dev)
            if done.any():
                tr.finish(done_t)
                idx = np.nonzero(done)[0]
                for e, i in zip(tr.done_eps[-len(idx):], idx):
                    e.update({"r_" + kk: float(vv[i]) for kk, vv in comp.items()})
                for vv in comp.values():
                    vv[done_t] = 0
                for i in np.nonzero(done)[0]:
                    snaps[i] = out[i][3]
                tr.reset(done_t, torch.tensor(np.stack([s["tcp"] for s in snaps]), device=dev))
                stage[done] = 0
                base.reset(obs, done_t)
                t_env[done_t] = 0
            t_env += 1
            base.push(obs)
            for key, val in (("x", x), ("u", uu), ("logp", logp), ("v", v), ("r", r), ("done", done_t.float())):
                buf[key].append(val)
        total += a.rollout * a.n_envs
        it += 1
        if train:
            with torch.no_grad():
                v_last = res.v(res.inp(obs, b, torch.full((a.n_envs,), k, device=dev, dtype=torch.long))).squeeze(1)
            R = {kk: torch.stack(vv) for kk, vv in buf.items()}
            adv = torch.zeros_like(R["r"])
            last = torch.zeros(a.n_envs, device=dev)
            for t in reversed(range(a.rollout)):
                nv = v_last if t == a.rollout - 1 else R["v"][t + 1]
                nonterm = 1 - R["done"][t]
                delta = R["r"][t] + a.gamma * nv * nonterm - R["v"][t]
                last = delta + a.gamma * a.lam * nonterm * last
                adv[t] = last
            ret = adv + R["v"]
            X, U, LP, ADV, RET = (z.reshape(-1, *z.shape[2:]) for z in (R["x"], R["u"], R["logp"], adv, ret))
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            for _ in range(a.epochs):
                perm = torch.randperm(len(X), device=dev)
                for i in range(0, len(X), a.mb):
                    j = perm[i:i + a.mb]
                    d = res.dist(X[j])
                    ratio = (d.log_prob(U[j]).sum(1) - LP[j]).exp()
                    pl = -torch.min(ratio * ADV[j], ratio.clamp(0.8, 1.2) * ADV[j]).mean()
                    vl = ((res.v(X[j]).squeeze(1) - RET[j]) ** 2).mean()
                    opt.zero_grad()
                    (pl + 0.5 * vl).backward()
                    nn.utils.clip_grad_norm_(res.parameters(), 0.5)
                    opt.step()
        if it % a.log_every == 0 or total >= a.steps:
            s = summarize(tr.done_eps[-3 * a.n_envs:])
            rec = dict(steps=total, sps=int(total / (time.time() - t0)), episodes_done=len(tr.done_eps),
                       mean_r=float(torch.stack(buf["r"]).mean()), std=float(res.log_std.exp().mean().detach()), **s)
            log.append(rec)
            print(json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in rec.items()}), flush=True)
            if train:
                torch.save(res.state_dict(), os.path.join(a.out, "residual.pt"))
                if total // a.save_every > (total - a.rollout * a.n_envs * a.log_every) // a.save_every:
                    torch.save(res.state_dict(), os.path.join(a.out, f"residual_{total // 1000:05d}k.pt"))
        if not train and len(tr.done_eps) >= a.n:
            break
    pool.close()
    eps = tr.done_eps[:a.n] if not train else tr.done_eps[-3 * a.n_envs:]
    json.dump(dict(summary=summarize(eps), log=log, episodes=eps),
              open(os.path.join(a.out, "train.json" if train else "eval.json"), "w"), indent=1)
    print("final:", json.dumps({kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in summarize(eps).items()}))
    if eps and "r_prog" in eps[0]:
        print("return:", json.dumps({kk: round(float(np.mean([e[kk] for e in eps])), 4)
                                     for kk in eps[0] if kk.startswith("r_")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--res")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--envs-per-proc", type=int, default=1, help="environments per worker process")
    ap.add_argument("--n", type=int, default=128, help="eval: episodes")
    ap.add_argument("--steps", type=float, default=4e6)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--res-scale", type=float, default=0.03)
    ap.add_argument("--init-log-std", type=float, default=-2.2)
    ap.add_argument("--w-prog", type=float, default=2.0)
    ap.add_argument("--w-time", type=float, default=0.0002)
    ap.add_argument("--w-jerk", type=float, default=0.0005, help="per mm/step^2 of tcp acceleration")
    ap.add_argument("--w-res", type=float, default=0.0005)
    ap.add_argument("--w-stop", type=float, default=0.0, help="per step the tcp is stopped after moving")
    ap.add_argument("--bonus", type=float, nargs=4, default=[1.0, 2.0, 3.0, 5.0],
                    help="door open, pot lifted, pot inside, success")
    ap.add_argument("--save-every", type=float, default=5e5)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true",
                    help="eval: sample the residual with its exploration noise, as in training")
    a = ap.parse_args()
    a.steps = int(a.steps) if a.cmd == "train" else 10 ** 9
    os.makedirs(a.out, exist_ok=True)
    run(a, train=(a.cmd == "train"))


if __name__ == "__main__":
    main()
