#!/usr/bin/env python
"""Residual RL on top of the DiT policy (sim/dit.py), towards the demonstrations' direct,
continuous motion and away from the solver's stop-and-go.

The DiT (frozen) proposes a chunk of joint targets every E steps, closed-loop on the
state. A small Gaussian residual policy, trained with PPO in GPU-parallel simulation,
adds up to --res-scale rad to each step's 7 joint targets (the gripper stays the DiT's).
Because the DiT re-reads the state every chunk, a residual that gets the arm ahead makes
the DiT move on to its next targets: the residual can straighten the path and change the
pace, not only nudge it.

Reward per step:
  task:   one-off bonuses as each stage is first reached: door > 60 deg (+1), pot lifted
          > 5 cm (+2), pot in the cavity (+3), success = pot in and door shut (+5)
  pace:   -w_time per step
  smooth: -w_jerk * |d2 p_tcp| (m/step^2): stop-and-go is acceleration and deceleration
  small:  -w_res * |residual / res_scale|^2

  python archive/state-policy/rl_residual.py base --ckpt <dit.pt> --out <dir>             # the DiT alone
  python archive/state-policy/rl_residual.py train --ckpt <dit.pt> --out <dir> [--steps 5e6]
  python archive/state-policy/rl_residual.py eval --ckpt <dit.pt> --res <dir>/residual.pt --out <dir>

Episode metrics (logged and in eval): success, stages, steps, tcp path length, stops
(the tcp nearly at rest, < 1 mm/step, for 10+ steps after moving), mean |d2 p_tcp|.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim.microwave_env as M  # noqa: E402
from dit import DiT, E, N_OBS  # noqa: E402

POT_REGION = ((0.421, -0.299), (0.524, -0.153))     # sim/generate.py's pick region
T_EP = 3000


def make_env(n, seed):
    import gymnasium as gym
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    env = gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", sim_backend="physx_cuda",
                   num_envs=n, pot_region=POT_REGION, max_episode_steps=T_EP)
    return ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)


class Base:
    """The frozen DiT: chunks of absolute joint targets + gripper, re-queried every E steps."""

    def __init__(self, ckpt, n, dev):
        c = torch.load(ckpt, map_location=dev)
        self.m = DiT(c["d_obs"]).to(dev)
        self.m.load_state_dict(c["state"])
        self.m.eval()
        self.no_time = c.get("no_time", False)
        self.n, self.dev = n, dev
        self.hist = None
        self.k = 0
        self.force = True

    def reset(self, obs, mask=None):
        if self.hist is None or mask is None:
            self.hist = obs[:, None].repeat(1, N_OBS, 1)
        else:
            self.hist[mask] = obs[mask][:, None].repeat(1, N_OBS, 1)
        # a fresh episode must not finish the previous episode's chunk: re-plan now
        self.force = True

    def push(self, obs):
        self.hist = torch.cat([self.hist[:, 1:], obs[:, None]], 1)

    @torch.no_grad()
    def actions(self, obs, t):
        """The base action for this step; re-plans at chunk boundaries."""
        if self.k % E == 0 or self.force:
            self.k, self.force = 0, False
            tf = torch.zeros(self.n, device=self.dev) if self.no_time else t.float() / 3000
            ch = self.m.sample(self.hist, tf)
            self.q = obs[:, None, :7] + ch[:, :, :7]
            self.g = ch[:, :, 7].clamp(-1, 1).round()
        a = torch.cat([self.q[:, self.k % E], self.g[:, self.k % E, None]], 1)
        self.k += 1
        return a, (self.k - 1) % E


class Residual(nn.Module):
    def __init__(self, d_obs=33, w=256, init_log_std=-2.2):
        super().__init__()
        d = d_obs + 8 + E
        self.pi = nn.Sequential(nn.Linear(d, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(), nn.Linear(w, 7))
        self.v = nn.Sequential(nn.Linear(d, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(), nn.Linear(w, 1))
        # small: exploration noise is per-step jitter on the joint targets, and at
        # exp(-1) * res_scale (~0.011 rad) it broke grasps -- success fell from 35 % to 7 %
        self.log_std = nn.Parameter(torch.full((7,), float(init_log_std)))
        nn.init.zeros_(self.pi[-1].weight)
        nn.init.zeros_(self.pi[-1].bias)                    # starts as "no residual"
        self.register_buffer("mu", torch.zeros(d_obs))
        self.register_buffer("sd", torch.ones(d_obs))

    def inp(self, obs, base, k):
        onehot = torch.nn.functional.one_hot(k, E).float()
        return torch.cat([(obs - self.mu) / self.sd, base, onehot], 1)

    def dist(self, x):
        return torch.distributions.Normal(self.pi(x), self.log_std.exp())


class Tracker:
    """Per-env episode bookkeeping: stage flags (for the one-off bonuses) and motion metrics."""

    def __init__(self, n, dev):
        self.n, self.dev = n, dev
        z = lambda dt=torch.float32: torch.zeros(n, device=dev, dtype=dt)
        self.flags = torch.zeros(n, 4, dtype=torch.bool, device=dev)
        self.steps, self.path, self.acc, self.still, self.stops = z(), z(), z(), z(), z()
        self.p = torch.zeros(n, 3, 3, device=dev)            # last three tcp positions
        self.moved = z(torch.bool)
        self.done_eps = []

    def reset(self, mask, tcp):
        for a in (self.steps, self.path, self.acc, self.still, self.stops):
            a[mask] = 0
        self.flags[mask] = False
        self.moved[mask] = False
        self.p[mask] = tcp[mask][:, None].repeat(1, 3, 1)

    def step(self, tcp, door_deg, pot_z, inside, success):
        v = tcp - self.p[:, 2]
        d2 = tcp - 2 * self.p[:, 2] + self.p[:, 1]
        self.p = torch.cat([self.p[:, 1:], tcp[:, None]], 1)
        speed = v.norm(dim=1)
        self.steps += 1
        self.path += speed
        self.acc += d2.norm(dim=1)
        self.moved |= speed > 0.003
        slow = speed < 0.001
        self.still = torch.where(slow, self.still + 1, torch.zeros_like(self.still))
        self.stops += ((self.still == 10) & self.moved).float()
        new = torch.stack([door_deg > 60, pot_z > 0.05, inside, success], 1) & ~self.flags
        self.flags |= new
        return new, d2.norm(dim=1)

    def finish(self, mask):
        for i in torch.nonzero(mask).flatten().tolist():
            self.done_eps.append(dict(success=bool(self.flags[i, 3]), door=bool(self.flags[i, 0]),
                                      lifted=bool(self.flags[i, 1]), inside=bool(self.flags[i, 2]),
                                      steps=int(self.steps[i]), path_m=float(self.path[i]),
                                      stops=int(self.stops[i]), mean_acc_mm=float(self.acc[i] / self.steps[i] * 1000)))


def summarize(eps):
    if not eps:
        return {}
    k = lambda key: float(np.mean([e[key] for e in eps]))
    ok = [e for e in eps if e["success"]]
    s = dict(n=len(eps), success=k("success"), door=k("door"), lifted=k("lifted"), inside=k("inside"))
    if ok:
        s.update({f"succ_{key}": float(np.mean([e[key] for e in ok])) for key in ("steps", "path_m", "stops", "mean_acc_mm")})
    return s


def run(a, train):
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    env = make_env(a.n_envs, a.seed)
    obs, _ = env.reset(seed=a.seed)
    u = env.unwrapped
    base = Base(a.ckpt, a.n_envs, dev)
    base.reset(obs)
    res = Residual(init_log_std=a.init_log_std).to(dev)
    if a.res:
        res.load_state_dict(torch.load(a.res, map_location=dev))
    else:
        res.mu.copy_(base.m.o_mu)
        res.sd.copy_(base.m.o_sd)
    opt = torch.optim.Adam(res.parameters(), lr=a.lr)
    tr = Tracker(a.n_envs, dev)
    tr.reset(torch.ones(a.n_envs, dtype=torch.bool, device=dev), u.tcp_in_mw())
    t_env = torch.zeros(a.n_envs, dtype=torch.long, device=dev)
    use_res = train or a.res is not None
    total, t0, it = 0, time.time(), 0
    log = []
    while total < a.steps:
        buf = {k: [] for k in ("x", "u", "logp", "v", "r", "done")}
        for _ in range(a.rollout):
            b, k = base.actions(obs, t_env)
            x = res.inp(obs, b, torch.full((a.n_envs,), k, device=dev, dtype=torch.long))
            with torch.no_grad():
                dist = res.dist(x)
                uu = dist.sample() if train else dist.mean
                logp = dist.log_prob(uu).sum(1)
                v = res.v(x).squeeze(1)
            r_act = uu.clamp(-1, 1) * a.res_scale if use_res else torch.zeros_like(uu)
            act = torch.cat([b[:, :7] + r_act, b[:, 7:]], 1)
            obs, _, term, trunc, info = env.step(act)
            t_env += 1
            done = term | trunc
            fin = info.get("final_info", info)
            new, acc = tr.step(u.tcp_in_mw(), fin["door_deg"], u.pot_in_mw()[:, 2], fin["pot_inside"], fin["success"])
            bonus = (new.float() * torch.tensor([1.0, 2.0, 3.0, 5.0], device=dev)).sum(1)
            r = bonus - a.w_time - a.w_jerk * acc * 1000 - a.w_res * (uu.clamp(-1, 1) ** 2).sum(1)
            if done.any():
                tr.finish(done)
                tr.reset(done, u.tcp_in_mw())
                base.reset(obs, done)
                t_env[done] = 0
            base.push(obs)
            for key, val in (("x", x), ("u", uu), ("logp", logp), ("v", v), ("r", r), ("done", done.float())):
                buf[key].append(val)
        total += a.rollout * a.n_envs
        it += 1
        if train:
            with torch.no_grad():
                x_last = res.inp(obs, base.actions(obs, t_env)[0] if False else b, torch.full((a.n_envs,), k, device=dev, dtype=torch.long))
                v_last = res.v(x_last).squeeze(1)
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
                    pl = -torch.min(ratio * ADV[j], ratio.clamp(1 - 0.2, 1 + 0.2) * ADV[j]).mean()
                    vl = ((res.v(X[j]).squeeze(1) - RET[j]) ** 2).mean()
                    loss = pl + 0.5 * vl
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(res.parameters(), 0.5)
                    opt.step()
        if it % a.log_every == 0 or total >= a.steps:
            recent = tr.done_eps[-4 * a.n_envs:]
            s = summarize(recent)
            rec = dict(steps=total, sps=int(total / (time.time() - t0)), episodes_done=len(tr.done_eps),
                       mean_r=float(torch.stack(buf["r"]).mean()), std=float(res.log_std.exp().mean()), **s)
            log.append(rec)
            print(json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in rec.items()}), flush=True)
            if train:
                torch.save(res.state_dict(), os.path.join(a.out, "residual.pt"))
        if not train and len(tr.done_eps) >= a.n_envs:
            break
    os.makedirs(a.out, exist_ok=True)
    first = tr.done_eps[:a.n_envs] if not train else tr.done_eps[-a.n_envs:]
    json.dump(dict(summary=summarize(first), log=log, episodes=first),
              open(os.path.join(a.out, "train.json" if train else "eval.json"), "w"), indent=1)
    print("final:", json.dumps({kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in summarize(first).items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "train", "eval"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--res")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=128)
    ap.add_argument("--steps", type=float, default=5e6)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--res-scale", type=float, default=0.03)
    ap.add_argument("--w-time", type=float, default=0.001)
    ap.add_argument("--w-jerk", type=float, default=0.002, help="per mm/step^2 of tcp acceleration")
    ap.add_argument("--w-res", type=float, default=0.001)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--init-log-std", type=float, default=-2.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.steps = int(a.steps)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "base":
        a.res = None
        a.steps = 10 ** 9
    run(a, train=(a.cmd == "train"))


if __name__ == "__main__":
    main()
