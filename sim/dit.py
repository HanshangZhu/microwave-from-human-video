#!/usr/bin/env python
"""Diffusion-transformer policy with action chunking (DiT + chunks, as in Diffusion Policy /
ACT) for MicrowavePot-v0, trained on the same generated demonstrations as sim/bc.py.

Condition: the last N_OBS states (33-d each, normalised) and the elapsed fraction of the
episode, embedded as tokens. Output: a chunk of H future actions, each the joint-target
offset from the current joints (7) and the gripper command (1), produced by denoising H
action tokens with a transformer (DDPM training, DDIM sampling). At run time the first E
of the H actions are executed, then the policy is queried again.

Why over the MLP of sim/bc.py: the MLP rollouts leave the demonstrations within ~350 steps
(the arm drifts in its redundant degree of freedom while holding the handle) and never
recover; longer chunks mean fewer closed-loop decisions for errors to compound over, and
a generative model does not average different valid continuations into an invalid one.

  python sim/dit.py train --out <dir>
  python sim/dit.py eval  --ckpt <dir>/dit.pt --out <dir>/eval --n 20
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))
from bc import T_MAX, load  # noqa: E402

N_OBS, H, E = 2, 16, 8
K_TRAIN, K_SAMPLE = 100, 10
D_ACT = 8


def windows(obs, act):
    """Per step t: the last N_OBS states (padded at the start), elapsed fraction, and the
    next H actions as (joint target - current joints, gripper)."""
    T = len(act)
    oi = np.maximum(np.arange(T)[:, None] - np.arange(N_OBS)[None, ::-1], 0)
    o = obs[oi]                                                     # (T, N_OBS, 33)
    ai = np.minimum(np.arange(T)[:, None] + np.arange(H)[None], T - 1)
    a = act[ai].copy()                                              # (T, H, 8)
    a[..., :7] -= obs[:, None, :7]
    return o, (np.arange(T) / T_MAX).astype(np.float32), a


def betas():
    s = 0.008                                                       # cosine schedule
    t = np.linspace(0, K_TRAIN, K_TRAIN + 1) / K_TRAIN
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return np.clip(1 - f[1:] / f[:-1], 1e-5, 0.999).astype(np.float32)


class DiT(nn.Module):
    def __init__(self, d_obs, d=256, layers=4, heads=4):
        super().__init__()
        self.obs_in = nn.Linear(d_obs, d)
        self.time_in = nn.Linear(1, d)
        self.k_in = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.act_in = nn.Linear(D_ACT, d)
        self.pos = nn.Parameter(torch.randn(1, N_OBS + 2 + H, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D_ACT))
        for n in ("o_mu", "o_sd", "a_mu", "a_sd"):
            self.register_buffer(n, torch.zeros(d_obs if n[0] == "o" else D_ACT))
        self.register_buffer("beta", torch.tensor(betas()))

    def forward(self, a_noisy, k, o, tf):
        """a_noisy (B,H,8) normalised; k (B,) diffusion step; o (B,N_OBS,33) raw; tf (B,)."""
        on = (o - self.o_mu) / self.o_sd
        x = torch.cat([self.obs_in(on), self.time_in(tf[:, None, None]),
                       self.k_in((k.float()[:, None] / K_TRAIN))[:, None], self.act_in(a_noisy)], 1)
        y = self.tf(x + self.pos)
        return self.out(y[:, -H:])                                  # predicted noise

    @torch.no_grad()
    def sample(self, o, tf):
        ab = torch.cumprod(1 - self.beta, 0)
        x = torch.randn(len(o), H, D_ACT, device=o.device)
        ks = torch.linspace(K_TRAIN - 1, 0, K_SAMPLE, device=o.device).round().long()
        for i, k in enumerate(ks):                                  # DDIM, eta = 0
            eps = self(x, k.repeat(len(o)), o, tf)
            a0 = (x - (1 - ab[k]).sqrt() * eps) / ab[k].sqrt()
            a0 = a0.clamp(-5, 5)
            if i == len(ks) - 1:
                x = a0
            else:
                kp = ks[i + 1]
                x = ab[kp].sqrt() * a0 + (1 - ab[kp]).sqrt() * eps
        return x * self.a_sd + self.a_mu


def train(a):
    eps = load(a.data, squeeze=a.squeeze)
    print(f"{len(eps)} episodes, {sum(len(e[1]) for e in eps):,} steps"
          + (" after dropping waiting steps" if a.squeeze else ""), flush=True)
    rng = np.random.default_rng(0)
    val = set(rng.permutation(len(eps))[:a.n_val].tolist())
    tr = {"o": [], "t": [], "a": []}
    va = {"o": [], "t": [], "a": []}
    for i, (o, ac, _) in enumerate(eps):
        ow, tw, aw = windows(o, ac)
        if a.no_time:
            tw = np.zeros_like(tw)
        d = va if i in val else tr
        d["o"].append(ow), d["t"].append(tw), d["a"].append(aw)
    tr = {k: np.concatenate(v) for k, v in tr.items()}
    va = {k: np.concatenate(v) for k, v in va.items()}
    dev = torch.device("cuda")
    m = DiT(tr["o"].shape[-1]).to(dev)
    m.o_mu.copy_(torch.tensor(tr["o"].reshape(-1, tr["o"].shape[-1]).mean(0)))
    m.o_sd.copy_(torch.tensor(tr["o"].reshape(-1, tr["o"].shape[-1]).std(0) + 1e-3))
    m.a_mu.copy_(torch.tensor(tr["a"].reshape(-1, D_ACT).mean(0)))
    m.a_sd.copy_(torch.tensor(tr["a"].reshape(-1, D_ACT).std(0) + 1e-3))
    O, Tf, A = (torch.tensor(tr[k], device=dev) for k in ("o", "t", "a"))
    An = (A - m.a_mu) / m.a_sd
    ab = torch.cumprod(1 - m.beta, 0)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(O) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    print(f"{len(eps)} episodes, {len(O):,} train windows; {steps} steps", flush=True)
    step = 0
    for ep in range(a.epochs):
        perm = torch.randperm(len(O), device=dev)
        for i in range(0, len(O) - a.batch + 1, a.batch):
            b = perm[i:i + a.batch]
            k = torch.randint(0, K_TRAIN, (len(b),), device=dev)
            noise = torch.randn_like(An[b])
            x = ab[k].sqrt()[:, None, None] * An[b] + (1 - ab[k]).sqrt()[:, None, None] * noise
            loss = ((m(x, k, O[b], Tf[b]) - noise) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
        if ep % 10 == 9 or ep == a.epochs - 1:
            with torch.no_grad():
                sel = torch.randint(0, len(va["o"]), (2048,))
                Ov, Tv, Av = (torch.tensor(va[k][sel.numpy()], device=dev) for k in ("o", "t", "a"))
                pred = m.sample(Ov, Tv)
                err = (pred - Av)[:, 0, :7].abs().mean().item()
                gerr = ((pred[:, :, 7].clamp(-1, 1).round() != Av[:, :, 7].round()).float().mean().item())
            print(f"epoch {ep + 1:3d}: noise loss {loss.item():.4f}  val first-step joint error "
                  f"{np.degrees(err):.2f} deg, gripper cmd wrong {gerr * 100:.1f}%", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(dict(state=m.state_dict(), d_obs=tr["o"].shape[-1], no_time=a.no_time),
               os.path.join(a.out, "dit.pt"))
    print("->", os.path.join(a.out, "dit.pt"))


def rollout(args):
    ckpt, pot, seed, video_dir = args
    import gymnasium as gym
    import sim.microwave_env as M  # noqa: F401
    torch.set_num_threads(2)
    c = torch.load(ckpt, map_location="cpu")
    m = DiT(c["d_obs"])
    m.load_state_dict(c["state"])
    m.eval()
    env = gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", pot_mw=(*pot, np.pi / 2),
                   render_mode="rgb_array" if video_dir else None, sim_backend="cpu", max_episode_steps=T_MAX)
    if video_dir:
        from mani_skill.utils.wrappers.record import RecordEpisode
        env = RecordEpisode(env, video_dir, save_video=True, save_trajectory=False, video_fps=30,
                            max_steps_per_video=T_MAX)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    hist = [obs.cpu().numpy().reshape(-1)] * N_OBS
    door_max, pot_max, info, t = 0.0, 0.0, {}, 0
    torch.manual_seed(seed)
    while t < T_MAX:
        o = hist[-1]
        tf = 0.0 if c.get("no_time") else t / T_MAX
        acts = m.sample(torch.tensor(np.stack(hist[-N_OBS:]))[None].float(), torch.tensor([tf]).float())[0].numpy()
        for k in range(E):
            act = np.r_[o[:7] + acts[k, :7], float(np.clip(np.round(acts[k, 7]), -1, 1))]
            obs, _, term, trunc, info = env.step(act)
            hist.append(obs.cpu().numpy().reshape(-1))
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
    gen = json.load(open(os.path.join(a.data.split(",")[0], "summary.json")))["runs"]
    pots = np.array([r["pot"] for r in gen])
    lo, hi = pots.min(0), pots.max(0)
    rng = np.random.default_rng(a.seed)
    jobs = [(a.ckpt, rng.uniform(lo, hi).round(4).tolist(), 1000 + i,
             os.path.join(a.out, f"video_{i:02d}") if i < a.videos else None) for i in range(a.n)]
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(rollout, jobs))
    keys = ["door_opened", "grasped_lifted", "pot_inside", "door_shut", "success"]
    for i, r in enumerate(res):
        print(f"  {i:2d} pot {r['pot']}: door max {r['door_max_deg']:5.1f}  pot max z {r['pot_max_z']:.3f}  "
              + "  ".join(k for k in keys if r[k]), flush=True)
    summ = {k: float(np.mean([r[k] for r in res])) for k in keys}
    print("rates:", {k: f"{v * 100:.0f}%" for k, v in summ.items()}, f"(n={len(res)})")
    json.dump(dict(rates=summ, episodes=res), open(os.path.join(a.out, "eval.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--data", default=os.path.join(ROOT, "Gitignore-but-important/sim/gen"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--squeeze", action="store_true", help="drop waiting steps (sim/bc.py compress)")
    ap.add_argument("--no-time", action="store_true", help="no elapsed-time input")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
