"""Where does the premature close come from? Offline probes of trained camera policies.

Several camera policies (A-low, A2m, Bc0, ...) fail by closing the gripper ~2-3 cm before
the grasp depth, then re-closing until the time limit. Two questions:

  openloop  Is the early close already in the learned function on states like the training
            data (the approach towards the pot, rendered as in training)? Each policy is
            queried on the same recorded approach windows (the last ~3 s before the solver's
            close, every rendered step), several noise draws per query; reported per bin of
            steps-to-the-true-close: P(a close among the 8 executed actions) and the
            tcp - pot offset (mw frame) at that step. Early closes here -> the policy has
            learned them; none -> they arise only after closed-loop drift.

  nn        Which training samples look most like the moments a policy closed early in
            closed loop? Failed evaluation episodes are re-run with every query's inputs
            kept; for each close command within 15 cm of the pot, the query that produced it
            is embedded with the policy's own conditioning (mean of its prop + image tokens)
            and matched to its nearest training samples, each labelled by phase relative to
            that demo's close: approach (> 16 steps before), boundary (the 16-action chunk
            contains the close), carry (after the close).

  python sim/probe_close.py openloop --ckpts vis_low,vis_a2m,... [--n-draw 8]
  python sim/probe_close.py nn --ckpt vis_low --data genlow,genlow_dart02,genlow_dart04 [--episodes 12]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))
import dit_vis as DV  # noqa: E402
from dit import N_OBS, H, E  # noqa: E402

G = os.path.join(ROOT, "Gitignore-but-important/sim")
BINS = [-64, -48, -32, -24, -16, -8, 1, 9]          # steps from the query to the true close


def run_obs(rd):
    if os.path.exists(os.path.join(rd, "demo.npz")):
        z = np.load(os.path.join(rd, "demo.npz"))
        return z["obs"].astype(np.float32), z["actions"].astype(np.float32)
    import h5py
    with h5py.File(os.path.join(rd, "solve.h5"), "r") as f:
        return f["traj_0/obs"][:-1].astype(np.float32), f["traj_0/actions"][:].astype(np.float32)


def successes(d, views="views_hq"):
    s = json.load(open(os.path.join(d, "summary.json")))
    return [r["run"] for r in s["runs"] if r["success"]
            and os.path.exists(os.path.join(d, f"run_{r['run']:03d}", views + ".npz"))]


def windows(rd, views="views_hq", lo=-64, hi=8, prop="robot"):
    """Inputs exactly as dit_vis._load_one builds them, at the rendered steps within [lo, hi]
    of the demo's close on the pot, plus the step offset and tcp - pot (mw frame, cm)."""
    import sim.microwave_env as M
    o, ac = run_obs(rd)
    keep = DV.idle_mask(o, ac)
    v = np.load(os.path.join(rd, views + ".npz"))
    g = ac[:, 7]
    flips = np.where((g[1:] < 0) & (g[:-1] >= 0))[0] + 1
    near = np.linalg.norm(o[flips, 18:21] - o[flips, 25:28], axis=1) < 0.15
    tc = int(flips[near][0])                             # the pot grasp (the first close is the handle)
    k = v["idx"] - tc
    sel = keep[v["idx"]] & (k >= lo) & (k <= hi)
    idx = v["idx"][sel]
    oc = o[keep]
    ci = (np.cumsum(keep) - 1)[idx]
    oi = np.maximum(ci[:, None] - np.arange(N_OBS)[None, ::-1], 0)
    rel = (o[idx, 18:21] - o[idx, 25:28]) @ np.asarray(M.R_W_MW) * 100
    return DV.featurize(oc[oi], prop), v["front"][sel], v["wrist"][sel], idx - tc, rel


@torch.no_grad()
def close_prob(m, P, F, W, dev, n_draw=8, bs=64):
    """P(any close among the first E predicted actions) per sample, over n_draw noise draws."""
    out = np.zeros(len(P))
    for i in range(0, len(P), bs):
        p = torch.tensor(P[i:i + bs], device=dev)
        f = torch.tensor(F[i:i + bs], device=dev)
        w = torch.tensor(W[i:i + bs], device=dev)
        hits = 0
        for _ in range(n_draw):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not m.fp32):
                a = m.sample(p, f, w).float()
            hits = hits + (a[:, :E, 7].clamp(-1, 1).round() < 0).any(1).float()
        out[i:i + bs] = (hits / n_draw).cpu().numpy()
    return out


def openloop(a):
    dev = torch.device("cuda")
    sets = {}
    for name, spec in (("heldout", a.heldout), ("low_train", a.low_train), ("a2_train", a.a2_train)):
        P, F, W, K, R = [], [], [], [], []
        for item in spec.split(","):
            d, sl = item.split(":")
            lo_, hi_ = (int(x) for x in sl.split("-"))
            for r in successes(os.path.join(G, d))[lo_:hi_]:
                p, f, w, k, rel = windows(os.path.join(G, d, f"run_{r:03d}"))
                P.append(p); F.append(f); W.append(w); K.append(k); R.append(rel)
        sets[name] = [np.concatenate(x) for x in (P, F, W, K, R)]
        print(f"probe set {name}: {len(sets[name][0])} queries", flush=True)
    res = {}
    for ck in a.ckpts.split(","):
        m, _ = DV.load_model(os.path.join(G, ck, "dit_vis.pt"), dev)
        torch.manual_seed(0)
        res[ck] = {}
        for name, (P, F, W, K, R) in sets.items():
            pc = close_prob(m, P, F, W, dev, a.n_draw)
            rows = []
            for b0, b1 in zip(BINS[:-1], BINS[1:]):
                s = (K >= b0) & (K < b1)
                rows.append(dict(bin=[b0, b1], n=int(s.sum()), p_close=round(float(pc[s].mean()), 3),
                                 rel_y_cm=round(float(R[s, 1].mean()), 2), rel_z_cm=round(float(R[s, 2].mean()), 2)))
            res[ck][name] = rows
        del m
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    names = list(sets)
    print("\nP(close among the 8 executed actions) by steps to the true close; tcp - pot y (cm) from the "
          f"first probe set\n{'bin':>12} {'y cm':>6} " + " ".join(f"{ck + '/' + n[:6]:>16}" for ck in res for n in names))
    for j, (b0, b1) in enumerate(zip(BINS[:-1], BINS[1:])):
        y = res[next(iter(res))][names[0]][j]["rel_y_cm"]
        print(f"{f'[{b0},{b1})':>12} {y:6.2f} " + " ".join(f"{res[ck][n][j]['p_close']:16.2f}" for ck in res for n in names))


# ---------------------------------------------------------------- nn: closed-loop early closes
_Q = []                                                  # (prop, front, wrist) per query of the current episode
_orig_sample = DV.VisDiT.sample


def _stash_sample(self, prop, front, wrist, x_init=None):
    _Q.append((prop.cpu().numpy(), front.cpu().numpy(), wrist.cpu().numpy()))
    return _orig_sample(self, prop, front, wrist, x_init)


def _episode(job):
    """One evaluation episode with every query's inputs kept; returns the result and, for
    each close command near the pot, the inputs of the query that produced it."""
    DV.VisDiT.sample = _stash_sample
    _Q.clear()
    res = DV.rollout_retry(job)
    ev = []
    for c in res["closes"]:
        q = (c["t"] - 1) // E                            # chunk mode: a query every E steps from t = 0
        if q < len(_Q):
            ev.append(dict(close=c, inputs=_Q[q], offset=(c["t"] - 1) % E))
    return res, ev


@torch.no_grad()
def embed(m, P, F, W, dev, bs=128):
    out = []
    for i in range(0, len(P), bs):
        c = m.cond(torch.tensor(P[i:i + bs], device=dev), torch.tensor(F[i:i + bs], device=dev),
                   torch.tensor(W[i:i + bs], device=dev))
        out.append(torch.nn.functional.normalize(c.float().mean(1), dim=1).cpu())
    return torch.cat(out)


def nn(a):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ck = os.path.join(G, a.ckpt, "dit_vis.pt")
    ev_json = json.load(open(os.path.join(G, a.ckpt, "eval128", "eval.json")))["episodes"]
    fails = [e for e in ev_json if not e["grasped_lifted"]][:a.episodes]
    jobs = [(ck, e["pot"], e["seed"], None, "chunk", None, E) for e in fails]
    events, results = [], []
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for res, ev in ex.map(_episode, jobs):
            results.append(res)
            events += [dict(e, seed=res["seed"]) for e in ev]
            print(f"  seed {res['seed']}: lifted {res['grasped_lifted']}, {len(ev)} closes near the pot", flush=True)
    print(f"{len(events)} close events from {len(results)} re-run episodes "
          f"({sum(not r['grasped_lifted'] for r in results)} again never lifted)", flush=True)
    dev = torch.device("cuda")
    m, _ = DV.load_model(ck, dev)
    # training samples with phase labels
    P, F, W, K, RUN = [], [], [], [], []
    for d in a.data.split(","):
        d, cap = (d.split(":")[0], int(d.split(":")[1])) if ":" in d else (d, None)
        for r in successes(os.path.join(G, d))[:cap]:
            p, f, w, k, _ = windows(os.path.join(G, d, f"run_{r:03d}"), lo=-10 ** 6, hi=10 ** 6)
            P.append(p); F.append(f); W.append(w); K.append(k); RUN += [f"{d}/{r}"] * len(k)
    P, F, W, K = (np.concatenate(x) for x in (P, F, W, K))
    print(f"{len(K):,} training samples embedded", flush=True)
    Z = embed(m, P, F, W, dev)
    del F, W
    phase = np.where(K < -H, "approach", np.where(K <= 0, "boundary", "carry"))
    rows = []
    for e in events:
        p, f, w = e["inputs"]
        z = embed(m, p, f, w, dev)
        sim = (Z @ z[0]).numpy()
        top = np.argsort(-sim)[:a.k]
        lab, cnt = np.unique(phase[top], return_counts=True)
        rows.append(dict(seed=e["seed"], t=e["close"]["t"], rel_cm=e["close"]["rel_cm"],
                         early=e["close"]["rel_cm"][1] < -2.0, nn_phase=dict(zip(lab.tolist(), cnt.tolist())),
                         nn_k_median=int(np.median(K[top])), nn_sim=round(float(sim[top].mean()), 3)))
    json.dump(dict(results=results, events=rows), open(a.out, "w"), indent=1)
    for grp in (True, False):
        g = [r for r in rows if r["early"] == grp]
        if not g:
            continue
        tot = {}
        for r in g:
            for kk, v in r["nn_phase"].items():
                tot[kk] = tot.get(kk, 0) + v
        n = sum(tot.values())
        print(f"{'early (y < -2 cm)' if grp else 'at depth (y >= -2 cm)'}: {len(g)} closes; nearest "
              f"{a.k} training samples by phase: " + ", ".join(f"{kk} {v / n:.0%}" for kk, v in sorted(tot.items()))
              + f"; median steps to their demo's close {np.median([r['nn_k_median'] for r in g]):.0f}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("openloop")
    o.add_argument("--ckpts", default="vis_low,vis_a2m,vis_a2c,vis_a2,vis_c0")
    # held out from A-low, A2m and A2c (A2 trained on them): gena2_dart02 / _dart04 successes past their caps
    o.add_argument("--heldout", default="gena2_dart02:21-29,gena2_dart04:3-8")
    o.add_argument("--low-train", default="genlow:0-10")
    o.add_argument("--a2-train", default="gena2:0-10")
    o.add_argument("--n-draw", type=int, default=8)
    o.add_argument("--out", default=os.path.join(G, "probe_close/openloop.json"))
    n = sp.add_parser("nn")
    n.add_argument("--ckpt", default="vis_low")
    n.add_argument("--data", default="genlow,genlow_dart02,genlow_dart04")
    n.add_argument("--episodes", type=int, default=12)
    n.add_argument("--jobs", type=int, default=4)
    n.add_argument("--k", type=int, default=20)
    n.add_argument("--out", default=os.path.join(G, "probe_close/nn_vis_low.json"))
    a = ap.parse_args()
    openloop(a) if a.cmd == "openloop" else nn(a)


if __name__ == "__main__":
    main()
