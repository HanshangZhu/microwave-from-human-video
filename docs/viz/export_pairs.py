"""Per configuration: A (solver only) and B (human-guided) re-solved with phase step logging
(Gitignore-but-important/sim/pairs/{A,B}/run_XXX) -> pairs/run_XXX.json for the explainer page:
tcp / pot / gripper / door per 4 steps, phases with their end steps, the source of every step
(solver / human approach / human carry), the human points commanded (after the solver's clamps)
and the raw warped human paths (before them), front + wrist renders every 12 steps."""
import base64
import json
import os
import sys

import cv2
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import sim.microwave_env as M  # noqa: E402
from sim import human_paths as HP  # noqa: E402

G = "Gitignore-but-important/sim"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
R = lambda a, n=3: np.round(np.asarray(a, float), n).tolist()  # noqa: E731
SUB, IMG = 4, 12


def jpg(img, q=62):
    ok, b = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(b.tobytes()).decode()


def to_mw(p):
    return (np.asarray(p) - M.T_W_MW[:3, 3]) @ M.R_W_MW


gen = json.load(open(f"{G}/gen/summary.json"))["runs"]
genh = json.load(open(f"{G}/genh/summary.json"))["runs"]
index = []
for run in sorted(int(d[4:]) for d in os.listdir(f"{G}/pairs/A") if d.startswith("run_") and not d.endswith(".log")):
    P = {"run": run, "pot": genh[run]["pot"], "place": genh[run]["place"], "take": genh[run]["human"],
         "train": {"A": gen[run]["success"], "B": genh[run]["success"]}}
    for arm in ("A", "B"):
        d = f"{G}/pairs/{arm}/run_{run:03d}"
        res = json.load(open(f"{d}/result.json"))
        z = np.load(f"{d}/demo.npz")
        o, a = z["obs"], z["actions"]
        tcp, pot = to_mw(o[:, 18:21]), to_mw(o[:, 25:28])
        n = len(o)
        # source of each step: 0 solver, 1 human approach, 2 human carry
        src = np.zeros(n, int)
        ph = res["phases"]
        bounds = {p["phase"]: (ph[i - 1]["t"] if i else 0, p["t"]) for i, p in enumerate(ph)}
        ht = res.get("human_target") or {}
        if "approach" in ht and "reach, grasp" in bounds:
            s0, s1 = bounds["reach, grasp"]
            ap = np.array(ht["approach"])
            seg = tcp[s0:s1]
            i0 = int(np.argmin(np.linalg.norm(seg - ap[0], axis=1)))
            i1 = i0 + int(np.argmin(np.linalg.norm(seg[i0:] - ap[-1], axis=1)))
            src[s0 + i0:s0 + i1 + 1] = 1
        if "carry" in ht and "carry (human path)" in bounds:
            s0, s1 = bounds["carry (human path)"]
            cp = np.array(ht["carry"])
            i0 = int(np.argmin(np.linalg.norm(tcp[s0:s1] - cp[0], axis=1)))
            src[s0 + i0:s1] = 2
        # raw human paths for this configuration, before the solver's clamps
        raw = {}
        if arm == "B" and res.get("human"):
            H = HP.load(res["human"])
            c = np.r_[P["pot"], 0]
            if H["approach"] is not None:
                raw["approach"] = R(H["approach"] + np.r_[c[:2], 0])
            if H["carry"] is not None:
                raw["carry"] = R(HP.warp_carry(H["carry"], H["pick"], H["place"], c[:2], P["place"]))
        # error of the executed tcp against the commanded human points
        track = {}
        for k in ("approach", "carry"):
            if k in ht:
                T = np.array(ht[k])
                m = src == (1 if k == "approach" else 2)
                if m.any():
                    track[k] = float(np.median([np.min(np.linalg.norm(tcp[m] - p, axis=1)) for p in T]))
        v = np.load(f"{d}/views_hq.npz")
        idx = v["idx"]
        frames = {int(i): [jpg(v["front"][j]), jpg(v["wrist"][j], 60)] for j, i in enumerate(idx) if i % IMG == 0}
        # jerk (third difference / dt^3, dt = 0.05 s) of the tcp and of the joint-target labels,
        # the max over each SUB-step block so single spikes survive the subsampling
        jt = np.r_[np.zeros(3), np.linalg.norm(np.diff(tcp, 3, axis=0), axis=1) / 0.05 ** 3]
        ja = np.r_[np.zeros(3), np.linalg.norm(np.diff(a[:, :7], 3, axis=0), axis=1) / 0.05 ** 3]
        blk = lambda x: [float(x[i:i + SUB].max()) for i in range(0, n, SUB)]  # noqa: E731
        grip = a[:, 7]
        carry = (grip == -1) & (np.linalg.norm(tcp - pot, axis=1) < 0.15)
        L = float(np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1)[carry[1:] & carry[:-1]]))
        P[arm] = dict(success=res["success"], n=n, tcp=R(tcp[::SUB]), pot=R(pot[::SUB]), grip=R(grip[::SUB], 0),
                      door=R(np.rad2deg(o[::SUB, 32]), 1), src=src[::SUB].tolist(),
                      phases=[dict(name=p["phase"], t=p["t"], ok=p["solved"]) for p in ph],
                      jerk_tcp=R(blk(jt), 2), jerk_act=R(blk(ja), 1),
                      jerk_med=dict(human_tcp=float(np.median(jt[src > 0])) if (src > 0).any() else None,
                                    solver_tcp=float(np.median(jt[src == 0])),
                                    human_act=float(np.median(ja[src > 0])) if (src > 0).any() else None,
                                    solver_act=float(np.median(ja[src == 0]))),
                      parts=res.get("human_parts"), target=ht, raw=raw, track=track, frames=frames,
                      stats=dict(carry_m=L, carry_zmax=float(tcp[carry, 2].max()) if carry.any() else None,
                                 steps=n, human_steps=int((src > 0).sum())))
    f = f"{OUT}/run_{run:03d}.json"
    json.dump(P, open(f, "w"), separators=(",", ":"))
    index.append(dict(run=run, take=P["take"], parts=P["B"]["parts"], A=P["A"]["success"], B=P["B"]["success"],
                      train=P["train"]))
    print(run, P["take"], P["B"]["parts"], "A", P["A"]["success"], "B", P["B"]["success"], "train", P["train"],
          "track", {k: round(x * 100, 1) for k, x in P["B"]["track"].items()}, os.path.getsize(f) // 1024, "KB")
json.dump(index, open(f"{OUT}/index.json", "w"))
