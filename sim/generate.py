#!/usr/bin/env python
"""Generate robot demonstrations for MicrowavePot-v0 from the human takes.

The three takes (recon/DEMOS.md) give where the pot was picked up and where it was put
down. New configurations are sampled in the region they span, widened by --margin:
the pot's start anywhere in that pick region, the place target anywhere in that place
region. Each configuration is solved and executed in physics by solve_microwave.py;
only runs the environment judges successful (pot on the turntable, door shut) are kept.

  python sim/generate.py --n 60 --out Gitignore-but-important/sim/gen [--jobs 10]

Writes <out>/run_XXX/ per configuration and <out>/summary.json (yield, failures).
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAKES = ["take_000", "take_001", "take_002"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--videos", type=int, default=3, help="record video for the first few runs")
    ap.add_argument("--dart-noise", type=float, default=0.0)
    ap.add_argument("--human", default="", help="comma-separated takes whose hand paths guide the solver "
                    "(sim/human_paths.py), used in turn by run index; empty: solver only")
    ap.add_argument("--human-smooth", action="store_true", help="--human: smoothing-spline the hand paths")
    ap.add_argument("--human-carry-only", action="store_true", help="--human: the hand's carry only, the solver's approach")
    ap.add_argument("--carry-z", type=float, default=None, help="tcp height of the solver's own carry")
    ap.add_argument("--python", default="python", help="interpreter for the solver runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tk = [json.load(open(os.path.join(ROOT, "Gitignore-but-important/recon/takes", t, "pot.json"))) for t in TAKES]
    pick = np.array([[t["pick"]["x"], t["pick"]["y"]] for t in tk])
    place = np.array([[t["end"]["x"], t["end"]["y"]] for t in tk])
    lo_p, hi_p = pick.min(0) - a.margin, pick.max(0) + a.margin
    lo_q, hi_q = place.min(0) - a.margin, place.max(0) + a.margin
    rng = np.random.default_rng(a.seed)
    cfgs = [dict(run=i, pot=rng.uniform(lo_p, hi_p).round(4).tolist(), place=rng.uniform(lo_q, hi_q).round(4).tolist())
            for i in range(a.n)]
    print(f"pick region x {lo_p[0]:.3f}..{hi_p[0]:.3f}, y {lo_p[1]:.3f}..{hi_p[1]:.3f}; "
          f"place region x {lo_q[0]:.3f}..{hi_q[0]:.3f}, y {lo_q[1]:.3f}..{hi_q[1]:.3f}", flush=True)

    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2")

    def run(c):
        d = os.path.join(a.out, f"run_{c['run']:03d}")
        cmd = [a.python, os.path.join(ROOT, "sim/solve_microwave.py"), "--out", d,
               "--pot", *map(str, c["pot"]), "--place", *map(str, c["place"])]
        if c["run"] >= a.videos:
            cmd.append("--no-video")
        if a.carry_z is not None:
            cmd += ["--carry-z", str(a.carry_z)]
        if a.dart_noise > 0:
            cmd += ["--dart-noise", str(a.dart_noise)]
        if a.human:
            takes = a.human.split(",")
            cmd += ["--human", takes[c["run"] % len(takes)]] + (["--human-smooth"] if a.human_smooth else []) \
                + (["--human-carry-only"] if a.human_carry_only else [])
        with open(d + ".log", "w") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            r = json.load(open(os.path.join(d, "result.json")))
        except Exception:
            r = dict(success=False, crashed=True, phases=[])
        fail = next((p["phase"] for p in r.get("phases", []) if not p["solved"]), None)
        out = dict(c, success=r["success"], pot_inside=r.get("pot_inside"), door_deg=r.get("door_deg"),
                   first_failed_phase=fail, crashed=r.get("crashed", False),
                   human=r.get("human"), human_parts=r.get("human_parts"))
        print(f"run {c['run']:3d}: {'OK ' if out['success'] else 'FAIL'} pot {c['pot']} place {c['place']}"
              + ("" if out["success"] else f"  ({fail or 'pot/door check'}; door {out['door_deg']})"), flush=True)
        return out

    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run, cfgs))
    ok = [r for r in res if r["success"]]
    json.dump(dict(n=len(res), success=len(ok), yield_=len(ok) / len(res), runs=res),
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"{len(ok)}/{len(res)} successful ({len(ok) / len(res) * 100:.0f}%) -> {a.out}")


if __name__ == "__main__":
    main()
