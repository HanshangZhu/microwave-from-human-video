"""Second, task-specific stance scan (after reach.py): with the actual hand tasks of the
solver -- the left-hand hook on the bar at door 0 and 60 deg, the right-hand pot grasp
(POT_OFF, fingers vertical, forearm yawed PHI deg from +x_mw) at the takes' pick and place
positions -- IK with the torso joint free; reports the torso angle each needs. A stance
where all are reachable with a small torso swing is preferred (the torso turns both arms).

  python sim/g1_spike/stance_scan.py
"""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import sim.microwave_env as M  # noqa: E402
import sim.g1_spike.g1_env as G  # noqa: E402
import sim.g1_spike.g1_solve as GS  # noqa: E402
from sim.g1_spike.reach import make_kin, body_clear, PICKS, PLACES  # noqa: E402
from sim.g1_spike.g1_arm import G1Kin  # noqa: E402


def main(phi_deg=0.0):
    """phi_deg = 0: the forearm's yaw is free (the pot turns with the hand during the carry)."""
    env, r, body = make_kin()
    lim = r.get_qlimits()
    rng = np.random.default_rng(0)
    seeds = [G.INIT_QPOS.copy()]
    for _ in range(12):
        s = G.INIT_QPOS.copy()
        s[:11] = rng.uniform(lim[:11, 0], lim[:11, 1]) * 0.8
        seeds.append(s)
    phi = np.deg2rad(phi_deg)
    rows = []
    for x, y, z, yaw in itertools.product([0.30, 0.35, 0.40, 0.45], [-0.45, -0.40, -0.35],
                                          ZS, [0.0, 0.3, 0.6]):
        sweep, inb = body_clear(body, (x, y), z, yaw, q_max=np.deg2rad(65))
        if sweep or inb:
            continue
        # torso front vs the pot at the takes' picks: pelvis-frame torso front 0.085 m
        G.set_g1_stance((x, y), z, yaw)
        kin = G1Kin(r, G.root_pose())
        res = {}
        S = GS.Solver.__new__(GS.Solver)
        tasks = {"hook0": ("left", S.hook_task(S.bar(0.0), 0.0)),
                 "hook60": ("left", S.hook_task(S.bar(np.deg2rad(60)), np.deg2rad(60)))}
        fdir = [np.cos(phi), np.sin(phi), 0.0]
        for i, (px, py) in enumerate(PICKS[:1] + PLACES[:1]):
            pz = 0.0 if i == 0 else DER_TT
            off, up = GS.pot_grasp(np.deg2rad(BETA))
            tasks["pick" if i == 0 else "place"] = ("right", dict(
                p=M.mw_to_world([px, py, pz]), offset=off,
                axes=[(up, GS.W([0, 0, 1]), 1.0)] + ([((1, 0, 0), GS.W(fdir), 1.0)] if phi_deg else [])))
        for name, (side, t) in tasks.items():
            js = [0] + (G.LEFT_ARM if side == "left" else G.RIGHT_ARM)
            link = kin.link(f"{side}_palm_link")
            best = None
            for s in seeds:
                q, pe, ae = kin.solve(s, js, link, w_reg=0.002, q_ref=np.zeros(len(js)), **t)
                if pe < 0.004 and ae < 4 and abs(q[js[1]]) < MAX_SH_PITCH and (best is None or abs(q[0]) < abs(best[0])):
                    T = kin.fk(q, link)
                    f = M.R_W_MW.T @ T[:3, 0]
                    best = (q[0], np.degrees(np.arctan2(f[1], f[0])))
            res[name] = None if best is None else round(float(np.degrees(best[0])))
            if best is not None and side == "right":
                res[name + "_phi"] = round(float(best[1]))
        ok = all(v is not None for v in res.values())
        tv = [v for k, v in res.items() if not k.endswith("_phi")]
        swing = (max(tv) - min(tv)) if ok else 999
        rows.append((ok, swing, (x, y, z, yaw), res))
        print(rows[-1], flush=True)
    print("BEST")
    for r_ in sorted(rows, key=lambda r_: (not r_[0], r_[1]))[:10]:
        print(r_)


DER_TT = GS.DER["turntable_top_z"]
BETA = float(os.environ.get("BETA", 0))            # deg, the pot grasp's forearm pitch
ZS = [float(v) for v in os.environ.get("ZS", "-0.15,-0.1").split(",")]
MAX_SH_PITCH = float(os.environ.get("MAX_SH_PITCH", 9))   # rad: reject over-the-head arm solutions
GS.BAR_Z = float(os.environ.get("BAR_Z", GS.BAR_Z))

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
