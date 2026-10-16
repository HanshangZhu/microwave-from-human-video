"""Which hand reaches which key pose from the chosen stance (g1_solve.STANCE)?

Key poses as in reach.py (tcp on the handle bar at door 0..90 deg / on the pot at the takes'
picks and places, finger axis vertical), each tried with each hand, torso joint free and
fixed at 0. Prints a table.

  python sim/g1_spike/reach_table.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import sim.g1_spike.g1_env as G  # noqa: E402
import sim.g1_spike.g1_solve as GS  # noqa: E402
from sim.g1_spike.reach import make_kin, key_poses, reach  # noqa: E402
from sim.g1_spike.g1_arm import G1Kin  # noqa: E402


def main():
    x, y, z, yaw = GS.STANCE
    G.set_g1_stance((x, y), z, yaw)
    env, r, body = make_kin()
    kin = G1Kin(r, G.root_pose())
    lim = r.get_qlimits()
    rng = np.random.default_rng(0)
    seeds = [G.INIT_QPOS.copy()]
    for _ in range(15):
        s = G.INIT_QPOS.copy()
        s[:11] = rng.uniform(lim[:11, 0], lim[:11, 1]) * 0.8
        seeds.append(s)
    print(f"stance {GS.STANCE}: pelvis (x, y) in mw, height above counter, yaw")
    print(f"{'pose':8s} {'left+torso':>11s} {'left':>6s} {'right+torso':>12s} {'right':>6s}")
    for name, (_, p, axes) in key_poses().items():
        row = []
        for side in ("left", "right"):
            for torso in (True, False):
                q, pe, ae = reach(kin, side, p, axes, (x, y), seeds, torso=torso)
                row.append("yes" if (pe < 0.005 and ae < 5) else f"{pe * 1000:.0f}mm")
        print(f"{name:8s} {row[0]:>11s} {row[1]:>6s} {row[2]:>12s} {row[3]:>6s}")


if __name__ == "__main__":
    main()
