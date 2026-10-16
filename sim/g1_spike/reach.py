"""Reachability scan: where can the G1 stand so its hands reach the task's key poses?

For each candidate stance (pelvis x, y in the microwave frame, height above the counter,
yaw) and each key pose, task IK (sim/g1_spike/g1_arm.py) from several seeds; a pose counts
as reached when the hand point is within 5 mm and the axis constraints within 5 deg.
Key poses (all from recon/DEMOS.md / parameters.json):
  door  (left hand): the handle bar (z = 0.17) at door angles 0, 30, 60, 75, 90 deg,
                     bar along the palm's z axis (the fingers curl round it)
  pick  (right hand): the pot at the three takes' pick positions, hand at z = 0.035,
                      palm z vertical
  place (right hand): the three takes' place positions on the turntable, same hand
Also a geometric check that the torso / shoulders stay out of the door's swept sector.

  python sim/g1_spike/reach.py [--quick]
"""
import argparse
import json
import os
import sys

import numpy as np
import sapien

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import sim.microwave_env as M  # noqa: E402
import sim.g1_spike.g1_env as G  # noqa: E402
from sim.g1_spike.g1_arm import G1Kin  # noqa: E402

PICKS = [(0.480, -0.241), (0.441, -0.279), (0.504, -0.173)]
PLACES = [(0.183, 0.095), (0.172, 0.079), (0.173, 0.111)]
HANDLE = np.array([0.343, -0.062, 0.17])
POT_Z, PLACE_Z = 0.035, 0.041 + 0.035


def door_rot(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def key_poses():
    """name -> (side, target position in mw, [(hand axis, mw direction, weight)])"""
    up = [((0, 0, 1), (0, 0, 1), 1.0)]
    k = {}
    for d in (0, 30, 60, 75, 90):
        k[f"door{d}"] = ("left", door_rot(np.deg2rad(d)) @ HANDLE, up)
    for i, (x, y) in enumerate(PICKS):
        k[f"pick{i}"] = ("right", np.array([x, y, POT_Z]), up)
    for i, (x, y) in enumerate(PLACES):
        k[f"place{i}"] = ("right", np.array([x, y, PLACE_Z]), up)
    return k


def make_kin():
    """A G1 articulation in an empty scene, for its pinocchio model and link geometry."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    env = gym.make("MicrowavePotG1-v0", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                   render_backend="none")
    env.reset(seed=0)
    r = env.unwrapped.agent.robot._objs[0]
    # torso + shoulder collision vertices in the root frame (for the door-sweep check)
    pts = []
    Tr = r.links[0].entity_pose.inv().to_transformation_matrix()
    for l in r.links:
        if not any(l.name.endswith(s) for s in ("torso_link", "shoulder_roll_link", "shoulder_yaw_link")):
            continue
        for s in l.collision_shapes:
            T = Tr @ l.entity_pose.to_transformation_matrix() @ s.local_pose.to_transformation_matrix()
            v = s.vertices * s.scale if hasattr(s, "vertices") else np.zeros((1, 3))
            pts.append((T[:3, :3] @ v.T).T + T[:3, 3])
    return env, r, np.concatenate(pts)


def stance_root(xy, z, yaw):
    """Root pose in WORLD for a stance (M's world: mw placed so ROBOT_IN_MW is the origin)."""
    return sapien.Pose(p=[0, 0, z], q=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def body_clear(body_root, xy, z, yaw, q_max=np.deg2rad(90)):
    """Torso/shoulder points (root frame) vs the door's swept sector and the microwave body."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    w = body_root @ Rz.T + [0, 0, z]                      # world (root at origin)
    mw = M.world_to_mw(w) if False else (w @ M.R_W_MW) + np.r_[xy, 0.0]  # world -> mw for this stance
    mw = mw[mw[:, 2] > 0.0]                                # above the counter
    r = np.hypot(mw[:, 0], mw[:, 1])
    ang = np.arctan2(-mw[:, 1], mw[:, 0])                 # door opens towards -y_mw: angle grows
    in_sweep = (r < 0.385) & (ang > -0.1) & (ang < q_max + 0.1) & (mw[:, 2] < 0.30)
    in_body = (mw[:, 0] > -0.02) & (mw[:, 0] < 0.48) & (mw[:, 1] > -0.07) & (mw[:, 2] < 0.29)
    return int(in_sweep.sum()), int(in_body.sum())


def reach(kin, side, p_mw, axes_mw, xy, seeds, torso=True):
    joints = ([0] if torso else []) + (G.LEFT_ARM if side == "left" else G.RIGHT_ARM)
    link = kin.link(f"{side}_tcp_link")
    p = (np.asarray(p_mw) - np.r_[xy, 0.0]) @ M.R_W_MW.T            # mw -> world for this stance
    axes = [(a, M.R_W_MW @ np.asarray(d, float), w) for a, d, w in axes_mw]
    best = None
    for s in seeds:
        q, pe, ae = kin.solve(s, joints, link, p=p, axes=axes, w_reg=0.002)
        if best is None or pe + 0.001 * ae < best[1] + 0.001 * best[2]:
            best = (q, pe, ae)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "Gitignore-but-important/g1_spike/reach.json"))
    a = ap.parse_args()
    env, robot, body = make_kin()
    rng = np.random.default_rng(0)
    lim = robot.get_qlimits()
    seeds = [G.INIT_QPOS.copy()] + [np.where(np.arange(25) < 11, rng.uniform(lim[:, 0], lim[:, 1]) * 0.6, 0)
                                    for _ in range(5)]
    keys = key_poses()
    xs = np.arange(0.20, 0.71, 0.05)
    ys = np.arange(-0.60, -0.14, 0.05)
    zs = [-0.10, -0.05, 0.0, 0.05, 0.10]
    yaws = [0.0, 0.4, 0.8]
    if a.quick:
        xs, ys, zs, yaws = xs[::2], ys[::2], zs[::2], yaws
    rows = []
    for z in zs:
        for yaw in yaws:
            kin = G1Kin(robot, stance_root(None, z, yaw))
            for x in xs:
                for y in ys:
                    sweep, inb = body_clear(body, (x, y), z, yaw)
                    if inb:
                        continue
                    res = {}
                    for name, (side, p, axes) in keys.items():
                        q, pe, ae = reach(kin, side, p, axes, (x, y), seeds)
                        res[name] = bool(pe < 0.005 and ae < 5)
                    n = sum(res.values())
                    rows.append(dict(x=round(x, 3), y=round(y, 3), z=z, yaw=yaw, door_sweep_hits=sweep,
                                     n=n, **res))
                    print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=0)
    good = sorted(rows, key=lambda r: (-(r["door_sweep_hits"] == 0), -r["n"]))[:15]
    print("BEST")
    for r in good:
        print(r)


if __name__ == "__main__":
    main()
