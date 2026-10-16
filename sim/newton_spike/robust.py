"""Full open-loop replay in N worlds with small perturbations of the pot start pose, to see how close to
the edge the Newton replay success is. World 0 is unperturbed.
python robust.py --worlds 64 --jitter-xy 0.005 --jitter-yaw-deg 3 --substeps 25 --contact-timeconst 0.004
"""
import argparse
import json
import os
import time

import numpy as np
import warp as wp

import newton
import scene as S
from replay import TAKE, OUT, gripper_target, set_initial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--jitter-xy", type=float, default=0.005)
    ap.add_argument("--jitter-yaw-deg", type=float, default=3.0)
    ap.add_argument("--substeps", type=int, default=25)
    ap.add_argument("--contact-timeconst", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="robust")
    ap.add_argument("--njmax", type=int, default=1000)
    ap.add_argument("--nconmax", type=int, default=400)
    args = ap.parse_args()
    n = args.worlds
    f = np.load(TAKE)
    actions, ms_panda, ms_pot = f["actions"], f["panda"], f["pot"]
    model, info = S.build_model(n, dict(finger_mu=1.5, pot_mu=0.65, contact_timeconst=args.contact_timeconst))
    set_initial(model, info, ms_panda[0], ms_pot[0], n)
    rng = np.random.default_rng(args.seed)
    q = model.joint_q.numpy().reshape(n, -1)
    dxy = rng.uniform(-args.jitter_xy, args.jitter_xy, (n, 2))
    dyaw = np.deg2rad(rng.uniform(-args.jitter_yaw_deg, args.jitter_yaw_deg, n))
    dxy[0] = 0
    dyaw[0] = 0
    pq = info["pot_q"]
    q[:, pq:pq + 2] += dxy
    q[:, pq + 3:pq + 7] = np.stack([np.zeros(n), np.zeros(n), np.sin(dyaw / 2), np.cos(dyaw / 2)], 1)
    model.joint_q.assign(q.reshape(-1))
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    solver = newton.solvers.SolverMuJoCo(model, solver="newton", integrator="implicitfast", cone="elliptic",
                                         impratio=10.0, iterations=50, ls_iterations=25, njmax=args.njmax, nconmax=args.nconmax,
                                         use_mujoco_contacts=True)
    dt = 1.0 / (20 * args.substeps)
    T = len(actions)
    nd = control.joint_target_pos.shape[0] // n
    full = np.zeros((T, n, nd), np.float32)
    full[:, :, :7] = actions[:, None, :7]
    full[:, :, 7:9] = gripper_target(actions[:, 7])[:, None, None]
    dev = wp.array(full.reshape(T, -1), dtype=wp.float32)
    wp.copy(control.joint_target_pos, dev[0])

    def simulate():
        a, b = s0, s1
        for _ in range(args.substeps):
            a.clear_forces()
            solver.step(a, b, control, None, dt)
            a, b = b, a
        if args.substeps % 2 == 1:
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(b, name), getattr(a, name))

    with wp.ScopedCapture() as cap:
        simulate()
    g = cap.graph
    nb = info["n_bodies"]
    door_max = np.zeros(n)
    z_max = np.zeros(n)
    t0 = time.time()
    for k in range(T):
        wp.copy(control.joint_target_pos, dev[k])
        wp.capture_launch(g)
        if k % 10 == 0 or k == T - 1:
            jq = s0.joint_q.numpy().reshape(n, -1)
            bq = s0.body_q.numpy().reshape(n, nb, 7)
            door_max = np.maximum(door_max, np.rad2deg(jq[:, info["door_q"]]))
            z_max = np.maximum(z_max, bq[:, info["pot_body"], 2])
    wall = time.time() - t0
    jq = s0.joint_q.numpy().reshape(n, -1)
    bq = s0.body_q.numpy().reshape(n, nb, 7)
    door = np.rad2deg(jq[:, info["door_q"]])
    pmw = S.world_to_mw(bq[:, info["pot_body"], 0:3])
    P = S.PARAMS
    (x0, x1), (y0, y1) = P["cavity_x_range"], P["cavity_y_range"]
    inside = (pmw[:, 0] > x0 + 0.03) & (pmw[:, 0] < x1 - 0.03) & (pmw[:, 1] > y0 + 0.03) & (pmw[:, 1] < y1) \
        & (np.abs(pmw[:, 2] - P["turntable_top_z"]) < 0.02)
    shut = door < 5
    res = dict(args=vars(args), wall_s=wall, n=n, success_rate=float((inside & shut).mean()),
               opened60=float((door_max > 60).mean()), lifted5cm=float((z_max > 0.05 + 0.001).mean()),
               inside=float(inside.mean()), shut=float(shut.mean()), world0_success=bool(inside[0] & shut[0]),
               per_world=[dict(dxy=dxy[i].round(4).tolist(), dyaw_deg=float(np.rad2deg(dyaw[i]).round(2)),
                               door_max=float(door_max[i].round(1)), pot_zmax=float(z_max[i].round(3)),
                               pot_final_mw=pmw[i].round(3).tolist(), door_final=float(door[i].round(1)),
                               success=bool(inside[i] & shut[i])) for i in range(n)])
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_world"}))


if __name__ == "__main__":
    main()
