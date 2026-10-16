"""Rough throughput of the MicrowavePot scene in Newton/MuJoCo-Warp with N replicated worlds.
All worlds replay the same take_000 actions (from step 0) for --steps control steps (20 Hz).
python throughput.py --worlds 1 64 256 1024 --substeps 25 --contact-timeconst 0.004 --steps 300
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


def run(n, args, actions, ms_panda, ms_pot):
    cfg = dict(finger_mu=1.5, pot_mu=0.65, contact_timeconst=args.contact_timeconst)
    model, info = S.build_model(n, cfg)
    set_initial(model, info, ms_panda[0], ms_pot[0], n)
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    solver = newton.solvers.SolverMuJoCo(model, solver="newton", integrator="implicitfast", cone="elliptic",
                                         impratio=10.0, iterations=50, ls_iterations=25,
                                         njmax=args.njmax, nconmax=args.nconmax, use_mujoco_contacts=True)
    dt = 1.0 / (20 * args.substeps)
    tgt = np.zeros((n, control.joint_target_pos.shape[0] // n), np.float32)
    tgt[:, 0:9] = ms_panda[0, 13:22]
    control.joint_target_pos.assign(tgt.reshape(-1))
    st = [s0, s1]

    def simulate():
        a, b = st
        for _ in range(args.substeps):
            a.clear_forces()
            solver.step(a, b, control, None, dt)
            a, b = b, a
        if args.substeps % 2 == 1:
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(b, name), getattr(a, name))

    with wp.ScopedCapture() as cap:
        simulate()
    graph = cap.graph
    # pre-stage all targets on device so the loop has no host->device copies besides one small copy
    all_t = np.repeat(np.concatenate([actions[:args.steps, :7],
                                      np.repeat(gripper_target(actions[:args.steps, 7:8]), 2, 1)], 1)[:, None, :],
                      n, 1)
    full = np.zeros((args.steps, n, tgt.shape[1]), np.float32)
    full[:, :, :9] = all_t
    dev = wp.array(full.reshape(args.steps, -1), dtype=wp.float32)
    wp.synchronize()
    t0 = time.time()
    for k in range(args.steps):
        wp.copy(control.joint_target_pos, dev[k])
        wp.capture_launch(graph)
    wp.synchronize()
    wall = time.time() - t0
    jq = s0.joint_q.numpy().reshape(n, -1)
    door = np.rad2deg(jq[:, info["door_q"]])
    return dict(worlds=n, steps=args.steps, substeps=args.substeps, wall_s=wall,
                ctrl_steps_per_s_per_world=args.steps / wall, env_steps_per_s=n * args.steps / wall,
                physics_steps_per_s=n * args.steps * args.substeps / wall,
                realtime_factor_total=n * args.steps * 0.05 / wall,
                door_deg_min_max=[float(door.min()), float(door.max())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, nargs="+", default=[1, 64, 256, 1024])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--substeps", type=int, default=25)
    ap.add_argument("--contact-timeconst", type=float, default=0.004)
    ap.add_argument("--njmax", type=int, default=400)
    ap.add_argument("--nconmax", type=int, default=200)
    ap.add_argument("--tag", default="throughput")
    args = ap.parse_args()
    f = np.load(TAKE)
    res = []
    for n in args.worlds:
        r = run(n, args, f["actions"], f["panda"], f["pot"])
        print(json.dumps(r), flush=True)
        res.append(r)
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
