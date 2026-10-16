#!/usr/bin/env python
"""DAgger-style data: the solver (privileged expert) takes over from states the student
policy actually visits, and its continuation is recorded as new demonstrations.

The student is a policy that sees only what it will see at deployment (now the state DiT,
later a camera policy); the expert is sim/solve_microwave.py's Solver, which reads the
simulator's true poses. The solver is a phase script, not a function of the state, so
"label this state" is done by handing the episode over: run the student from a start
state up to a branch time, give the arm to the solver there, let it finish the task,
keep the (observation, planned action) pairs from the hand-over on if it succeeds.
That is intervention-style DAgger: the first states of every new demonstration are the
student's own, the rest are the expert's.

Branch times are taken where the student fails most -- between turning towards the pot
(after the door is open) and the lift: K points in that window, three of them in its last
15 % where the grasp is decided, the last one just before the lift if the student grasped
it, or just after the pot was first shoved if the student pushed it.

At the hand-over the expert's options are tried most local first, each from a fresh
deterministic replay of the student to the same state; the first that succeeds is kept
(see hand_over): hold (lift what the hand has closed on), continue (straight in from
here), regrasp (open, back off 8 cm, in again), restart (open, tuck, the whole pick).
A pot that was knocked over is skipped.

  python archive/state-policy/dagger.py --ckpt <student dit.pt> --out <dir> [--n 96] [--branches 5] [--jobs 24]

Writes <out>/run_XXX/demo.npz and <out>/summary.json in sim/generate.py's layout, so
sim/dit.py train --data ...,<out> reads it directly.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAKES = ["take_000", "take_001", "take_002"]


def regions(margin=0.02):
    """Pick and place regions spanned by the human takes (as sim/generate.py)."""
    tk = [json.load(open(os.path.join(ROOT, "Gitignore-but-important/recon/takes", t, "pot.json"))) for t in TAKES]
    pick = np.array([[t["pick"]["x"], t["pick"]["y"]] for t in tk])
    place = np.array([[t["end"]["x"], t["end"]["y"]] for t in tk])
    return (pick.min(0) - margin, pick.max(0) + margin), (place.min(0) - margin, place.max(0) + margin)


class Student:
    """The DiT stepped exactly as sim/dit.py's rollout does (same seeding, same chunking),
    so replaying it from the same reset reaches the same state."""

    def __init__(self, ckpt):
        import torch
        from dit import DiT
        self.torch = torch
        self.c = torch.load(ckpt, map_location="cpu")
        self.m = DiT(self.c["d_obs"])
        self.m.load_state_dict(self.c["state"])
        self.m.eval()

    def run(self, env, obs, seed, t_stop, on_step=None):
        """Step the student from the reset observation for t_stop steps."""
        from dit import N_OBS, E
        from bc import T_MAX
        torch = self.torch
        torch.manual_seed(seed)
        hist = [obs.cpu().numpy().reshape(-1)] * N_OBS
        t, grip = 0, 1.0
        while t < t_stop:
            o = hist[-1]
            tf = 0.0 if self.c.get("no_time") else t / T_MAX
            acts = self.m.sample(torch.tensor(np.stack(hist[-N_OBS:]))[None].float(), torch.tensor([tf]).float())[0].numpy()
            for k in range(E):
                grip = float(np.clip(np.round(acts[k, 7]), -1, 1))
                act = np.r_[o[:7] + acts[k, :7], grip]
                obs, _, _, _, info = env.step(act)
                hist.append(obs.cpu().numpy().reshape(-1))
                t += 1
                if on_step is not None:
                    on_step(t, info)
                if t >= t_stop:
                    break
        return obs, grip


def geometry(env, grip):
    """Where the hand is relative to the pot's grasp point, at the hand-over."""
    from transforms3d.quaternions import qmult, qinverse, quat2axangle
    import solve_microwave as SM
    import sim.microwave_env as M
    u = env.unwrapped
    pot = u.pot_in_mw().cpu().numpy()[0]
    q_pot = u.pot.pose.raw_pose.cpu().numpy()[0, 3:]
    tilt = np.degrees(np.arccos(np.clip(abs(1 - 2 * (q_pot[1] ** 2 + q_pot[2] ** 2)), -1, 1)))
    tcp_pose = u.agent.tcp.pose.sp
    tcp = M.world_to_mw(tcp_pose.p)
    c = np.array([pot[0], pot[1], SM.POT_GRASP_Z])
    want = u.agent.build_grasp_pose(SM.mw_dir([0, 1.0, 0]), SM.mw_dir([1.0, 0, 0]), M.mw_to_world(c)).q
    d = qmult(want, qinverse(tcp_pose.q))
    ang = np.degrees(abs(quat2axangle(d if d[0] >= 0 else -d)[1]))
    return dict(pot_z=round(float(pot[2]), 4), tilt_deg=round(float(tilt), 1), grip=grip,
                d_mm=np.round((tcp - c) * 1000).astype(int).tolist(), ang_deg=round(float(ang), 1))


def modes_for(g):
    """The expert's options at this state, most local first."""
    dx, dy, dz = np.array(g["d_mm"]) / 1000
    near = abs(dx) < 0.03 and abs(dz) < 0.06 and -0.06 < dy < 0.01 and g["ang_deg"] < 15
    in_front = abs(dx) < 0.03 and abs(dz) < 0.06 and -0.25 < dy < -0.01 and g["ang_deg"] < 25
    if g["tilt_deg"] > 10 or (g["pot_z"] > 0.01 and not (g["grip"] < -0.5 and near)):
        return []                                          # knocked over (or already carried off)
    if g["grip"] < -0.5 and near:
        return ["hold", "regrasp", "restart"]
    if g["grip"] > 0.5 and in_front:
        return ["continue", "restart"]
    if near:
        return ["regrasp", "restart"]
    return ["restart"]


def hand_over(env, obs, grip, place, mode):
    """The solver finishes the task from the current state in the given mode:
      hold      the hand has closed on the pot: lift and carry from here
      continue  hand open, in front of the pot on its approach line: straight in, grasp
      regrasp   open, back off 8 cm along the approach, straight in again, grasp
      restart   open, up clear if low, tuck, the whole pick again
    then shut the door. Returns (ok, arm)."""
    import sapien
    import solve_microwave as SM
    import sim.microwave_env as M
    u = env.unwrapped
    arm = SM.Arm(env, obs=obs, record_states=True)
    arm.grip = grip
    S = SM.Solver(env, arm)
    if mode == "hold":
        ok = S.pick_and_place(place, hold=True)
    elif mode == "continue":
        ok = S.pick_and_place(place, from_tcp=True)
    elif mode == "regrasp":
        arm.gripper(SM.OPEN)
        tcp = M.world_to_mw(arm.tcp().p)
        back = S.grasp_pose(tcp - [0, 0.08, 0], [0, 1.0, 0], [1.0, 0, 0])
        ok = arm.go([back]) and S.pick_and_place(place, from_tcp=True)
    else:
        if grip < 0.5:
            arm.gripper(SM.OPEN)
        if M.world_to_mw(arm.tcp().p)[2] < 0.20:
            arm.go([sapien.Pose(arm.tcp().p + [0, 0, 0.12], arm.tcp().q)], high=True)
        q_tuck = S.tuck_q()
        ok = q_tuck is not None and arm.move_joint(q_tuck, step=0.02)
        ok = ok and S.pick_and_place(place)
    ok = ok and S.shut_door()
    return bool(ok and u.evaluate()["success"][0]), arm


def branch_times(rec, k):
    """K branch times between turning to the pot and the lift (see the module docstring)."""
    t = np.array([r["t"] for r in rec])
    door = np.array([r["door"] for r in rec])
    tcp = np.array([r["tcp"] for r in rec])
    pot = np.array([r["pot"] for r in rec])
    p0 = pot[0]
    opened = np.nonzero(door > 60)[0]
    if not len(opened):
        return []
    near = np.nonzero((np.arange(len(t)) > opened[0]) & (np.linalg.norm(tcp[:, :2] - p0[:2], axis=1) < 0.30))[0]
    if not len(near):
        return []
    i0 = near[0]
    after = np.arange(len(t)) > i0
    lift = np.nonzero(after & (pot[:, 2] > 0.003))[0]                                # in the hand
    push = np.nonzero(after & (np.linalg.norm(pot[:, :2] - p0[:2], axis=1) > 0.01))[0]  # shoved
    if len(lift) and (not len(push) or lift[0] <= push[0]):
        i1 = lift[0] - 1                       # the grasp worked: branch up to the lift, not into the carry
    elif len(push):
        i1 = min(push[0] + 20, len(t) - 1)     # just after the student started pushing it
    else:
        i1 = len(t) - 1
    if i1 <= i0:
        return []
    # denser towards the end of the window, where the grasp is decided
    frac = np.r_[np.linspace(0, 0.66, max(k - 3, 1) + 1)[1:], 0.85, 0.93, 1.0] if k >= 4 else np.linspace(0, 1, k + 1)[1:]
    return sorted(set(int(t[int(round(i0 + f * (i1 - i0)))]) for f in frac))


def job(args):
    ckpt, pot, place, seed, k, out_dir, run0 = args
    import torch
    import gymnasium as gym
    import sim.microwave_env as M  # noqa: F401
    from bc import T_MAX
    torch.set_num_threads(1)
    st = Student(ckpt)

    def make():
        return gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", pot_mw=(*pot, np.pi / 2),
                        sim_backend="cpu", render_backend="none", max_episode_steps=10 ** 5)

    # 1. the student alone, to find where to branch
    env = make()
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    rec = []

    def log(t, info):
        rec.append(dict(t=t, door=float(info["door_deg"][0]), tcp=M.world_to_mw(u.agent.tcp.pose.sp.p),
                        pot=u.pot_in_mw().cpu().numpy()[0].copy(), success=bool(info["success"][0])))
    st.run(env, obs, seed, T_MAX, on_step=log)
    student = dict(lifted=bool(max(r["pot"][2] for r in rec) > 0.05), success=any(r["success"] for r in rec))
    env.close()

    # 2. replay to each branch time and hand over; the expert's options are tried most
    # local first, each from a fresh deterministic replay; the first that succeeds is kept
    runs = []
    for b, t_b in enumerate(branch_times(rec, k)):
        tried, kept = [], None
        modes = None
        for attempt in range(3):
            env = make()
            u = env.unwrapped
            obs, _ = env.reset(seed=seed)
            obs, grip = st.run(env, obs, seed, t_b)
            if modes is None:
                drift = float(np.linalg.norm(u.pot_in_mw().cpu().numpy()[0] - rec[t_b - 1]["pot"]))
                geo = geometry(env, grip)
                modes = modes_for(geo)
            if attempt >= len(modes):
                env.close()
                break
            ok, arm = hand_over(env, obs, grip, place, modes[attempt])
            tried.append(modes[attempt])
            env.close()
            if ok:
                kept = (modes[attempt], arm)
                break
        run = run0 + b
        if kept is not None:
            d = os.path.join(out_dir, f"run_{run:03d}")
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "demo.npz"), obs=np.array(kept[1].rec_obs, np.float32),
                                actions=np.array(kept[1].rec_act, np.float32),
                                states=np.array(kept[1].rec_state, np.float32))
        runs.append(dict(run=run, pot=pot, place=place, seed=seed, t_branch=t_b, tried=tried,
                         mode=kept[0] if kept else ("knocked over" if not modes else "none worked"),
                         success=kept is not None, steps=len(kept[1].rec_act) if kept else 0,
                         replay_drift_m=drift, student=student, at_hand_over=geo))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the student (sim/dit.py checkpoint)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96, help="start states")
    ap.add_argument("--branches", type=int, default=5, help="hand-overs per start state")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7, help="start states (keep apart from the eval seeds)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (lo_p, hi_p), (lo_q, hi_q) = regions()
    rng = np.random.default_rng(a.seed)
    jobs = [(a.ckpt, rng.uniform(lo_p, hi_p).round(4).tolist(), rng.uniform(lo_q, hi_q).round(4).tolist(),
             10000 * a.seed + i, a.branches, a.out, i * a.branches) for i in range(a.n)]
    env = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.update(env)
    runs = []
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for i, rs in enumerate(ex.map(job, jobs)):
            runs += rs
            for r in rs:
                print(f"start {i:3d} t {r['t_branch']:4d} {r['mode']:12s} {'OK ' if r['success'] else 'fail'} "
                      f"student {'ok' if r['student']['success'] else ('no lift' if not r['student']['lifted'] else 'fail')} "
                      f"drift {r['replay_drift_m'] * 1000:.2f} mm  {r['at_hand_over']}", flush=True)
    ok = [r for r in runs if r["success"]]
    by = {m: [r["success"] for r in runs if r["mode"] == m] for m in sorted({r["mode"] for r in runs})}
    summ = dict(n_starts=a.n, n_branches=len(runs), success=len(ok), yield_=len(ok) / max(len(runs), 1),
                by_mode={m: f"{sum(v)}/{len(v)}" for m, v in by.items()},
                student_success_on_branched_starts=float(np.mean([r["student"]["success"] for r in runs
                                                                  if r["t_branch"] == min(q["t_branch"] for q in runs if q["seed"] == r["seed"])]))
                if runs else 0.0,
                runs=runs)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print({k: v for k, v in summ.items() if k != "runs"})


if __name__ == "__main__":
    main()
