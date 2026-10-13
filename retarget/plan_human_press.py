#!/usr/bin/env python
"""Planning pass over the human press specs, one attempt per spec.

RoboTwin's own collector retries a failed episode with a fresh seed until it has
`episode_num` successes. That is right for randomly sampled scenes and wrong
here: every seed *is* a specific recorded press, so a retry would re-attempt the
same unreachable press forever. This pass tries each spec exactly once and
records why the failures failed.

That failure list is a result, not just bookkeeping. click_bell refuses to place
its bell within 50 mm of the centre line; this tells us, from real press
locations, what that exclusion is actually protecting against.

Run from the RoboTwin root with its own env:

  cd $ROBOTWIN && HP_SPECS=<specs.json> \
    python <this>/plan_human_press.py human_press_bell <task_config>
"""
import argparse
import json
import os
import sys
import traceback

sys.path.append("./")

import yaml  # noqa: E402
import importlib  # noqa: E402

CONFIGS = "./task_config"


def load_args(task_name, task_config):
    with open(f"{CONFIGS}/{task_config}.yml", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config
    with open(f"{CONFIGS}/_embodiment_config.yml", encoding="utf-8") as f:
        emb = yaml.safe_load(f)

    def efile(t):
        p = emb[t]["file_path"]
        if p is None:
            raise SystemExit(f"missing embodiment files for {t}")
        return p

    et = args["embodiment"]
    if len(et) == 1:
        args["left_robot_file"] = args["right_robot_file"] = efile(et[0])
        args["dual_arm_embodied"] = True
        name = str(et[0])
    elif len(et) == 3:
        args["left_robot_file"], args["right_robot_file"] = efile(et[0]), efile(et[1])
        args["embodiment_dis"], args["dual_arm_embodied"] = et[2], False
        name = f"{et[0]}+{et[1]}"
    else:
        raise SystemExit("embodiment must have 1 or 3 entries")
    for side in ("left", "right"):
        with open(os.path.join(args[f"{side}_robot_file"], "config.yml"), encoding="utf-8") as f:
            args[f"{side}_embodiment_config"] = yaml.safe_load(f)
    args["embodiment_name"] = name
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    args["need_plan"] = True
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_name")
    ap.add_argument("task_config")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many specs")
    a = ap.parse_args()

    specs_path = os.environ.get("HP_SPECS")
    if not specs_path:
        raise SystemExit("set HP_SPECS")
    specs = json.load(open(specs_path))["specs"]
    n = a.limit or len(specs)

    args = load_args(a.task_name, a.task_config)
    os.makedirs(args["save_path"], exist_ok=True)
    task = getattr(importlib.import_module(f"envs.{a.task_name}"), a.task_name)()

    seeds, report = [], []
    for i in range(n):
        s = specs[i]
        why, ok = "", False
        try:
            task.setup_demo(now_ep_num=len(seeds), seed=i, **args)
            task.play_once()
            if task.plan_success is False:
                why = "motion planning failed (unreachable or in collision)"
            elif not task.check_success():
                why = "planned, but the task's success check did not trigger"
            else:
                ok = True
                task.save_traj_data(len(seeds))
                seeds.append(i)
        except Exception as e:                       # noqa: BLE001
            why = f"{type(e).__name__}: {e}"
            traceback.print_exc(limit=1)
        finally:
            try:
                task.close_env()
            except Exception:                        # noqa: BLE001
                pass
        report.append(dict(spec=i, take=s["take"], bell_xy=s["bell_xy"], arm=s["arm"],
                           outside=s["outside_task_distribution"], ok=ok, why=why))
        print(f"[{'ok ' if ok else 'FAIL'}] spec {i:3d}  {s['take']}  "
              f"bell ({s['bell_xy'][0]*1000:6.0f},{s['bell_xy'][1]*1000:6.0f}) mm  "
              f"{s['arm']:5s}  {why}", flush=True)

    with open(os.path.join(args["save_path"], "seed.txt"), "w") as f:
        f.write(" ".join(str(s) for s in seeds))
    rp = os.path.join(args["save_path"], "plan_report.json")
    json.dump(dict(n_specs=n, n_ok=len(seeds), seeds=seeds, report=report),
              open(rp, "w"), indent=1)

    out = [r for r in report if r["outside"]]
    ins = [r for r in report if not r["outside"]]
    print(f"\nplanned {len(seeds)}/{n}")
    if ins:
        print(f"  inside the task's sampling region : {sum(r['ok'] for r in ins)}/{len(ins)}")
    if out:
        print(f"  outside it                        : {sum(r['ok'] for r in out)}/{len(out)}")
    print(f"  seeds -> {args['save_path']}/seed.txt")
    print(f"  report -> {rp}")
    print(f"\nnow run the stock collector with use_seed: true and "
          f"episode_num: {len(seeds)} to render these.")


if __name__ == "__main__":
    main()
