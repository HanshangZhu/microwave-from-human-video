#!/usr/bin/env python
"""Render the planned episodes with several RoboTwin processes at once.

RoboTwin's collector is single-process and its phase-2 loop resumes from the
first missing `episode{i}.hdf5`, so two processes pointed at one directory race
for the same index. Each shard therefore gets its own `task_config` — which is
what decides `save_path` — its own `seed.txt`, and its own cache; afterwards the
episodes are merged into one directory with a manifest recording which recorded
press each came from.

Run from the RoboTwin root:

  HP_SPECS=<specs.json> python <this>/render_parallel.py --mapping direct --shards 3
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.getcwd()
PY = os.environ.get("ROBOTWIN_PY", sys.executable)


def shard_config(base_cfg, name, n_ep):
    src = f"./task_config/{base_cfg}.yml"
    dst = f"./task_config/{name}.yml"
    txt = open(src).read()
    out = []
    for line in txt.splitlines():
        if line.startswith("episode_num:"):
            line = f"episode_num: {n_ep}"
        elif line.startswith("use_seed:"):
            line = "use_seed: true"
        out.append(line)
    open(dst, "w").write("\n".join(out) + "\n")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="human_press_bell")
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--shards", type=int, default=3)
    a = ap.parse_args()

    base = f"hp_{a.mapping}"
    plan = json.load(open(f"./data/{a.task}/{base}/plan_report.json"))
    seeds = plan["seeds"]
    if not seeds:
        raise SystemExit("no planned seeds")
    k = min(a.shards, len(seeds))
    chunks = [seeds[i::k] for i in range(k)]          # round-robin keeps shards even

    procs = []
    for i, ch in enumerate(chunks):
        name = f"{base}_s{i}"
        d = f"./data/{a.task}/{name}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "seed.txt"), "w").write(" ".join(str(s) for s in ch))
        # Phase 2 replays the joint paths cached by the planning pass, which live
        # under the *planning* save_path. Each shard needs its own copy, renumbered
        # to the shard's episode order.
        os.makedirs(f"{d}/_traj_data", exist_ok=True)
        for j, seed in enumerate(ch):
            src = f"./data/{a.task}/{base}/_traj_data/episode{seeds.index(seed)}.pkl"
            if not os.path.exists(src):
                raise SystemExit(f"missing planned trajectory {src}; run the planning pass first")
            shutil.copy2(src, f"{d}/_traj_data/episode{j}.pkl")
        shard_config(base, name, len(ch))
        log = open(f"/tmp/claude-1000/render_{name}.log", "w")
        procs.append((name, ch, subprocess.Popen(
            [PY, "script/collect_data.py", a.task, name],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
            env={**os.environ, "PYTHONWARNINGS": "ignore::UserWarning"})))
        print(f"shard {i}: {len(ch)} episodes -> {name}", flush=True)

    t0 = time.time()
    for name, ch, p in procs:
        rc = p.wait()
        n = len([f for f in os.listdir(f"./data/{a.task}/{name}/data")
                 if f.endswith(".hdf5")]) if os.path.isdir(f"./data/{a.task}/{name}/data") else 0
        print(f"shard {name}: exit {rc}, {n}/{len(ch)} episodes "
              f"({time.time()-t0:.0f}s)", flush=True)

    # merge, renumbering episodes and recording where each came from
    out = f"./data/{a.task}/{base}_all"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(f"{out}/data", exist_ok=True)
    os.makedirs(f"{out}/video", exist_ok=True)
    manifest, idx = [], 0
    for name, ch, _ in procs:
        d = f"./data/{a.task}/{name}"
        for j, seed in enumerate(ch):
            src = f"{d}/data/episode{j}.hdf5"
            if not os.path.exists(src):
                continue
            shutil.copy2(src, f"{out}/data/episode{idx}.hdf5")
            v = f"{d}/video/episode{j}.mp4"
            if os.path.exists(v):
                shutil.copy2(v, f"{out}/video/episode{idx}.mp4")
            manifest.append(dict(episode=idx, spec=seed, shard=name, shard_index=j))
            idx += 1
    # scene_info is keyed by episode index, so it has to be renumbered on merge,
    # not copied from one shard -- the instruction generator reads it per episode
    # and silently produces nothing for the episodes it cannot find.
    scene = {}
    for m in manifest:
        src = f"./data/{a.task}/{m['shard']}/scene_info.json"
        if os.path.exists(src):
            si = json.load(open(src))
            key = f"episode_{m['shard_index']}"
            if key in si:
                scene[f"episode_{m['episode']}"] = si[key]
    json.dump(scene, open(f"{out}/scene_info.json", "w"), ensure_ascii=False, indent=4)
    json.dump(dict(mapping=a.mapping, n_episodes=idx, manifest=manifest),
              open(f"{out}/manifest.json", "w"), indent=1)
    print(f"\nmerged {idx} episodes -> {out}")


if __name__ == "__main__":
    main()
