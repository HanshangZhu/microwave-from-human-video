#!/usr/bin/env python
"""Read the evaluation cells and report paired comparisons.

Every arm runs the same seeds in the same order, so the comparisons are paired:
what matters is not two success rates but the seeds on which the two arms
disagree. With 60 seeds an unpaired difference of 15 pp is inside the noise,
while 0-vs-9 discordant pairs is not; McNemar uses only the discordant ones.

Episodes with `success: null` errored (a crashed renderer, say) and are excluded
from both arms of any pair they appear in -- counting them as failures is how a
2-server GPU OOM once read as "1.7% success".

  python eval/analyse.py <run dir>
"""
import argparse
import glob
import json
import os
from math import comb


def load(run):
    """{(model, task, arm): {seed: bool}}"""
    out = {}
    for f in glob.glob(os.path.join(run, "*", "*", "*", "episodes.jsonl")):
        parts = f.split(os.sep)
        model_task, arm = parts[-4], parts[-3]
        if "__" not in model_task:
            continue
        model, task = model_task.split("__", 1)
        d = {}
        for line in open(f):
            r = json.loads(line)
            if r.get("success") is not None:
                d[r["seed"]] = bool(r["success"])
        if d:
            out[(model, task, arm)] = d
    return out


def mcnemar(a, b):
    """Exact two-sided McNemar over the seeds both arms completed."""
    seeds = sorted(set(a) & set(b))
    n01 = sum(1 for s in seeds if not a[s] and b[s])      # b wins
    n10 = sum(1 for s in seeds if a[s] and not b[s])      # a wins
    n = n01 + n10
    if n == 0:
        return len(seeds), 0, 0, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return len(seeds), n10, n01, p


def rate(d):
    return 100.0 * sum(d.values()) / len(d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    cells = load(a.run)
    if not cells:
        raise SystemExit(f"no cells under {a.run}")

    print(f"{'model':12s} {'task':16s} {'arm':9s} {'success':>12s}")
    for (m, t, arm), d in sorted(cells.items()):
        print(f"{m:12s} {t:16s} {arm:9s} {sum(d.values()):3d}/{len(d):<3d} = {rate(d):5.1f}%")

    # The comparisons the experiment exists to make. Each pair changes exactly
    # one thing; the label says what.
    pairs = [
        ("bf16bug", "bf16", "time grid: BF16 -> FP32, upstream path, same speed"),
        ("fast", "leang", "time grid: BF16 -> FP32, fast path, same speed"),
        ("bf16", "leang", "speed: 126 ms -> 79 ms, both grids fixed"),
        ("bf16bug", "fast", "speed: 126 ms -> 79 ms, both grids buggy"),
        ("fp32", "bf16", "precision: FP32 -> BF16, grid fixed"),
    ]
    for m in sorted({k[0] for k in cells}):
        for t in sorted({k[1] for k in cells if k[0] == m}):
            head = False
            for x, y, what in pairs:
                A, B = cells.get((m, t, x)), cells.get((m, t, y))
                if not A or not B:
                    continue
                if not head:
                    print(f"\n--- {m} / {t} ---")
                    head = True
                n, w_x, w_y, p = mcnemar(A, B)
                print(f"  {x:8s} {rate(A):5.1f}%  ->  {y:8s} {rate(B):5.1f}%   "
                      f"{rate(B)-rate(A):+6.1f} pp   discordant {w_x} v {w_y}  p = {p:.4f}"
                      f"{'  *' if p < 0.05 else ''}")
                print(f"           {what}")

    # Same arm, different model: what the post-training did.
    models = sorted({k[0] for k in cells})
    if len(models) > 1:
        base = "base" if "base" in models else models[0]
        for other in [m for m in models if m != base]:
            head = False
            for (m, t, arm) in sorted(cells):
                if m != base:
                    continue
                A, B = cells.get((base, t, arm)), cells.get((other, t, arm))
                if not A or not B:
                    continue
                if not head:
                    print(f"\n--- post-training: {base} -> {other} ---")
                    head = True
                n, w_a, w_b, p = mcnemar(A, B)
                print(f"  {t:16s} {arm:9s} {rate(A):5.1f}% -> {rate(B):5.1f}%   "
                      f"{rate(B)-rate(A):+6.1f} pp   discordant {w_a} v {w_b}  p = {p:.4f}"
                      f"{'  *' if p < 0.05 else ''}")


if __name__ == "__main__":
    main()
