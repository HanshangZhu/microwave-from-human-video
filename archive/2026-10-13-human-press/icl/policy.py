#!/usr/bin/env python
"""In-context press policy: a VLM conditioned on k demonstrations, no training.

The prompt carries no task description. It is k pairs of (head-camera image,
where the press went) followed by a query image, and the instruction is only
"continue the pattern". Everything the model knows about the task comes from the
demonstrations, so k = 0 has to fail and any success at k > 0 is attributable to
the recorded data and nothing else. That is the point of the design: you cannot
get this curve from a prompt that already says "press the bell".

Two output spaces, because they ask different questions:

  table   millimetres on the table. The demonstrations have to teach the whole
          image-to-world mapping.
  pixel   a point in the image, converted to the table by us using the camera
          matrices. The demonstrations only have to teach what to point at.

  python icl/policy.py --bank <dir> --k 8 --space table --self-test
"""
import argparse
import json
import os
import random
import re

import numpy as np

QWEN = "/home/hanszhu/Research/VLA/lingbot-vla-v2-accel/models/Qwen3-VL-4B-Instruct"
SYSTEM = "You continue patterns from examples. Answer with the requested format and nothing else."
ASK = {
    "table": "x and y in millimetres, as `x=<number> y=<number>`",
    "pixel": "the pixel column and row, as `u=<number> v=<number>`",
}


class ICLPolicy:
    def __init__(self, bank_dir, k, space="table", seed=0, model_path=QWEN,
                 max_new_tokens=24, device="cuda"):
        self.bank_dir, self.k, self.space = bank_dir, k, space
        self.rng = random.Random(seed)
        b = json.load(open(os.path.join(bank_dir, "bank.json")))
        self.entries = b["entries"]
        self.model_path, self.max_new_tokens, self.device = model_path, max_new_tokens, device
        self._m = self._p = None

    # -- lazy load so prompt construction can be tested without a GPU ---------
    def _load(self):
        if self._m is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._p = AutoProcessor.from_pretrained(self.model_path)
        self._m = AutoModelForImageTextToText.from_pretrained(
            self.model_path, dtype=torch.bfloat16, device_map=self.device).eval()

    def _answer(self, e):
        if self.space == "table":
            x, y = e["target_xy_mm"]
            return f"x={x:.0f} y={y:.0f}"
        u, v = e["target_uv"]
        return f"u={u:.0f} v={v:.0f}"

    def choose(self, exclude_episodes=()):
        pool = [e for e in self.entries if e["episode"] not in exclude_episodes
                and (self.space == "table" or "target_uv" in e)]
        self.rng.shuffle(pool)
        return pool[: self.k]

    def build_messages(self, query_image, demos):
        """Interleaved image/answer pairs, then the query. No task description."""
        content = []
        for e in demos:
            content.append({"type": "image", "image": os.path.join(self.bank_dir, e["image"])})
            content.append({"type": "text", "text": self._answer(e)})
        content.append({"type": "image", "image": query_image})
        content.append({"type": "text", "text": f"Continue the pattern. Reply with {ASK[self.space]}."})
        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": content}]

    def predict(self, query_image, demos=None):
        import torch
        self._load()
        demos = self.choose() if demos is None else demos
        msgs = self.build_messages(query_image, demos)
        inputs = self._p.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self._m.device)
        with torch.no_grad():
            out = self._m.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self._p.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self.parse(text), text, len(demos)

    @staticmethod
    def parse(text):
        """-> (a, b) or None. Accepts `x=.. y=..`, `u=.. v=..`, or two bare numbers."""
        m = re.findall(r"[xu]\s*=\s*(-?\d+\.?\d*)\D+[yv]\s*=\s*(-?\d+\.?\d*)", text)
        if m:
            return float(m[0][0]), float(m[0][1])
        n = re.findall(r"-?\d+\.?\d*", text)
        return (float(n[0]), float(n[1])) if len(n) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--space", choices=("table", "pixel"), default="table")
    ap.add_argument("--self-test", action="store_true",
                    help="leave-one-out over the bank: predict each demo from the others")
    ap.add_argument("--n", type=int, default=8, help="self-test queries")
    a = ap.parse_args()

    pol = ICLPolicy(a.bank, a.k, a.space)
    if not a.self_test:
        msgs = pol.build_messages("<query.png>", pol.choose())
        n_img = sum(1 for c in msgs[1]["content"] if c["type"] == "image")
        print(f"prompt: {n_img} images ({a.k} demos + 1 query), space={a.space}")
        print("last two blocks:", json.dumps(msgs[1]["content"][-2:])[:200])
        return

    errs, raw_fail = [], 0
    for e in pol.entries[: a.n]:
        demos = pol.choose(exclude_episodes={e["episode"]})
        q = os.path.join(a.bank, e["image"])
        pred, text, k = pol.predict(q, demos)
        truth = e["target_xy_mm"] if a.space == "table" else e.get("target_uv")
        if pred is None:
            raw_fail += 1
            print(f"  ep{e['episode']:02d}  unparseable: {text!r}")
            continue
        d = float(np.hypot(pred[0] - truth[0], pred[1] - truth[1]))
        errs.append(d)
        print(f"  ep{e['episode']:02d}  truth {truth[0]:7.1f},{truth[1]:7.1f}   "
              f"pred {pred[0]:7.1f},{pred[1]:7.1f}   err {d:6.1f}")
    if errs:
        unit = "mm" if a.space == "table" else "px"
        print(f"\nk={a.k} {a.space}: median error {np.median(errs):.1f} {unit}, "
              f"mean {np.mean(errs):.1f}, unparseable {raw_fail}/{a.n}")
        if a.space == "table":
            print(f"  (click_bell needs the press within 25 mm to count)")


if __name__ == "__main__":
    main()
