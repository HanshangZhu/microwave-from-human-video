#!/usr/bin/env python
"""More frames of a sweep, posed in the frame of an existing sparse run.

The kitchen splat (recon/gs_kitchen.py) was trained on 60 of the sweep's 1200 frames; its
training views fit far better than held-out ones (novel views are coverage-limited). This
takes the sharpest frame of every --step frames plus the 60 already used, aligns their
depth to colour (capture_to_r2s.align_depth) and poses the new frames in the sparse run's
metric frame: batches of --batch new frames, each solved by Pi3X in one global pass with
the --anchors sparse frames nearest in time, then moved by the similarity (Umeyama) that
maps those anchors onto their sparse poses -- so the sparse run's registration to the
microwave still applies. The anchors' residual per batch is printed. (Pi3X's chunked
long-sequence mode over all frames drifted: 18 cm / 27 deg median at the shared cameras.)

  python recon/sweep_dense.py <capture dir> <tag> <sparse cap> --out <cap> [--step 4]    (env 3dgrut)

Writes <out>/input, depth, intrinsics.json (as capture_to_r2s.py) and <out>/run/poses.npz
(c2w metric, OpenCV; names; src = source frame; shared = in the sparse set) +
<out>/run/registration.json (the sparse run's).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_to_r2s import align_depth, sharpness  # noqa: E402


def umeyama(src, dst):
    """s, R, t with dst ~ s R src + t"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("tag")
    ap.add_argument("sparse", help="capture_to_r2s.py dir of the sparse set, with run/poses.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24, help="new frames per Pi3X pass")
    ap.add_argument("--anchors", type=int, default=14, help="sparse frames per pass (nearest in time)")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.capture, a.tag + ".json")))
    ci, d = meta["intrinsics"], meta["depth"]
    R_dc = np.array(d["depth_to_color"]["rotation_colmajor"]).reshape(3, 3).T
    t_dc = np.array(d["depth_to_color"]["translation_m"])
    sp_intr = json.load(open(os.path.join(a.sparse, "intrinsics.json")))
    old = [int(i) for i in sp_intr["source_frames"]]
    cap = cv2.VideoCapture(os.path.join(a.capture, a.tag + ".avi"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    bins = np.array_split(np.arange(len(frames)), len(frames) // a.step)
    picks = {int(b[np.argmax([sharpness(frames[i]) for i in b])]) for b in bins}
    picks = sorted(picks | set(old))
    for sub in ("input", "depth", "run"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    names = []
    for k, i in enumerate(picks):
        name = f"{k:04d}.png"
        names.append(name)
        cv2.imwrite(os.path.join(a.out, "input", name), frames[i])
        z16 = cv2.imread(os.path.join(a.capture, d["folder"], f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        za = align_depth(z16, d["depth_scale_m"], d["intrinsics"], ci, R_dc, t_dc)
        cv2.imwrite(os.path.join(a.out, "depth", name), np.clip(za * 1000.0, 0, 65535).astype(np.uint16))
    intr = dict(sp_intr, source_frames=picks, source=sp_intr["source"] + f" (dense, step {a.step})")
    json.dump(intr, open(os.path.join(a.out, "intrinsics.json"), "w"), indent=2)
    print(f"{len(picks)} frames ({len(old)} shared with the sparse set)", flush=True)

    # poses: Pi3X over the whole sequence in chunks drifts (VO mode: shared-camera residual 18 cm
    # median, 27 deg -- unusable). Instead every batch of new frames is solved in one global
    # pass together with the sparse frames around it (anchors), and moved into the sparse
    # frame by the similarity that fits its anchors.
    import torch
    sys.path.insert(0, os.environ.get("PI3_ROOT", os.path.expanduser("~/Research/R2S/external/Pi3")))
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_multimodal_data
    dev = torch.device("cuda")
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(dev).eval()
    ref = np.load(os.path.join(a.sparse, "run/poses.npz"))["c2w"]            # metric, sparse frame, in old order
    K = np.array([[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]], np.float32)
    new_ids = [k for k, i in enumerate(picks) if i not in old]
    Q = np.zeros((len(picks), 4, 4))
    res_all = []
    for k, i in enumerate(picks):
        if i in old:
            Q[k] = ref[old.index(i)]
    tmp = os.path.join(a.out, "_batch")
    for b0 in range(0, len(new_ids), a.batch):
        ids = new_ids[b0:b0 + a.batch]
        lo, hi = picks[ids[0]], picks[ids[-1]]
        anc = sorted(old, key=lambda i: 0 if lo <= i <= hi else min(abs(i - lo), abs(i - hi)))[:a.anchors]
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        order = [("n", k) for k in ids] + [("a", i) for i in anc]
        for j, (kind, x) in enumerate(order):
            src = names[x] if kind == "n" else names[picks.index(x)]
            shutil.copy(os.path.join(a.out, "input", src), os.path.join(tmp, f"{j:04d}.png"))
        n = len(order)
        imgs, c = load_multimodal_data(tmp, conditions=dict(intrinsics=np.repeat(K[None], n, 0), depths=None),
                                       interval=1, device=dev, verbose=False)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(imgs, intrinsics=c["intrinsics"])
        P = out["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        del out, imgs
        torch.cuda.empty_cache()
        ai = np.arange(len(ids), n)
        R_ref = np.stack([ref[old.index(i)] for i in anc])
        s_, R, t = umeyama(P[ai, :3, 3], R_ref[:, :3, 3])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = s_ * R, t
        def move(M):
            o = np.eye(4)
            o[:3, :3] = R @ M[:3, :3]
            o[:3, 3] = s_ * R @ M[:3, 3] + t
            return o
        r = np.array([np.linalg.norm(move(P[j])[:3, 3] - R_ref[q, :3, 3]) for q, j in enumerate(ai)])
        ang = [np.degrees(np.arccos(np.clip((np.trace(move(P[j])[:3, :3].T @ R_ref[q, :3, :3]) - 1) / 2, -1, 1)))
               for q, j in enumerate(ai)]
        res_all += list(r)
        print(f"batch {b0 // a.batch}: frames {lo}-{hi}, {len(anc)} anchors, residual median {np.median(r)*100:.1f} cm "
              f"max {r.max()*100:.1f} cm, rot median {np.median(ang):.2f} deg", flush=True)
        for q, k in enumerate(ids):
            Q[k] = move(P[q])
    shutil.rmtree(tmp, ignore_errors=True)
    res = np.array(res_all)
    sh = np.array([i in old for i in picks])
    np.savez(os.path.join(a.out, "run/poses.npz"), c2w=Q, names=np.array(names), src=np.array(picks), shared=sh,
             residual_m=res)
    shutil.copy(os.path.join(a.sparse, "run/registration.json"), os.path.join(a.out, "run/registration.json"))
    print("->", a.out)


if __name__ == "__main__":
    main()
