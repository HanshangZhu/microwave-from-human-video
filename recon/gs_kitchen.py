#!/usr/bin/env python
"""The kitchen as a 3D Gaussian splat, in the microwave frame, for rendering backgrounds.

Inputs are what the mesh backdrop (kitchen_backdrop.py) was built from, plus one view:
  - the kitchen sweep's 60 frames (recon/kitchen/closed/input) at their Pi3X poses scaled to
    metres (fuse_depth.py run/poses.npz), with their RealSense depth;
  - take_000's last frame (the scene at rest), from the front camera the policy sees through,
    at its registered pose -- so that view is reproduced closely;
  - initial points: Pi3X's dense cloud of the sweep (pi3x_poisson.py run_poisson/cloud.ply).
Everything is moved into the microwave frame with the sweep's registration, so a splat
render and the simulator's cameras share one frame.

Training is gsplat's rasterizer with its default densification strategy; loss = 0.8 L1 +
0.2 (1 - SSIM) on colour + 0.2 L1 on depth (metres) where the RealSense depth is valid.

  train:  python recon/gs_kitchen.py train --out <dir> [--steps 7000] [--inpainted <gs_inpaint.py dir>]  (env sam3d)
  render: python recon/gs_kitchen.py render --ckpt <dir>/ckpt.pt --cams <npz> --out <npz>
          cams npz: K (n,3,3), w2c (n,4,4) OpenCV in the microwave frame, width, height;
          the scanned microwave and anything above the counter's workspace are removed
          first (the simulator draws its own microwave and pot) -- unless --keep-all, for a
          splat trained on images with the microwave painted out (gs_inpaint.py).
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, "Gitignore-but-important")
KIT = os.path.join(G, "recon/kitchen/closed")
MICROWAVE = ((-0.06, -0.10, -0.02), (0.52, 0.40, 0.32))   # as kitchen_backdrop.py
WORKSPACE = ((-0.10, -0.60, 0.04), (0.80, 0.0, 0.50))
# the microwave itself (assets/microwave URDF: body 0.46 x 0.32 x 0.28 m, door + handle ~8 cm in
# front), with 2-3 cm for the sweep's registration error: what gs_inpaint.py paints out
MW_TIGHT = ((-0.03, -0.10, -0.01), (0.49, 0.345, 0.30))


def load_views(inpainted=None, sweep=KIT):
    """inpainted: recon/gs_inpaint.py's output dir -- its images instead of the originals, and no
    measured depth inside its masks (the depth there is of the removed microwave).
    sweep: the sweep's frames (the 60 of recon/kitchen/closed, or recon/sweep_dense.py's), posed
    in recon/kitchen/closed's metric frame; each view keeps its source frame in the video"""
    T = np.linalg.inv(np.array(json.load(open(os.path.join(KIT, "run/registration.json")))["T_model_to_world"]))
    intr = json.load(open(os.path.join(sweep, "intrinsics.json")))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float32)
    P = np.load(os.path.join(sweep, "run/poses.npz"))
    views = []
    for name, c2w, src in zip(P["names"], P["c2w"], intr["source_frames"]):
        img = cv2.imread(os.path.join(sweep, "input", str(name)))[..., ::-1]
        dep = cv2.imread(os.path.join(sweep, "depth", str(name)), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        views.append(dict(img=img, depth=dep, K=K, c2w=T @ c2w, name=str(name), src=int(src)))
    # take_000 at rest: the front camera's own view
    take = os.path.join(G, "recon/takes/take_000")
    reg = np.array(json.load(open(os.path.join(take, "run/registration.json")))["T_model_to_world"])
    seg = json.load(open(os.path.join(take, "anchor.json")))["segments"][-1]
    ci = json.load(open(os.path.join(G, "laptop/microwave01/take_000.json")))["intrinsics"]
    Kt = np.array([[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]], np.float32)
    img = cv2.imread(os.path.join(take, "rest_frame.png"))[..., ::-1]
    dep = np.zeros(img.shape[:2], np.float32)      # the take's depth is not aligned to colour: colour only
    views.append(dict(img=img, depth=dep, K=Kt, c2w=np.linalg.inv(reg) @ np.array(seg["T_last_from_cam"]),
                      name="take_000_rest", src=-10 ** 6))
    if inpainted:
        for v in views:
            f = v["name"] if v["name"].endswith(".png") else v["name"] + ".png"
            v["img"] = cv2.imread(os.path.join(inpainted, "images", f))[..., ::-1]
            m = cv2.imread(os.path.join(inpainted, "masks", f), cv2.IMREAD_GRAYSCALE) > 0
            v["depth"] = np.where(m, 0, v["depth"]).astype(np.float32)
    return views, T


def load_points(T, n=400000, drop_box=None):
    import open3d as o3d
    pc = o3d.io.read_point_cloud(os.path.join(G, "recon/kitchen/closed/run_poisson/cloud.ply"))
    p, c = np.asarray(pc.points), np.asarray(pc.colors)
    k = np.random.default_rng(0).choice(len(p), min(n, len(p)), replace=False)
    p, c = (p[k] @ T[:3, :3].T + T[:3, 3]).astype(np.float32), c[k].astype(np.float32)
    if drop_box is not None:
        keep = ~np.all((p >= np.array(drop_box[0])) & (p <= np.array(drop_box[1])), axis=1)
        p, c = p[keep], c[keep]
    return p, c


def inside(p, box):
    lo, hi = torch.tensor(box[0], device=p.device), torch.tensor(box[1], device=p.device)
    return ((p >= lo) & (p <= hi)).all(-1)


# what a render shows where the sweep saw nothing (the ceiling, behind the camera): a light
# grey like the room's walls and ceiling, not black -- the simulator shows its grey there
EMPTY = (0.75, 0.75, 0.73)


def render(params, viewmats, Ks, W, H, sh_degree=3, mode="RGB+ED", bg=None):
    from gsplat import rasterization
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    kw = {}
    if bg is not None:
        c = torch.tensor(bg, device=viewmats.device, dtype=torch.float32)
        kw["backgrounds"] = c[None]                     # gsplat pads the depth channel itself
    return rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                         torch.sigmoid(params["opacities"]), colors, viewmats, Ks, W, H,
                         sh_degree=sh_degree, render_mode=mode, packed=False, near_plane=0.01, far_plane=20.0, **kw)


def train(a):
    from gsplat.strategy import DefaultStrategy
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    dev = torch.device("cuda")
    views, T = load_views(a.inpainted, a.sweep)
    pts, cols = load_points(T, drop_box=MW_TIGHT if a.inpainted else None)
    print(f"{len(views)} views, {len(pts):,} initial points", flush=True)
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=4).fit(pts).kneighbors(pts)
    s0 = np.log(np.clip(d[:, 1:].mean(1), 1e-4, 0.05))
    N = len(pts)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(pts, device=dev)),
        "scales": torch.nn.Parameter(torch.tensor(s0, device=dev, dtype=torch.float32)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(((torch.tensor(cols, device=dev) - 0.5) / 0.28209479)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device=dev)),
    })
    cams = np.stack([v["c2w"][:3, 3] for v in views])
    scene_scale = float(np.linalg.norm(cams - cams.mean(0), axis=1).max()) * 1.1
    lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / a.steps))
    strategy = DefaultStrategy(refine_stop_iter=min(15000, int(a.steps * 0.6)), reset_every=3000, verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)
    data = [dict(img=torch.tensor(v["img"].copy(), device=dev).float() / 255,
                 depth=torch.tensor(v["depth"], device=dev),
                 K=torch.tensor(v["K"], device=dev),
                 w2c=torch.tensor(np.linalg.inv(v["c2w"]), device=dev, dtype=torch.float32)) for v in views]
    rng = np.random.default_rng(0)
    os.makedirs(a.out, exist_ok=True)
    # --holdout k: every k-th sweep view is never trained on (the front view always is), for a
    # novel-view PSNR -- the display cameras are novel views
    # held out: every k-th of the 60 sparse frames (the same views whichever sweep set is trained),
    # and no training view within 8 video frames of one (near-duplicates)
    held_src = json.load(open(os.path.join(KIT, "intrinsics.json")))["source_frames"][::a.holdout] if a.holdout else []
    held = {i for i, v in enumerate(views) if v["src"] in held_src}
    near = {i for i, v in enumerate(views) if any(abs(v["src"] - h) <= 8 for h in held_src)}
    train_ids = [i for i in range(len(data) - 1) if i not in near]

    # --pose-opt: a learnable 6-DoF correction per training view (axis-angle, metres), applied on the
    # left of its world-to-camera; the front view stays fixed -- its pose is the registration
    dpose = torch.nn.Parameter(torch.zeros(len(data), 6, device=dev))
    popt = torch.optim.Adam([dpose], lr=a.pose_lr) if a.pose_opt else None

    def w2c_of(i):
        if not a.pose_opt or i == len(data) - 1:
            return data[i]["w2c"]
        w, t = dpose[i, :3], dpose[i, 3:]
        th = w.norm() + 1e-12
        k = w / th
        Kx = torch.zeros(3, 3, device=dev)
        Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
        R = torch.eye(3, device=dev) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx
        D = torch.eye(4, device=dev)
        D[:3, :3], D[:3, 3] = R, t
        return D @ data[i]["w2c"]
    evals = {int(x) for x in a.eval_at.split(",") if x} | {a.steps}

    def psnr(ids):
        out = []
        with torch.no_grad():
            for i in ids:
                v = data[i]
                H, W = v["img"].shape[:2]
                r, _, _ = render(params, w2c_of(i)[None], v["K"][None], W, H, sh_degree=3)
                out.append(float(-10 * torch.log10(((r[0, ..., :3].clamp(0, 1) - v["img"]) ** 2).mean())))
        return float(np.mean(out)) if out else float("nan")

    for step in range(a.steps):
        if step in evals:
            print(f"EVAL step {step:6d}  PSNR train {psnr(train_ids):.2f}  front {psnr([len(data) - 1]):.2f}  "
                  f"held-out {psnr(sorted(held)):.2f}  ({len(held)} views)", flush=True)
        i = len(data) - 1 if rng.random() < a.p_front else int(rng.choice(train_ids))
        v = data[i]
        H, W = v["img"].shape[:2]
        sh = min(3, step // 1000)
        out, alpha, info = render(params, w2c_of(i)[None], v["K"][None], W, H, sh_degree=sh)
        rgb, dep = out[0, ..., :3], out[0, ..., 3]
        strategy.step_pre_backward(params, opts, state, step, info)
        l1 = (rgb - v["img"]).abs().mean()
        ss = 1 - ssim(rgb.permute(2, 0, 1)[None], v["img"].permute(2, 0, 1)[None], data_range=1.0)
        m = (v["depth"] > 0.1) & (v["depth"] < 3.0)
        ld = (dep[m] - v["depth"][m]).abs().mean() if m.any() else torch.zeros((), device=dev)
        loss = 0.8 * l1 + 0.2 * ss + a.depth_w * ld
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if popt is not None:
            if step >= 500:
                popt.step()
            popt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, opts, state, step, info, packed=False)
        if step % 500 == 0 or step == a.steps - 1:
            print(f"step {step:5d}  l1 {l1.item():.4f}  ssim-loss {ss.item():.4f}  depth {ld.item():.4f} m  "
                  f"gaussians {len(params['means']):,}", flush=True)
    print(f"EVAL step {a.steps:6d}  PSNR train {psnr(train_ids):.2f}  front {psnr([len(data) - 1]):.2f}  "
          f"held-out {psnr(sorted(held)):.2f}  ({len(held)} views)", flush=True)
    torch.save({k: p.detach().cpu() for k, p in params.items()}, os.path.join(a.out, "ckpt.pt"))
    if a.pose_opt:
        dp = dpose.detach().cpu().numpy()
        print(f"pose corrections: rotation median {np.degrees(np.median(np.linalg.norm(dp[:, :3], axis=1))):.2f} deg "
              f"max {np.degrees(np.linalg.norm(dp[:, :3], axis=1).max()):.2f}; translation median "
              f"{np.median(np.linalg.norm(dp[:, 3:], axis=1)) * 100:.1f} cm", flush=True)
        np.save(os.path.join(a.out, "pose_corrections.npy"), dp)
    # training views re-rendered, for a look
    with torch.no_grad():
        for i in (0, len(data) // 3, 2 * len(data) // 3, len(data) - 1):
            v = data[i]
            H, W = v["img"].shape[:2]
            out, _, _ = render(params, w2c_of(i)[None], v["K"][None], W, H)
            im = np.hstack([v["img"].cpu().numpy(), out[0, ..., :3].clamp(0, 1).cpu().numpy()])
            cv2.imwrite(os.path.join(a.out, f"train_view_{i:02d}.jpg"), (im[..., ::-1] * 255).astype(np.uint8))
    print("->", a.out)


def render_cmd(a):
    dev = torch.device("cuda")
    params = {k: v.to(dev) for k, v in torch.load(a.ckpt).items()}
    if not a.keep_all:
        box = MW_TIGHT if a.tight else MICROWAVE
        keep = ~(inside(params["means"], box) | inside(params["means"], WORKSPACE))
        params = {k: v[keep] for k, v in params.items()}
    c = np.load(a.cams)
    W, H = int(c["width"]), int(c["height"])
    rgb, dep = [], []
    with torch.no_grad():
        for K, w2c in zip(c["K"], c["w2c"]):
            out, alpha, _ = render(params, torch.tensor(w2c, device=dev, dtype=torch.float32)[None],
                                   torch.tensor(K, device=dev, dtype=torch.float32)[None], W, H, bg=EMPTY)
            rgb.append((out[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy())
            d = out[0, ..., 3].cpu().numpy()
            d[alpha[0, ..., 0].cpu().numpy() < 0.5] = np.inf
            dep.append(d)
    np.savez_compressed(a.out, rgb=np.stack(rgb), depth=np.stack(dep).astype(np.float32))
    print("->", a.out)


def heldout_cmd(a):
    """PSNR on the held-out sweep views (every --holdout-th of the 60 sparse frames), each after
    aligning its pose to the frozen splat (300 Adam steps on a 6-DoF correction, colour L1) --
    without it a splat trained with --pose-opt is scored against poses it moved away from."""
    dev = torch.device("cuda")
    params = {k: v.to(dev) for k, v in torch.load(a.ckpt).items()}
    views, _ = load_views(a.inpainted)
    held_src = json.load(open(os.path.join(KIT, "intrinsics.json")))["source_frames"][::a.holdout]
    out = []
    for v in views:
        if v["src"] not in held_src:
            continue
        img = torch.tensor(v["img"].copy(), device=dev).float() / 255
        K = torch.tensor(v["K"], device=dev)
        w2c0 = torch.tensor(np.linalg.inv(v["c2w"]), device=dev, dtype=torch.float32)
        H, W = img.shape[:2]
        dp = torch.nn.Parameter(torch.zeros(6, device=dev))
        opt = torch.optim.Adam([dp], lr=1e-3)

        def w2c():
            w, t = dp[:3], dp[3:]
            th = w.norm() + 1e-12
            k = w / th
            Kx = torch.zeros(3, 3, device=dev)
            Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
            R = torch.eye(3, device=dev) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx
            D = torch.eye(4, device=dev)
            D[:3, :3], D[:3, 3] = R, t
            return D @ w2c0

        def score():
            with torch.no_grad():
                r, _, _ = render(params, w2c()[None], K[None], W, H)
                return float(-10 * torch.log10(((r[0, ..., :3].clamp(0, 1) - img) ** 2).mean()))
        before = score()
        for _ in range(a.align_steps):
            r, _, _ = render(params, w2c()[None], K[None], W, H)
            loss = (r[0, ..., :3] - img).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        out.append((v["src"], before, score(), float(np.degrees(dp[:3].norm().item())), float(dp[3:].norm().item())))
    for src, b, c, dr, dt in out:
        print(f"  frame {src:5d}: PSNR {b:.2f} -> {c:.2f} after aligning ({dr:.2f} deg, {dt * 100:.1f} cm)")
    print(f"HELDOUT {a.ckpt}: PSNR {np.mean([o[1] for o in out]):.2f} as posed, "
          f"{np.mean([o[2] for o in out]):.2f} pose-aligned ({len(out)} views)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "render", "heldout"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--p-front", type=float, default=0.15, help="train: how often the front camera's view is drawn")
    ap.add_argument("--depth-w", type=float, default=0.2)
    ap.add_argument("--holdout", type=int, default=0, help="train: hold out every k-th sweep view (novel-view PSNR)")
    ap.add_argument("--eval-at", default="1000,5000,10000,20000", help="train: steps to print the PSNRs at")
    ap.add_argument("--inpainted", help="train: recon/gs_inpaint.py output (the microwave painted out)")
    ap.add_argument("--sweep", default=KIT, help="train: sweep frames (recon/kitchen/closed or a sweep_dense.py dir)")
    ap.add_argument("--pose-opt", action="store_true", help="train: refine every sweep view's pose")
    ap.add_argument("--pose-lr", type=float, default=1e-4)
    ap.add_argument("--align-steps", type=int, default=300, help="heldout: pose-alignment steps per view")
    ap.add_argument("--ckpt")
    ap.add_argument("--tight", action="store_true", help="render: delete only MW_TIGHT (a splat trained --inpainted)")
    ap.add_argument("--keep-all", action="store_true",
                    help="render: delete nothing (a splat trained --inpainted has no microwave to delete, and a view "
                         "seen by one camera only, like the front one, is explained partly by Gaussians floating in "
                         "those boxes -- deleting them uncovers junk)")
    ap.add_argument("--cams")
    a = ap.parse_args()
    {"train": train, "render": render_cmd, "heldout": heldout_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
