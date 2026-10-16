#!/usr/bin/env python
"""Camera-based student: the DiT of sim/dit.py, conditioned on images instead of the
simulator's poses.

What it sees -- only what a real robot would have:
  front   the RealSense image from where the human take's camera stood (sim/views.py)
  wrist   the hand camera
  robot   joint positions and velocities and the tcp pose (forward kinematics), the
          last N_OBS steps
Not the pot's pose and not the door's angle: those it has to see.

Each image goes through an encoder -- a small CNN trained from scratch, or (--encoder
resnet18) an ImageNet-pretrained ResNet-18 fine-tuned at a tenth of the learning rate,
whose last feature map is pooled to a few tokens, or (--encoder dinov2_s) a frozen DINOv2
ViT-S/14 whose patch tokens are all kept -- with random-shift augmentation; those tokens, the robot-state tokens
and the noisy action chunk are denoised by the same transformer as sim/dit.py (DDPM
training, DDIM sampling, chunks of H, E executed). The images are encoded once per query
and reused over the DDIM steps.

Data: the same demonstrations as the state DiT, with sim/render_views.py's views.npz
(every --stride-th step rendered); training samples are the rendered steps.

  python sim/dit_vis.py train --data <dir>[,<dir>...] --out <dir> [--epochs 60] [--encoder resnet18 --views views_hq]
  python sim/dit_vis.py eval  --ckpt <dir>/dit_vis.pt --out <dir>/eval --n 128 [--jobs 12]
  python sim/dit_vis.py show  --ckpt <dir>/dit_vis.pt --out <dir>/show --index 0   # one annotated episode
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))
from bc import T_MAX  # noqa: E402
from dit import N_OBS, H, E, K_TRAIN, K_SAMPLE, D_ACT, betas  # noqa: E402

D_PROP = 25                          # qpos 9, qvel 9, tcp pose 7 -- the first 25 of the 33-d state
# controls: "privileged" adds the pot pose (7) and the door angle (1) -- the state DiT's input
PROP_DIMS = {"robot": 25, "privileged": 33, "privileged_rel": 39, "privileged_trim": 28, "privileged_nodoor": 32}


def featurize(o, prop):
    """The policy's non-image input from the 33-d state (qpos 9, qvel 9, tcp pose 7, pot pose
    7, door angle 1), numpy (..., 33) -> (..., PROP_DIMS[prop]). Controls for why images beat
    the privileged state:
      privileged_rel   + tcp - pot and tcp - handle (world frame): the relative geometry
                       the wrist camera sees directly
      privileged_trim  robot + pot x, y + door angle: without the near-constant pot height
                       and quaternion, whose tiny changes normalisation blows up
      privileged_nodoor  robot + pot pose, no door angle: the door angle tracks task progress,
                       a candidate nuisance cue (causal confusion, de Haan et al. 2019)"""
    if prop in ("robot", "privileged", "privileged_nodoor"):
        return o[..., :PROP_DIMS[prop]]
    if prop == "privileged_trim":
        return np.concatenate([o[..., :25], o[..., 25:27], o[..., 32:33]], -1)
    import sim.microwave_env as M
    hx = json.load(open(M.MW_PARAMS))["derived"]["handle_bar_centre_closed"]
    q = o[..., 32]
    c, s_ = np.cos(q), np.sin(q)
    hm = np.stack([c * hx[0] + s_ * hx[1], -s_ * hx[0] + c * hx[1], np.full_like(q, 0.17)], -1)   # door_rot(q) @ handle
    hw = hm @ M.R_W_MW.T + M.T_W_MW[:3, 3]
    tcp, pot = o[..., 18:21], o[..., 25:28]
    return np.concatenate([o, tcp - pot, tcp - hw], -1).astype(np.float32)
FRONT_TOK, WRIST_TOK = (4, 7), (4, 4)


def idle_mask(obs, act, keep=2, still=0.05):
    """sim/bc.py's compress as a mask: drop waiting steps (action unchanged, all joints
    still) beyond the first `keep` of each run."""
    same = np.r_[False, np.all(np.abs(np.diff(act, axis=0)) < 1e-6, axis=1)]
    idle = same & np.all(np.abs(obs[:, 9:18]) < still, axis=1)
    run, m = 0, np.ones(len(act), bool)
    for i, d in enumerate(idle):
        run = run + 1 if d else 0
        m[i] = run <= keep
    return m


def load(data_dirs, views="views", prop="robot"):
    """views may be comma-separated (e.g. two random scenes per run): each is loaded as its
    own copy of the run's samples."""
    if "," in views:
        return sum((load(data_dirs, v, prop) for v in views.split(",")), [])
    return _load_one(data_dirs, views, prop)


def _load_one(data_dirs, views, prop):
    """Per successful run with views: (prop windows (n, N_OBS, 25), actions (n, H, 8),
    front (n, h, w, 3), wrist (n, h, w, 3)) at the rendered steps that are not waiting."""
    import h5py
    out = []
    for d in data_dirs.split(","):
        d, cap = (d.split(":")[0], int(d.split(":")[1])) if ":" in d else (d, None)   # dir:N = first N runs
        s = json.load(open(os.path.join(d, "summary.json")))
        n_used = 0
        for r in s["runs"]:
            rd = os.path.join(d, f"run_{r['run']:03d}")
            if not r["success"] or (views != "none" and not os.path.exists(os.path.join(rd, views + ".npz"))):
                continue
            if cap is not None and n_used >= cap:
                break
            n_used += 1
            if os.path.exists(os.path.join(rd, "demo.npz")):
                z = np.load(os.path.join(rd, "demo.npz"))
                o, ac = z["obs"].astype(np.float32), z["actions"].astype(np.float32)
            else:
                with h5py.File(os.path.join(rd, "solve.h5"), "r") as f:
                    o, ac = f["traj_0/obs"][:-1].astype(np.float32), f["traj_0/actions"][:].astype(np.float32)
            keep = idle_mask(o, ac)
            if views == "none":                          # control: every step, no images
                z1 = np.zeros((len(o), 1, 1, 3), np.uint8)
                v = {"idx": np.arange(len(o)), "front": z1, "wrist": z1}
            else:
                v = np.load(os.path.join(rd, views + ".npz"))
            sel = keep[v["idx"]]
            idx = v["idx"][sel]
            # windows on the compressed sequence, as the state DiT is trained (--squeeze)
            oc, acc = o[keep], ac[keep]
            pos = np.cumsum(keep) - 1                    # raw step -> compressed index
            ci = pos[idx]
            T = len(acc)
            oi = np.maximum(ci[:, None] - np.arange(N_OBS)[None, ::-1], 0)
            ai = np.minimum(ci[:, None] + np.arange(H)[None], T - 1)
            a = acc[ai].copy()
            a[..., :7] -= oc[ci][:, None, :7]
            out.append((featurize(oc[oi], prop), a, v["front"][sel], v["wrist"][sel]))
    return out


class Enc(nn.Module):
    """Small CNN -> a pooled grid of tokens."""

    def __init__(self, d, grid):
        super().__init__()
        ch = [3, 32, 64, 128, 128]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1), nn.GroupNorm(8, ch[i + 1]), nn.SiLU(),
                       nn.Conv2d(ch[i + 1], ch[i + 1], 3, padding=1), nn.GroupNorm(8, ch[i + 1]), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.proj = nn.Linear(ch[-1], d)

    def forward(self, img):                              # (B, h, w, 3) uint8
        x = img.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        f = self.pool(self.net(x))                       # (B, C, gh, gw)
        return self.proj(f.flatten(2).transpose(1, 2))   # (B, gh*gw, d)


class ResEnc(nn.Module):
    """ImageNet-pretrained ResNet-18 up to layer4 (stride 32) -> a pooled grid of tokens.
    Fine-tuned at a lower learning rate than the rest (see train)."""

    def __init__(self, d, grid):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.proj = nn.Linear(512, d)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):                              # (B, h, w, 3) uint8
        x = (img.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
        f = self.pool(self.net(x))
        return self.proj(f.flatten(2).transpose(1, 2))


class DinoEnc(nn.Module):
    """DINOv2 ViT-S/14, frozen: every 14 x 14 patch stays a token (no pooling), so the
    transformer can attend to the few patches the pot or the handle is in. The wrist
    image is resized to 126 x 126 (9 x 9 patches); the front's 126 x 224 is 9 x 16."""

    def __init__(self, d, size):
        super().__init__()
        self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, verbose=False)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.size = size                                 # (h, w), multiples of 14
        self.proj = nn.Linear(384, d)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()                                  # frozen: no dropout / drop-path
        return self

    def forward(self, img):
        x = (img.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
        if tuple(x.shape[-2:]) != self.size:
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False, antialias=True)
        with torch.no_grad():
            tok = self.net.forward_features(x)["x_norm_patchtokens"]
        return self.proj(tok)


DINO_SIZE = {"front": (126, 224), "wrist": (126, 126)}


def n_tokens(encoder):
    if encoder == "none":
        return 0
    if encoder == "dinov2_s":
        return sum(h // 14 * w // 14 for h, w in DINO_SIZE.values())
    return FRONT_TOK[0] * FRONT_TOK[1] + WRIST_TOK[0] * WRIST_TOK[1]


def erase(img, p):
    """Random erasing: with probability p per image, one rectangle (5-25 % of the image,
    aspect 0.3-3) is filled with a random colour -- no single region can be relied on."""
    if p <= 0:
        return img
    B, h, w, _ = img.shape
    dev = img.device
    area = torch.empty(B, device=dev).uniform_(0.05, 0.25) * h * w
    ar = torch.exp(torch.empty(B, device=dev).uniform_(np.log(0.3), np.log(3.3)))
    rh = (area * ar).sqrt().clamp(1, h).long()
    rw = (area / ar).sqrt().clamp(1, w).long()
    y0 = (torch.rand(B, device=dev) * (h - rh + 1)).long()
    x0 = (torch.rand(B, device=dev) * (w - rw + 1)).long()
    rows = torch.arange(h, device=dev)[None, :, None]
    cols = torch.arange(w, device=dev)[None, None, :]
    m = (rows >= y0[:, None, None]) & (rows < (y0 + rh)[:, None, None]) \
        & (cols >= x0[:, None, None]) & (cols < (x0 + rw)[:, None, None]) \
        & (torch.rand(B, device=dev) < p)[:, None, None]
    col = torch.randint(0, 256, (B, 1, 1, 3), device=dev, dtype=img.dtype)
    return torch.where(m[..., None], col, img)


def color_aug(img, on):
    """Per image: brightness, contrast and saturation x U(0.7, 1.3), a per-channel gain
    U(0.9, 1.1) (white balance), and with p 0.3 a 3x3 box blur."""
    if not on:
        return img
    B = img.shape[0]
    dev = img.device
    x = img.float() / 255.0
    u = lambda lo, hi, *s_: torch.empty(B, *s_, device=dev).uniform_(lo, hi)
    x = x * u(0.7, 1.3, 1, 1, 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * u(0.7, 1.3, 1, 1, 1) + m
    g = x.mean(-1, keepdim=True)
    x = (x - g) * u(0.7, 1.3, 1, 1, 1) + g
    x = x * u(0.9, 1.1, 1, 1, 3)
    blur = (torch.rand(B, device=dev) < 0.3)
    if blur.any():
        xb = x[blur].permute(0, 3, 1, 2)
        xb = F.avg_pool2d(F.pad(xb, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        x[blur] = xb.permute(0, 2, 3, 1)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def shift(img, pad):
    """Random shift augmentation (DrQ): pad by replication, crop back at a random offset."""
    B, h, w, _ = img.shape
    x = F.pad(img.permute(0, 3, 1, 2).float(), (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (B,), device=img.device)
    oy = torch.randint(0, 2 * pad + 1, (B,), device=img.device)
    ar_h = torch.arange(h, device=img.device)
    ar_w = torch.arange(w, device=img.device)
    rows = (oy[:, None] + ar_h[None])[:, :, None].expand(B, h, w)
    cols = (ox[:, None] + ar_w[None])[:, None, :].expand(B, h, w)
    out = x[torch.arange(B, device=img.device)[:, None, None], :, rows, cols]   # (B, h, w, 3)
    return out.round().clamp(0, 255).to(torch.uint8)


class VisDiT(nn.Module):
    def __init__(self, d=256, layers=4, heads=4, encoder="cnn", d_prop=D_PROP):
        super().__init__()
        self.encoder, self.d_prop = encoder, d_prop
        if encoder == "none":                            # control: no images at all
            self.front = self.wrist = None
        elif encoder == "dinov2_s":
            self.front, self.wrist = DinoEnc(d, DINO_SIZE["front"]), DinoEnc(d, DINO_SIZE["wrist"])
        else:
            E_ = ResEnc if encoder == "resnet18" else Enc
            self.front, self.wrist = E_(d, FRONT_TOK), E_(d, WRIST_TOK)
        self.prop_in = nn.Linear(d_prop, d)
        self.k_in = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.act_in = nn.Linear(D_ACT, d)
        n_img = n_tokens(encoder)
        self.pos = nn.Parameter(torch.randn(1, N_OBS + n_img + 1 + H, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D_ACT))
        for n, dim in (("p_mu", d_prop), ("p_sd", d_prop), ("a_mu", D_ACT), ("a_sd", D_ACT)):
            self.register_buffer(n, torch.zeros(dim))
        self.register_buffer("beta", torch.tensor(betas()))

    def cond(self, prop, front, wrist, prop_noise=0.0):
        pn = (prop - self.p_mu) / self.p_sd
        if prop_noise > 0:
            pn = pn + prop_noise * torch.randn_like(pn)
        if self.front is None:
            return self.prop_in(pn)
        return torch.cat([self.prop_in(pn), self.front(front), self.wrist(wrist)], 1)

    def forward(self, a_noisy, k, c):
        x = torch.cat([c, self.k_in((k.float()[:, None] / K_TRAIN))[:, None], self.act_in(a_noisy)], 1)
        y = self.tf(x + self.pos)
        return self.out(y[:, -H:])

    @torch.no_grad()
    def sample(self, prop, front, wrist, x_init=None):
        """x_init: the starting noise (B, H, D_ACT); fixed across queries, DDIM (eta 0) is then a
        deterministic function of the observation, so consecutive chunks agree more."""
        c = self.cond(prop, front, wrist)
        ab = torch.cumprod(1 - self.beta, 0)
        x = torch.randn(len(prop), H, D_ACT, device=prop.device) if x_init is None else x_init.clone()
        ks = torch.linspace(K_TRAIN - 1, 0, K_SAMPLE, device=prop.device).round().long()
        for i, k in enumerate(ks):                       # DDIM, eta = 0
            eps = self(x, k.repeat(len(prop)), c)
            a0 = ((x - (1 - ab[k]).sqrt() * eps) / ab[k].sqrt()).clamp(-5, 5)
            x = a0 if i == len(ks) - 1 else ab[ks[i + 1]].sqrt() * a0 + (1 - ab[ks[i + 1]]).sqrt() * eps
        return x * self.a_sd + self.a_mu


def train(a):
    eps = load(a.data, a.views, a.prop)
    n = sum(len(e[0]) for e in eps)
    print(f"{len(eps)} runs with views, {n:,} samples", flush=True)
    rng = np.random.default_rng(0)
    val = set(rng.permutation(len(eps))[:a.n_val].tolist())
    cat = lambda j, s: np.concatenate([e[j] for i, e in enumerate(eps) if (i in val) == s])
    dev = torch.device("cuda")
    P, A = (torch.tensor(cat(j, False), device=dev) for j in range(2))
    Pv, Av, Fv, Wv = (torch.tensor(cat(j, True), device=dev) for j in range(4))
    # training images: on the GPU, or (--images-cpu) in pinned host memory, one batch at a time
    if a.images_cpu:
        Fr, Wr = (torch.from_numpy(cat(j, False)).pin_memory() for j in (2, 3))
    else:
        Fr, Wr = (torch.tensor(cat(j, False), device=dev) for j in (2, 3))
    img = lambda T_, b: T_[b.cpu()].to(dev, non_blocking=True) if a.images_cpu else T_[b]
    dp = PROP_DIMS[a.prop]
    m = VisDiT(d=a.width, layers=a.layers, heads=a.heads, encoder=a.encoder, d_prop=dp).to(dev)
    if a.init:                                        # continue from a checkpoint (its EMA weights)
        c0 = torch.load(a.init, map_location=dev)
        m.load_state_dict(c0["state"])
        print(f"initialised from {a.init}", flush=True)
    m.p_mu.copy_(P.reshape(-1, dp).mean(0)); m.p_sd.copy_(P.reshape(-1, dp).std(0) + 1e-3)
    m.a_mu.copy_(A.reshape(-1, D_ACT).mean(0)); m.a_sd.copy_(A.reshape(-1, D_ACT).std(0) + 1e-3)
    An = (A - m.a_mu) / m.a_sd
    ab = torch.cumprod(1 - m.beta, 0)
    # EMA of the weights (Diffusion Policy's schedule: decay 1 - (1 + step)^-0.75, at most
    # 0.9999); evaluation and the saved checkpoints use the averaged weights
    import copy
    ema = copy.deepcopy(m).eval() if a.ema else None
    if ema is not None:
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    n_step = 0
    # a pretrained encoder is fine-tuned 10x slower than the freshly initialised parts
    enc = [p for n, p in m.named_parameters() if n.startswith(("front.net", "wrist.net")) and p.requires_grad]
    rest = [p for n, p in m.named_parameters() if not n.startswith(("front.net", "wrist.net"))]
    lr_enc = a.lr * (0.1 if a.encoder == "resnet18" else 1.0)
    groups = ([dict(params=enc, lr=lr_enc)] if enc else []) + [dict(params=rest, lr=a.lr)]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    if a.steps:                                       # a step budget instead of an epoch count
        a.epochs = int(round(a.steps / (len(P) // a.batch)))
    steps = a.epochs * (len(P) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.05)
    print(f"{len(P):,} train / {len(Pv):,} val samples; {steps} steps; encoder {a.encoder}, views {a.views}", flush=True)
    for ep in range(a.epochs):
        perm = torch.randperm(len(P), device=dev)
        for i in range(0, len(P) - a.batch + 1, a.batch):
            b = perm[i:i + a.batch]
            k = torch.randint(0, K_TRAIN, (len(b),), device=dev)
            noise = torch.randn_like(An[b])
            x = ab[k].sqrt()[:, None, None] * An[b] + (1 - ab[k]).sqrt()[:, None, None] * noise
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.fp32):
                c = m.cond(P[b], *((erase(color_aug(shift(img(Fr, b), 6), a.color_aug), a.erase),
                                    erase(color_aug(shift(img(Wr, b), 4), a.color_aug), a.erase))
                                   if m.front is not None else (None, None)), prop_noise=a.prop_noise)
                loss = ((m(x, k, c).float() - noise) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            n_step += 1
            if ema is not None:
                dcy = min(0.9999, 1 - (1 + n_step) ** -0.75)
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), m.parameters()):
                        if pm.requires_grad:
                            pe.mul_(dcy).add_(pm.detach(), alpha=1 - dcy)
                    for be, bm in zip(ema.buffers(), m.buffers()):
                        be.copy_(bm)
        mv = ema if ema is not None else m                   # the model that is evaluated / saved
        if a.save_every and (ep + 1) % a.save_every == 0 and ep < a.epochs - 1:
            os.makedirs(a.out, exist_ok=True)
            torch.save(dict(state=mv.state_dict(), encoder=a.encoder, views=a.views, prop=a.prop, fp32=a.fp32,
                            ema=a.ema, step=n_step, width=a.width, layers=a.layers, heads=a.heads),
                       os.path.join(a.out, f"dit_vis_ep{ep + 1:04d}.pt"))
        if ep % 5 == 4 or ep == a.epochs - 1:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.fp32):
                sel = torch.randint(0, len(Pv), (1024,), device=dev)
                pred = mv.sample(Pv[sel], Fv[sel], Wv[sel]).float()
                err = (pred - Av[sel])[:, 0, :7].abs().mean().item()
                gerr = (pred[:, :, 7].clamp(-1, 1).round() != Av[sel][:, :, 7].round()).float().mean().item()
            print(f"epoch {ep + 1:3d}: noise loss {loss.item():.4f}  val first-step joint error "
                  f"{np.degrees(err):.2f} deg, gripper cmd wrong {gerr * 100:.1f}%", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(dict(state=mv.state_dict(), encoder=a.encoder, views=a.views, prop=a.prop, fp32=a.fp32,
                    ema=a.ema, step=n_step, width=a.width, layers=a.layers, heads=a.heads),
               os.path.join(a.out, "dit_vis.pt"))
    print("->", os.path.join(a.out, "dit_vis.pt"))


def prop_of(obs, prop="robot"):
    if not isinstance(obs, dict):                        # obs_mode "state": the flat 33-d vector
        return torch.from_numpy(np.asarray(featurize(obs[0].cpu().numpy(), prop), np.float32))
    parts = [obs["agent"]["qpos"], obs["agent"]["qvel"], obs["extra"]["tcp_pose"]]
    if prop != "robot":
        parts += [obs["extra"]["pot_pose"], obs["extra"]["door_qpos"]]
        o = torch.cat(parts, 1)[0].cpu().numpy()
        return torch.from_numpy(np.asarray(featurize(o, prop), np.float32))
    return torch.cat(parts, 1)[0].cpu().float()


def images(obs, dev, bg=None, wsplat=None):
    """The policy's two images at its resolution (supersampled renders are area-downsampled).
    bg = (image, ids, depth): the front camera's pixels of those segmentation ids (the ground and
    empty space) replaced by the image -- a real photo taken from the same camera, or a render
    of the splatted kitchen (recon/gs_kitchen.py render --keep-all); with depth (m, the
    render's), also every pixel where the background is more than 5 cm nearer than what the
    simulator drew there. The splat renders are used without depth: nothing real stands
    between these cameras and the simulated objects, and a splat's depth is unreliable
    exactly where the front view is the only view.
    wsplat: a sim/gs_client.SplatRenderer -- the wrist camera's ground / empty pixels replaced
    by the splat rendered from its current pose (the same segmentation ids as bg)."""
    if not isinstance(obs, dict):
        return None, None
    from sim.views import downsample
    sd = obs["sensor_data"]
    out = []
    for n in ("front", "wrist"):
        img = sd[n]["rgb"][0].cpu().numpy()
        if n == "front" and bg is not None:
            m = np.isin(sd[n]["segmentation"][0, ..., 0].cpu().numpy(), bg[1])
            if bg[2] is not None:
                d = sd[n]["depth"][0, ..., 0].cpu().numpy().astype(np.float32) / 1000
                m |= (d > 0) & (bg[2] < d - 0.05)      # 5 cm: the splat's counter sits ~1-2 cm high
            img = img.copy()
            img[m] = bg[0][m]
        if n == "wrist" and wsplat is not None:
            from sim.gs_client import wrist_background
            m = np.isin(sd[n]["segmentation"][0, ..., 0].cpu().numpy(), bg[1])
            img = img.copy()
            img[m] = wrist_background(obs, wsplat)[m]
        out.append(torch.tensor(downsample(img, n))[None].to(dev))
    return tuple(out)


def real_background(path, env):
    """(the background at the front camera's render size, RGB; the segmentation ids it replaces;
    its depth or None). path: a photo, or an .npz from recon/gs_kitchen.py render (rgb, depth)."""
    import cv2
    h, w = env.unwrapped._sensors["front"].config.height, env.unwrapped._sensors["front"].config.width
    ids = [0, *env.unwrapped.ground.per_scene_id.tolist()]
    if path.endswith(".npz"):
        z = np.load(path)
        assert z["rgb"].shape[1:3] == (h, w), f"render {z['rgb'].shape[1:3]} for a {h}x{w} camera"
        return z["rgb"][0], ids, None
    img = cv2.cvtColor(cv2.resize(cv2.imread(path), (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return img, ids, None


def bg_obs_mode(path):
    return "rgb+segmentation"


def load_model(ckpt, dev):
    c = torch.load(ckpt, map_location=dev)
    prop = c.get("prop", "robot")
    m = VisDiT(d=c.get("width", 256), layers=c.get("layers", 4), heads=c.get("heads", 4),
               encoder=c.get("encoder", "cnn"), d_prop=PROP_DIMS[prop]).to(dev)
    m.load_state_dict(c["state"])
    m.eval()
    m.prop = prop
    m.fp32 = c.get("fp32", False)
    # hq cameras for anything rendered with them: views_hq, and the --dr renders (views_dr, views_drm<k>)
    return m, any(v == "views_hq" or v.startswith("views_dr") for v in c.get("views", "views").split(","))


def rollout_retry(args, tries=3):
    """rollout(), retried when Vulkan fails to come up (seen on the Jetson Thor: an occasional
    vk::createInstanceUnique ErrorIncompatibleDriver after ~100 episodes ended the whole eval)."""
    for k in range(tries):
        try:
            return rollout(args)
        except RuntimeError as e:
            if "vk::" not in str(e) or k == tries - 1:
                raise
            print(f"  retry {args[2]} after: {e}", flush=True)
            time.sleep(5)


def rollout(args):
    ckpt, pot, seed, video_dir = args[:4]
    infer = args[4] if len(args) > 4 else "chunk"
    n_exec = args[6] if len(args) > 6 and args[6] else E    # actions executed per query (trained chunk: H)
    backdrop = args[5] if len(args) > 5 else None
    pot_z = args[7] if len(args) > 7 else 0.0               # pot start height (0: on the counter)
    real_bg = args[8] if len(args) > 8 else None            # a photo behind the front camera's sim objects
    splat_wrist = args[9] if len(args) > 9 else None        # a splat checkpoint behind the wrist camera's
    import gymnasium as gym
    import sim.microwave_env as M  # noqa: F401
    torch.set_num_threads(2)
    dev = torch.device("cuda")
    m, hq = load_model(ckpt, dev)
    vis = m.front is not None
    # no images and no video: no renderer at all (a Vulkan device per process ran a 30-process
    # eval out of GPU memory)
    norender = {} if (vis or video_dir) else dict(render_backend="none")
    env = gym.make("MicrowavePot-v0", obs_mode=(bg_obs_mode(real_bg) if real_bg else "rgb") if vis else "state",
                   control_mode="pd_joint_pos",
                   pot_mw=(*pot, np.pi / 2), views=("hq" if hq else True) if vis else False,
                   render_mode="rgb_array" if video_dir else None, sim_backend="cpu",
                   max_episode_steps=T_MAX, backdrop=backdrop, pot_z=pot_z, **norender)
    if video_dir:
        from mani_skill.utils.wrappers.record import RecordEpisode
        env = RecordEpisode(env, video_dir, save_video=True, save_trajectory=False, video_fps=30,
                            max_steps_per_video=T_MAX)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    bg = real_background(real_bg, env) if real_bg else None
    wsplat = None
    if splat_wrist:
        from sim.gs_client import SplatRenderer
        wsplat = SplatRenderer(splat_wrist)
    hist = [prop_of(obs, m.prop)] * N_OBS
    door_max, pot_max, info, t = 0.0, 0.0, {}, 0
    tcp_path, tcp_prev, tcp_zmax, stops, still = 0.0, None, 0.0, 0, 0
    # the grasp: every time the gripper command flips to close with the tcp within 15 cm of the pot
    closes, grip_prev = [], 1.0
    torch.manual_seed(seed)
    x_fix = torch.randn(1, H, D_ACT, device=dev) if infer == "fixed_noise" else None

    def predict():
        fr_, wr_ = images(obs, dev, bg, wsplat)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not m.fp32):
            return m.sample(torch.stack(hist[-N_OBS:])[None].to(dev), fr_, wr_, x_fix)[0].float().cpu().numpy()

    # infer: "chunk" -- query every E steps, execute E of the H actions (as trained/evaluated so far);
    # "fixed_noise" -- the same, from one fixed starting noise per episode; "ensemble" -- query
    # every step and execute, per step, the exponentially weighted mean of every chunk's
    # prediction for that step (ACT's temporal ensembling, weights exp(-0.01 * age), oldest first)
    preds = {}                                           # step -> list of absolute 8-d actions, oldest first
    done = False
    while t < T_MAX and not done:
        q = hist[-1][:7].numpy()
        acts = predict()
        absa = np.c_[q + acts[:, :7], acts[:, 7]]
        if infer == "ensemble":
            for h in range(H):
                preds.setdefault(t + h, []).append(absa[h])
            P_ = np.array(preds.pop(t))
            w = np.exp(-0.01 * np.arange(len(P_)))
            a_t = (w[:, None] * P_).sum(0) / w.sum()
            steps = [a_t]
        else:
            steps = absa[:n_exec]
        for a_t in steps:
            act = np.r_[a_t[:7], float(np.clip(np.round(a_t[7]), -1, 1))]
            obs, _, _, _, info = env.step(act)
            hist.append(prop_of(obs, m.prop))
            t += 1
            door_max = max(door_max, float(info["door_deg"][0]))
            pot_max = max(pot_max, float(u.pot_in_mw()[0, 2]))
            tcp = u.tcp_in_mw()[0].cpu().numpy()
            if tcp_prev is not None:
                sp = float(np.linalg.norm(tcp - tcp_prev))
                tcp_path += sp
                still = still + 1 if sp < 0.001 else 0
                stops += still == 10                     # stood still for 10 steps (as sim/rl_residual.Tracker)
            tcp_prev, tcp_zmax = tcp, max(tcp_zmax, float(tcp[2]))
            if act[7] < 0 <= grip_prev:
                pp = u.pot_in_mw()[0].cpu().numpy()
                if np.linalg.norm(tcp - pp) < 0.15:
                    closes.append(dict(t=t, rel_cm=np.round((tcp - pp) * 100, 2).tolist(),
                                       door=round(float(info["door_deg"][0]), 1)))
            grip_prev = act[7]
            if bool(info["success"][0]):
                done = True
                break
    p = u.pot_in_mw().cpu().numpy()[0]
    res = dict(pot=list(pot), seed=seed, steps=t, door_opened=door_max > 60, door_max_deg=door_max,
               grasped_lifted=pot_max > 0.05 + pot_z, pot_max_z=pot_max, pot_inside=bool(info["pot_inside"][0]),
               door_shut=bool(info["door_shut"][0]), success=bool(info["success"][0]), pot_final=p.round(3).tolist(),
               tcp_path_m=round(tcp_path, 3), tcp_zmax=round(tcp_zmax, 3), stops=int(stops), closes=closes)
    env.close()
    if wsplat is not None:
        wsplat.close()
    return res


def show(a):
    """One episode as a video of what the policy sees: front and wrist images (its only
    inputs besides the robot's own joints), a third-person view, and the episode's state
    per step (from the simulator, for the viewer -- the policy never gets it)."""
    import subprocess
    import cv2
    import gymnasium as gym
    import sim.microwave_env as M  # noqa: F401
    gen = json.load(open(os.path.join(a.data.split(",")[0], "summary.json")))["runs"]
    pots = np.array([r["pot"] for r in gen])
    rng = np.random.default_rng(a.seed)
    pot = [rng.uniform(pots.min(0), pots.max(0)).round(4).tolist() for _ in range(a.index + 1)][-1]
    seed = 1000 + a.index
    dev = torch.device("cuda")
    m, hq = load_model(a.ckpt, dev)
    env = gym.make("MicrowavePot-v0", obs_mode=bg_obs_mode(a.real_bg) if a.real_bg else "rgb", control_mode="pd_joint_pos",
                   pot_mw=(*pot, np.pi / 2), views="hq" if hq else True, render_mode="rgb_array", sim_backend="cpu",
                   max_episode_steps=T_MAX, backdrop=a.backdrop)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    bg = real_background(a.real_bg, env) if a.real_bg else None
    hist = [prop_of(obs, m.prop)] * N_OBS
    torch.manual_seed(seed)
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, f"episode_{a.index:03d}_raw.avi")
    W = 640 + 448
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, 427))
    flags = dict(door=False, lifted=False, inside=False, success=False)
    t, info, grip = 0, {}, 1.0
    while t < T_MAX and not flags["success"]:
        q = hist[-1][:7].numpy()
        fr_, wr_ = images(obs, dev, bg)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not m.fp32):
            acts = m.sample(torch.stack(hist[-N_OBS:])[None].to(dev), fr_, wr_)[0].float().cpu().numpy()
        for k in range(E):
            grip = float(np.clip(np.round(acts[k, 7]), -1, 1))
            obs, _, _, _, info = env.step(np.r_[q + acts[k, :7], grip])
            hist.append(prop_of(obs, m.prop))
            t += 1
            door, pz = float(info["door_deg"][0]), float(u.pot_in_mw()[0, 2])
            flags["door"] |= door > 60
            flags["lifted"] |= pz > 0.05
            flags["inside"] |= bool(info["pot_inside"][0])
            flags["success"] |= bool(info["success"][0])
            if t % 2 == 0 or flags["success"]:
                fr_, wr_ = images(obs, dev, bg)
                fr = cv2.resize(fr_[0].cpu().numpy(), (448, 252), interpolation=cv2.INTER_NEAREST)
                wr = cv2.resize(wr_[0].cpu().numpy(), (175, 175), interpolation=cv2.INTER_NEAREST)
                tp = cv2.resize(env.render()[0].cpu().numpy(), (640, 427))
                right = np.zeros((427, 448, 3), np.uint8)
                right[:252] = fr
                right[252:, :175] = wr
                txt = [f"step {t}  ({t / 20:.1f} s)", f"gripper cmd {grip:+.0f}", f"door {door:5.1f} deg",
                       f"pot z {pz * 100:4.1f} cm",
                       "  ".join(k for k, v in flags.items() if v) or "-"]
                for j, line in enumerate(txt):
                    cv2.putText(right, line, (185, 280 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                cv2.putText(right, "policy input: front", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(right, "wrist", (6, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(tp, "third person (not a policy input)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 0), 1)
                vw.write(cv2.cvtColor(np.hstack([tp, right]), cv2.COLOR_RGB2BGR))
            if flags["success"]:
                break
    vw.release()
    out = raw.replace("_raw.avi", ".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out],
                   check=True)
    os.remove(raw)
    print(json.dumps(dict(pot=pot, seed=seed, steps=t, **flags)))
    print("->", out)


def evaluate(a):
    """Same start states as sim/dit.py eval (pot region of --data's first dir, --seed)."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    gen = json.load(open(os.path.join(a.data.split(",")[0], "summary.json")))["runs"]
    pots = np.array([r["pot"] for r in gen])
    lo, hi = pots.min(0), pots.max(0)
    if a.pot_shift:                                     # out-of-distribution starts: the region moved (mw frame, m)
        sh = np.array([float(v) for v in a.pot_shift.split(",")])
        lo, hi = lo + sh, hi + sh
    rng = np.random.default_rng(a.seed)
    jobs = [(a.ckpt, rng.uniform(lo, hi).round(4).tolist(), 1000 + i,
             os.path.join(a.out, f"video_{i:02d}") if i < a.videos else None, a.infer, a.backdrop, a.exec,
             a.pot_z, a.real_bg, a.splat_wrist) for i in range(a.n)]
    os.makedirs(a.out, exist_ok=True)
    from concurrent.futures import as_completed
    keys = ["door_opened", "grasped_lifted", "pot_inside", "door_shut", "success"]
    res = [None] * len(jobs)
    # --tasks-per-child: a fresh pool every jobs x tasks-per-child episodes (on the Thor every worker's
    # Vulkan stops coming up after ~9 episodes in the same process, and retrying in that process does
    # not help; Python 3.12.3's own max_tasks_per_child deadlocked there after 40 episodes)
    per_pool = a.jobs * a.tasks_per_child if a.tasks_per_child else len(jobs)
    n_done = 0
    for c0 in range(0, len(jobs), per_pool):
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
            fut = {ex.submit(rollout_retry, jobs[i]): i for i in range(c0, min(c0 + per_pool, len(jobs)))}
            for f in as_completed(fut):    # stream: every episode as it finishes
                n_done += 1
                i = fut[f]
                res[i] = f.result()
                done = [r for r in res if r is not None]
                run = {k: float(np.mean([r[k] for r in done])) for k in keys}
                print(f"[{n_done:3d}/{len(jobs)}] episode {i:3d}: "
                      + ("SUCCESS" if res[i]["success"] else "  ".join(k for k in keys if res[i][k]) or "nothing")
                      + "   | running: " + "  ".join(f"{k} {v * 100:.0f}%" for k, v in run.items()), flush=True)
                json.dump(dict(partial=True, n=len(done), rates=run), open(os.path.join(a.out, "partial.json"), "w"))
    for i, r in enumerate(res):
        print(f"  {i:3d} pot {r['pot']}: door max {r['door_max_deg']:5.1f}  pot max z {r['pot_max_z']:.3f}  "
              + "  ".join(k for k in keys if r[k]), flush=True)
    summ = {k: float(np.mean([r[k] for r in res])) for k in keys}
    print("rates:", {k: f"{v * 100:.0f}%" for k, v in summ.items()}, f"(n={len(res)})")
    ok = [r for r in res if r["success"]]
    if ok and "tcp_path_m" in ok[0]:
        motion = {k: round(float(np.mean([r[k] for r in ok])), 3) for k in ("steps", "tcp_path_m", "tcp_zmax", "stops")}
        print("successful episodes, mean:", motion)
        summ["motion_successful"] = motion
    json.dump(dict(rates=summ, episodes=res), open(os.path.join(a.out, "eval.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "show"])
    ap.add_argument("--data", default=os.path.join(ROOT, "Gitignore-but-important/sim/gen"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=0, help="train: about this many gradient steps (overrides --epochs)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--width", type=int, default=256, help="train: transformer width (and image token size)")
    ap.add_argument("--layers", type=int, default=4, help="train: transformer layers")
    ap.add_argument("--heads", type=int, default=4, help="train: attention heads")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--tasks-per-child", type=int, default=0, help="eval: episodes per worker process (0: unlimited)")
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--index", type=int, default=0, help="show: which eval episode (same start state)")
    ap.add_argument("--encoder", choices=["cnn", "resnet18", "dinov2_s", "none"], default="cnn",
                    help="none: no images (control)")
    ap.add_argument("--fp32", action="store_true", help="train and sample in fp32 instead of bf16 autocast")
    ap.add_argument("--erase", type=float, default=0.0, help="random-erasing probability per training image")
    ap.add_argument("--color-aug", action="store_true", help="brightness/contrast/saturation/white-balance jitter + blur")
    ap.add_argument("--images-cpu", action="store_true", help="keep training images in pinned host memory")
    ap.add_argument("--prop-noise", type=float, default=0.0, help="std of Gaussian noise on the normalised state (training)")
    ap.add_argument("--ema", action="store_true", help="exponential moving average of the weights (as Diffusion Policy)")
    ap.add_argument("--pot-z", type=float, default=0.0, help="eval: the pot's start height (m), e.g. on top of the microwave")
    ap.add_argument("--pot-shift", default="", help="eval: dx,dy -- move the pot's start region (microwave frame, metres)")
    ap.add_argument("--backdrop", help="eval/show: a visual-only scene mesh (recon/kitchen_backdrop.py) instead of the grey ground")
    ap.add_argument("--real-bg", help="eval/show: behind the front camera's sim objects, in place of the grey ground: a "
                    "photo from its pose (recon/takes/<take>/rest_frame.png) or a splat render with depth "
                    "(recon/gs_kitchen.py render, .npz); the wrist camera is unchanged")
    ap.add_argument("--splat-wrist", help="eval: a splat checkpoint (recon/gs_kitchen.py, trained --inpainted) rendered "
                    "behind the wrist camera's sim objects every step (needs --real-bg for the ids)")
    ap.add_argument("--exec", type=int, default=0, help="eval: actions executed per query (default E = 8 of H = 16)")
    ap.add_argument("--infer", choices=["chunk", "fixed_noise", "ensemble"], default="chunk",
                    help="eval: how chunks become actions (see rollout)")
    ap.add_argument("--init", help="train: start from this checkpoint's weights (e.g. fine-tune on re-rendered views)")
    ap.add_argument("--save-every", type=int, default=0, help="also save a checkpoint every N epochs (for closed-loop selection)")
    ap.add_argument("--prop", choices=list(PROP_DIMS), default="robot",
                    help="privileged: also the pot pose and door angle (control)")
    ap.add_argument("--views", default="views",
                    help="views (minimal shader), views_hq (sim/render_views.py --hq), none (control: every step, no images)")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "show": show}[a.cmd](a)


if __name__ == "__main__":
    main()
