#!/usr/bin/env python
"""Render the splatted kitchen on request, for a camera that moves (the wrist camera).

The simulator's env (maniskill: torch 2.11) and gsplat's (sam3d: torch 2.7) cannot share a
process, so the simulator starts this as a child (sim/gs_client.py) and talks over pipes:
request = 16 float64 (world(mw)-to-camera, OpenCV) + 9 float64 (K) + 2 int32 (width, height);
reply = height x width x 3 uint8. Every Gaussian is kept (a splat trained on images with the
microwave painted out; see gs_kitchen.py render --keep-all).

  python recon/gs_server.py <ckpt.pt>                                   (env sam3d)
"""
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs_kitchen import EMPTY, render  # noqa: E402


def main():
    dev = torch.device("cuda")
    params = {k: v.to(dev) for k, v in torch.load(sys.argv[1]).items()}
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    out.write(b"ok\n")
    out.flush()
    n = 25 * 8 + 2 * 4
    while True:
        buf = inp.read(n)
        if len(buf) < n:
            return
        v = np.frombuffer(buf[:200], np.float64)
        W, H = struct.unpack("<ii", buf[200:])
        w2c = torch.tensor(v[:16].reshape(4, 4), device=dev, dtype=torch.float32)
        K = torch.tensor(v[16:].reshape(3, 3), device=dev, dtype=torch.float32)
        with torch.no_grad():
            r, _, _ = render(params, w2c[None], K[None], W, H, mode="RGB", bg=EMPTY)
        out.write((r[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy().tobytes())
        out.flush()


if __name__ == "__main__":
    main()
