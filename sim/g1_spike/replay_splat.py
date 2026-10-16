#!/usr/bin/env python
"""Re-film a recorded G1 run (g1_solve.py's trajectory) in the splatted kitchen.

Kinematic: every recorded simulator state (pot, microwave, robot) is set and rendered -- no
physics is re-run, so this is exactly the recorded run. Cameras: sim/gs_cams-style look-at
poses in the microwave frame, the splat rendered once per camera (recon/gs_kitchen.py render
--keep-all), the simulator's objects drawn over it by segmentation. The upper-body G1 has no
legs and stands with its pelvis at counter height; whatever of it is below the counter top
is hidden behind the splatted table (depth-tested against z = 0 in the microwave frame).

  python sim/g1_spike/replay_splat.py <run dir with g1_solve.h5> --cams <cams npz> --bg <render npz> \\
      --out <dir> [--every 4]                                                         (env maniskill)
"""
import argparse
import os
import subprocess
import sys

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sim.microwave_env as M  # noqa: E402
import sim.g1_spike.g1_env as G  # noqa: E402
from sim.g1_spike.g1_solve import STANCE  # noqa: E402
from sim.gs_demo import sapien_pose  # noqa: E402
from mani_skill.sensors.camera import CameraConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cams", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--speed", type=float, default=7.0, help="GIF / video speed-up")
    a = ap.parse_args()
    cams = np.load(a.cams)
    names = [str(n) for n in cams["names"]]
    bg = np.load(a.bg)
    W, H = int(cams["width"]), int(cams["height"])

    base = G.MicrowavePotG1Env._default_sensor_configs.fget

    def sensors(self):
        cfg = list(base(self))
        for n, K, w2c in zip(names, cams["K"], cams["w2c"]):
            cfg.append(CameraConfig("demo_" + n, sapien_pose(w2c), W, H, intrinsic=K, near=0.01, far=20,
                                    shader_pack="default"))
        return cfg
    G.MicrowavePotG1Env._default_sensor_configs = property(sensors)
    x, y, z, ry = STANCE
    G.set_g1_stance((x, y), z, ry)
    import gymnasium as gym
    env = gym.make("MicrowavePotG1-v0", obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
                   sim_backend="cpu", max_episode_steps=6000)
    env.reset(seed=0)
    u = env.unwrapped
    ids = [0, *u.ground.per_scene_id.tolist()]
    f = h5py.File(os.path.join(a.run, "g1_solve.h5"), "r")["traj_0"]
    S = f["env_states"]
    T = S["actors/pot"].shape[0]
    art = {k: S["articulations"][k] for k in S["articulations"]}
    T_mw_w = np.linalg.inv(M.T_W_MW)
    Kinv = {n: np.linalg.inv(K) for n, K in zip(names, cams["K"])}
    c2w = {n: np.linalg.inv(w2c) for n, w2c in zip(names, cams["w2c"])}          # camera -> mw
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3).astype(np.float64)
    os.makedirs(a.out, exist_ok=True)
    frames = {n: [] for n in names}
    for t in range(0, T, a.every):
        state = dict(actors={"pot": torch.tensor(S["actors/pot"][t])[None]},
                     articulations={k: torch.tensor(v[t])[None] for k, v in art.items()})
        u.set_state_dict(state)
        obs = u.get_obs()
        door = float(np.degrees(art["microwave"][t][13]))
        for n in names:
            sd = obs["sensor_data"]["demo_" + n]
            rgb = sd["rgb"][0].cpu().numpy().copy()
            seg = sd["segmentation"][0, ..., 0].cpu().numpy()
            d = sd["depth"][0, ..., 0].cpu().numpy().astype(np.float64) / 1000
            m = np.isin(seg, ids)
            # below the counter top: hidden by the splatted table
            pc = (Kinv[n] @ pix.T) * d.reshape(-1)                              # camera (OpenCV)
            pw = c2w[n][:3, :3] @ pc + c2w[n][:3, 3:4]                          # mw frame
            m |= ((d > 0) & (pw[2].reshape(H, W) < -0.01))
            rgb[m] = bg["rgb"][names.index(n)][m]
            frames[n].append((t, door, rgb))
    env.close()
    ff = os.environ.get("FFMPEG", "ffmpeg")
    for n in names:
        d = os.path.join(a.out, n)
        os.makedirs(d, exist_ok=True)
        for i, (t, door, rgb) in enumerate(frames[n]):
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            img[0:30, 0:W] = (img[0:30, 0:W] * 0.35).astype(np.uint8)
            cv2.putText(img, f"Unitree G1 upper body, scripted from take 0's keyframes, in the splatted kitchen "
                        f"({a.speed:.0f}x)   t = {t / 20:5.1f} s   door {door:5.1f} deg", (10, 21),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(d, f"{i:04d}.png"), img)
        fps = 20.0 / a.every * a.speed
        subprocess.run([ff, "-v", "error", "-y", "-framerate", f"{fps:.2f}", "-i", os.path.join(d, "%04d.png"),
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(a.out, f"{n}.mp4")], check=True)
        print("->", os.path.join(a.out, f"{n}.mp4"), len(frames[n]), "frames", flush=True)


if __name__ == "__main__":
    main()
