"""One still image of the G1 scene (the default stance or --stance x y z yaw).

  python sim/g1_spike/still.py --out Gitignore-but-important/g1_spike/scene.png
"""
import argparse
import os
import sys

import gymnasium as gym
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import sim.g1_spike.g1_env as G  # noqa: E402
import sim.g1_spike.g1_solve as GS  # noqa: E402

STANCE = GS.STANCE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stance", type=float, nargs=4, default=STANCE)
    a = ap.parse_args()
    x, y, z, yaw = a.stance
    G.set_g1_stance((x, y), z, yaw)
    env = gym.make("MicrowavePotG1-v0", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                   render_mode="rgb_array", pot_mw=(0.480, -0.241, GS.POT_YAW),
                   render_backend=os.environ.get("G1_RENDER_BACKEND", "gpu"))
    env.reset(seed=0)
    img = env.render()
    img = np.asarray(img.cpu() if hasattr(img, "cpu") else img)[0]
    from PIL import Image
    Image.fromarray(img).save(a.out)
    print("saved", a.out, img.shape)
    env.close()


if __name__ == "__main__":
    main()
