"""Client of recon/gs_server.py: the splatted kitchen behind a moving camera (the wrist camera).

    r = SplatRenderer(ckpt)                 # starts the server in the sam3d env
    img = r.render(w2c_mw, K, width, height)
"""
import os
import struct
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("SPLAT_PYTHON", os.path.expanduser("~/miniforge3/envs/sam3d/bin/python"))


class SplatRenderer:
    def __init__(self, ckpt):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_LIBRARY_PATH")}
        self.p = subprocess.Popen([PY, os.path.join(ROOT, "recon/gs_server.py"), ckpt], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, env=env)
        assert self.p.stdout.readline().strip() == b"ok", "splat server did not start"

    def render(self, w2c, K, width, height):
        self.p.stdin.write(np.asarray(w2c, np.float64).tobytes() + np.asarray(K, np.float64).tobytes()
                           + struct.pack("<ii", width, height))
        self.p.stdin.flush()
        n = width * height * 3
        buf = b""
        while len(buf) < n:
            buf += self.p.stdout.read(n - len(buf))
        return np.frombuffer(buf, np.uint8).reshape(height, width, 3)

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def wrist_background(obs, renderer):
    """the splat from the wrist camera's current pose, at its render size"""
    import sim.microwave_env as M
    p = obs["sensor_param"]["wrist"]
    E = np.eye(4)
    E[:3] = p["extrinsic_cv"][0].cpu().numpy()
    h, w = obs["sensor_data"]["wrist"]["rgb"].shape[1:3]
    return renderer.render(E @ M.T_W_MW, p["intrinsic_cv"][0].cpu().numpy(), w, h)
