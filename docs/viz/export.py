"""Export the real data behind the raw-video -> training-chunk pipeline for the explainer page."""
import base64
import json
import os
import sys

import cv2
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import sim.microwave_env as M  # noqa: E402
from sim import human_paths  # noqa: E402
from sim.dit_vis import idle_mask  # noqa: E402

G = "Gitignore-but-important"
OUT = sys.argv[1]
R = lambda a, n=4: np.round(np.asarray(a, float), n).tolist()  # noqa: E731


def jpg(img_rgb, q=72):
    ok, b = cv2.imencode(".jpg", img_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(b.tobytes()).decode()


def to_mw(p_world):
    return (np.asarray(p_world) - M.T_W_MW[:3, 3]) @ M.R_W_MW   # R^T (p - t)


D = {}
P = json.load(open(M.MW_PARAMS))["derived"]
D["mw"] = dict(turntable=P["turntable_centre_xy"], cavity_x=P["cavity_x_range"], cavity_y=P["cavity_y_range"],
               handle=P["handle_bar_centre_closed"], robot=M.ROBOT_IN_MW.tolist(), width=0.46, depth=0.32,
               door_len=0.356)

# ---- takes: tracking time series (10 Hz) + events
takes = {}
for tk in ("take_000", "take_001", "take_002"):
    T = f"{G}/recon/takes/{tk}"
    h, d, pt = np.load(f"{T}/hand.npz"), np.load(f"{T}/door.npz"), np.load(f"{T}/pot_track.npz")
    pj = json.load(open(f"{T}/pot.json"))
    s = slice(0, None, 3)
    hv = h["valid"].astype(bool)
    ee = np.where(hv[:, None], h["ee"], np.nan)
    lid = pt["lid"]
    ins = pt["inside"]
    hp = human_paths.load(tk)
    takes[tk] = dict(
        t=R(h["t"][s], 2), door=R(np.rad2deg(d["q"][s]), 1), door_raw=R(np.rad2deg(d["q_raw"][s]), 1),
        hand=[None if np.isnan(p[0]) else R(p, 3) for p in ee[s]],
        lid=[None if np.isnan(p[0]) else R(p[:2], 3) for p in lid[s]],
        inside=[None if np.isnan(p[0]) else R(p, 3) for p in ins[s]],
        lifted=[bool(x) for x in pt["lifted"][s]],
        start=R([pj["start"]["x"], pj["start"]["y"]], 3), pick=R([pj["pick"]["x"], pj["pick"]["y"]], 3),
        place=R([pj["end"]["x"], pj["end"]["y"]], 3), t_pick=pj["t_pick"], t_place=pj["t_place"],
        approach=None if hp["approach"] is None else R(hp["approach"], 3),
        carry=None if hp["carry"] is None else R(hp["carry"], 3),
        valid_frac=float(hv.mean()))
D["takes"] = takes

# ---- take_000 raw frames every 0.5 s with the 2D hand keypoints
cap = cv2.VideoCapture(f"{G}/laptop/microwave01/take_000.avi")
h0 = np.load(f"{G}/recon/takes/take_000/hand.npz")
frames, kps, i = [], [], 0
W0 = H0 = None
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i % 15 == 0:
        H0, W0 = fr.shape[:2]
        small = cv2.resize(fr, (384, int(384 * H0 / W0)), interpolation=cv2.INTER_AREA)
        frames.append(jpg(small[..., ::-1], 62))
        kps.append(R(h0["kp2d"][i] / [W0, H0], 4) if h0["valid"][i] else None)
    i += 1
D["video"] = dict(frames=frames, kp=kps, dt=0.5, size=[W0, H0])
print("frames", len(frames), W0, H0, sum(len(f) for f in frames) // 1024, "KB")

# ---- generator sampling (gen) + regions
gs = json.load(open(f"{G}/sim/gen/summary.json"))["runs"]
D["gen"] = [dict(pot=r["pot"], place=r["place"], ok=r["success"]) for r in gs]

# ---- A vs B on one shared configuration (human take_000, approach + carry, both successful)
hs = json.load(open(f"{G}/sim/genh/summary.json"))["runs"]
pick_run = next(r["run"] for r in hs if r["success"] and r.get("human") == "take_000"
                and r["human_parts"] == {"approach": True, "carry": True} and gs[r["run"]]["success"])


def traj(d):
    with h5py.File(f"{d}/solve.h5") as f:
        o, a = f["traj_0/obs"][:-1], f["traj_0/actions"][:]
    tcp = to_mw(o[:, 18:21])
    pot = to_mw(o[:, 25:28])
    return o, a, tcp, pot


AB = {}
for k, d in (("A", f"{G}/sim/gen/run_{pick_run:03d}"), ("B", f"{G}/sim/genh/run_{pick_run:03d}")):
    o, a, tcp, pot = traj(d)
    g = a[:, 7]
    AB[k] = dict(tcp=R(tcp[::4], 3), pot=R(pot[::4], 3), grip=R(g[::4], 2), door=R(np.rad2deg(o[::4, 32]), 1))
AB["run"] = pick_run
AB["cfg"] = dict(pot=gs[pick_run]["pot"], place=gs[pick_run]["place"])
D["ab"] = AB

# ---- DART: one perturbed demo, executed joints vs recorded (clean) targets
z = np.load(f"{G}/sim/gen_dart02/run_000/demo.npz")
o, a = z["obs"], z["actions"]
D["dart"] = dict(q=R(o[:1600:2, :7], 4), a=R(a[:1600:2, :7], 4), noise=float(z["dart_noise"]))
o2, a2, _, _ = traj(f"{G}/sim/gen/run_000")
D["clean"] = dict(q=R(o2[:1600:2, :7], 4), a=R(a2[:1600:2, :7], 4))

# ---- chunking: gen/run_000 full action/joint arrays, idle mask, render indices, thumbnails
keep = idle_mask(o2, a2)
v = np.load(f"{G}/sim/gen/run_000/views_hq.npz")
idx = v["idx"]
thumb_every = 6
th = {}
for j in range(0, len(idx), thumb_every):
    th[int(idx[j])] = dict(front=jpg(v["front"][j], 70), wrist=jpg(v["wrist"][j], 70))
D["chunk"] = dict(q=R(o2[:, :7], 3), grip_q=R(o2[:, 7], 4), a=R(a2, 3), keep=[int(x) for x in keep],
                  idx=[int(x) for x in idx], thumbs=th, N_OBS=2, H=16, E=8, n=int(len(a2)))
print("thumbs", len(th), sum(len(t["front"]) + len(t["wrist"]) for t in th.values()) // 1024, "KB")

# ---- numbers
D["counts"] = dict(A=[58, 34, 4], B=[55, 29, 2], gen=[60, 80, 40], samples_A=56387)
json.dump(D, open(OUT, "w"), separators=(",", ":"))
print("total", os.path.getsize(OUT) // 1024, "KB; run", pick_run)
