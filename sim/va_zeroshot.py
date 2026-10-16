#!/usr/bin/env python
"""Zero-shot LingBot-VA (LeRobot's lerobot/lingbot_va_libero_long) on MicrowavePot-v0.

LingBot-VA is a ~5B autoregressive video-action world model (Wan2.2 stack): per chunk it
denoises 4 future video-latent frames of its two cameras and the 16 actions that go with
them, and feeds the really observed frames + the executed actions back into its KV cache.
The LIBERO-Long checkpoint was post-trained on LIBERO (single Franka, 20 Hz OSC control):

  cameras  observation.images.image  = agentview (3rd person), observation.images.image2 = wrist,
           128 x 128, latents concatenated on width
  action   7-d: EEF delta xyz, EEF delta rotation (axis-angle), gripper; in LIBERO's controller
           units, i.e. robosuite OSC_POSE inputs in [-1, 1]: xyz * 0.05 m, rotation * 0.5 rad
           (world frame, pre-multiplied), gripper -1 open / +1 close

Mapping onto our env (no training, nothing tuned to the task):
  front    sim/views.py's RealSense pose ("front") -> observation.images.image
  wrist    sim/views.py's hand camera ("wrist")    -> observation.images.image2
  images   upright as rendered. LeRobot's LIBERO env processor rotates robosuite's raw images by
           180 deg and the checkpoint (image_hflip=True) flips them back horizontally, so the
           model itself sees upright, non-mirrored images; we hand it hflip(upright) exactly as
           the LIBERO env processor would, and keep the checkpoint's config untouched.
  front    is 16:9 (224 x 126); --front square (default) re-renders it from the same pose
           with the same horizontal field of view and a square image (more floor / wall, no
           distortion); --front squash resizes 224 x 126 -> 128 x 128; --front crop centre-crops.
  control  pd_ee_delta_pose (ManiSkill: target = current tcp pose + delta, root-frame
           translation, rotation about root-aligned axes at the tcp -- the same as OSC's).
           xyz: a * 0.05 m -> action a * 0.05 / 0.1. rotation: axis-angle a * 0.5 rad ->
           XYZ Euler angles / ROT_SCALE (ManiSkill scales the normalised rotation by rot_lower
           = -0.1, i.e. -0.1 rad per unit; norm-clipped to 1 => at most 0.1 rad per step).
           gripper: ours +1 open / -1 close -> -a.
  rate     both 20 Hz.

  mkdir -p ext/vklib && ln -sf /lib/aarch64-linux-gnu/libvulkan.so.1 ext/vklib/   # Thor (aarch64)
  LD_LIBRARY_PATH=$PWD/ext/vklib VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  .venv-va/bin/python sim/va_zeroshot.py --episodes 2 --prompt long
On Thor, do NOT put /lib/aarch64-linux-gnu itself on LD_LIBRARY_PATH: it holds the system
cuDNN 9.20, whose sub-libraries then get loaded into torch's cuDNN 9.24 and every conv fails
(CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED). A directory with only libvulkan.so.1 is enough for
SAPIEN to pick the system Vulkan loader (otherwise it falls back to its bundled x86-64 one).
Outputs (per episode) in --out: env video (third person | front | wrist, annotated), the
imagined video (the policy's decoded predicted latents) next to what it really saw, per-step
log (jsonl) and a summary json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CKPT = "lerobot/lingbot_va_libero_long"
PROMPTS = {"long": "open the microwave, put the pot inside the microwave, and close the microwave door",
           "short": "put the pot in the microwave"}
LIBERO_POS, LIBERO_ROT = 0.05, 0.5      # robosuite OSC_POSE output_max (m, rad) per unit input
MS_POS = 0.1                            # Panda pd_ee_delta_pose pos_upper (m per unit action)
ROT_SCALE = -0.1                        # Panda pd_ee_delta_pose: rotation = action * rot_lower (rad); checked by --check-rot
SIZE = 128


# ---- cameras --------------------------------------------------------------------------
def square_front_patch():
    """Replace sim.views.camera_configs' 16:9 front camera by a square one from the same pose
    with the same horizontal field of view (the principal point kept, the extra rows split
    evenly above / below). Patched at runtime; sim/views.py itself is unchanged."""
    import sim.views as V
    from mani_skill.sensors.camera import CameraConfig
    orig = V.camera_configs

    def camera_configs(env, take="take_000", hq=False):
        cams = orig(env, take, hq)
        k = V.SUPERSAMPLE if hq else 1
        w = V.FRONT_W * k
        pose, K = V.front_camera(take, w)
        K = K.copy()
        K[1, 2] += (w - V.FRONT_H * k) / 2.0
        front = CameraConfig("front", pose, w, w, intrinsic=K, near=0.01, far=10,
                             shader_pack="default" if hq else "minimal")
        return [front] + [c for c in cams if c.uid != "front"]

    V.camera_configs = camera_configs


def to_model_image(img, mode):
    """(H, W, 3) uint8 render -> (128, 128, 3) uint8, upright."""
    import cv2
    h, w = img.shape[:2]
    if mode == "crop" and w != h:
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(np.ascontiguousarray(img), (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def to_batch_tensor(img_up):
    """upright uint8 HWC -> [1, 3, H, W] float in [0, 1], horizontally flipped like LeRobot's
    LIBERO env processor output (the checkpoint's image_hflip flips it back)."""
    x = torch.from_numpy(np.ascontiguousarray(img_up[:, ::-1])).permute(2, 0, 1).float() / 255.0
    return x[None]


# ---- action mapping -------------------------------------------------------------------
def libero_to_maniskill(a, rot_scale=ROT_SCALE):
    """a: 7-d LIBERO action (unnormalised = OSC inputs) -> 7-d pd_ee_delta_pose action."""
    from transforms3d.axangles import axangle2mat
    from transforms3d.euler import mat2euler
    a = np.clip(np.asarray(a, np.float64), -1, 1)
    pos = a[:3] * LIBERO_POS / MS_POS
    rv = a[3:6] * LIBERO_ROT
    ang = np.linalg.norm(rv)
    R = axangle2mat(rv / ang, ang) if ang > 1e-9 else np.eye(3)
    # pytorch3d's euler_angles_to_matrix(e, "XYZ") = Rx(e0) @ Ry(e1) @ Rz(e2) = transforms3d "rxyz"
    e = np.array(mat2euler(R, "rxyz"))
    rot = e / rot_scale
    n = np.linalg.norm(rot)
    if n > 1:
        rot = rot / n
    grip = -float(a[6])
    return np.r_[pos, rot, grip].astype(np.float32)


# ---- env -------------------------------------------------------------------------------
def pot_starts(n, seed=1):
    """Pot (x, y) in mw of sim/dit_vis.py eval's episodes 0..n-1 (same rng, same seeds 1000+i)."""
    gen = json.load(open(os.path.join(ROOT, "Gitignore-but-important/sim/gen/summary.json")))["runs"]
    pots = np.array([r["pot"] for r in gen])
    rng = np.random.default_rng(seed)
    return [rng.uniform(pots.min(0), pots.max(0)).round(4).tolist() for _ in range(n)]


def make_env(pot, views, max_steps):
    import gymnasium as gym
    import sim.microwave_env  # noqa: F401  (registers MicrowavePot-v0)
    return gym.make("MicrowavePot-v0", obs_mode="rgb", control_mode="pd_ee_delta_pose",
                    pot_mw=(*pot, np.pi / 2), views=views, render_mode="rgb_array", sim_backend="cpu",
                    max_episode_steps=max_steps + 10)


def cam_images(obs, front_mode):
    sd = obs["sensor_data"]
    fr = sd["front"]["rgb"][0].cpu().numpy()
    wr = sd["wrist"]["rgb"][0].cpu().numpy()
    return to_model_image(fr, front_mode), to_model_image(wr, "squash")


# ---- policy ----------------------------------------------------------------------------
def load_policy(dev):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
    policy = LingBotVAPolicy.from_pretrained(CKPT)
    policy.config.device = str(dev)
    policy.config.save_predicted_video = True
    policy.to(dev)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=CKPT,
                                         preprocessor_overrides={"device_processor": {"device": str(dev)}})
    return policy, pre, post


def check_rot():
    """Which sign does pd_ee_delta_pose give a +1 rotation action? (tcp yaw change per step)."""
    from transforms3d.quaternions import quat2mat
    from transforms3d.axangles import mat2axangle
    env = make_env(pot_starts(1)[0], False, 50)
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    q0 = u.agent.tcp.pose.q[0].cpu().numpy()
    for _ in range(5):
        env.step(np.array([0, 0, 0, 0, 0, 1.0, 1.0], np.float32))
    q1 = u.agent.tcp.pose.q[0].cpu().numpy()
    ax, ang = mat2axangle(quat2mat(q1) @ quat2mat(q0).T)
    print(f"5 steps of rot action +z: world rotation axis {np.round(ax, 3)} angle {np.rad2deg(ang):.1f} deg")
    env.close()


def run_episode(i, pot, seed, prompt, a, policy, pre, post, out):
    import cv2
    import imageio
    tag = f"ep{i:02d}_{a.prompt_name}"
    env = make_env(pot, a.views, a.max_steps)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    policy.reset()
    torch.manual_seed(seed)
    logf = open(os.path.join(out, f"{tag}_steps.jsonl"), "w")
    vw = imageio.get_writer(os.path.join(out, f"{tag}_env.mp4"), fps=20, codec="libx264", quality=7,
                            macro_block_size=1)
    seen, pred_latents, chunk_times = [], [], []
    t, t0 = 0, time.time()
    door_max, pot_zmax, pot0 = 0.0, 0.0, u.pot_in_mw()[0].cpu().numpy()
    tcp0 = u.tcp_in_mw()[0].cpu().numpy()
    handle_mw = None
    tcp_min_pot, info, stop = 1e9, {}, "max_steps"
    while t < a.max_steps:
        if time.time() - t0 > a.budget_min * 60:
            stop = "time_budget"
            break
        fr, wr = cam_images(obs, a.front)
        seen.append(np.hstack([fr, wr]))
        batch = {"observation.images.image": to_batch_tensor(fr), "observation.images.image2": to_batch_tensor(wr),
                 "task": [prompt]}
        batch = pre(batch)
        refill = len(policy._action_queue) == 0
        tc = time.time()
        with torch.inference_mode():
            act = policy.select_action(batch)
        if refill:
            torch.cuda.synchronize()
            chunk_times.append(time.time() - tc)
            print(f"  [{tag}] step {t:4d} chunk {len(chunk_times):3d}: {chunk_times[-1]:.1f} s  "
                  f"door {door_max:.1f} deg  pot z {pot_zmax * 100:.1f} cm", flush=True)
        if policy.last_predicted_latents is not None:
            pred_latents.append(policy.last_predicted_latents)
            policy.last_predicted_latents = None
        a_lib = post(act)[0].float().cpu().numpy()
        a_env = libero_to_maniskill(a_lib)
        obs, _, _, _, info = env.step(a_env)
        t += 1
        door = float(info["door_deg"][0])
        p = u.pot_in_mw()[0].cpu().numpy()
        tcp = u.tcp_in_mw()[0].cpu().numpy()
        door_max, pot_zmax = max(door_max, door), max(pot_zmax, float(p[2]))
        tcp_min_pot = min(tcp_min_pot, float(np.linalg.norm(tcp - p)))
        logf.write(json.dumps(dict(t=t, door_deg=round(door, 2), pot_mw=p.round(4).tolist(), tcp_mw=tcp.round(4).tolist(),
                                   a_libero=a_lib.round(4).tolist(), a_env=a_env.round(4).tolist(),
                                   pot_inside=bool(info["pot_inside"][0]), success=bool(info["success"][0]))) + "\n")
        # env video: third person | front (as fed, upscaled) | wrist, annotated
        tp = cv2.resize(env.render()[0].cpu().numpy(), (480, 320))
        side = np.vstack([cv2.resize(fr, (160, 160), interpolation=cv2.INTER_NEAREST),
                          cv2.resize(wr, (160, 160), interpolation=cv2.INTER_NEAREST)])
        frame = np.hstack([tp, side])
        for j, s in enumerate([f"step {t} ({t / 20:.1f} s)", f"door {door:5.1f} deg  max {door_max:.1f}",
                               f"pot z {p[2] * 100:4.1f} cm  max {pot_zmax * 100:.1f}",
                               f"grip cmd {a_env[6]:+.2f}", f"prompt: {a.prompt_name}"]):
            cv2.putText(frame, s, (6, 18 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        vw.append_data(frame)
        if bool(info["success"][0]):
            stop = "success"
            break
    vw.close()
    logf.close()
    # imagined video: all chunks' predicted latents, decoded once (as lerobot-eval does)
    n_pred = 0
    if pred_latents:
        with torch.inference_mode():
            pv = policy.decode_predicted_latents(torch.cat(pred_latents, dim=2)).numpy()
        n_pred = len(pv)
        imageio.mimwrite(os.path.join(out, f"{tag}_imagined.mp4"), pv, fps=20, codec="libx264", quality=7,
                         macro_block_size=1)
        # side by side: what it saw (top) vs what it imagined (bottom), frame k ~ env step k
        n = min(len(pv), len(seen))
        h, w = seen[0].shape[:2]
        cmp = [np.vstack([seen[k], pv[k] if pv[k].shape[:2] == (h, w) else cv2.resize(pv[k], (w, h))])
               for k in range(n)]
        cmp = [cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2), interpolation=cv2.INTER_NEAREST) for c in cmp]
        for c in cmp:
            cv2.putText(c, "seen: front | wrist", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(c, "imagined", (4, c.shape[0] // 2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        imageio.mimwrite(os.path.join(out, f"{tag}_seen_vs_imagined.mp4"), cmp, fps=20, codec="libx264", quality=7,
                         macro_block_size=1)
    pf = u.pot_in_mw()[0].cpu().numpy()
    res = dict(episode=i, prompt=prompt, pot=pot, seed=seed, steps=t, stop=stop, wall_s=round(time.time() - t0, 1),
               chunks=len(chunk_times), s_per_chunk_mean=round(float(np.mean(chunk_times[1:] or chunk_times)), 2),
               s_per_chunk_first=round(chunk_times[0], 2) if chunk_times else None,
               s_per_env_step=round((time.time() - t0) / max(t, 1), 3),
               door_max_deg=round(door_max, 2), pot_z0=round(float(pot0[2]), 4), pot_zmax=round(pot_zmax, 4),
               pot_moved_cm=round(float(np.linalg.norm(pf[:2] - pot0[:2])) * 100, 2), pot_final=pf.round(3).tolist(),
               tcp_start=tcp0.round(3).tolist(), tcp_final=u.tcp_in_mw()[0].cpu().numpy().round(3).tolist(),
               tcp_min_dist_to_pot=round(tcp_min_pot, 3), pot_inside=bool(info.get("pot_inside", [False])[0]),
               success=bool(info.get("success", [False])[0]), imagined_frames=n_pred, front=a.front)
    json.dump(res, open(os.path.join(out, f"{tag}_summary.json"), "w"), indent=1)
    env.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--first", type=int, default=0, help="first episode index (pot start / seed)")
    ap.add_argument("--prompt", default="long", help="long | short | any instruction")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--budget-min", type=float, default=20.0, help="wall-clock budget per episode")
    ap.add_argument("--front", choices=["square", "squash", "crop"], default="square")
    ap.add_argument("--views", default="hq", help="sim/views.py: True (minimal shader) or hq")
    ap.add_argument("--out", default=os.path.join(ROOT, "Gitignore-but-important/va_zeroshot"))
    ap.add_argument("--check-rot", action="store_true")
    a = ap.parse_args()
    a.views = True if a.views in ("True", "true", "1") else a.views
    a.prompt_name = a.prompt if a.prompt in PROMPTS else "custom"
    prompt = PROMPTS.get(a.prompt, a.prompt)
    if a.front == "square":
        square_front_patch()
    if a.check_rot:
        return check_rot()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    tl = time.time()
    policy, pre, post = load_policy(dev)
    print(f"policy loaded in {time.time() - tl:.0f} s", flush=True)
    pots = pot_starts(a.first + a.episodes)
    allres = []
    for i in range(a.first, a.first + a.episodes):
        r = run_episode(i, pots[i], 1000 + i, prompt, a, policy, pre, post, a.out)
        print(json.dumps(r), flush=True)
        allres.append(r)
    json.dump(allres, open(os.path.join(a.out, f"summary_{a.prompt_name}_{a.first}-{a.first + a.episodes - 1}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
