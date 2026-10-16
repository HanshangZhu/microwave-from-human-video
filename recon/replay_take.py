#!/usr/bin/env python
"""Side-by-side replay of a take: the recorded frame with the detected hand skeleton,
and the simulation scene rendered from the same camera (textured microwave URDF,
the 21 hand keypoints as spheres, joined by bones, and the Snap Pot where
pot_track.py put it). If registration and the hand's
3D placement are right, the two halves line up frame by frame.

The door follows door_track.py's angle per frame (<anchor>/door.npz) when it exists;
otherwise it stays at the angle the still view registered.

Runs in the env with sapien (maniskill); writes an H.264 mp4 via ffmpeg:
  python recon/replay_take.py <capture dir> <tag> --anchor <dir> [--fps 15]
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np
import sapien

LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def pot_path(anchor, kpm, valid, n_frames):
    """Pot pose per frame (x, y, z of its base centre, yaw) from pot_track.py: on the
    counter where the lid was seen resting; carried with the hand (its grip point
    minus half a pot) between pick and place; on the turntable after the place."""
    jp, tp = os.path.join(anchor, "pot.json"), os.path.join(anchor, "pot_track.npz")
    if not (os.path.exists(jp) and os.path.exists(tp)):
        return None
    info, tr = json.load(open(jp)), np.load(tp)
    fps = 1 / (tr["t"][1] - tr["t"][0])
    i_pick = int(round(info["t_pick"] * fps)) if info["t_pick"] is not None else n_frames
    i_place = int(round(info["t_place"] * fps)) if info["t_place"] is not None else n_frames
    P = np.full((n_frames, 4), np.nan)
    lid = tr["lid"]
    for i in range(n_frames):
        if i < i_pick and np.isfinite(lid[i, 0]):
            P[i] = [lid[i, 0], lid[i, 1], 0.0, lid[i, 2]]
        elif i_pick <= i < i_place and valid[i]:
            g = (kpm[i, 4] + kpm[i, 8]) / 2                  # thumb-index grip point
            P[i] = [g[0], g[1], max(g[2] - 0.035, 0.0), info["pick"]["yaw"]]
        elif i >= i_place and info["end"] is not None:
            e = info["end"]
            P[i] = [e["x"], e["y"], e["z"], info["pick"]["yaw"]]
    P[0] = P[0] if np.isfinite(P[0, 0]) else [info["start"]["x"], info["start"]["y"], 0, info["start"]["yaw"]]
    t = np.arange(n_frames)
    for c in range(4):                                       # fill gaps by interpolation
        ok = np.isfinite(P[:, c])
        P[:, c] = np.interp(t, t[ok], P[ok, c])
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("tag")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--urdf", default="assets/microwave/microwave_textured.urdf")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--pot-glb", default="assets/heinz_snap_pot/visual.glb")
    ap.add_argument("--w", type=int, default=640)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.capture, a.tag + ".json")))
    ci = meta["intrinsics"]
    src_fps = meta.get("fps", 30)
    h = np.load(os.path.join(a.anchor, "hand.npz"))
    kp2, kpm, valid = h["kp2d"], h["kp3d_model"], h["valid"]
    reg = json.load(open(os.path.join(a.anchor, "run", "registration.json")))
    T_last_model = np.linalg.inv(np.array(reg["T_model_to_world"]))
    segs = json.load(open(os.path.join(a.anchor, "anchor.json")))["segments"]

    def cam_to_model(i):
        sg = next((s for s in segs if s["start"] <= i < s["end"]), segs[-1])
        return T_last_model @ np.array(sg["T_last_from_cam"])

    W, H = a.w, int(a.w * ci["height"] / ci["width"])
    s = W / ci["width"]
    scene = sapien.Scene()
    scene.set_ambient_light([0.8, 0.8, 0.8])
    scene.add_directional_light([0.2, 0.6, -1], [0.4, 0.4, 0.4])
    scene.add_ground(0, render_material=sapien.render.RenderMaterial(base_color=[0.72, 0.72, 0.7, 1]))
    ld = scene.create_urdf_loader()
    ld.fix_root_link = True
    mw = ld.load(a.urdf)
    mw.set_qpos([reg["joint_angle_rad"]])
    balls = []
    for j in range(21):
        b = scene.create_actor_builder()
        b.add_sphere_visual(radius=0.006 if j in (4, 8) else 0.004, material=sapien.render.RenderMaterial(
            base_color=[1, 0.2, 0.2, 1] if j in (4, 8) else [0.1, 0.9, 0.2, 1]))
        balls.append(b.build_kinematic(name=f"kp{j}"))
    cam = scene.add_camera("cam", W, H, 2 * np.arctan(ci["height"] / 2 / ci["fy"]), 0.02, 5)

    pot = pot_path(a.anchor, kpm, valid, n_frames=len(valid))
    if pot is not None:
        b = scene.create_actor_builder()
        b.add_visual_from_file(a.pot_glb)
        pot_actor = b.build_kinematic(name="pot")
    dpath = os.path.join(a.anchor, "door.npz")
    door_q = np.load(dpath)["q"] if os.path.exists(dpath) else None
    cap = cv2.VideoCapture(os.path.join(a.capture, a.tag + ".avi"))
    n = len(valid)
    tmp = os.path.join(a.anchor, "replay_raw.avi")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), a.fps, (2 * W, H))
    K = np.array([[ci["fx"] * s, 0, ci["ppx"] * s], [0, ci["fy"] * s, ci["ppy"] * s], [0, 0, 1]])
    for i in range(0, n, max(1, src_fps // a.fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, H))
        C = cam_to_model(i)                                  # OpenCV camera -> model
        G = C[:3, :3] @ np.diag([1, -1, -1])
        T = np.eye(4)
        T[:3, :3] = np.stack([-G[:, 2], -G[:, 0], G[:, 1]], 1)
        T[:3, 3] = C[:3, 3]
        cam.entity.set_pose(sapien.Pose(T))
        if door_q is not None:
            mw.set_qpos([door_q[min(i, len(door_q) - 1)]])
        if pot is not None:
            x, y, z, yaw = pot[i]
            pot_actor.set_pose(sapien.Pose([x, y, z], [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        for j, b in enumerate(balls):
            b.set_pose(sapien.Pose(kpm[i, j] if valid[i] else [0, 0, -10]))
        scene.update_render()
        cam.take_picture()
        r = (cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)[..., ::-1].copy()
        if valid[i]:
            w2c = np.linalg.inv(C)
            P = kpm[i] @ w2c[:3, :3].T + w2c[:3, 3]
            uv = (P @ K.T)[:, :2] / P[:, 2:3]
            for p, q in LINKS:
                cv2.line(r, tuple(uv[p].astype(int)), tuple(uv[q].astype(int)), (40, 220, 40), 2)
                cv2.line(f, tuple((kp2[i, p] * s).astype(int)), tuple((kp2[i, q] * s).astype(int)),
                         (40, 220, 40), 2)
        cv2.putText(f, f"{a.tag}  {i / src_fps:5.1f}s  recorded" + ("" if valid[i] else "  (no hand)"),
                    (8, 24), 0, 0.6, (0, 255, 255), 2)
        cv2.putText(r, "simulation, same camera" + (f"  door {np.degrees(door_q[min(i, len(door_q) - 1)]):3.0f} deg"
                                                    if door_q is not None else ""), (8, 24), 0, 0.6, (0, 255, 255), 2)
        vw.write(np.hstack([f, r]))
    vw.release()
    out = os.path.join(a.anchor, "replay.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "23", out], check=True)
    os.remove(tmp)
    print("replay ->", out)


if __name__ == "__main__":
    main()
