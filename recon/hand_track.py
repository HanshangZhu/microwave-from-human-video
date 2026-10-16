#!/usr/bin/env python
"""The demonstrator's hand through a take, in the microwave's (= simulation) frame.

Per frame: MediaPipe's hand landmarker gives 21 keypoints in the image and the same
21 in metres around the hand's centre ("world" landmarks, camera-aligned axes).
The measured depth, aligned to colour, places the hand: at the five palm keypoints
(wrist and the four knuckles, the flattest and least occluded part) the depth gives a
3D point each, and the world landmarks are shifted onto them (median offset). The
camera pose comes from take_anchor.py (segments, if it was knocked) and
register_model.py on the take's still view.

A gripper is read off the hand: position = midpoint of thumb tip and index tip,
aperture = their distance, approach axis = wrist -> middle knuckle, closing axis =
index tip -> thumb tip made orthogonal to it.

Writes <anchor dir>/hand.npz and hand_check.png (keypoints over frames, and the
trajectory around the microwave from the top and the side).

Runs in the env with mediapipe (handtrack):
  python recon/hand_track.py <capture dir> <tag> --anchor <dir from take_anchor.py>
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_to_r2s import align_depth  # noqa: E402

PALM = [0, 5, 9, 13, 17]
MODEL = os.path.expanduser("~/.cache/handtrack/hand_landmarker.task")


def depth_at(z, u, v, r=4):
    h, w = z.shape
    u, v = int(round(u)), int(round(v))
    if not (r <= u < w - r and r <= v < h - r):
        return np.nan
    win = z[v - r:v + r + 1, u - r:u + r + 1]
    win = win[win > 0]
    return float(np.median(win)) if len(win) > 10 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("tag")
    ap.add_argument("--anchor", required=True)
    a = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    meta = json.load(open(os.path.join(a.capture, a.tag + ".json")))
    ci, d = meta["intrinsics"], meta["depth"]
    R = np.array(d["depth_to_color"]["rotation_colmajor"]).reshape(3, 3).T
    t = np.array(d["depth_to_color"]["translation_m"])
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["ppx"], ci["ppy"]
    fps = meta.get("fps", 30)
    ddir = os.path.join(a.capture, d["folder"])

    reg = json.load(open(os.path.join(a.anchor, "run", "registration.json")))
    T_last_model = np.linalg.inv(np.array(reg["T_model_to_world"]))   # last still camera -> model
    segs = json.load(open(os.path.join(a.anchor, "anchor.json")))["segments"]

    def cam_to_model(i):
        sg = next(s for s in segs if s["start"] <= i < s["end"]) if i < segs[-1]["end"] else segs[-1]
        return T_last_model @ np.array(sg["T_last_from_cam"])

    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4))
    cap = cv2.VideoCapture(os.path.join(a.capture, a.tag + ".avi"))
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(os.listdir(ddir)))
    kp2 = np.full((n, 21, 2), np.nan)
    kpm = np.full((n, 21, 3), np.nan)
    score = np.zeros(n)
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=np.ascontiguousarray(bgr[..., ::-1])), int(i * 1000 / fps))
        if not res.hand_landmarks:
            continue
        # the largest hand in the image (one hand is used)
        k = max(range(len(res.hand_landmarks)), key=lambda j: np.ptp(
            [p.x for p in res.hand_landmarks[j]]) * np.ptp([p.y for p in res.hand_landmarks[j]]))
        P2 = np.array([[p.x * ci["width"], p.y * ci["height"]] for p in res.hand_landmarks[k]])
        Wl = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[k]])
        z16 = cv2.imread(os.path.join(ddir, f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        za = align_depth(z16, d["depth_scale_m"], d["intrinsics"], ci, R, t)
        off = []
        for j in PALM:
            zz = depth_at(za, *P2[j])
            if np.isfinite(zz) and 0.15 < zz < 2.0:
                p3 = np.array([(P2[j, 0] - cx) / fx * zz, (P2[j, 1] - cy) / fy * zz, zz])
                off.append(p3 - Wl[j])
        if len(off) < 3:
            continue
        Xc = Wl + np.median(off, 0)
        T = cam_to_model(i)
        kp2[i], kpm[i] = P2, Xc @ T[:3, :3].T + T[:3, 3]
        score[i] = res.handedness[k][0].score
    lm.close()

    valid = np.isfinite(kpm[:, 0, 0])
    thumb, index, wrist, mid = kpm[:, 4], kpm[:, 8], kpm[:, 0], kpm[:, 9]
    ee = (thumb + index) / 2
    aperture = np.linalg.norm(thumb - index, axis=1)
    approach = mid - wrist
    approach /= np.linalg.norm(approach, axis=1, keepdims=True)
    close = thumb - index
    close -= (close * approach).sum(1, keepdims=True) * approach
    close /= np.linalg.norm(close, axis=1, keepdims=True)
    out = os.path.join(a.anchor, "hand.npz")
    np.savez(out, t=np.arange(n) / fps, valid=valid, kp2d=kp2, kp3d_model=kpm, score=score,
             ee=ee, aperture=aperture, approach=approach, closing=close)
    print(f"{a.tag}: hand found in {valid.sum()}/{n} frames ({valid.mean()*100:.0f}%); "
          f"ee x {np.nanmin(ee[:, 0]):.2f}..{np.nanmax(ee[:, 0]):.2f}, y {np.nanmin(ee[:, 1]):.2f}.."
          f"{np.nanmax(ee[:, 1]):.2f}, z {np.nanmin(ee[:, 2]):.2f}..{np.nanmax(ee[:, 2]):.2f} m; "
          f"aperture {np.nanpercentile(aperture, 5)*100:.1f}-{np.nanpercentile(aperture, 95)*100:.1f} cm -> {out}",
          flush=True)
    check(a, cap, kp2, ee, aperture, valid, fps, n)


def check(a, cap, kp2, ee, aperture, valid, fps, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    links = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
             (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
    tiles = []
    for i in np.linspace(int(2 * fps), n - 1, 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if valid[i]:
            for p, q in links:
                cv2.line(f, tuple(kp2[i, p].astype(int)), tuple(kp2[i, q].astype(int)), (0, 255, 0), 3)
        cv2.putText(f, f"{i / fps:.1f}s", (10, 40), 0, 1.3, (0, 255, 255), 3)
        tiles.append(cv2.resize(f, (480, 270)))
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])

    fig, ax = plt.subplots(1, 3, figsize=(19.2, 5.4), dpi=100)
    tt = np.arange(n) / fps
    box = dict(x=[0, 0.46, 0.46, 0, 0], y=[-0.036, -0.036, 0.315, 0.315, -0.036])
    sc = ax[0].scatter(ee[valid, 0], ee[valid, 1], c=tt[valid], s=4, cmap="viridis")
    ax[0].plot(box["x"], box["y"], "k-")
    ax[0].plot([0.03, 0.32], [0, 0], "k:")
    ax[0].set(title="top view (model frame)", xlabel="x m", ylabel="y m (into microwave)", aspect="equal")
    ax[1].scatter(ee[valid, 1], ee[valid, 2], c=tt[valid], s=4, cmap="viridis")
    ax[1].plot([-0.036, 0.315, 0.315, -0.036, -0.036], [0, 0, 0.278, 0.278, 0], "k-")
    ax[1].set(title="side view", xlabel="y m", ylabel="z m", aspect="equal")
    ax[2].plot(tt, aperture * 100, ".", ms=2)
    ax[2].set(title="thumb-index aperture", xlabel="s", ylabel="cm")
    fig.colorbar(sc, ax=ax[0], label="s")
    fig.tight_layout()
    fig.canvas.draw()
    plot = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1]
    plot = cv2.resize(plot, (sheet.shape[1], int(plot.shape[0] * sheet.shape[1] / plot.shape[1])))
    p = os.path.join(a.anchor, "hand_check.png")
    cv2.imwrite(p, np.vstack([sheet, plot]))
    print("check ->", p)


if __name__ == "__main__":
    main()
