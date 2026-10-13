#!/usr/bin/env python
"""Fix the reference frame once per capture session, with nothing in the scene.

Two things are measured here instead of per clip:

  * the table plane, from a clean frame -- during a press the hand is always
    next to the brick, so a plane fitted from the brick's surroundings mid-clip
    is fitted partly to the hand and its depth shadow;
  * the vertical, from the IMU. A fitted normal off this device wandered over
    28-52 deg across frames; gravity is good to 0.2% and does not care how the
    depth field is distorted.

Writes session.json next to the recordings. extract.py --session reads it.

  python capture/calibrate_session.py --out <session dir>
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def grab_with_imu(w, h, fps, warmup, n_accel=120):
    """One aligned colour+depth frame, plus gravity expressed in the colour frame."""
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.accel)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    acc, got = [], None
    try:
        for i in range(warmup + n_accel):
            fs = pipe.wait_for_frames()
            for fr in fs:
                m = fr.as_motion_frame()
                if m and fr.profile.stream_type() == rs.stream.accel:
                    d = m.get_motion_data()
                    acc.append([d.x, d.y, d.z])
            if i >= warmup:
                a = align.process(fs)
                c, d = a.get_color_frame(), a.get_depth_frame()
                if c and d:
                    got = (np.asanyarray(c.get_data()).copy(),
                           np.asanyarray(d.get_data()).astype(np.float32) * scale,
                           C.intrinsics_dict(c.profile.as_video_stream_profile().intrinsics))
            if len(acc) >= n_accel and got:
                break
        ex = prof.get_stream(rs.stream.accel).get_extrinsics_to(prof.get_stream(rs.stream.color))
        R = np.array(ex.rotation).reshape(3, 3).T      # librealsense stores column-major
    finally:
        pipe.stop()
    # an accelerometer at rest reports specific force, which points UP: flip it
    g_accel = -np.mean(acc, 0)
    g_cam = R @ g_accel
    return got, g_accel, g_cam, scale


def desk_mask(dep, brick, K, radius, band_m=0.10):
    """Pixels plausibly on the desk: near the brick in image space, at a similar
    depth, and not the brick itself."""
    h, w = dep.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cy, cx = np.nonzero(brick)
    u0, v0 = cx.mean(), cy.mean()
    near = ((xs - u0) ** 2 + (ys - v0) ** 2) < radius ** 2
    z0 = np.median(dep[brick > 0][dep[brick > 0] > 0])
    band = (dep > 0) & (np.abs(dep - z0) < band_m)
    off = cv2.dilate(brick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) == 0
    return (near & band & off).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="session directory (session.json goes here)")
    ap.add_argument("--w", type=int, default=848)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--colour", choices=("red", "yellow", "green", "blue"), default="red",
                    help="which object colour band to use (see PALETTE in common.py)")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    default=None, help="crop to the desk; keeps background red out of the search")
    ap.add_argument("--z-range", type=float, nargs=2, default=(0.30, 1.20),
                    metavar=("LO", "HI"), help="plausible brick depth in metres")
    ap.add_argument("--pick", type=int, nargs=2, metavar=("U", "V"), default=None,
                    help="choose the candidate nearest this pixel instead of the largest "
                         "one; largest-blob is not a reliable identity test")
    ap.add_argument("--radius", type=int, default=0,
                    help="desk radius in px around the brick (0 = pick the best of a sweep)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    (bgr, dep, K), g_accel, g_cam, scale = grab_with_imu(a.w, a.h, a.fps, a.warmup)
    gn = g_cam / np.linalg.norm(g_cam)
    print(f"|g| = {np.linalg.norm(g_accel):.3f} m/s^2   gravity in colour frame "
          f"{np.round(gn, 4).tolist()}")

    roi = tuple(a.roi) if a.roi else None
    band = C.PALETTE[a.colour]
    print(f"colour band '{a.colour}': h1={band['h1']} h2={band['h2']} "
          f"s>{band['s_min']} v>{band['v_min']}")
    cands = C.red_candidates(bgr, dep, band, roi=roi)
    print(f"{len(cands)} blob(s) in search region"
          + (f" (roi {roi})" if roi else " (whole frame)") + ":")
    for bm, uv, ar, z in cands[:6]:
        ok = a.z_range[0] <= z <= a.z_range[1] if np.isfinite(z) else False
        print(f"   ({uv[0]:4.0f},{uv[1]:4.0f})  {ar:6d} px  z {z:6.3f} m   "
              f"{'<- in depth band' if ok else 'rejected by depth band'}")
    brick, c, area = C.pick_brick(bgr, dep, band, roi, tuple(a.z_range))
    if not area:
        print("FAIL: no red blob inside the ROI and depth band")
        return 1
    print(f"chosen object: {area} px at ({c[0]:.0f},{c[1]:.0f})")

    best = None
    print(f"{'radius':>8} {'desk px':>9} {'rms mm':>8} {'normal vs gravity':>19}")
    for r in ([a.radius] if a.radius else (120, 180, 250, 330)):
        dm = desk_mask(dep, brick, K, r)
        pts = C.pointcloud(dep, K, dm)
        fit = C.fit_plane(pts)
        if fit is None:
            print(f"{r:8d} {len(pts):9d} {'--':>8} {'--':>19}")
            continue
        n, p0, rms, inl, tot = fit
        ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(n, gn)), 0, 1))))
        print(f"{r:8d} {inl:9d} {rms*1000:8.2f} {ang:18.1f}d")
        if best is None or rms < best[2]:
            best = (n, p0, rms, inl, tot, r, ang)
    obj_pts = C.pointcloud(dep, K, cv2.erode(brick, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5))))
    obj_rest = np.median(obj_pts, 0) if len(obj_pts) >= 20 else None
    if best is None:
        # The plane only ever supplied an origin, and press depth is a difference,
        # so the origin cancels. Fall back to the object's own rest position and
        # keep going -- at a near-horizontal view the desk is edge-on and will
        # never fit well, which is not a reason to refuse to calibrate.
        if obj_rest is None:
            print("FAIL: no plane fit and no object point cloud")
            return 1
        print("plane fit unusable (desk is near edge-on) -- using the object's own "
              "rest position as the origin; the vertical is gravity either way")
        n, p0, rms, inl, tot, r, ang = (-gn, obj_rest, float("nan"), 0, 0, 0, 0.0)
    else:
        n, p0, rms, inl, tot, r, ang = best

    # The vertical we actually use is gravity. The plane only supplies an origin
    # and a sanity check; press depth is a difference, so the origin cancels.
    up = -gn
    sess = dict(
        intrinsics=K, depth_scale_m=scale, size=[a.w, a.h],
        gravity_accel_frame=g_accel.tolist(), gravity_colour_frame=gn.tolist(),
        up_colour_frame=up.tolist(),
        plane=dict(normal=n.tolist(), point=p0.tolist(), rms_mm=rms * 1000,
                   inliers=inl, total=tot, radius_px=r,
                   incidence_deg=C.incidence_deg(n),
                   normal_vs_gravity_deg=ang),
        object_rest=dict(u=c[0], v=c[1], area_px=area,
                         xyz_cam=obj_rest.tolist() if obj_rest is not None else None),
        colour=a.colour, band=band,
        roi=list(roi) if roi else None, z_range=list(a.z_range),
        note="vertical axis = IMU gravity; plane supplies the origin only")
    C.save_json(os.path.join(a.out, "session.json"), sess)

    vis = bgr.copy()
    cnts, _ = cv2.findContours(brick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    cv2.circle(vis, (int(c[0]), int(c[1])), r, (255, 0, 0), 1)
    if roi:
        cv2.rectangle(vis, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (0, 200, 0), 2)
    cv2.imwrite(os.path.join(a.out, "session_ref.png"), vis)

    lever = abs(c[1] - K["ppy"]) / K["fy"]
    print(f"\nheight lever arm (v - ppy)/fy = {lever:.3f}  ->  depth noise contributes "
          f"~{lever * 3.0:.2f} mm to height; centre the object vertically to kill it")
    print(f"best radius {r} px: plane rms {rms*1000:.2f} mm, "
          f"incidence {C.incidence_deg(n):.1f} deg")
    print(f"plane normal vs IMU gravity: {ang:.1f} deg "
          f"({'agree -- both references are sound' if ang < 5 else 'DISAGREE -- the depth plane is distorted; gravity is the one to trust'})")
    npx = int((cv2.erode(brick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0).sum())
    print(f"expected height noise ~{rms*1000/max(np.sqrt(npx),1):.3f} mm over {npx} brick pixels")
    print(f"\nwrote {os.path.join(a.out, 'session.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
