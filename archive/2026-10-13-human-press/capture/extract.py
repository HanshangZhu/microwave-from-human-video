#!/usr/bin/env python
"""Extract a press trajectory from a recorded .db3 (rosbag2).

Per frame: segment the red brick, fit (once) the table plane from a ring of
table pixels around it, then take the *median* signed height of the brick's
visible top-face pixels above that plane. The median over ~10^3 pixels is what
makes a few-millimetre press measurable with a sensor whose single-pixel noise
is ~3 mm.

Writes alongside the recording:
  <tag>.traj.npz      per-frame trajectory
  <tag>.extract.json  plane fit, rest height, thresholds, QA counters
  <tag>.qa.mp4        overlay video
  <tag>.qa.png        height / press-depth plot

  python capture/extract.py <bag>... [--out DIR]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def read_bag(path):
    """Yield (t_s, bgr, depth_metres) for every frame in the bag, in order."""
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device_from_file(path, repeat_playback=False)
    prof = pipe.start(cfg)
    dev = prof.get_device().as_playback()
    dev.set_real_time(False)                 # deliver every frame, drop none
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    K = seen = t0 = None
    try:
        while True:
            ok, fs = pipe.try_wait_for_frames(2000)
            if not ok:
                break
            fs = align.process(fs)
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if not c or not d:
                continue
            # Dedup on the COLOUR frame number: the composite frameset's number
            # is not monotonic under non-realtime playback (observed 4,4,4,1,2,3)
            # and using it silently drops a third of the clip.
            n = c.get_frame_number()
            if seen is not None and n <= seen:
                continue
            seen = n
            if K is None:
                K = C.intrinsics_dict(c.profile.as_video_stream_profile().intrinsics)
                t0 = c.get_timestamp()
            # .copy() is load-bearing: np.asanyarray wraps the frame's buffer
            # without copying, so holding the array holds the frame. Exhaust the
            # frame pool and librealsense drops frames -- silently, mid-clip.
            yield ((c.get_timestamp() - t0) / 1000.0,
                   np.asanyarray(c.get_data()).copy(),
                   np.asanyarray(d.get_data()).astype(np.float32) * scale)
    finally:
        pipe.stop()
    read_bag.K, read_bag.scale = K, scale


def extract(bag, out_dir, hsv=C.RED_DEFAULT, press_mm=2.0, rest_frac=0.15,
            ring=(60, 220), session=None, max_jump=60.0, area_cap=2.5):
    tag = os.path.basename(bag)[:-4]
    frames = list(read_bag(bag))
    K, scale = read_bag.K, read_bag.scale
    if not frames:
        raise RuntimeError(f"{tag}: no frames in bag")

    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    if session:
        hsv = C.band_from_session(session)
    roi = tuple(session["roi"]) if (session and session.get("roi")) else None
    zr = tuple(session["z_range"]) if (session and session.get("z_range")) else None

    def in_band(cs):
        return [c for c in cs
                if zr is None or c[3] is None
                or (np.isfinite(c[3]) and zr[0] <= c[3] <= zr[1])]

    # Anchor the object from the frames before the hand arrives, then TRACK it.
    # Detecting the largest blob globally every frame is what produced 300-560 mm
    # of "drift": a forearm is far bigger than the object and its hue overlaps
    # both bands, so the moment the hand enters, the centroid jumps to the arm.
    # Prefer the anchor the recorder locked during GET READY: at that moment the
    # object is in its new place and the hand has not arrived. Re-deriving it
    # here from the opening frames anchors on whatever is biggest, which in a
    # clip that opens with a hand in shot is the hand -- that inflates the
    # reference area 2-3x and then nothing counts as "unoccluded" afterwards.
    rec_meta = bag[:-4] + ".meta.json"
    anchor = anchor_area = None
    if os.path.exists(rec_meta):
        am = (json.load(open(rec_meta)) or {}).get("anchor") or {}
        if am.get("uv") and am.get("area_px"):
            anchor, anchor_area = np.array(am["uv"], float), float(am["area_px"])
            anchor_src = "recorder (locked in GET READY)"
    if anchor is None:
        # Fall back to the whole clip, not its opening: the object is present in
        # nearly every frame while the hand is only there for part of one, so the
        # median area is the object's and the opening second is the worst sample.
        obs = [(c[1], c[2]) for _, bgr, dep in frames
               for c in in_band(C.red_candidates(bgr, dep, hsv, roi))[:1]]
        if len(obs) < 10:
            raise RuntimeError(f"{tag}: object seen in only {len(obs)} frames")
        ar = np.array([o[1] for o in obs], float)
        anchor_area = float(np.median(ar))
        near = np.abs(ar - anchor_area) < 0.4 * anchor_area
        anchor = np.median(np.array([o[0] for o in obs], float)[near], 0)
        anchor_src = "median over the clip"
    n_gated = [0]

    def pick(bgr, dep):
        cs = in_band(C.red_candidates(bgr, dep, hsv, roi))
        ok = [c for c in cs
              if np.linalg.norm(np.array(c[1]) - anchor) <= max_jump
              and c[2] <= area_cap * anchor_area]
        if not ok:
            n_gated[0] += len(cs) > 0
            return np.zeros(bgr.shape[:2], np.uint8), None, 0
        b = max(ok, key=lambda c: c[2])
        return b[0], b[1], b[2]

    if session:
        # Reference frame comes from calibrate_session: the vertical is IMU
        # gravity (a fitted normal off this device wandered 28-52 deg between
        # frames) and the plane only supplies an origin, which cancels anyway
        # because press depth is a difference. Fitting per clip is impossible:
        # during a press the hand is always inside the ring.
        n = np.array(session["up_colour_frame"], float)
        p0 = np.array(session["plane"]["point"], float)
        rms, inl, tot = session["plane"]["rms_mm"] / 1000, session["plane"]["inliers"], \
            session["plane"]["total"]
        plane_src = "session (IMU gravity vertical)"
    else:
        ring_pts, used = [], 0
        for _, bgr, dep in frames:
            m, c, area = pick(bgr, dep)
            if not area:
                continue
            ring_pts.append(C.pointcloud(dep, K, C.annulus(m, *ring)))
            used += 1
            if used >= 5:
                break
        if not ring_pts:
            raise RuntimeError(f"{tag}: brick never segmented -- check HSV / lighting")
        fit = C.fit_plane(np.concatenate(ring_pts))
        if fit is None:
            raise RuntimeError(f"{tag}: table plane fit failed")
        n, p0, rms, inl, tot = fit
        plane_src = f"per-clip ring {ring}"
    R, origin = C.table_basis(n, p0)

    # ---- per-frame brick height ---------------------------------------------
    rows = []
    for t, bgr, dep in frames:
        m, c, area = pick(bgr, dep)
        if not area or c is None:
            rows.append((t, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        # Erode to drop edge pixels that bleed onto the table -- but only while it
        # leaves most of the blob. A fingertip can take the object down to ~150 px,
        # where a 7x7 erosion removes nearly everything and the frame is lost for
        # no reason; `core.any()` was not a strong enough guard.
        core = cv2.erode(m, erode)
        pts = C.pointcloud(dep, K, core if core.sum() > 0.4 * m.sum() else m)
        if len(pts) < 30:
            rows.append((t, area, len(pts), c[0], c[1], np.nan, np.nan, np.nan))
            continue
        tab = (R @ (pts - origin).T).T       # x, y in-plane; z = height above table
        rows.append((t, area, len(pts), c[0], c[1],
                     float(np.median(tab[:, 0])), float(np.median(tab[:, 1])),
                     float(np.median(tab[:, 2]))))

    a = np.array(rows, float)
    t, area, npx, u, v, x, y, h = a.T

    # ---- rest height and press depth ----------------------------------------
    # Rest height from the brick's UPPER height mode, not from the first frames:
    # a clip can start mid-approach or with the brick still occluded, and a press
    # can only push the brick down, so the top of the distribution is the rest state.
    fin = np.isfinite(h)
    if fin.sum() < 10:
        raise RuntimeError(f"{tag}: only {fin.sum()} frames with a usable object height")
    # Unoccluded area first, then everything else keys off it. A fingertip takes
    # the cap from ~420 px to ~110, and statistics mixing the two are meaningless.
    rest_area = anchor_area if anchor_area else (
        float(np.percentile(area[area > 0], 85)) if (area > 0).any() else 0.0)
    clear = fin & (area >= VIS_FRAC * rest_area)
    if clear.sum() < 5:
        raise RuntimeError(f"{tag}: only {clear.sum()} frames with the object unoccluded")
    hi = h[clear] >= np.percentile(h[clear], 100 * (1 - rest_frac))
    rest_slice = h[clear][hi]
    rest_h, rest_sd = float(np.median(rest_slice)), float(rest_slice.std())
    k = int(rest_slice.size)
    thr = max(press_mm / 1000.0, 3 * rest_sd)
    press = rest_h - h                       # metres, positive = pushed down
    # In-plane drift: a clean press moves the brick down, not sideways. A brick
    # that slides or gets picked up is a demo to throw away, and this is the
    # cheapest way to see it without watching every clip.
    xy_drift = drift_between(x, y, clear, 0, len(t))
    pressed = press > thr

    events = segment_events(t, press, thr, x, y, area, rest_area, clear)

    meta = dict(
        tag=tag, bag=os.path.abspath(bag), frames=len(frames), depth_scale_m=scale,
        intrinsics=K,
        plane=dict(source=plane_src, normal=n.tolist(), point=p0.tolist(),
                   rms_mm=rms * 1000, inliers=inl, total=tot,
                   incidence_deg=C.incidence_deg(n)),
        clear_frames=int(clear.sum()), vis_frac=VIS_FRAC,
        rest=dict(height_m=rest_h, noise_sd_mm=rest_sd * 1000,
                  brick_area_px=rest_area, n_samples=int(k),
                  n_frames_with_height=int(fin.sum())),
        press=dict(threshold_mm=thr * 1000, max_depth_mm=float(np.nanmax(press)) * 1000,
                   n_pressed_frames=int(np.nansum(pressed)),
                   contact_frac=float(np.nansum(pressed) / len(h))),
        tracker=dict(anchor_uv=anchor.tolist(), anchor_area_px=anchor_area,
                     anchor_source=anchor_src,
                     max_jump_px=max_jump, area_cap=area_cap,
                     frames_gated_out=int(n_gated[0])),
        events=events,
        n_events=len(events), n_events_ok=int(sum(e["ok"] for e in events)),
        qa=dict(xy_drift_mm=xy_drift * 1000,
                frames_without_brick=int((area == 0).sum()),
                min_brick_area_px=float(area.min()),
                max_occlusion=float(1 - area[area > 0].min() / rest_area) if rest_area else None,
                brick_px_median=float(np.median(npx[npx > 0])) if (npx > 0).any() else 0.0))

    np.savez(os.path.join(out_dir, tag + ".traj.npz"),
             t_s=t, area_px=area, n_brick_px=npx, u=u, v=v,
             x_tab=x, y_tab=y, h_tab=h, press_m=press, pressed=pressed,
             plane_normal=n, plane_point=p0, R_table=R, origin=origin,
             rest_h=rest_h, threshold_m=thr,
             event_bounds=np.array([[e["i0"], e["i1"]] for e in events], int).reshape(-1, 2),
             event_ok=np.array([e["ok"] for e in events], bool))
    C.save_json(os.path.join(out_dir, tag + ".extract.json"), meta)
    qa_video(bag, frames, hsv, press, pressed, h, os.path.join(out_dir, tag + ".qa.mp4"),
             roi, zr)
    qa_plot(t, press, area, rest_area, thr, tag, os.path.join(out_dir, tag + ".qa.png"),
            events)
    return meta

# A frame counts as "object unoccluded" above this fraction of its clear area.
VIS_FRAC = 0.7


def drift_between(x, y, clear, i0, i1, edge=0.35):
    """How far the object actually moved across a window: the gap between its
    position early and late in that window, both measured on unoccluded frames
    only. Comparing every frame to a median instead measures the fingertip
    crescent, not the object."""
    idx = np.flatnonzero(clear[i0:i1]) + i0
    if idx.size < 6:
        return np.nan
    k = max(2, int(idx.size * edge))
    a, b = idx[:k], idx[-k:]
    pa = np.array([np.median(x[a]), np.median(y[a])])
    pb = np.array([np.median(x[b]), np.median(y[b])])
    return float(np.linalg.norm(pb - pa))


def segment_events(t, press, thr, x, y, area, rest_area, clear, min_dur=0.20, gap=0.35):
    """Split a clip into individual press events.

    A clip is not the unit that matters -- a press is. If the object slides after
    the second of three presses, the first two are still good data, and judging
    the whole clip throws them away. Retargeting consumes single presses anyway,
    so events are also the shape the next stage wants.

    An event spans the descent, the contact and the release: the contiguous run
    above `thr`, grown outwards while the signal stays above 0.3 * thr.
    """
    on = np.nan_to_num(press, nan=-1e9) > thr
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return []
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    merged = []
    for r in runs:
        if merged and t[r[0]] - t[merged[-1][-1]] < gap:
            merged[-1] = np.concatenate([merged[-1], r])
        else:
            merged.append(r)

    lo_thr, out = 0.3 * thr, []
    for r in merged:
        if t[r[-1]] - t[r[0]] < min_dur:
            continue
        a, b = int(r[0]), int(r[-1])
        while a > 0 and np.nan_to_num(press[a - 1], nan=-1e9) > lo_thr:
            a -= 1
        while b < len(t) - 1 and np.nan_to_num(press[b + 1], nan=-1e9) > lo_thr:
            b += 1
        sl = slice(a, b + 1)
        if np.isfinite(press[sl]).sum() < 4:
            continue
        # Press location: where the object sits when nothing is on top of it,
        # taken from the approach window just before this event.
        pre = np.flatnonzero(clear[max(0, a - 45):a]) + max(0, a - 45)
        post = np.flatnonzero(clear[b:min(len(t), b + 45)]) + b
        if pre.size < 3:
            continue
        loc = np.array([np.median(x[pre]), np.median(y[pre])])
        at_end = b >= len(t) - 3          # the press ran into the end of the clip
        drift = (float(np.linalg.norm(
            np.array([np.median(x[post]), np.median(y[post])]) - loc)) * 1000
            if post.size >= 3 else np.nan)
        lost = int((area[sl] == 0).sum())
        n_clear = int(clear[sl].sum())
        n = b - a + 1
        peak = float(np.nanmax(press[sl])) * 1000
        bad = []
        if np.isfinite(drift) and drift > 8:
            bad.append(f"slid {drift:.0f} mm")
        if not np.isfinite(drift) and not at_end:
            bad.append("no clear view after release")
        if lost / n > 0.25:
            bad.append(f"lost {lost}/{n}f")
        if peak < 1.5 * thr * 1000:
            bad.append(f"shallow {peak:.1f} mm")
        out.append(dict(i0=a, i1=b, t0=float(t[a]), t1=float(t[b]),
                        dur_s=float(t[b] - t[a]), peak_mm=peak, drift_mm=drift,
                        lost_frames=lost, n_frames=n, truncated=bool(at_end),
                        x_m=float(loc[0]), y_m=float(loc[1]),
                        ok=not bad, why="; ".join(bad)))
    return out


def qa_video(bag, frames, hsv, press, pressed, h, path, roi=None, zr=None):
    hh, ww = frames[0][1].shape[:2]
    fourcc = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
    vw = cv2.VideoWriter(path, fourcc(*"mp4v"), 30, (ww, hh))
    for i, (t, bgr, dep) in enumerate(frames):
        m, c, area = C.pick_brick(bgr, dep, hsv, roi, zr)
        vis = bgr.copy()
        if area:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
        down = bool(pressed[i]) if np.isfinite(press[i]) else False
        cv2.putText(vis, f"t={t:5.2f}s  press={press[i]*1000:6.2f} mm  area={int(area)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, "CONTACT" if down else "free", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if down else (0, 200, 0), 2)
        bar = int(np.clip(press[i] * 1000 / 15.0, 0, 1) * (hh - 60)) if np.isfinite(press[i]) else 0
        cv2.rectangle(vis, (ww - 40, hh - 30 - bar), (ww - 15, hh - 30), (0, 0, 255), -1)
        vw.write(vis)
    vw.release()


def qa_plot(t, press, area, rest_area, thr, tag, path, events=()):
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, height_ratios=[2, 1])
    ax[0].plot(t, press * 1000, lw=1.5)
    ax[0].axhline(thr * 1000, color="r", ls="--", lw=1, label=f"contact {thr*1000:.1f} mm")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel("press depth (mm)")
    ax[0].legend(loc="upper right", fontsize=8)
    for e in events:
        ax[0].axvspan(e["t0"], e["t1"], color="tab:green" if e["ok"] else "tab:red", alpha=0.15)
        ax[0].annotate(f"{e['peak_mm']:.0f}mm" + ("" if e["ok"] else "\n" + e["why"]),
                       ((e["t0"] + e["t1"]) / 2, e["peak_mm"]), ha="center", va="bottom",
                       fontsize=7, color="k" if e["ok"] else "tab:red")
    ok = sum(e["ok"] for e in events)
    ax[0].set_title(f"{tag}: object displacement   --   {ok}/{len(events)} press events usable")
    ax[1].plot(t, area / rest_area if rest_area else area, lw=1, color="tab:orange")
    ax[1].set_ylabel("brick area\n/ rest")
    ax[1].set_xlabel("t (s)")
    ax[1].axhline(1.0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--out", default=None, help="default: alongside each bag")
    ap.add_argument("--max-jump", type=float, default=60.0,
                    help="px the object may be from its anchor before a frame is dropped")
    ap.add_argument("--session", default=None,
                    help="session.json from calibrate_session.py (strongly recommended)")
    ap.add_argument("--ring", type=int, nargs=2, default=(60, 220),
                    help="inner/outer radius (px) of the desk ring used for the plane")
    ap.add_argument("--press-mm", type=float, default=2.0,
                    help="minimum displacement counted as contact")
    a = ap.parse_args()
    sess = json.load(open(a.session)) if a.session else None
    for b in a.bags:
        out = a.out or os.path.dirname(os.path.abspath(b))
        os.makedirs(out, exist_ok=True)
        try:
            m = extract(b, out, press_mm=a.press_mm, ring=tuple(a.ring), session=sess, max_jump=a.max_jump)
        except Exception as e:
            print(f"[FAIL] {os.path.basename(b)}: {e}")
            continue
        p, q, r, rest = m["plane"], m["qa"], m["press"], m["rest"]
        bad = []
        if q["frames_without_brick"] / m["frames"] > 0.15:
            bad.append(f"object lost {q['frames_without_brick']}/{m['frames']}f")
        if q["xy_drift_mm"] > 15:
            bad.append(f"object slid {q['xy_drift_mm']:.0f} mm")
        if rest["noise_sd_mm"] > 1.5:
            bad.append(f"rest unstable {rest['noise_sd_mm']:.1f} mm")
        if r["max_depth_mm"] < 1.5:
            bad.append(f"no press ({r['max_depth_mm']:.1f} mm)")
        ev = m["events"]
        verdict = f"{m['n_events_ok']}/{m['n_events']} ev" if ev else "0 ev  "
        if not ev:
            bad.append("no press events found")
        print(f"[{verdict}] {m['tag']}  {m['frames']}f  plane {p['rms_mm']:.1f} mm "
              f"@ {p['incidence_deg']:.0f}deg  rest sd {rest['noise_sd_mm']:.2f} mm  "
              f"press max {r['max_depth_mm']:.1f} mm  drift {q['xy_drift_mm']:.1f} mm  "
              f"obj {q['brick_px_median']:.0f} px  gated {m['tracker']['frames_gated_out']}f"
              + ("   <- " + "; ".join(bad) if bad else ""))


if __name__ == "__main__":
    main()
