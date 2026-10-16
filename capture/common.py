"""Shared pieces for scan.py and record.py: streams, the take writer, preview, speech.

A take on disk:
  <tag>.avi          colour, MJPG
  <tag>_depth/       depth, 16-bit lossless PNG, one per colour frame
  <tag>.json         both intrinsics, depth scale, depth->colour extrinsics, depth
                     preset, per-frame colour/depth timestamps, depth-valid fraction

Colour as MJPG and depth as PNG is ~12 MB/s; a raw colour+depth .db3 is ~60 MB/s,
which the laptop's disk cannot hold for a full session.
"""
import json
import os
import queue
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_VALID = 0.6          # fraction of the image with depth; below this, look again
PRESETS = {"default": rs.rs400_visual_preset.default,
           "high_accuracy": rs.rs400_visual_preset.high_accuracy,
           "high_density": rs.rs400_visual_preset.high_density}


def say(text, enabled=True):
    """Spoken cue -- whoever is holding the camera cannot see the screen."""
    if not enabled:
        return
    for cmd in (["spd-say", "-r", "20", text],
                ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"]):
        if shutil.which(cmd[0]):
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return


def start(w, h, fps, depth=True, bag=None, preset="default"):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    if depth or bag:
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)  # native res
    if bag:
        cfg.enable_record_to_file(bag)
    prof = pipe.start(cfg)
    if depth or bag:
        s = prof.get_device().first_depth_sensor()
        if s.supports(rs.option.visual_preset):
            # Right after a stop/start the UVC control can still be held by the
            # previous session ("Device or resource busy"); it frees within ~1 s.
            for attempt in range(20):
                try:
                    s.set_option(rs.option.visual_preset, int(PRESETS[preset]))
                    break
                except RuntimeError as e:
                    if "busy" not in str(e).lower() or attempt == 19:
                        pipe.stop()
                        raise
                    time.sleep(0.1)
    return pipe, prof


def intr_dict(i):
    return dict(fx=i.fx, fy=i.fy, ppx=i.ppx, ppy=i.ppy, width=i.width, height=i.height,
                model=str(i.model), coeffs=list(i.coeffs))


class PngWriter:
    """Background writer so PNG compression never stalls the capture loop."""

    def __init__(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.folder, self.q, self.dropped = folder, queue.Queue(maxsize=600), 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def put(self, idx, z16):
        try:
            self.q.put_nowait((idx, z16))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while (item := self.q.get()) is not None:
            idx, z = item
            cv2.imwrite(os.path.join(self.folder, f"{idx:06d}.png"), z,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def close(self):
        self.q.put(None)
        self.t.join()


def grab(pipe):
    fs = pipe.wait_for_frames()
    c, d = fs.get_color_frame(), fs.get_depth_frame()
    return (np.asanyarray(c.get_data()) if c else None,
            np.asanyarray(d.get_data()) if d else None)


def overlay(bgr, status, colour, z16=None, scale=0.001, big=None, note=None):
    vis = bgr.copy()
    h, w = vis.shape[:2]
    if z16 is not None:
        v = z16 > 0
        frac = float(v.mean())
        med = float(np.median(z16[v]) * scale) if v.any() else 0.0
        d = cv2.applyColorMap(cv2.convertScaleAbs(z16, alpha=255.0 / 2000), cv2.COLORMAP_TURBO)
        d[~v] = 0
        tw, th = w // 3, h // 3
        vis[h - th:, w - tw:] = cv2.resize(d, (tw, th))
        ok = frac >= MIN_VALID and med > 0.2
        cv2.putText(vis, f"depth valid {frac*100:3.0f}%   median {med:.2f} m", (12, 64), FONT,
                    0.7, (0, 220, 0) if ok else (0, 0, 255), 2)
    else:
        frac = None
    cv2.putText(vis, status, (12, 32), FONT, 0.8, colour, 2)
    if note:
        cv2.putText(vis, note, (12, 96), FONT, 0.65, (255, 255, 255), 2)
    if big:
        (tw, th), _ = cv2.getTextSize(big, FONT, 5, 10)
        cv2.putText(vis, big, ((w - tw) // 2, (h + th) // 2), FONT, 5, colour, 10)
    return vis, frac


def quit_key():
    return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)


def timed(pipe, win, secs, label, colour, note=None):
    """Live preview with a big countdown. False if the user quit."""
    t0 = time.time()
    while (left := secs - (time.time() - t0)) > 0:
        bgr, z = grab(pipe)
        if bgr is None:
            continue
        vis, _ = overlay(bgr, label, colour, z, big=str(int(np.ceil(left))), note=note)
        cv2.imshow(win, vis)
        if quit_key():
            return False
    return True


def wait_space(pipe, win, label, note=None):
    """Live preview until SPACE. False if the user quit."""
    while True:
        bgr, z = grab(pipe)
        if bgr is None:
            continue
        vis, _ = overlay(bgr, label, (0, 255, 0), z, note=note)
        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            return True
        if k in (ord("q"), 27):
            return False


def record_take(out, tag, secs, w, h, fps, win, depth=True, raw=False, preset="default",
                speak=True, extra=None):
    """Record one take. Returns (meta, quit)."""
    bag = os.path.join(out, tag + ".db3") if raw else None
    pipe, prof = start(w, h, fps, depth=depth, bag=bag, preset=preset)
    cprof = prof.get_stream(rs.stream.color).as_video_stream_profile()
    dmeta, pngw, scale = None, None, 0.001
    if depth or bag:
        dprof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        ex = dprof.get_extrinsics_to(cprof)
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        dmeta = dict(intrinsics=intr_dict(dprof.intrinsics), depth_scale_m=scale, preset=preset,
                     depth_to_color=dict(rotation_colmajor=list(ex.rotation),
                                         translation_m=list(ex.translation)),
                     folder=tag + "_depth" if (depth and not bag) else None)
        if depth and not bag:
            pngw = PngWriter(os.path.join(out, tag + "_depth"))
    video = os.path.join(out, tag + ".avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter.fourcc(*"MJPG"), fps, (w, h))
    vw.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    stamps, valid, quit_ = [], [], False
    say("go", speak)
    t0 = time.time()
    try:
        while time.time() - t0 < secs:
            fs = pipe.wait_for_frames()
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if not c:
                continue
            bgr = np.asanyarray(c.get_data()).copy()
            vw.write(bgr)
            stamps.append([c.get_frame_number(), c.get_timestamp(),
                           d.get_timestamp() if d else None])
            z = np.asanyarray(d.get_data()).copy() if d else None
            if z is not None and pngw:
                pngw.put(len(stamps) - 1, z)
            vis, frac = overlay(bgr, f"REC {tag}  {secs - (time.time() - t0):.1f}s",
                                (0, 0, 255), z, scale)
            if frac is not None:
                valid.append(frac)
            cv2.imshow(win, vis)
            if quit_key():
                quit_ = True
                break
    finally:
        pipe.stop()
        vw.release()
        if pngw:
            pngw.close()
    say("stop", speak)

    meta = dict(tag=tag, video=os.path.basename(video), bag=os.path.basename(bag) if bag else None,
                n_frames=len(stamps), fps=fps, size=[w, h], duration_s=secs,
                intrinsics=intr_dict(cprof.intrinsics), depth=dmeta,
                depth_frames_dropped=pngw.dropped if pngw else None,
                depth_valid_frac={"median": float(np.median(valid)) if valid else None,
                                  "min": float(np.min(valid)) if valid else None},
                frames_index=["color_frame_number", "color_ts_ms", "depth_ts_ms"],
                frames=stamps, recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))
    json.dump(meta, open(os.path.join(out, tag + ".json"), "w"))

    mb = os.path.getsize(video) / 1e6
    dd = os.path.join(out, tag + "_depth")
    if os.path.isdir(dd):
        mb += sum(os.path.getsize(os.path.join(dd, f)) for f in os.listdir(dd)) / 1e6
    dv = meta["depth_valid_frac"]["median"]
    warn = f"   <- depth only {dv*100:.0f}% valid, check" if dv is not None and dv < MIN_VALID else ""
    drop = f", {pngw.dropped} depth frames dropped" if pngw and pngw.dropped else ""
    print(f"[{tag}] {len(stamps)} frames ({len(stamps)/max(secs,1e-6):.1f} fps), {mb:.0f} MB, "
          f"depth valid {(dv or 0)*100:.0f}%{drop}{warn}", flush=True)
    return meta, quit_


def disk_ok(path, min_gb, speak=True):
    free = shutil.disk_usage(path).free / 1e9
    if free < min_gb:
        print(f"stopping: only {free:.1f} GB free", flush=True)
        say("disk full, stopping", speak)
        return False
    return True
