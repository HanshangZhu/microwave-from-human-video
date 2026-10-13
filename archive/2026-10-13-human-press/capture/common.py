"""Shared helpers for the press-capture pipeline.

The prop: a red Duplo brick resting on folded tissue. A human finger presses the
brick; the tissue gives it a few millimetres of travel. We therefore track the
*brick*, not the fingertip -- the brick is a few thousand pixels of saturated red,
so the median of its top-face height above the table averages the ~3 mm per-pixel
depth noise down to well under a millimetre. Contact is a displacement, not a
sub-millimetre threshold.
"""
import json
import numpy as np
import cv2

# HSV bands for the pressed object, measured on this desk under its warm lamp.
#
# red Duplo brick: s_min=180 is measured, not guessed. Skin under this lamp sits
# at hue ~9 with saturation 120-190, inside any loose red band, so a finger on
# the brick merges into the blob and drags the centroid ~35 px. Sweeping s_min
# over a real press clip, 170-190 all give xy drift 8 mm / rest noise 0.4 mm,
# while 110 gives 85 mm / 16 mm and 210+ loses the brick in dim frames.
#
# yellow cap: hue 23-27, but skin is hue 17-24 AND more saturated (193 vs 130),
# so the bands overlap at the edges and the glossy top falls below s_min where it
# is specular. Narrow hue is what separates it, not saturation.
# min_area_px is deliberately low. It used to be a crude stand-in for "is this
# really the object", but a fingertip on the cap takes its visible area from
# ~420 px down to ~110, so a 250 px floor reported the object as LOST for 80% of
# a clip in which it was visible in every frame. The anchor gate in Tracker is
# what rejects impostors now; this floor only has to reject noise speckle.
PALETTE = {
    "red":    dict(h1=(0, 10), h2=(170, 180), s_min=180, v_min=60, min_area_px=100),
    "yellow": dict(h1=(22, 34), h2=None, s_min=150, v_min=70, min_area_px=70),
    # The desk lamp pushes every surface in this room into H 17-27. Green and
    # blue are empty bands here, so a coloured sticker in one of them needs no
    # saturation tuning at all -- nothing else competes.
    "green":  dict(h1=(40, 85), h2=None, s_min=90, v_min=50, min_area_px=70),
    "blue":   dict(h1=(95, 130), h2=None, s_min=110, v_min=50, min_area_px=70),
}
RED_DEFAULT = PALETTE["red"]


def red_binary(bgr, p=RED_DEFAULT, roi=None):
    """Cleaned red-hue binary image, optionally restricted to roi=(x0,y0,x1,y1)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo_s, lo_v = p["s_min"], p["v_min"]
    m = cv2.inRange(hsv, (p["h1"][0], lo_s, lo_v), (p["h1"][1], 255, 255))
    if p.get("h2"):                      # red wraps around H=0 and needs a second band
        m = m | cv2.inRange(hsv, (p["h2"][0], lo_s, lo_v), (p["h2"][1], 255, 255))
    if roi is not None:
        x0, y0, x1, y1 = (int(v) for v in roi)
        keep = np.zeros_like(m)
        keep[y0:y1, x0:x1] = 255
        m &= keep
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def red_candidates(bgr, dep=None, p=RED_DEFAULT, roi=None, min_area=None):
    """All red blobs, largest first: [(mask, (u,v), area, median_depth_m)].

    Red hue alone does not identify the brick -- warm-lit cardboard in the
    background scores higher saturation than the Duplo does. Depth and a desk
    ROI are what actually separate them.
    """
    m = red_binary(bgr, p, roi)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < (min_area if min_area is not None else p["min_area_px"]):
            continue
        bm = (lab == i).astype(np.uint8) * 255
        # None  = no depth map was supplied, so the gate cannot be evaluated
        # NaN   = a depth map was supplied and this blob has no valid depth,
        #         which is evidence against it being a real object
        z = None
        if dep is not None:
            d = dep[(bm > 0) & (dep > 0)]
            z = float(np.median(d)) if d.size else np.nan
        out.append((bm, (float(cent[i][0]), float(cent[i][1])), area, z))
    out.sort(key=lambda r: -r[2])
    return out


def pick_brick(bgr, dep=None, p=RED_DEFAULT, roi=None, z_range=None):
    """Largest red blob that also sits at a plausible depth. -> (mask, uv, area)"""
    for bm, uv, area, z in red_candidates(bgr, dep, p, roi):
        if z_range is not None and z is not None and not (
                np.isfinite(z) and z_range[0] <= z <= z_range[1]):
            continue
        return bm, uv, area
    return np.zeros(bgr.shape[:2], np.uint8), None, 0


def red_mask(bgr, p=RED_DEFAULT, roi=None):
    """Backwards-compatible colour-only pick (no depth gate)."""
    return pick_brick(bgr, None, p, roi)


def band_from_session(session, colour=None):
    """Hue/saturation come from the session (lighting- and scene-specific); the
    area floor comes from the code (a detector knob we keep re-tuning)."""
    name = colour or (session or {}).get("colour") or "red"
    band = dict(PALETTE[name])
    if session and session.get("band"):
        sb = session["band"]
        band["h1"] = tuple(sb["h1"])
        band["h2"] = tuple(sb["h2"]) if sb.get("h2") else None
        band["s_min"], band["v_min"] = sb["s_min"], sb["v_min"]
    return band


class Tracker:
    """Anchored single-object tracker over the colour candidates.

    Detecting the largest blob globally every frame is what made the preview
    "start flying" the moment a hand entered: a forearm is far larger than the
    object and its hue overlaps both the red and the yellow band, so the pick
    jumps to the arm. Once anchored, only blobs near the anchor and of a
    plausible size are accepted; everything else is reported as not visible,
    which is the honest answer.
    """

    def __init__(self, band=None, roi=None, z_range=None, max_jump=60.0, area_cap=2.5):
        self.band = band or RED_DEFAULT
        self.roi, self.z_range = roi, z_range
        self.max_jump, self.area_cap = max_jump, area_cap
        self.uv = self.area = None
        self._seed = []

    def _in_band(self, cands):
        # A blob with no valid depth is not evidence of the object -- it is the
        # absence of evidence. Letting NaN through picked a 591 px patch of
        # background over the real 388 px cap.
        return [c for c in cands
                if self.z_range is None or c[3] is None
                or (np.isfinite(c[3]) and self.z_range[0] <= c[3] <= self.z_range[1])]

    def observe(self, bgr, dep=None):
        """Largest in-band blob, ignoring the anchor. Used while seeding."""
        cs = self._in_band(red_candidates(bgr, dep, self.band, self.roi))
        return cs[0] if cs else None

    def seed(self, bgr, dep=None):
        """Collect one anchor sample. Call over a few frames with the object
        in place and the hand clear, then lock()."""
        c = self.observe(bgr, dep)
        if c:
            self._seed.append((c[1], c[2]))
        return c

    def lock(self):
        if len(self._seed) < 3:
            return False
        self.uv = np.median(np.array([s[0] for s in self._seed], float), 0)
        self.area = float(np.median([s[1] for s in self._seed]))
        self._seed = []
        return True

    def reset(self):
        self.uv = self.area = None
        self._seed = []

    @property
    def locked(self):
        return self.uv is not None

    def __call__(self, bgr, dep=None):
        cs = self._in_band(red_candidates(bgr, dep, self.band, self.roi))
        if self.locked:
            cs = [c for c in cs
                  if np.linalg.norm(np.array(c[1]) - self.uv) <= self.max_jump
                  and c[2] <= self.area_cap * self.area]
        if not cs:
            return np.zeros(bgr.shape[:2], np.uint8), None, 0
        b = max(cs, key=lambda c: c[2])
        return b[0], b[1], b[2]


def deproject(u, v, z, K):
    """Pixel + metric depth -> camera-frame XYZ. Aligned colour stream has zero
    distortion coefficients on this device, so the pinhole form is exact."""
    return np.stack([(u - K["ppx"]) / K["fx"] * z,
                     (v - K["ppy"]) / K["fy"] * z, z], -1)


def pointcloud(depth_m, K, mask=None):
    """(H,W) metric depth -> (N,3) camera-frame points with depth > 0."""
    h, w = depth_m.shape
    vs, us = np.mgrid[0:h, 0:w]
    ok = depth_m > 0
    if mask is not None:
        ok &= mask > 0
    return deproject(us[ok].astype(np.float32), vs[ok].astype(np.float32), depth_m[ok], K)


def fit_plane(pts, iters=8, sigma=2.5):
    """Robust least-squares plane through (N,3) points.

    Returns (normal, point_on_plane, rms_m, n_inliers, n_total). The normal is
    flipped to face the camera (-Z), i.e. it points 'up' out of the table.
    """
    if len(pts) < 50:
        return None
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    z, keep = pts[:, 2], np.ones(len(pts), bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        s = res[keep].std()
        if not np.isfinite(s) or s == 0:
            break
        keep = np.abs(res) < sigma * s
        if keep.sum() < 50:
            return None
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    if n[2] > 0:          # make it point back towards the camera
        n = -n
    p0 = pts[keep].mean(0)
    return n, p0, float(res[keep].std()), int(keep.sum()), int(len(pts))


def table_basis(n, p0):
    """Right-handed table frame: +z along the plane normal (up out of the table),
    +x = camera +x projected into the plane. Returns (R, origin) with
    p_table = R @ (p_cam - origin)."""
    x = np.array([1.0, 0.0, 0.0]) - n * np.dot(np.array([1.0, 0.0, 0.0]), n)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - n * n[1]
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.stack([x, y, n]), p0


def incidence_deg(n):
    """Angle between the plane normal and the camera optical axis. 0 = the camera
    looks straight down the normal (best case for depth)."""
    return float(np.degrees(np.arccos(abs(n[2]))))


def annulus(mask, inner=15, outer=70):
    """Ring of table pixels around the brick, used to fit the table plane."""
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner * 2 + 1,) * 2)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer * 2 + 1,) * 2)
    return cv2.subtract(cv2.dilate(mask, k_out), cv2.dilate(mask, k_in))


def intrinsics_dict(intr):
    return dict(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy,
                width=intr.width, height=intr.height,
                model=str(intr.model), coeffs=list(intr.coeffs))


def read_gravity(n=120):
    """Mean accelerometer reading in the accel frame, with the sign flipped so the
    vector points along gravity (an accelerometer at rest reports specific force,
    which points *up*). Returns None if the IMU is unavailable."""
    import pyrealsense2 as rs
    try:
        p, c = rs.pipeline(), rs.config()
        c.enable_stream(rs.stream.accel)
        p.start(c)
        a = []
        while len(a) < n:
            for fr in p.wait_for_frames():
                m = fr.as_motion_frame()
                if m and fr.profile.stream_type() == rs.stream.accel:
                    d = m.get_motion_data()
                    a.append([d.x, d.y, d.z])
        p.stop()
        return (-np.mean(a, 0)).tolist()
    except Exception:
        return None


def save_json(path, obj):
    def enc(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=enc)
