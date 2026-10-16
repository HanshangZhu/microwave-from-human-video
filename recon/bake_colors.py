#!/usr/bin/env python
"""Bake the scanned colours onto a measured primitive URDF (e.g. assets/microwave).

Geometry comes from the measured model; colour comes from the original scan frames,
not from the fused point cloud (TSDF voxels average the colour at their own 3-4 mm;
a frame resolves ~0.5 mm at half a metre). Every box face becomes a texture tile of
--texel metres per texel; each texel is projected into every frame of every shot
(registered by register_model.py, which also gives the door angle per shot).
A frame counts for a texel when the texel faces it (cos > --min-cos), the model
itself does not hide it (ray-cast against the known model, so holes in the measured
depth on glossy steel or glass do not matter), and nothing else does: the measured
depth there is not more than --occlusion nearer than the texel (a kettle, a hand). The texel's colour is the
per-channel median over those frames, which suppresses most specular highlights.
Texels no frame saw (hidden faces, behind the door window) are inpainted from their
tile, or take the visual's median colour. Cylinders get one colour.

Writes into --out: <link>.glb (textured visual mesh per link), coverage.json, and
<name>_textured.urdf next to the URDF, or --urdf-out (visuals -> the GLBs,
collisions unchanged).

  python recon/bake_colors.py --urdf assets/microwave/microwave.urdf \\
      --shot <cap closed> <run closed>/frames --shot <cap open> <run open>/frames \\
      --out assets/microwave/textured

<run>/frames comes from frame_poses.py (per-frame poses; frames it dropped are skipped).
--decal paints a straight-on photo onto the front instead (finer texels there, see
paint_decal); the scan still sets the colour balance.
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_model import Model  # noqa: E402


def box_tiles(v, texel_for):
    """Six faces of a box visual: texel centres + normal in the visual frame.
    texel_for(face centre, outward normal) gives the texel size of that face."""
    hs = np.array(v["size"]) / 2
    tiles = []
    for ax in range(3):
        for sgn in (-1, 1):
            u_ax, v_ax = (ax + 1) % 3, (ax + 2) % 3    # e_u x e_v = e_ax
            if sgn < 0:
                u_ax, v_ax = v_ax, u_ax              # keep U x V = outward normal
            texel = texel_for(np.eye(3)[ax] * sgn * hs[ax], np.eye(3)[ax] * sgn)
            nu = max(1, int(np.ceil(2 * hs[u_ax] / texel)))
            nv = max(1, int(np.ceil(2 * hs[v_ax] / texel)))
            U, V, N = np.eye(3)[u_ax], np.eye(3)[v_ax], np.eye(3)[ax] * sgn
            ii, jj = np.meshgrid((np.arange(nv) + 0.5) / nv, (np.arange(nu) + 0.5) / nu, indexing="ij")
            P = (N * hs[ax])[None, None] + ((jj - 0.5) * 2 * hs[u_ax])[..., None] * U \
                + ((ii - 0.5) * 2 * hs[v_ax])[..., None] * V
            corners = [N * hs[ax] + (cu - 0.5) * 2 * hs[u_ax] * U + (cv - 0.5) * 2 * hs[v_ax] * V
                       for cu, cv in ((0, 0), (1, 0), (1, 1), (0, 1))]
            tiles.append(dict(shape=(nv, nu), P=P.reshape(-1, 3), N=N, corners=np.array(corners)))
    return tiles


def cylinder_tile(v, n=600):
    m = trimesh.creation.cylinder(radius=v["radius"], height=v["length"], sections=24)
    P, fi = trimesh.sample.sample_surface_even(m, n, seed=0)
    return dict(shape=(1, 1), P=P, N=m.face_normals[fi], mesh=m)


def group_median(idx, vals, n):
    """Per-channel lower median of vals (k,3) grouped by idx in [0,n); nan where empty."""
    out = np.full((n, 3), np.nan)
    if len(idx) == 0:
        return out
    cnt = np.bincount(idx, minlength=n)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    has = cnt > 0
    for c in range(3):
        o = np.lexsort((vals[:, c], idx))
        s = vals[o, c]
        out[has, c] = s[start[has] + (cnt[has] - 1) // 2]
    return out


def paint_decal(a, tiles, off, med, nobs):
    """Front texels take their colour from a straight-on photo through the homography
    of its four front corners, colour-matched to the scan by a per-channel gain over
    the texels both saw. Parallax of parts standing proud of the front (handle, knob)
    is ignored."""
    raw = cv2.imread(a.decal)
    clean = raw
    if a.decal_mask:
        clean = cv2.inpaint(raw, (cv2.imread(a.decal_mask, cv2.IMREAD_GRAYSCALE) > 127)
                            .astype(np.uint8), 9, cv2.INPAINT_TELEA)
    raw, clean = raw[..., ::-1].astype(np.float32), clean[..., ::-1].astype(np.float32)
    px = np.array([[float(c) for c in xy.split(",")] for xy in a.decal_px.split()])
    x0, z0, x1, z1 = [float(c) for c in a.decal_rect.split()]
    Hm = cv2.getPerspectiveTransform(np.float32([[x0, z1], [x1, z1], [x1, z0], [x0, z0]]),
                                     np.float32(px))
    Pl = np.concatenate([t["Pl"] for t in tiles])
    Nl = np.concatenate([t["Nl"] for t in tiles])
    el = (Nl[:, 1] < -0.8) & (Pl[:, 1] < a.decal_max_y) & (Pl[:, 0] >= x0) & (Pl[:, 0] <= x1) \
        & (Pl[:, 2] >= z0) & (Pl[:, 2] <= z1)
    uv = cv2.perspectiveTransform(Pl[el][:, [0, 2]][None].astype(np.float32), Hm)[0]
    h, w = raw.shape[:2]
    u = np.clip(uv[:, 0], 0, w - 2)
    v = np.clip(uv[:, 1], 0, h - 2)
    u0, v0 = u.astype(int), v.astype(int)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]

    def sample(img):
        return (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv)
                + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)
    # the front surface itself from the cleaned photo; the parts standing on it
    # (cylinders: handle, knob, buttons) from the photo as taken
    is_cyl = np.concatenate([np.full(len(t["Pl"]), t["visual"]["kind"] != "box") for t in tiles])[el]
    col = np.where(is_cyl[:, None], sample(raw), sample(clean))
    both = np.isfinite(med[el][:, 0])
    gain = np.median(med[el][both], 0) / np.maximum(np.median(col[both], 0), 1)
    med = med.copy()
    med[el] = np.clip(col * gain, 0, 255)
    nobs = nobs.copy()
    nobs[el] += 1
    dec = np.zeros(len(med), bool)
    dec[el] = True
    print(f"decal: {el.sum():,} front texels from {os.path.basename(a.decal)}, colour gain "
          f"{np.round(gain, 2)} (from {both.sum():,} texels the scan also saw)", flush=True)
    return med, nobs, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--shot", nargs=2, action="append", required=True, metavar=("CAP", "RUN"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--texel", type=float, default=0.0015)
    ap.add_argument("--min-cos", type=float, default=0.25)
    ap.add_argument("--urdf-out", help="default: <urdf>_textured.urdf next to the input")
    ap.add_argument("--occlusion", type=float, default=0.02,
                    help="drop a view of a texel when the measured depth is this much nearer")
    ap.add_argument("--decal", help="a straight-on photo of the front, painted onto it")
    ap.add_argument("--decal-px", help="its front-face corners in pixels, TL TR BR BL: "
                                       "'x,y x,y x,y x,y'")
    ap.add_argument("--decal-rect", help="the same corners in the model: 'x0 z0 x1 z1' "
                                         "(TL = x0,z1 ... BL = x0,z0)")
    ap.add_argument("--decal-max-y", type=float, default=-0.02,
                    help="decal texels are those facing -y and in front of this y")
    ap.add_argument("--decal-mask", help="white where parts stand proud of the front (handle, "
                                         "knob): inpainted out of the decal; those parts "
                                         "take their colour from the unmasked photo")
    ap.add_argument("--texel-front", type=float, default=0.0004,
                    help="texel size on the decal faces")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = Model(a.urdf)

    # every texel of every visual, in its link's frame
    tiles = []                                   # dicts with link, visual, P, N (link frame)

    def is_front(P, N):                          # link frame at q = 0 (= model frame)
        return N[..., 1] < -0.8 and P[..., 1] < a.decal_max_y

    for link, vs in model.visuals.items():
        for v in vs:
            R, p = v["T"][:3, :3], v["T"][:3, 3]
            fine = (lambda c, n, R=R, p=p: a.texel_front if a.decal and is_front(R @ c + p, R @ n)
                    else a.texel)
            ts = box_tiles(v, fine) if v["kind"] == "box" else [cylinder_tile(v)]
            for t in ts:
                R, p = v["T"][:3, :3], v["T"][:3, 3]
                t.update(link=link, visual=v, Pl=t["P"] @ R.T + p,
                         Nl=(np.broadcast_to(t["N"], t["P"].shape) if t["N"].ndim == 1 else t["N"]) @ R.T)
                tiles.append(t)
    off = np.cumsum([0] + [len(t["Pl"]) for t in tiles])
    n_tex = off[-1]
    print(f"{len(tiles)} tiles, {n_tex:,} texels at {a.texel*1000:.1f} mm"
          + (f" ({a.texel_front*1000:.1f} mm on the front)" if a.decal else ""), flush=True)

    obs_i, obs_c = [], []
    for cap, run in a.shot:
        reg = json.load(open(os.path.join(run, "registration.json")))
        T = np.array(reg["T_model_to_world"])
        q = {reg["joint"]: reg["joint_angle_rad"]}
        pz = np.load(os.path.join(run, "poses.npz"))
        c2w = pz["c2w"]
        kept = pz["kept"] if "kept" in pz.files else np.ones(len(c2w), bool)   # frame_poses.py
        intr = json.load(open(os.path.join(cap, "intrinsics.json")))
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
        W, H = intr["width"], intr["height"]
        names = sorted(os.listdir(os.path.join(cap, "input")))
        # texels in the world for this shot's door angle
        Pw, Nw = np.empty((n_tex, 3)), np.empty((n_tex, 3))
        for k, t in enumerate(tiles):
            L = T @ model.link_T(t["link"], q)
            Pw[off[k]:off[k + 1]] = t["Pl"] @ L[:3, :3].T + L[:3, 3]
            Nw[off[k]:off[k + 1]] = t["Nl"] @ L[:3, :3].T
        mw = model.mesh(q).apply_transform(T)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mw.vertices), o3d.utility.Vector3iVector(mw.faces))))
        n_shot = 0
        for n, name in enumerate(names):
            if not kept[n]:
                continue
            img = cv2.imread(os.path.join(cap, "input", name))[..., ::-1].astype(np.float32)
            Dm = cv2.imread(os.path.join(cap, "depth", name), cv2.IMREAD_UNCHANGED) / 1000.0
            c = c2w[n][:3, 3]
            d = Pw - c
            dist = np.linalg.norm(d, axis=1)
            d /= dist[:, None]
            ok = -(d * Nw).sum(1) > a.min_cos
            w2c = np.linalg.inv(c2w[n])
            Xc = Pw[ok] @ w2c[:3, :3].T + w2c[:3, 3]
            front = Xc[:, 2] > 0.05
            idx = np.nonzero(ok)[0][front]
            uv = (Xc[front] @ K.T)[:, :2] / Xc[front][:, 2:3]
            inb = (uv[:, 0] > 1) & (uv[:, 0] < W - 2) & (uv[:, 1] > 1) & (uv[:, 1] < H - 2)
            idx, uv, zc = idx[inb], uv[inb], Xc[front][inb, 2]
            # something else in front (a kettle, a hand): the measured depth is nearer than
            # the texel. Missing depth (0) is not evidence either way.
            dm = Dm[np.rint(uv[:, 1]).astype(int), np.rint(uv[:, 0]).astype(int)]
            clear = ~((dm > 0) & (dm < zc - a.occlusion))
            idx, uv = idx[clear], uv[clear]
            rays = np.concatenate([np.broadcast_to(c, (len(idx), 3)), d[idx]], 1).astype(np.float32)
            hit = rc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            vis = hit > dist[idx] - 0.003
            idx, uv = idx[vis], uv[vis]
            x0, y0 = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
            fx, fy = (uv[:, 0] - x0)[:, None], (uv[:, 1] - y0)[:, None]
            col = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
                   + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
            obs_i.append(idx)
            obs_c.append(col.astype(np.uint8))
            n_shot += len(idx)
        print(f"{os.path.basename(os.path.normpath(cap))}: door {reg['joint_angle_deg']:.0f} deg, "
              f"{len(names)} frames, {n_shot:,} texel observations", flush=True)
    obs_i, obs_c = np.concatenate(obs_i), np.concatenate(obs_c)
    med = group_median(obs_i, obs_c, n_tex)
    nobs = np.bincount(obs_i, minlength=n_tex)
    dec = np.zeros(n_tex, bool)
    if a.decal:
        med, nobs, dec = paint_decal(a, tiles, off, med, nobs)

    # visual-level fallback colour: median of all its observations, else the URDF colour
    vis_key = np.concatenate([np.full(len(t["Pl"]), id(t["visual"])) for t in tiles])
    fallback = {}
    for t in tiles:
        v = t["visual"]
        if id(v) in fallback:
            continue
        m = vis_key[obs_i] == id(v)
        fallback[id(v)] = (np.median(obs_c[m], 0) if m.sum() > 20
                           else np.array(v["rgba"][:3]) * 255)

    # tile images: observed median, inpaint the rest
    coverage = {}
    for k, t in enumerate(tiles):
        nv, nu = t["shape"]
        v = t["visual"]
        if v["kind"] != "box":
            c = med[off[k]:off[k + 1]]
            dk = dec[off[k]:off[k + 1]]
            if dk.any():                         # a part on the front: its colour in the photo
                col = c[dk].mean(0)              # chrome: the mean, not the highlight
                if np.ptp(col) < 0.2 * col.max():
                    col = np.full(3, col.mean())  # near-neutral metal: drop the colour cast
            else:
                col = np.nanmedian(c, 0) if np.isfinite(c).any() else fallback[id(v)]
            t["img"] = np.tile(np.asarray(col, np.uint8), (2, 2, 1))
        else:
            c = med[off[k]:off[k + 1]].reshape(nv, nu, 3)
            seen = np.isfinite(c[..., 0])
            img = np.where(seen[..., None], np.nan_to_num(c), fallback[id(v)]).astype(np.uint8)
            if 0.05 < seen.mean() < 1:
                img = cv2.inpaint(img, (~seen).astype(np.uint8), 5, cv2.INPAINT_TELEA)
            t["img"] = img
        cov = coverage.setdefault(v["name"], [0, 0])
        cov[0] += int((nobs[off[k]:off[k + 1]] > 0).sum())
        cov[1] += int(off[k + 1] - off[k])

    # per link: atlas + mesh -> GLB
    glbs = {}
    for link in model.visuals:
        lt = [t for t in tiles if t["link"] == link]
        if not lt:
            continue
        AW = max(2048, max(t["img"].shape[1] for t in lt) + 2)
        x = y = row_h = 0
        for t in lt:                               # shelf packing, 1 px gutter
            h, w = t["img"].shape[:2]
            if x + w + 1 > AW:
                x, y, row_h = 0, y + row_h + 1, 0
            t["at"] = (y, x)
            x, row_h = x + w + 1, max(row_h, h)
        AH = y + row_h + 1
        atlas = np.zeros((AH, AW, 3), np.uint8)
        V_, F_, UV_ = [], [], []
        for t in lt:
            r0, c0 = t["at"]
            h, w = t["img"].shape[:2]
            atlas[r0:r0 + h, c0:c0 + w] = t["img"]
            vR, vp = t["visual"]["T"][:3, :3], t["visual"]["T"][:3, 3]
            if "corners" in t:
                verts = t["corners"] @ vR.T + vp
                uv = np.array([[c0 + cu * w, r0 + cv * h] for cu, cv in ((0, 0), (1, 0), (1, 1), (0, 1))])
                faces = np.array([[0, 1, 2], [0, 2, 3]])
            else:
                m = t["mesh"]
                verts = m.vertices @ vR.T + vp
                uv = np.tile([c0 + w / 2, r0 + h / 2], (len(verts), 1))
                faces = m.faces
            F_.append(faces + sum(len(v) for v in V_))
            V_.append(verts)
            UV_.append(np.stack([uv[:, 0] / AW, 1 - uv[:, 1] / AH], 1))
        mesh = trimesh.Trimesh(np.concatenate(V_), np.concatenate(F_), process=False)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.concatenate(UV_), material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(atlas), metallicFactor=0.0, roughnessFactor=0.8))
        p = os.path.join(a.out, f"{link}.glb")
        mesh.export(p)
        Image.fromarray(atlas).save(os.path.join(a.out, f"{link}_atlas.png"))
        glbs[link] = p
        print(f"{link}: atlas {AW}x{AH} -> {p}", flush=True)

    json.dump({k: dict(seen=s, texels=n, fraction=s / n) for k, (s, n) in coverage.items()},
              open(os.path.join(a.out, "coverage.json"), "w"), indent=1)

    # textured URDF: visuals -> GLB, collisions unchanged
    tree = ET.parse(a.urdf)
    udir = os.path.dirname(os.path.abspath(a.urdf))
    for link in tree.getroot().findall("link"):
        for v in link.findall("visual"):
            link.remove(v)
        if link.get("name") in glbs:
            v = ET.SubElement(link, "visual", name=f"{link.get('name')}_textured")
            g = ET.SubElement(v, "geometry")
            ET.SubElement(g, "mesh", filename=os.path.relpath(glbs[link.get("name")], udir))
    up = a.urdf_out or os.path.join(udir, os.path.splitext(os.path.basename(a.urdf))[0]
                                    + "_textured.urdf")
    for m in tree.getroot().iter("mesh"):          # mesh paths relative to the written URDF
        m.set("filename", os.path.relpath(os.path.join(udir, m.get("filename")),
                                          os.path.dirname(os.path.abspath(up))))
    tree.write(up)
    print("urdf ->", up)
    for k, (s, n) in coverage.items():
        print(f"  {k:20s} seen {s / n * 100:5.1f}%")


if __name__ == "__main__":
    main()
