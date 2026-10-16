#!/usr/bin/env python
"""A synthetic scan of the microwave with a known answer, for testing
register_model.py and bake_colors.py before (and independently of) a real scan.

SAPIEN renders the URDF -- every visual repainted a distinct colour -- on a counter
in front of a wall, with a dark box (a "kettle") standing in front of its right side
that hides part of it from some views, from a hand-held-like arc of 40 views. Output is
the layout capture_to_r2s.py writes (input/, depth/ in mm aligned to colour,
intrinsics.json), plus:

  gt_poses.npz   c2w (OpenCV) in the first camera's frame, as Pi3X would give them
  gt.json        model->world transform, door angle, and every visual's colour as
                 rendered (median pixel over its segmentation mask, 0-255 RGB)

Depth gets 2 mm Gaussian noise. Runs in the env with sapien (maniskill):

  python recon/tests/synth_scan.py --urdf assets/microwave/microwave.urdf --out <dir> [--door 85]
"""
import argparse
import colorsys
import json
import os
import re
import tempfile

import cv2
import numpy as np
import sapien


def look_at(eye, target):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    l = np.cross([0, 0, 1], f)
    l /= np.linalg.norm(l)
    T = np.eye(4)
    T[:3, :3] = np.stack([f, l, np.cross(f, l)], 1)
    T[:3, 3] = eye
    return sapien.Pose(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--door", type=float, default=0.0, help="door angle, degrees")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, "input"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)

    # repaint every visual a distinct, saturated colour
    src = open(a.urdf).read()
    names = re.findall(r'<visual name="([^"]+)"', src)
    colours = {}
    for i, n in enumerate(names):
        r, g, b = colorsys.hsv_to_rgb(i / len(names), 0.75, 0.85)
        colours[n] = [r, g, b]
        src = re.sub(rf'(<material name="{n}_m"><color rgba=")[^"]+', rf"\g<1>{r:.3f} {g:.3f} {b:.3f} 1", src)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False)
    tmp.write(src)
    tmp.close()

    scene = sapien.Scene()
    scene.set_ambient_light([1.0, 1.0, 1.0])               # flat light: one colour per visual
    scene.add_ground(0, render_half_size=[3, 3], render_material=sapien.render.RenderMaterial(
        base_color=[0.9, 0.9, 0.88, 1]))
    wb = scene.create_actor_builder()                      # wall behind the microwave
    wb.add_box_visual(half_size=[1.0, 0.01, 0.5], material=sapien.render.RenderMaterial(
        base_color=[0.8, 0.35, 0.1, 1]))
    wb.build_static().set_pose(sapien.Pose([0, 0.9, 0.5]))
    kb = scene.create_actor_builder()                      # an occluder in front: a "kettle"
    kb.add_box_visual(half_size=[0.06, 0.06, 0.1], material=sapien.render.RenderMaterial(
        base_color=[0.1, 0.1, 0.1, 1]))
    ld = scene.create_urdf_loader()
    ld.fix_root_link = True
    mw = ld.load(tmp.name)
    yaw = np.deg2rad(25)                                   # model not aligned with the world
    T_wm = np.eye(4)
    T_wm[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    T_wm[:3, 3] = [-0.2, 0.1, 0.0]
    mw.set_root_pose(sapien.Pose(T_wm))
    kb.build_static().set_pose(sapien.Pose(T_wm[:3, :3] @ np.array([0.40, -0.25, 0.1]) + T_wm[:3, 3]))
    mw.set_qpos([np.deg2rad(a.door)])
    scene.step()

    W, H, fy = 1280, 720, 910.0
    cam = scene.add_camera("cam", W, H, 2 * np.arctan(H / 2 / fy), 0.05, 5)
    centre = T_wm[:3, :3] @ np.array([0.23, 0.0, 0.14]) + T_wm[:3, 3]
    shape_name = {}
    for l in mw.get_links():
        for sh in l.entity.find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
            shape_name[sh.per_scene_id] = sh.name
    seen = {n: [] for n in shape_name.values()}
    c2w = []
    for i in range(a.n):
        ang = np.deg2rad(-70 + 140 * i / (a.n - 1)) + rng.normal(0, 0.03)
        h = [0.25, 0.45][i % 2] + rng.normal(0, 0.02)
        r = 0.6 + rng.normal(0, 0.03)
        d = T_wm[:3, :3] @ np.array([np.sin(ang), -np.cos(ang), 0])
        eye = centre + r * d + [0, 0, h]
        if a.door > 30:                                    # look into the cavity / at the door
            target = centre + T_wm[:3, :3] @ np.array([-0.05 + 0.1 * (i % 3), 0.1, -0.03])
        else:
            target = centre + rng.normal(0, 0.03, 3)
        cam.entity.set_pose(look_at(eye, target))
        scene.update_render()
        cam.take_picture()
        rgb = (cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
        z = -cam.get_picture("Position")[..., 2]
        seg = cam.get_picture("Segmentation")[..., 0]
        for sid, n in shape_name.items():
            m = seg == sid
            if m.sum() > 50:
                seen[n].append(rgb[m][::max(1, m.sum() // 2000)])
        z = np.where(z > 0, z + rng.normal(0, 0.002, z.shape), 0)
        cv2.imwrite(os.path.join(a.out, "input", f"{i:04d}.png"), rgb[..., ::-1])
        cv2.imwrite(os.path.join(a.out, "depth", f"{i:04d}.png"),
                    np.clip(z * 1000, 0, 65535).astype(np.uint16))
        c2w.append(cam.get_model_matrix() @ np.diag([1, -1, -1, 1]))   # GL -> OpenCV
    K = cam.get_intrinsic_matrix()
    c2w = np.array(c2w)
    w0 = np.linalg.inv(c2w[0])                             # the first camera's frame, like Pi3X
    np.savez(os.path.join(a.out, "gt_poses.npz"), c2w=w0[None] @ c2w)
    json.dump(dict(width=W, height=H, fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]),
                   cy=float(K[1, 2]), model="none", coeffs=[0] * 5, depth_unit="mm (uint16)",
                   aligned_to="color", source="synthetic"),
              open(os.path.join(a.out, "intrinsics.json"), "w"), indent=1)
    rendered = {n: np.median(np.concatenate(v), 0).tolist() for n, v in seen.items() if v}
    json.dump(dict(T_model_to_world=(w0 @ T_wm).tolist(), door_deg=a.door, colours=colours,
                   rendered_rgb=rendered),
              open(os.path.join(a.out, "gt.json"), "w"), indent=1)
    os.unlink(tmp.name)
    print(f"{a.n} views -> {a.out}")


if __name__ == "__main__":
    main()
