"""Visual domain randomisation for the camera policy's training images.

Two styles (MicrowavePotEnv(dr_seed=..., dr_style=...)):
  procedural  random floor and walls (procedural textures) and random lights only
  doorman     after DoorMan (arXiv 2512.01061; its ablation: no randomisation 5-20 %
              success, solid colours 66-70 %, + real dome-light environments another
              15-30 %): a real indoor environment map (Poly Haven HDRIs) as background
              and lighting, a random floor (and in 40 % of scenes walls), random lights,
              the microwave, pot and robot tinted, the cameras jittered

The camera policy trained on one clean scene (grey grid floor, black sky, fixed lights)
fails when only the background changes (the scanned kitchen as backdrop: 97 % -> ~0 %).
So the training images are re-rendered with a random scene per demonstration:

  floor    a random texture (solid, low-frequency noise, checker, stripes, speckle) and
           colour under the counter's collision plane (which stays at z = 0)
  walls    behind and beside the microwave at random distances, random textures
  lights   random ambient level and tint, one or two directional lights from random
           directions and strengths

Physics is untouched: everything here is visual only. The scanned kitchen is NOT one of
the backgrounds, so it stays an unseen test scene.

  MicrowavePotEnv(dr_seed=<int>)   -- sim/render_views.py --dr picks one seed per run
"""
import os
import tempfile

import numpy as np
import sapien

_TMP = None


def _tmpdir():
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix="mw_dr_")
    return _TMP


def random_texture(rng, size=512):
    """A random RGB texture image (uint8), one of a few procedural families."""
    import cv2
    c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    kind = rng.integers(5)
    yy, xx = np.mgrid[0:size, 0:size] / size
    if kind == 0:                                          # solid
        w = np.zeros((size, size))
    elif kind == 1:                                        # low-frequency noise
        n = rng.uniform(0, 1, (rng.integers(3, 9),) * 2)
        w = cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    elif kind == 2:                                        # checker
        k = rng.integers(4, 120)                            # the texture spans metres: up to cm-sized squares
        w = ((np.floor(xx * k) + np.floor(yy * k)) % 2).astype(float)
    elif kind == 3:                                        # stripes at a random angle
        a, k = rng.uniform(0, np.pi), rng.integers(5, 200)
        w = (np.sin((xx * np.cos(a) + yy * np.sin(a)) * k * 2 * np.pi) > 0).astype(float)
    else:                                                  # speckle (counter-top like)
        w = cv2.GaussianBlur(rng.uniform(0, 1, (size, size)), (0, 0), rng.uniform(0.6, 4))
        w = (w - w.min()) / (np.ptp(w) + 1e-6)
    img = (c0[None, None] * (1 - w[..., None]) + c1[None, None] * w[..., None]).clip(0, 255).astype(np.uint8)
    return img


def _material(rng):
    import cv2
    img = random_texture(rng)
    path = os.path.join(_tmpdir(), f"tex_{rng.integers(1 << 62)}.png")
    cv2.imwrite(path, img[..., ::-1])
    mat = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], roughness=float(rng.uniform(0.3, 1.0)),
                                       metallic=float(rng.uniform(0, 0.3)))
    mat.base_color_texture = sapien.render.RenderTexture2D(filename=path)
    return mat


def build_scene(scene, seed, T_W_MW, Q_W_MW, walls=True):
    """Visual-only floor (and walls) in the microwave frame (static actor at the mw pose)."""
    rng = np.random.default_rng(seed)
    b = scene.create_actor_builder()
    # floor: a thin slab just under the counter plane; smaller without walls, so the
    # environment map (the room) shows beyond the counter
    half = 4.0 if walls else rng.uniform(0.6, 1.5)
    b.add_box_visual(sapien.Pose(p=[0.2, -0.3, -0.006]), [half, half, 0.005], material=_material(rng))
    if walls:
        # behind the microwave (+y), left (-x) and right (+x), random distances
        back = rng.uniform(0.45, 2.0)
        left, right = rng.uniform(0.8, 2.5), rng.uniform(1.0, 2.5)
        b.add_box_visual(sapien.Pose(p=[0.2, back, 1.0]), [4.0, 0.02, 1.5], material=_material(rng))
        b.add_box_visual(sapien.Pose(p=[-left, -0.5, 1.0]), [0.02, 4.0, 1.5], material=_material(rng))
        b.add_box_visual(sapien.Pose(p=[0.46 + right, -0.5, 1.0]), [0.02, 4.0, 1.5], material=_material(rng))
    b.add_box_collision(sapien.Pose(p=[0, 0, -5]), [0.01, 0.01, 0.01])   # ManiSkill static actors need one
    b.initial_pose = sapien.Pose(p=T_W_MW[:3, 3], q=Q_W_MW)
    return b.build_static(name="dr_scene")


def set_lighting(scene, seed):
    rng = np.random.default_rng(seed + 7919)
    amb = rng.uniform(0.15, 0.55) * rng.uniform(0.8, 1.2, 3)
    scene.set_ambient_light(amb.clip(0, 1).tolist())
    for _ in range(rng.integers(1, 3)):
        d = np.r_[rng.uniform(-1, 1, 2), -rng.uniform(0.3, 1.0)]
        col = rng.uniform(0.4, 1.4) * rng.uniform(0.85, 1.15, 3)
        scene.add_directional_light((d / np.linalg.norm(d)).tolist(), col.tolist())


HDRI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Gitignore-but-important/hdri")


def hdri_faces(path, size=256):
    """An equirectangular HDR -> 6 tonemapped 8-bit cube faces (px nx py ny pz nz, z up),
    cached next to it."""
    import cv2
    out = os.path.splitext(path)[0] + "_faces"
    names = [os.path.join(out, f"{k}.png") for k in ("px", "nx", "py", "ny", "pz", "nz")]
    if all(os.path.exists(n) for n in names):
        return names
    os.makedirs(out, exist_ok=True)
    im = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    im = im / (1 + im)                                     # Reinhard tonemap
    im = np.clip(im ** (1 / 2.2) * 255, 0, 255).astype(np.uint8)
    H, W = im.shape[:2]
    a = (np.arange(size) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(a, -a)
    one = np.ones_like(u)
    dirs = [(one, -u, v), (-one, u, v), (u, one, v), (-u, -one, v), (-v, u, one), (v, u, -one)]
    for (x, y, z), n in zip(dirs, names):
        r = np.sqrt(x * x + y * y + z * z)
        lon, lat = np.arctan2(y, x), np.arcsin(z / r)
        mx = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32) % W
        my = ((0.5 - lat / np.pi) * H).astype(np.float32).clip(0, H - 1)
        cv2.imwrite(n, cv2.remap(im, mx, my, cv2.INTER_LINEAR)[..., ::-1])
    return names


def set_environment(scene, seed):
    rng = np.random.default_rng(seed + 104729)
    hdris = sorted(f for f in os.listdir(HDRI_DIR) if f.endswith(".hdr"))
    faces = hdri_faces(os.path.join(HDRI_DIR, hdris[rng.integers(len(hdris))]))
    cube = sapien.render.RenderCubemap(*faces)
    for sc in scene.sub_scenes:
        sc.set_environment_map(cube)


def tint(env, seed):
    """Multiply the base colour of every visual part of the microwave, the pot and the robot
    by a random per-object tint (brightness 0.7-1.3, per channel +-15 %)."""
    rng = np.random.default_rng(seed + 15485863)
    from sapien.render import RenderBodyComponent
    groups = {"microwave": [l._objs[0].entity for l in env.mw.links],
              "pot": list(env.pot._objs),
              "robot": [l._objs[0].entity for l in env.agent.robot.links]}
    for name, ents in groups.items():
        k = rng.uniform(0.7, 1.3) * rng.uniform(0.85, 1.15, 3)
        for e in ents:
            for comp in e.components:
                if not isinstance(comp, RenderBodyComponent):
                    continue
                for shape in comp.render_shapes:
                    for part in (shape.parts if hasattr(shape, "parts") else [shape]):
                        m = part.material
                        c = np.array(m.base_color, float)
                        m.base_color = np.r_[np.clip(c[:3] * k, 0, 1), c[3]].tolist()


def camera_jitter(seed):
    """(front: dp metres, drot radians), (wrist: dp, drot) -- 'aligned and slightly randomised'."""
    rng = np.random.default_rng(seed + 32452843)
    return ((rng.normal(0, 0.008, 3).clip(-0.015, 0.015), rng.normal(0, np.deg2rad(0.8), 3).clip(-0.026, 0.026)),
            (rng.normal(0, 0.003, 3).clip(-0.005, 0.005), rng.normal(0, np.deg2rad(1.2), 3).clip(-0.035, 0.035)))
