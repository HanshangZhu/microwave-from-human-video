#!/usr/bin/env python
"""Check the microwave asset in SAPIEN, with the Snap Pot on the turntable:
door hinge direction and limits, the pot coming to rest on the turntable, the door
holding an opening (damping only), and a render at 0 / 46 / 90 deg.

Runs in the env with sapien (maniskill):
  python assets/microwave/check_microwave.py      # -> assets/microwave/renders/check.png
"""
import json, os, sys, numpy as np, sapien
from PIL import Image
A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
der = json.load(open(f"{A}/microwave/parameters.json"))["derived"]
scene = sapien.Scene()
scene.set_timestep(1 / 240)
scene.add_ground(0)
scene.set_ambient_light([0.5, 0.5, 0.5]); scene.add_directional_light([0.3, 1, -1], [1, 1, 1])
ld = scene.create_urdf_loader(); ld.fix_root_link = True
mw = ld.load(f"{A}/microwave/microwave.urdf")
j = [x for x in mw.get_active_joints()][0]
print("joint", j.name, "limits", j.limits)
door = [l for l in mw.get_links() if l.name == "door"][0]

# kinematics: where is the handle at 0 / 90 deg
for q in (0.0, np.pi / 2):
    mw.set_qpos([q]); scene.step()
    T = door.entity_pose.to_transformation_matrix()
    h = T[:3, :3] @ np.array(der["handle_bar_centre_closed"]) + T[:3, 3]
    print(f"q={np.degrees(q):5.1f} deg  handle at {np.round(h, 3)}")

# pot on the turntable
b = scene.create_actor_builder()
b.add_multiple_convex_collisions_from_file(f"{A}/heinz_snap_pot/collision.obj", decomposition="none",
    material=sapien.physx.PhysxMaterial(0.5, 0.5, 0))
b.add_visual_from_file(f"{A}/heinz_snap_pot/visual.glb")
b.set_mass_and_inertia(0.2, sapien.Pose(), [3e-4, 3e-4, 3e-4])
pot = b.build(name="pot")
cx, cy = der["turntable_centre_xy"]
pot.set_pose(sapien.Pose([cx, cy, der["turntable_top_z"] + 0.005]))
mw.set_qpos([0.0])
for _ in range(480): scene.step()
p = pot.pose.p
print(f"pot after 2 s: {np.round(p, 4)}  (turntable top z {der['turntable_top_z']:.3f})")
ok = abs(p[2] - der["turntable_top_z"]) < 0.003 and np.hypot(p[0] - cx, p[1] - cy) < 0.003
print("pot rests on turntable:", ok)

# door left open stays roughly where it is (damping only, no spring)
mw.set_qpos([1.0]); mw.set_qvel([0.0])
for _ in range(240): scene.step()
print(f"door after 1 s from 57.3 deg: {np.degrees(mw.get_qpos()[0]):.1f} deg")

# renders
cam = scene.add_camera("cam", 960, 640, np.deg2rad(55), 0.01, 10)
def look_at(eye, target):
    f = np.array(target, float) - eye; f /= np.linalg.norm(f)
    l = np.cross([0, 0, 1], f); l /= np.linalg.norm(l); u = np.cross(f, l)
    T = np.eye(4); T[:3, :3] = np.stack([f, l, u], 1); T[:3, 3] = eye
    return sapien.Pose(T)
cam.entity.set_pose(look_at(np.array([-0.15, -0.85, 0.55]), [0.2, 0.12, 0.1]))
ims = []
for q in (0.0, 0.8, 1.57):
    mw.set_qpos([q]); scene.step(); scene.update_render(); cam.take_picture()
    ims.append((cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8))
out = os.path.join(A, "microwave", "renders", "check.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
Image.fromarray(np.hstack(ims)).save(out)
print("render ->", out)
sys.exit(0 if ok else 1)
