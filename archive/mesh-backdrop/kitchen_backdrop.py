#!/usr/bin/env python
"""The scanned kitchen as a backdrop for the simulation's cameras (visual only).

The kitchen sweep's fused mesh (recon/fuse_depth.py) is put into the microwave frame with
the sweep's registration to the measured microwave (register_model.py; 3-5 cm off, see
recon/ASSET_ISSUES.md), cropped to the room around the microwave, decimated, and cleared
of what the simulation itself provides: the microwave (the scan's door is shut; the
simulated one opens) and anything standing on the counter in the task's workspace (the
simulated pot is the only pot). The counter surface itself stays. Exported as a vertex-
coloured GLB in the microwave frame; MicrowavePotEnv(backdrop=...) places it.

Runs in the env with open3d (3dgrut):
  python archive/mesh-backdrop/kitchen_backdrop.py <run dir with mesh.ply + registration.json> --out <file.glb>
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d

MICROWAVE = ((-0.06, -0.10, -0.02), (0.52, 0.40, 0.32))   # mw-frame box incl. door, handle, back bump
WORKSPACE = ((-0.10, -0.60, 0.015), (0.80, 0.0, 0.50))     # above the counter where the pot and hand move
# (--cut-z raises the 1.5 cm floor of that box: the scanned counter itself is +-1-2 cm rough, so the
# 1.5 cm cut punches holes in it through which the renderer's clear colour shows -- the "white
# speckles" of the evaluated backdrop; 0.04 with --flatten 0.03 removes them)
ROOM = ((-1.6, -1.8, -1.0), (1.6, 1.3, 1.4))


def inside(v, box):
    lo, hi = np.array(box[0]), np.array(box[1])
    return np.all((v >= lo) & (v <= hi), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--faces", type=int, default=300000)
    ap.add_argument("--cut-z", type=float, default=WORKSPACE[0][2],
                    help="lowest height of the workspace cut, m (1.5 cm as evaluated; 0.04 keeps the counter whole)")
    ap.add_argument("--flatten", type=float, default=0.0,
                    help="snap vertices within this distance of the counter plane (z = 0) onto it, m")
    a = ap.parse_args()
    m = o3d.io.read_triangle_mesh(os.path.join(a.run, "mesh.ply"))
    T = np.linalg.inv(np.array(json.load(open(os.path.join(a.run, "registration.json")))["T_model_to_world"]))
    m.transform(T)                                          # scan -> microwave frame
    v = np.asarray(m.vertices)
    drop = ~inside(v, ROOM) | inside(v, MICROWAVE) | inside(v, (WORKSPACE[0][:2] + (a.cut_z,), WORKSPACE[1]))
    m.remove_vertices_by_mask(drop)
    if a.flatten:                                           # the counter is a flat board; the scan is not
        v = np.asarray(m.vertices).copy()
        near = np.abs(v[:, 2]) < a.flatten
        v[near, 2] = -0.002                                 # just under the sim's counter: the pot stands on it
        m.vertices = o3d.utility.Vector3dVector(v)
        print(f"flattened {near.sum():,} counter vertices")
    m.remove_unreferenced_vertices()
    print(f"after crop: {len(m.triangles):,} faces")
    if len(m.triangles) > a.faces:
        m = m.simplify_quadric_decimation(a.faces)
    m.compute_vertex_normals()
    print(f"decimated: {len(m.triangles):,} faces, {len(m.vertices):,} vertices")
    import trimesh
    tm = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles),
                         vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8), process=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tm.export(a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
