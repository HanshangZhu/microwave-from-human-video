#!/usr/bin/env python
"""Put a Pi3X cloud from an orbit capture upright and cut out the object.

Pi3X returns points in the first camera's frame. For an orbit around an object the
camera centres lie roughly on a circle; its axis is the table normal. That axis
becomes +z, the table top (the densest height under the orbit centre) z = 0, and the
object is whatever stands on the table inside a box under the orbit centre. Faint
floaters are dropped with a statistical outlier filter.

Writes, next to the input:
  scene_up.ply    whole cloud, upright
  object.ply      the object alone

View with R2S's viewer (Filament; the legacy Open3D viewer drops colours here):
  python ~/Research/R2S/scene_recon/scripts/viewers/view_pi3x.py --ply <run>/object.ply --interactive

Works on a Pi3X run (points.ply + pi3x.npz, Pi3 units) or a fuse_depth.py run
(fused.ply + poses.npz, metres; pass --ply fused.ply --crop 0.08).

Runs in the env that has open3d (3dgrut):
  python recon/orbit_frame.py <run dir> [--ply points.ply] [--crop 0.15]
"""
import argparse
import os

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ply", default="points.ply")
    ap.add_argument("--crop", type=float, default=0.15,
                    help="half-width of the object box, in the cloud's units")
    a = ap.parse_args()

    pc = o3d.io.read_point_cloud(os.path.join(a.run, a.ply))
    P, C = np.asarray(pc.points), np.asarray(pc.colors)
    if os.path.exists(os.path.join(a.run, "pi3x.npz")):
        cams = np.load(os.path.join(a.run, "pi3x.npz"), allow_pickle=True)["camera_poses"]
    else:
        cams = np.load(os.path.join(a.run, "poses.npz"))["c2w"]
    cams = cams[:, :3, 3]
    ctr = cams.mean(0)
    _, _, vt = np.linalg.svd(cams - ctr)
    L = (P - ctr) @ vt.T                       # orbit plane x, y; orbit axis z
    F = np.eye(3)
    if np.median(L[:, 2]) > 0:                 # scene is below the orbit -> z up
        L[:, 1:] *= -1
        F = np.diag([1.0, -1.0, -1.0])

    # the orbit axis is only roughly vertical; level the cloud on the table plane itself
    box = (np.abs(L[:, 0]) < 2 * a.crop) & (np.abs(L[:, 1]) < 2 * a.crop)
    tol = 0.01 * np.ptp(L[box, 2])
    q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(L[box]))
    (nx, ny, nz, d), _ = q.segment_plane(tol, 3, 2000)
    n = np.array([nx, ny, nz]) * np.sign(nz)
    v, c = np.cross(n, [0, 0, 1]), n[2]          # rotate n onto +z (Rodrigues)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K / (1 + c)
    L = L @ R.T
    table_z = -d * np.sign(nz)                   # plane n.x + d = 0 -> z = -d after R
    print(f"table tilt vs orbit axis: {np.degrees(np.arccos(c)):.1f} deg", flush=True)

    box = (np.abs(L[:, 0]) < a.crop) & (np.abs(L[:, 1]) < a.crop)
    above = box & (L[:, 2] > table_z + tol)
    origin = np.r_[np.median(L[above, :2], 0), table_z]
    L -= origin

    def save(name, m, nb):
        q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(L[m]))
        q.colors = o3d.utility.Vector3dVector(C[m])
        n0 = len(q.points)
        q, _ = q.remove_statistical_outlier(nb_neighbors=nb, std_ratio=2.0)
        o3d.io.write_point_cloud(os.path.join(a.run, name), q)
        print(f"{name}: {len(q.points):,} of {n0:,} points kept", flush=True)

    save("scene_up.ply", np.ones(len(L), bool), 8)
    save("object.ply", above, 16)

    T = np.eye(4)                              # input frame -> this frame
    T[:3, :3] = R @ F @ vt
    T[:3, 3] = -T[:3, :3] @ ctr - origin
    np.save(os.path.join(a.run, "T_upright.npy"), T)
    mp = os.path.join(a.run, "mesh.ply")       # fuse_depth.py output: crop it the same way
    if os.path.exists(mp):
        mesh = o3d.io.read_triangle_mesh(mp).transform(T)
        V = np.asarray(mesh.vertices)
        out = (np.abs(V[:, 0] + origin[0]) > a.crop) | (np.abs(V[:, 1] + origin[1]) > a.crop) \
            | (V[:, 2] < tol)                  # same box and table cut as the points
        mesh.remove_vertices_by_mask(out)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(a.run, "object_mesh.ply"), mesh)
        print(f"object_mesh.ply: {len(mesh.triangles):,} triangles", flush=True)


if __name__ == "__main__":
    main()
