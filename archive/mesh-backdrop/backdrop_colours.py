"""Give the kitchen backdrop its scanned colours in SAPIEN.

archive/mesh-backdrop/kitchen_backdrop.py exports a vertex-coloured GLB, but SAPIEN's add_visual_from_file
ignores vertex colours and renders the mesh in its default white material. This clusters
the face colours (mean of each face's vertices) into k colours by k-means and writes one
sub-mesh per cluster with a PBR base colour, which SAPIEN does read. k = 64 gives a mean
colour error of ~2 / 255 on the kitchen scan.

  python archive/mesh-backdrop/backdrop_colours.py backdrop_mw.glb backdrop_mw_color.glb 64 [gain=0.8] [unlit]   (env maniskill)

The scan's colours already contain the room's light, and the simulator lights them again:
the counter (vertex colour ~155 / 255, as in the real frame) renders at ~189 from the front
camera. gain=g scales the base colours by g (0.8 brings the counter back to the real frame).
unlit: the colour as emission on a black base instead -- SAPIEN renders that brighter still
(~209), so it is not used.
"""
import sys, numpy as np, trimesh
src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
opts = sys.argv[4:]
unlit = "unlit" in opts
gain = float(next((o[5:] for o in opts if o.startswith("gain=")), 1.0))
m = trimesh.load(src, force="mesh")
vc = m.visual.vertex_colors[:, :3].astype(np.float32)
fc = vc[m.faces].mean(1)                                   # face colour = mean of its vertices
rng = np.random.default_rng(0)
C = fc[rng.choice(len(fc), k, replace=False)]
samp = fc[rng.choice(len(fc), min(60000, len(fc)), replace=False)]
for _ in range(25):                                         # k-means on a sample
    lab = ((samp[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    C = np.stack([samp[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
lab = np.concatenate([((fc[i:i + 50000, None] - C[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(fc), 50000)])
err = np.abs(fc - C[lab]).mean()
scene = trimesh.Scene()
for j in range(k):
    f = m.faces[lab == j]
    if not len(f):
        continue
    sub = trimesh.Trimesh(m.vertices, f, process=False)
    sub.remove_unreferenced_vertices()
    if unlit:
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[0, 0, 0, 255], emissiveFactor=C[j] / 255,
                                                  metallicFactor=0.0, roughnessFactor=1.0)
    else:
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=np.r_[C[j] * gain, 255].astype(np.uint8), metallicFactor=0.0,
                                                  roughnessFactor=0.9)
    sub.visual = trimesh.visual.TextureVisuals(material=mat)
    scene.add_geometry(sub, geom_name=f"c{j:02d}")
scene.export(dst)
print("clusters", k, "mean abs colour error", round(float(err), 1), "->", dst)
