"""Assemble the four matched-camera comparisons and the auditable QA report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True);a=ap.parse_args();out=a.asset
    c=json.loads((out/'parameters.json').read_text());r=json.loads((out/'verification.json').read_text())
    if c.get('measurement_quality')=='APPROXIMATE_USER_MEASUREMENTS':
        from report_measured import generate
        generate(out,c,r)
        return
    smoke=json.loads((out/'maniskill_smoke.json').read_text()) if (out/'maniskill_smoke.json').exists() else {'passed':False}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    sheet=Image.new('RGB',(1600,960),(245,247,249));draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(['front','side','top','three_quarter']):
        x=(i%2)*800;y=(i//2)*480
        label=name.replace('_',' ').title()
        draw.text((x+16,y+10),label+' | same camera, scale and rigid alignment',font=font,fill=(27,39,50))
        for j,mode in enumerate(['model','cloud']):
            im=Image.open(out/'renders'/f'{name}_{mode}.png').convert('RGB').resize((394,394),getattr(Image,'Resampling',Image).LANCZOS)
            sheet.paste(im,(x+j*400+3,y+43))
            draw.text((x+j*400+14,y+445),'Parametric model' if mode=='model' else 'All 3,560 original cloud points',font=small,fill=(27,39,50))
    sheet.save(out/'comparison_4views.png')
    data=np.genfromtxt(out/'cloud_residuals.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    xyz=np.stack([data['x_m'],data['y_m'],data['z_m']],axis=1)*1000;d=data['distance_mm']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,inds,title in zip(axes[:2],[(0,1),(0,2)],['Top: XY','Front: XZ']):
        sc=ax.scatter(xyz[:,inds[0]],xyz[:,inds[1]],c=d,s=5,cmap='turbo',vmin=0,vmax=20)
        ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('X (mm)');ax.set_ylabel('YZ'[inds[1]-1]+' (mm)')
        ax.grid(alpha=.2)
    fig.colorbar(sc,ax=axes[:2],label='Exact raw point-to-mesh distance (mm)',shrink=.8)
    axes[2].hist(d,bins=40,color='#316d8a');axes[2].axvline(5,c='#c44e36',linestyle='--',label='5 mm')
    axes[2].set(xlabel='Distance (mm)',ylabel='Original points',title='Untrimmed residuals');axes[2].legend()
    fig.savefig(out/'residual_map.png',dpi=160);plt.close(fig)
    dims='\n'.join(f"| {x['dimension']} | {x['point_cloud_fit_mm']:.2f} | {x['exported_mm']:.2f} | {x['export_error_mm']:+.5f} | unavailable | UNVERIFIED |" for x in r['dimensions'])
    rs=[]
    for p in sorted((out/'verification_rounds').glob('round_*.json')):
        v=json.loads(p.read_text());n=v['round'];cfg=json.loads((out/'verification_rounds'/f'parameters_{n}.json').read_text())
        rs.append(f"| {n} | {cfg['fit']['fit_envelope_band_mm']:.0f} | {cfg['fit'].get('rim_support_percentile',95):.1f} | {v['chamfer']['mean_mm']:.3f} | {v['chamfer']['p95_mm']:.3f} | {v['structural_pass']} |")
    region='\n'.join(f"| {name} | {v['count']} | {v.get('mean_mm',0):.3f} | {v.get('p95_mm',0):.3f} | {v.get('over_5mm_count',0)} |" for name,v in r['cloud_to_mesh_by_nearest_region'].items())
    quality=json.loads((out/'input_quality.json').read_text())
    qrows='\n'.join(f"| {Path(q['source']).parent.name} | {q['dimensions_mm']['height_mm']:.2f} | {q['dimensions_mm']['rim_long_mm']:.2f} x {q['dimensions_mm']['rim_short_mm']:.2f} | {q['dimensions_mm']['base_long_mm']:.2f} x {q['dimensions_mm']['base_short_mm']:.2f} |" for q in quality)
    dm=r['directional'];cv=r['coverage_and_outliers'];g=c['geometry'];assumptions='\n'.join('- '+s for s in c['assumptions'])
    text=f'''# Heinz Snap Pot: generated asset and verification

**Status: usable rigid simulation proxy; metric accuracy is NOT independently verified.**
Three fitting/verification rounds were completed. Round 2 is selected: it has the
lowest raw symmetric mean, improved rim coverage, and stable body dimensions.
The original ruler tolerance cannot be certified: the ruler fields were blank and
the user explicitly authorized extracting dimensions from the point cloud instead.
No estimate below is relabeled as a ruler measurement.

The complete raw-cloud symmetric Chamfer is **{r['chamfer']['mean_mm']:.3f} mm mean,
{r['chamfer']['p95_mm']:.3f} mm P95**. The raw P95 does not meet a 5 mm target.
The source contains duplicate interior sheets and lacks the base surface. These
limitations remain visible in the saved comparison images and residual files.

## Deliverables and coordinates

- `visual.glb`: closed, connected exterior, {r['structural']['visual_triangles']} triangles; white plastic and photo-sampled teal/orange lid palette.
- `collision.obj`: **two named closed convex solids**, body and rim/lid cap; analytic decomposition, not a single hull bridging the flange overhang.
- `snap_pot.blend`: native Blender scene, named dimension properties on the scene, packed texture, comparison cameras and raw point cloud.
- `parameters.json`: all physical dimensions, corner radii, thickness assumptions, source hash and cloud-to-model transform.
- `comparison_4views.png`, `renders/`, `cameras.json`: four matched orthographic cameras, identical pose/scale for model and all raw points.
- `verification.json`, `dimensions.csv`, `verification_rounds/`: exported mesh QA and three recorded candidates.
- `cloud_residuals.csv`, `residuals_over_5mm.ply`, `residual_map.png`: every source point and its exact residual; red PLY points exceed 5 mm.
- `maniskill_smoke.json`: actual CPU collision loading and drop-test result.

All mesh vertex coordinates are **metres**, +X is the long axis, +Z is up;
the origin is the centre of the flat base at z=0. Import with scale `[1,1,1]`
and no axis correction. The GLB deliberately disables Blender's usual Y-up export
conversion (`export_yup=False`), to satisfy this asset's Z-up convention.
See [Blender export API](https://docs.blender.org/api/5.0/bpy.ops.export_scene.html).
Some generic glTF viewers assume Y-up and may display this explicitly Z-up asset sideways.

## Dimensions (mm)

"Body top" means the body immediately below the flange. "Rim" is its outer
footprint. Height includes the assumed film thickness. Error below measures
export reproduction only, not physical-object accuracy.

| Dimension | Cloud fit | Exported mesh | Export minus fit | Ruler | Ruler ±2 mm |
|---|---:|---:|---:|---|---|
{dims}

Body top/base corner radii: {g['body_top_corner_radius_mm']:.2f}/{g['base_corner_radius_mm']:.2f} mm.
Rim radius: {g['rim_corner_radius_mm']:.2f} mm. Rim thickness **1.2 mm**, film
thickness **0.15 mm**, and film inset **1.0 mm** are explicit unresolved assumptions.
Bootstrap intervals in `parameters.json` measure sampling stability only; they
exclude calibration/pose errors and cannot establish ±2 mm accuracy.

## Inputs and fitting

Selected cloud: `{c['source_cloud']}` (3,560 points). SHA256:
`{c['source_cloud_sha256']}`.
This is the metric RealSense TSDF reconstruction with both intrinsics and depth
conditioning. All 40 per-view metric scale ratios are finite; p10–p90 is
0.50418–0.51026. The cloud is never rescaled. Alignment is only a Z rotation and
XY translation; table z=0 remains fixed. The transform is recorded in the JSON.

The repo contains a 40-frame orbit capture, rather than a separately identified
six-photo attachment set. Six representative original frames are shown in
`source_views.jpg`. Colours are sampled from frame 0037; the label is simplified
to its observed white/teal/orange regions. Printed text, artwork details, specular
glare and illumination-independent material colour are not reconstructed.

The fit uses 5 mm horizontal slices, a {c['fit']['fit_envelope_band_mm']:.0f} mm band inside each
slice's convex envelope, and a Cauchy robust loss. This avoids copying ghost
surfaces into the body. **No point is removed from verification or renders.**
Top plane height uses central high points; the rim uses 2.5/97.5-percentile supports.
At least 1 mm per side of flange overhang is imposed where sparse samples cannot
separate the rim from the body; this is an additional model constraint, not a measurement.

The alternative reconstructions disagree substantially:

| Reconstruction | Height mm | Rim L x S mm | Base L x S mm |
|---|---:|---:|---:|
{qrows}

These are alternative fits of the same capture, not independent measurements.
Their 11–13 mm lateral disagreements show that the stated ±3–5 mm unbiased-noise
assumption is not established for these fused files. K+depth was selected for
depth-constrained poses and the most consistent per-view scale; that choice is
not proof of absolute accuracy. `beans_global40` was excluded because its source
is Pi3X reconstruction units, not the measured-depth metric TSDF.

## Chamfer definition and untrimmed results

Distances are unsquared Euclidean millimetres. Cloud-to-mesh uses exact closest
points on exported triangles, for every original cloud point. Mesh-to-cloud
uses 200,000 area-uniform exterior mesh samples (seed 20261014) and nearest
original points. The symmetric mean is the average of the two directional means;
P95 is the quantile of their equally weighted (0.5/0.5) distributions, so unequal
sample counts do not bias the metric toward the dense mesh sample set. Surface
sampling and source-cloud coverage still limit the reverse estimate.

| Direction | Mean mm | P95 mm | Fraction >5 mm |
|---|---:|---:|---:|
| All cloud → mesh | {dm['cloud_to_mesh']['mean_mm']:.3f} | {dm['cloud_to_mesh']['p95_mm']:.3f} | {dm['cloud_to_mesh']['over_5mm_fraction']:.1%} |
| Full exterior mesh → cloud | {dm['mesh_to_cloud']['mean_mm']:.3f} | {dm['mesh_to_cloud']['p95_mm']:.3f} | {dm['mesh_to_cloud']['over_5mm_fraction']:.1%} |
| Equal-direction symmetric | {r['chamfer']['mean_mm']:.3f} | {r['chamfer']['p95_mm']:.3f} | — |

Raw cloud points, grouped by their nearest mesh region:

| Region | Points | Mean mm | P95 mm | Points >5 mm |
|---|---:|---:|---:|---:|
{region}

**Regions above 5 mm:** {cv['points_more_than_5mm_inside_closed_model']} raw points are more than 5 mm
inside the model. The XY/XZ slice map shows repeated internal sheets; they are
consistent with fusion/registration artifacts, not evidence for hidden geometry.
Another {cv['points_more_than_5mm_outside_closed_model']} points lie more than 5 mm outside the model,
around sparse/distorted outer regions. Per-point coordinates and nearest-region
labels are in the CSV; this classification does not claim to prove the true
surface location.

The entire unobserved flat base is >5 mm from available cloud samples:
base-only mesh→cloud mean **{cv['mesh_base_to_cloud']['mean_mm']:.3f} mm**, P95
**{cv['mesh_base_to_cloud']['p95_mm']:.3f} mm**. Minimum observed z is
**{cv['min_observed_cloud_z_mm']:.3f} mm**. Lowest sidewall and top-edge coverage gaps
also contribute. For context only, mesh→cloud **excluding the base** is
{cv['mesh_without_base_to_cloud']['mean_mm']:.3f} mm mean / {cv['mesh_without_base_to_cloud']['p95_mm']:.3f} mm P95.
This coverage-limited number does not replace the full metric above.

## Three rounds

| Round | Fit envelope band mm | Rim percentile | Symmetric mean mm | Symmetric P95 mm | Structural checks |
|---|---:|---:|---:|---:|---|
{chr(10).join(rs)}

Round 1 established the body and exposed sparse rim support. Round 2 expanded
rim support and reduced external outliers. Round 3 tightened the fitting band
as a sensitivity check: body dimensions changed <1 mm but the full metric got
slightly worse. The selected round 2 was re-exported for final renders; this is
not a fourth geometry-fit candidate. Further deformation to fit interior ghosts
would conflict with the requested clean outer shape.

## Simulation checks and limits

Visual and collision meshes have consistent winding, positive volume and closed
topology. GLB material/UV/normal seam vertices are welded by equal position for
topology checks. OBJ parts are individually convex; Blender meters and exported
bounds agree to floating-point precision. The collision lid cap replaces the
film's 1 mm inset step with a convex bevel over its 0.15 mm height.

ManiSkill CPU collision import + 4 cm drop: **{'PASS' if smoke.get('passed') else 'NOT PASSED'}**.
See `maniskill_smoke.json` for measured shape count, final pose/velocity, and test
parameters. This uses a **test-only 0.2 kg mass and 0.5 friction**, with uniform
filled-envelope inertia. These values were not measured. The headless test does
not load a Vulkan visual component; Blender and the GLB parser check visuals.
No Panda grasp trial or force calibration was performed.

PhysX cooked the two shapes to {smoke.get('cooked_shape_vertices',[])} vertices;
their overall bounds differ from the asset parameters by
{smoke.get('cooked_bounds_max_error_mm',float('nan')):.6f} mm. Local curved faces may
be simplified by cooking even though bounding dimensions are retained.

At the nominal rim, an 80 mm opening has **{80-g['rim_short_mm']:.2f} mm short-axis clearance**
and **{80-g['rim_long_mm']:.2f} mm long-axis clearance**. These are geometric
differences, not a successful grasp or collision-free approach guarantee.
`recon/snap_pot/maniskill_asset.py` loads existing convex pieces without an
additional decomposition; its `mass_kg` and `friction` arguments are required.
See [SAPIEN builder API](https://sapien-sim.github.io/docs/_modules/sapien/wrapper/actor_builder.html).

## Assumptions and unverified properties

{assumptions}
- Symmetric centered section profiles and minimum 1 mm-per-side flange overhang are modeling constraints.
- Ruler-grounded accuracy, physical mass/centre of mass/inertia, friction, stiffness, film peeling and grasp success are unverified.
- Replacing ghost-contaminated fused depth or adding independent dimensions is required to certify physical accuracy.

![Four matched camera pairs](comparison_4views.png)

![Untrimmed residual map](residual_map.png)
'''
    (out/'REPORT.md').write_text(text)
    (out/'selection.json').write_text(json.dumps(dict(selected_round=2,geometry_fit_rounds=3,
        selection='Lowest untrimmed symmetric mean among three candidates; stable body, improved sparse rim coverage.',
        absolute_accuracy_verified=False),indent=2)+'\n')
    print('Wrote REPORT.md, comparison_4views.png and residual_map.png')


if __name__=='__main__':main()
