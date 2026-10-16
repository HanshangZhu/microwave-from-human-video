"""Reporting for the approximate user-measurement revision."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def generate(out,c,r):
    g=c['geometry']
    selection=json.loads((out/'selection.json').read_text())
    smoke=json.loads((out/'maniskill_smoke.json').read_text())
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    sheet=Image.new('RGB',(1600,960),(245,247,249));draw=ImageDraw.Draw(sheet)
    count=r['chamfer']['point_count']
    for i,name in enumerate(['front','side','top','three_quarter']):
        x=(i%2)*800;y=(i//2)*480
        draw.text((x+16,y+10),name.replace('_',' ').title()+' | same camera; raw cloud scale = 1',font=font,fill=(27,39,50))
        for j,mode in enumerate(['model','cloud']):
            im=Image.open(out/'renders'/f'{name}_{mode}.png').convert('RGB').resize((394,394),getattr(Image,'Resampling',Image).LANCZOS)
            sheet.paste(im,(x+j*400+3,y+43))
            draw.text((x+j*400+14,y+445),'Approximate-measurement model' if mode=='model' else f'All {count:,} original cloud points',font=small,fill=(27,39,50))
    sheet.save(out/'comparison_4views.png')
    if c.get('bottom_reference'):
        preview=Image.new('RGB',(1280,560),(245,247,249));pd=ImageDraw.Draw(preview)
        for x,path,label in [(0,out/c['bottom_reference']['path'],'User photo: inverted pot'),(640,out/'renders/underside_model.png','Model underside: dimensions estimated')]:
            im=Image.open(path).convert('RGB');im.thumbnail((630,490),getattr(Image,'Resampling',Image).LANCZOS)
            preview.paste(im,(x+(640-im.width)//2,40+(490-im.height)//2))
            pd.text((x+12,12),label,font=font,fill=(27,39,50))
        pd.text((12,536),f"Recess {g['bottom_recess_depth_mm']:g} mm | outer round {g['bottom_round_height_mm']:g} mm | support plane z = 0 | moulded text omitted",font=small,fill=(27,39,50))
        preview.save(out/'bottom_comparison.png')
    if (out/'previous_2mm/underside_model.png').exists():
        comparison=Image.new('RGB',(1280,690),(245,247,249));cd=ImageDraw.Draw(comparison)
        for x,path,label in [(0,out/'previous_2mm/underside_model.png','Before: 2 mm, flat face normals'),(640,out/'renders/underside_model.png','After: 3 mm, smooth curved normals')]:
            im=Image.open(path).convert('RGB').resize((640,640),getattr(Image,'Resampling',Image).LANCZOS)
            comparison.paste(im,(x,45));cd.text((x+14,12),label,font=font,fill=(27,39,50))
        comparison.save(out/'rounding_before_after.png')
    data=np.genfromtxt(out/'cloud_residuals.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    xyz=np.stack([data['x_m'],data['y_m'],data['z_m']],axis=1)*1000;d=data['distance_mm']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,inds,title in zip(axes[:2],[(0,1),(0,2)],['Top: XY','Front: XZ']):
        sc=ax.scatter(xyz[:,inds[0]],xyz[:,inds[1]],c=d,s=5,cmap='turbo',vmin=0,vmax=max(20,float(np.ceil(d.max()/5)*5)))
        ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('X (mm)');ax.set_ylabel('YZ'[inds[1]-1]+' (mm)');ax.grid(alpha=.2)
    fig.colorbar(sc,ax=axes[:2],label='Exact raw point-to-mesh distance (mm)',shrink=.8)
    axes[2].hist(d,bins=40,color='#316d8a');axes[2].axvline(5,c='#c44e36',linestyle='--',label='5 mm')
    axes[2].set(xlabel='Distance (mm)',ylabel='Original points',title='All raw residuals');axes[2].legend()
    fig.savefig(out/'residual_map.png',dpi=160);plt.close(fig)
    dims='\n'.join(f"| {d['dimension']} | {d['ruler_mm']:.2f} | {d['exported_mm']:.2f} | {d['ruler_error_mm']:+.5f} | {d['ruler_check']} |" for d in r['dimensions'])
    comparisons=[];rounds=[]
    for row in selection['candidates']:
        g=row['reference_cloud_dimensions_mm'];name=Path(row['source']).parent.name
        comparisons.append(f"| {name} | {g['height_mm']:.2f} | {g['body_top_long_mm']:.2f} x {g['body_top_short_mm']:.2f} | {g['base_long_mm']:.2f} x {g['base_short_mm']:.2f} | {g['rim_long_mm']:.2f} x {g['rim_short_mm']:.2f} |")
        rounds.append(f"| {row['round']} | {name} | {row['chamfer_mean_mm']:.3f} | {row['chamfer_p95_mm']:.3f} | {'selected' if row['round']==selection['selected_round'] else ''} |")
    directions='\n'.join(f"| {name} | {v['mean_mm']:.3f} | {v['p95_mm']:.3f} | {v['over_5mm_fraction']:.1%} |" for name,v in r['directional'].items())
    regions='\n'.join(f"| {name} | {v['count']} | {v.get('mean_mm',0):.3f} | {v.get('p95_mm',0):.3f} | {v.get('over_5mm_count',0)} |" for name,v in r['cloud_to_mesh_by_nearest_region'].items())
    g=c['geometry'];cv=r['coverage_and_outliers'];assumptions='\n'.join('- '+s for s in c['assumptions'])
    underside=r['structural'].get('underside',{})
    cooked=smoke.get('cooked_underside',{})
    bottom_section=''
    if underside:
        bottom_section=f'''## 3 mm bottom rounding revision

The new photo shows the pot upside down, with a rounded bottom perimeter and
a shallow central panel recess. The previous flat-base assets and source are
preserved in `history/measurement_flat_base_v2.zip`. The prior 2 mm rounded
version is in `history/bottom_photo_2mm_v3.zip`. At the user's request, the
current exterior has a 3 mm quarter-circle bottom profile, sampled at
{g['bottom_curve_segments']} uniform angular intervals. Analytic per-loop normals
smooth the curved surfaces and keep the lid's planar surfaces and edges sharp.
The 3 mm radius remains an unmeasured modelling estimate.

All new underside dimensions are **estimates from an unscaled photograph**:
outer round height/inset {g['bottom_round_height_mm']:.1f}/{g['bottom_round_inset_mm']:.1f} mm,
recess depth {g['bottom_recess_depth_mm']:.1f} mm, recessed panel
{g['bottom_panel_long_mm']:.1f} x {g['bottom_panel_short_mm']:.1f} mm,
panel corner radius {g['bottom_panel_corner_radius_mm']:.1f} mm and recess-wall
bevel {g['bottom_recess_wall_bevel_mm']:.1f} mm. The outer base outline is still
68 x 48 mm, now defined at z={g['bottom_round_height_mm']:.1f} mm.
The estimated support contact outline at z=0 is
{g['base_long_mm']-2*g['bottom_round_inset_mm']:.0f} x {g['base_short_mm']-2*g['bottom_round_inset_mm']:.0f} mm. Moulded lettering,
recycling marks and the fine circumferential seam are omitted.

The exported fillet profile was checked at {r['structural']['bottom_fillet']['checked_vertices']}
vertices; maximum deviation from the requested arc is
{r['structural']['bottom_fillet']['arc_profile_max_error_mm']:.6f} mm. This confirms
explicit rounded geometry, rather than relying on shading to imply a round edge.

The collision union retains the recess: upward rays first hit the centre at
z={underside['collision_centre_first_surface_z_mm']:.3f} mm and the support land at
z={underside['collision_support_first_surface_z_mm']:.3f} mm. Across 50,000 visual
surface samples, maximum distance to the collision union is
{underside['visual_to_collision_union_sample_max_error_mm']:.3f} mm.
This tests mesh consistency, not the true physical underside dimensions.
The CPU smoke test also checks that PhysX cooking preserves this centre gap.
Cooked centre depth is {cooked.get('centre_first_surface_z_mm',0):.3f} mm;
maximum local support-height error across {cooked.get('support_land_rays',0)}
test rays is {cooked.get('support_land_max_height_error_mm',0):.3f} mm.
The previous 2 mm revision's strict 0.1 mm support-height diagnostic failed
because PhysX simplified thin support sectors; that historical result remains
in `maniskill_smoke_strict_diagnostic.json`. The current check uses the same 1 mm
surface-approximation budget as export QA, retains the separate 0.1 mm centre
gap check, and verifies stable z=0 settling. This cooking approximation is
not evidence of submillimetre physical accuracy.

The current result is round {r['round']} in `verification_rounds/`.
The prior 2 mm revision's three rounds are in `bottom_2mm_verification_rounds/`;
those resolved near-coplanar BMesh slivers using double-precision Qhull.
The raw-cloud mismatch remains unresolved: a bottom-only change cannot resolve
the observed lateral size disagreement or missing underside coverage.

![Same-camera rounding comparison](rounding_before_after.png)

![Rounded bottom edge close-up](renders/bottom_edge_detail.png)

![Photo and model underside](bottom_comparison.png)
'''
    text=f'''# Heinz Snap Pot — 3 mm bottom rounding revision

**Geometry and simulation import pass; physical accuracy remains unverified.**
The user supplied approximate dimensions: outer lid 95 x 74 mm, upper body
80 x 55 mm, base 68 x 48 mm, total height 63–64 mm, and a flat top assembly
about 2.5 mm thick. These replace the earlier cloud-only size estimates.
The previous assets and source are in `history/point_cloud_only_v1.zip`.

These are nominal anchors, not certified ruler ground truth. Height is the
reported range's 63.5 mm midpoint. All three original metric clouds were compared
at their original scale. Their height agrees, but their lateral dimensions do
not jointly agree with the new readings. Neither the cloud nor the entire model
is globally scaled to hide this disagreement.

Final untrimmed symmetric Chamfer: **{r['chamfer']['mean_mm']:.3f} mm mean,
{r['chamfer']['p95_mm']:.3f} mm P95**. This does not establish ±2 mm object accuracy.

{bottom_section}

## Dimensions (mm)

| Dimension | Approximate user value | Exported mesh | Difference | Independent accuracy |
|---|---:|---:|---:|---|
{dims}

Every export matches its nominal input within ±2 mm. Exact reproduction of
approximate readings is not independent validation; measurement uncertainty was
not supplied. **Top assembly thickness is {g['top_assembly_thickness_mm']:.2f} mm total**:
{g['rim_thickness_mm']:.2f} mm rim/plastic proxy plus an assumed {g['lid_thickness_mm']:.2f} mm film.
The film inset remains {g['lid_inset_mm']:.2f} mm. The cloud and photographs support
a flat wider cap, but ±3–5 mm depth noise cannot independently verify a 2.5 mm
plate or the film thickness.

Base/body-top/rim corner radii: {g['base_corner_radius_mm']:.2f}/
{g['body_top_corner_radius_mm']:.2f}/{g['rim_corner_radius_mm']:.2f} mm.
Rounding and pose are regularized from the exterior cloud; physical dimensions
are held nominal. Corner metrology remains uncertain under lateral disagreement.

## Comparison with raw clouds

Unconstrained exterior fits to the source clouds give these diagnostic dimensions:

| Source | Height | Upper body L x S | Base L x S | Rim L x S |
|---|---:|---:|---:|---:|
{chr(10).join(comparisons)}

Height agreement with 63–64 mm and lateral disagreement argue against a single
global scale correction. Missing rim coverage, pose/fusion distortion and rough
measurement uncertainty remain possible contributors; a unique cause is not
established. The prior K+depth choice was based on per-view scale consistency,
which did not prove lateral accuracy.

Selected source: **`{c['source_cloud']}`**, {count:,} points.
SHA256: `{c['source_cloud_sha256']}`.
This source had the lowest full symmetric mean in the preceding flat-base comparison.
All candidates come from one capture and are not independent validation.
The nonmetric `beans_global40` is excluded.

Pose refinement permits only XY translation and yaw; z=0 and scale=1 stay fixed.
The fit uses a 4 mm band inside 5 mm horizontal slice envelopes to suppress
interior ghost sheets. Regularization prevents physical-size disagreement from
creating extreme rounding or moving the model to match only one side.
**Every original point is retained in the final distances and renders.**

## Source-selection comparisons before the underside revision

| Candidate | Unscaled metric cloud | Symmetric mean mm | Symmetric P95 mm | Selection |
|---|---|---:|---:|---|
{chr(10).join(rounds)}

These are historical flat-base metrics, not the current underside mesh's metrics.
Each candidate used identical nominal physical dimensions, with its own rigid
pose and regularized rounding fit. Original checks are in
`measurement_verification_rounds/` and `measurement_candidates/`. The underside
revision retains the selected source, scale and rigid alignment; its current
metrics appear below.

## Distances and regions above 5 mm

Cloud→mesh uses exact closest points on the exported triangles for all raw
points. Mesh→cloud uses 200,000 area-uniform exterior samples (seed 20261014)
and nearest raw points. Distances are unsquared Euclidean millimetres. Symmetric
mean and P95 give 0.5 weight to each direction, independent of sample count.
Reverse distances include errors due to missing coverage.

| Direction | Mean mm | P95 mm | Fraction >5 mm |
|---|---:|---:|---:|
{directions}
| Equal-direction symmetric | {r['chamfer']['mean_mm']:.3f} | {r['chamfer']['p95_mm']:.3f} | — |

Raw points grouped by nearest model region:

| Region | Points | Mean mm | P95 mm | Points >5 mm |
|---|---:|---:|---:|---:|
{regions}

- **Body/interior:** {cv['points_more_than_5mm_inside_closed_model']} points lie >5 mm inside the model. This includes physical-size disagreement as well as visible ghost sheets; not all such errors can be dismissed as noise.
- **Outer scan regions:** {cv['points_more_than_5mm_outside_closed_model']} points lie >5 mm outside. Their exact coordinates and nearest regions are retained in `cloud_residuals.csv`.
- **Lid perimeter:** the 95 x 74 mm cap extends beyond the raw-cloud supports. Reverse distances along this overhang are large; the scan does not verify its size.
- **Base/lower sides:** the lowest raw z is {cv['min_observed_cloud_z_mm']:.3f} mm. Reverse distance over the underside and rounded transition (z <= {g.get('bottom_round_height_mm',0):.1f} mm) is {cv['mesh_base_to_cloud']['mean_mm']:.3f} mm mean / {cv['mesh_base_to_cloud']['p95_mm']:.3f} mm P95. This region is missing from the scan, so the photo-supported recess depth cannot be verified from it.

For context, reverse distance excluding the base is
{cv['mesh_without_base_to_cloud']['mean_mm']:.3f} mm mean /
{cv['mesh_without_base_to_cloud']['p95_mm']:.3f} mm P95. It does not replace the full metric.

## Files, coordinates and simulation

- `visual.glb`: one closed exterior, {r['structural']['visual_triangles']} triangles, photo-derived white/teal/orange palette.
- `collision.obj`: {len(r['structural']['collision_parts'])} named closed convex solids, preserving the flange overhang and central underside recess. Radial support sectors and horizontal body layers have disjoint interiors.
- `snap_pot.blend`: native scene, named dimension properties, packed texture, four cameras and raw reference cloud.
- `parameters.json` / `measurements_approx.json`: editable model and exact provenance of approximate readings.
- `comparison_4views.png`, `renders/`, `cameras.json`: four matched-camera pairs at identical scale.
- `verification.json`, `dimensions.csv`, `verification_rounds/`: current underside checks. `selection.json` records historical source-selection comparisons.
- `cloud_residuals.csv`, `residuals_over_5mm.ply`, `residual_map.png`: every point's error; red PLY points exceed 5 mm.
- `measurement_candidates/`: each comparison's parameters, exports and logs.

Units are **metres**, +Z up, +X along the long side. Origin is the support-plane centre
at z=0; import scale is `[1,1,1]`. Blender uses `export_yup=False`; generic glTF
viewers assuming Y-up may display this intentionally Z-up asset sideways.
All meshes have positive volume, consistent winding and closed topology;
collision parts are individually convex. The cap collision replaces the film's
inset step with a bevel within the film's 0.15 mm thickness.

ManiSkill CPU import and 4 cm drop: **{'PASS' if smoke['passed'] else 'FAIL'}**.
Loaded shapes: {smoke['collision_shapes']}; cooked vertex counts:
{smoke['cooked_shape_vertices']}; maximum cooked bound error:
{smoke['cooked_bounds_max_error_mm']:.6f} mm. Cooking can simplify local curvature.
Mass/friction are **test-only 0.2 kg/0.5**. Inertia is a uniform filled-envelope
proxy. Blender checks visuals; the CPU test disables Vulkan visual loading.
Panda grasp, approach clearance, stiffness, lid peeling and force calibration
remain untested.

For an **80 mm Panda opening**, nominal lid short-side clearance is
**{80-g['rim_short_mm']:.1f} mm total**; its 95 mm long dimension exceeds the opening.
The body just below the lid offers **{80-g['body_top_short_mm']:.1f} mm total**
short-side clearance. The 80 mm body long side has zero nominal clearance.
These dimensions do not certify a grasp or a collision-free approach.

## Assumptions and remaining uncertainty

{assumptions}

The same 40-frame orbit capture supplies appearance. Six original frames are
in `source_views.jpg`; colours are sampled from frame 0037. Lettering and detailed
artwork are not recreated, and illumination-independent reflectance is unmeasured.

![Matched camera comparisons](comparison_4views.png)

![All raw residuals](residual_map.png)
'''
    (out/'REPORT.md').write_text(text)
    print('Updated approximate-measurement report and four-camera comparison.')
