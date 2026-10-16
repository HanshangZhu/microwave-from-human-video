"""Independent exported-mesh QA and untrimmed, unsquared Chamfer in millimetres."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


def stats(d):
    d=np.asarray(d)*1000
    return dict(count=len(d),mean_mm=float(d.mean()),p95_mm=float(np.percentile(d,95)),
                max_mm=float(d.max()),over_5mm_count=int((d>5).sum()),over_5mm_fraction=float((d>5).mean())) if len(d) else dict(count=0)


def load_obj_parts(path):
    # Respect explicit OBJ objects, independent of importer material merging.
    vertices=[];parts={};name=None
    for line in path.read_text().splitlines():
        words=line.split()
        if not words:continue
        if words[0]=='o':name=words[1];parts[name]=[]
        elif words[0]=='v':vertices.append([float(x) for x in words[1:4]])
        elif words[0]=='f':parts[name].append([int(x.split('/')[0])-1 for x in words[1:]])
    result={}
    for name,faces in parts.items():
        m=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        m.remove_unreferenced_vertices();result[name]=m
    return result


def sample_surface(mesh,count=200000):
    rng=np.random.default_rng(20261014)
    ids=rng.choice(len(mesh.faces),size=count,p=mesh.area_faces/mesh.area)
    u=rng.random(count);v=rng.random(count);u=np.sqrt(u)
    tri=mesh.triangles[ids]
    p=(1-u)[:,None]*tri[:,0]+(u*(1-v))[:,None]*tri[:,1]+(u*v)[:,None]*tri[:,2]
    return p,ids


def weighted_p95(a,b):
    x=np.r_[a,b];w=np.r_[np.full(len(a),.5/len(a)),np.full(len(b),.5/len(b))]
    idx=np.argsort(x);return float(x[idx][np.searchsorted(np.cumsum(w[idx]),.95)])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--asset',required=True,type=Path)
    ap.add_argument('--round',type=int,default=1);a=ap.parse_args();out=a.asset
    c=json.loads((out/'parameters.json').read_text());g=c['geometry']
    mesh=trimesh.load(out/'visual.glb',force='mesh',process=False)
    # glTF intentionally duplicates vertices at material/normal/UV seams.
    # Weld equal positions for topology QA, without changing any coordinate.
    mesh.merge_vertices(merge_tex=True,merge_norm=True);mesh.remove_unreferenced_vertices()
    parts=load_obj_parts(out/'collision.obj')
    cv=np.concatenate([m.vertices for m in parts.values()])
    contact=mesh.vertices[np.isclose(mesh.vertices[:,2],0,atol=1e-7)]
    base_z=g.get('bottom_round_height_mm',0)/1000
    base=mesh.vertices[np.isclose(mesh.vertices[:,2],base_z,atol=1e-7)]
    body=parts['body_convex'];top=body.vertices[np.isclose(body.vertices[:,2],body.bounds[1,2],atol=1e-7)]
    measured=dict(height_mm=np.ptp(mesh.vertices[:,2])*1000,
        body_top_long_mm=np.ptp(top[:,0])*1000,body_top_short_mm=np.ptp(top[:,1])*1000,
        rim_long_mm=np.ptp(mesh.vertices[:,0])*1000,rim_short_mm=np.ptp(mesh.vertices[:,1])*1000,
        base_long_mm=np.ptp(base[:,0])*1000,base_short_mm=np.ptp(base[:,1])*1000)
    if 'top_assembly_thickness_mm' in g:
        measured['top_assembly_thickness_mm']=(mesh.bounds[1,2]-body.bounds[1,2])*1000
    approximate=c.get('measurement_quality')=='APPROXIMATE_USER_MEASUREMENTS'
    dimensions=[]
    for name,value in measured.items():
        ruler=c['measurements_mm'].get(name)
        dimensions.append(dict(dimension=name,parameter_mm=g[name],exported_mm=float(value),
            export_error_mm=float(value-g[name]),ruler_mm=ruler,
            ruler_error_mm=float(value-ruler) if ruler is not None else None,
            measurement_quality=c.get('measurement_quality','UNAVAILABLE'),
            nominal_2mm_check=('PASS' if abs(value-ruler)<=2 else 'FAIL') if ruler is not None else 'UNVERIFIED',
            ruler_check='APPROXIMATE_ONLY' if approximate and ruler is not None else (('PASS' if abs(value-ruler)<=2 else 'FAIL') if ruler is not None else 'UNVERIFIED')))
    print('Dimension                    Model mm Export mm Delta mm   Independent ruler check')
    for d in dimensions:print(f"{d['dimension']:28s} {d['parameter_mm']:7.2f} {d['exported_mm']:10.2f} {d['export_error_mm']:9.4f}  {d['ruler_check']}")
    with (out/'dimensions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=dimensions[0].keys());w.writeheader();w.writerows(dimensions)
    checks={}
    for name,m in parts.items():
        checks[name]=dict(watertight=bool(m.is_watertight),convex=bool(m.is_convex),
            winding_consistent=bool(m.is_winding_consistent),volume_m3=float(m.volume),
            vertices=len(m.vertices),triangles=len(m.faces),
            hull_relative_volume_error=float(abs(m.volume-m.convex_hull.volume)/m.convex_hull.volume))
    all_bounds_error=float(np.max(np.abs(mesh.bounds-np.stack([cv.min(0),cv.max(0)]))))
    structural=dict(visual_watertight=bool(mesh.is_watertight),visual_winding_consistent=bool(mesh.is_winding_consistent),
        visual_connected_components=len(mesh.split(only_watertight=False)),visual_volume_m3=float(mesh.volume),
        visual_triangles=len(mesh.faces),base_z_m=float(mesh.bounds[0,2]),
        base_centre_xy_m=((contact.min(0)+contact.max(0))/2)[:2].tolist(),
        visual_collision_bounds_max_difference_mm=all_bounds_error*1000,collision_parts=checks)
    solid=trimesh.util.concatenate(list(parts.values()))
    structural['uniform_filled_proxy']=dict(centre_of_mass_m=solid.center_mass.tolist(),
        principal_inertia_per_kg_m2=(np.diag(solid.moment_inertia)/solid.volume).tolist(),
        note='Uniform rigid envelope approximation; actual mass distribution unmeasured. Symmetry axes X/Y/Z are principal axes.')
    raw=trimesh.load(c['source_cloud']);points=np.asarray(raw.vertices)
    T=np.array(c['cloud_to_model']);points=points@T[:3,:3].T+T[:3,3]
    om=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(mesh.vertices.astype(np.float32)),
                                   o3d.core.Tensor(mesh.faces.astype(np.uint32)))
    ray=o3d.t.geometry.RaycastingScene();ray.add_triangles(om)
    tensor=o3d.core.Tensor(points.astype(np.float32))
    nearest=ray.compute_closest_points(tensor)
    closest=nearest['points'].numpy();face_ids=nearest['primitive_ids'].numpy().astype(int)
    pm=np.linalg.norm(points-closest,axis=1)
    signed=ray.compute_signed_distance(tensor).numpy()
    samples,sample_faces=sample_surface(mesh)
    mp,ids=cKDTree(points).query(samples,workers=4)
    if 'bottom_recess_depth_mm' in g:
        # Verify the actual exported collision UNION, including the open air gap.
        queries=o3d.core.Tensor(samples[::4].astype(np.float32))
        union_signed=np.full(len(queries),np.inf)
        combined=o3d.t.geometry.RaycastingScene()
        for part in parts.values():
            triangle=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(part.vertices.astype(np.float32)),o3d.core.Tensor(part.faces.astype(np.uint32)))
            pr=o3d.t.geometry.RaycastingScene();pr.add_triangles(triangle);combined.add_triangles(triangle)
            union_signed=np.minimum(union_signed,pr.compute_signed_distance(queries).numpy())
        contact_y=(g['bottom_panel_short_mm']/2+g['bottom_recess_wall_bevel_mm']+g['base_short_mm']/2-g['bottom_round_inset_mm'])/2/1000
        rays=o3d.core.Tensor(np.array([[0,0,-.01,0,0,1],[0,contact_y,-.01,0,0,1]],dtype=np.float32))
        actual=combined.cast_rays(rays)['t_hit'].numpy()-.01
        expected=np.array([g['bottom_recess_depth_mm']/1000,0])
        ray_error=float(np.max(abs(actual-expected))*1000)
        boundary_error=float(np.max(abs(union_signed))*1000)
        # Inspect exported outer-arc vertices at the +X tangent of each XY corner.
        # This distinguishes a geometric fillet from a shading-only change.
        corner_y=(g['base_short_mm']/2-g['base_corner_radius_mm'])/1000
        height=g['bottom_round_height_mm']/1000;inset=g['bottom_round_inset_mm']/1000
        arc=mesh.vertices[(abs(abs(mesh.vertices[:,1])-corner_y)<1e-7)&
                          (mesh.vertices[:,0]>=(g['base_long_mm']/2000-inset)-1e-7)&
                          (mesh.vertices[:,2]<=height+1e-7)]
        expected_x=g['base_long_mm']/2000-inset+inset*np.sqrt(np.maximum(0,1-(1-arc[:,2]/height)**2))
        arc_error=float(np.max(abs(arc[:,0]-expected_x))*1000)
        structural['bottom_fillet']=dict(height_mm=g['bottom_round_height_mm'],inset_mm=g['bottom_round_inset_mm'],
            arc_profile_max_error_mm=arc_error,checked_vertices=len(arc),
            distinct_height_levels=len(np.unique(np.round(arc[:,2],9))),
            passed=bool(arc_error<.01 and len(arc)>=2*g['bottom_curve_segments']))
        structural['underside']=dict(recess_depth_estimated_mm=g['bottom_recess_depth_mm'],
            collision_centre_first_surface_z_mm=float(actual[0]*1000),collision_support_first_surface_z_mm=float(actual[1]*1000),
            support_outer_dimensions_mm=(np.ptp(contact[:,:2],axis=0)*1000).tolist(),
            base_outline_reference_z_mm=base_z*1000,ray_error_mm=ray_error,
            visual_to_collision_union_sample_max_error_mm=boundary_error,surface_samples=50000,
            passed=bool(ray_error<.01 and boundary_error<=1))
    zbody=(g['height_mm']-g['rim_thickness_mm']-g['lid_thickness_mm'])/1000
    regions=np.where(closest[:,2]<=base_z+1e-6,'base',np.where(closest[:,2]>=zbody-1e-6,'rim_or_lid','body'))
    direction=dict(cloud_to_mesh=stats(pm),mesh_to_cloud=stats(mp))
    balanced=dict(mean_mm=float((pm.mean()+mp.mean())*500),p95_mm=weighted_p95(pm,mp)*1000,
        definition='Unsquared Euclidean distance; equal 0.5 weight per direction, p95 of this equally weighted mixture.',
        cloud_to_mesh_method='Exact point-to-exported-triangle closest distance; every original finite point; no trimming.',
        mesh_to_cloud_method='200000 area-uniform exterior mesh samples, fixed seed 20261014, nearest raw cloud point.',
        point_count=len(points),surface_sample_count=len(samples))
    samples_base=samples[:,2]<=base_z+1e-7
    over=pm>.005;interior=over&(signed<-.005)
    regions_report={r:stats(pm[regions==r]) for r in ['base','body','rim_or_lid']}
    missing=dict(mesh_base_to_cloud=stats(mp[samples_base]),
        mesh_without_base_to_cloud=stats(mp[~samples_base]),
        min_observed_cloud_z_mm=float(points[:,2].min()*1000),
        points_more_than_5mm_inside_closed_model=int(interior.sum()),
        points_more_than_5mm_outside_closed_model=int((over&~interior).sum()))
    # Save every residual and source index so no bad region is silently discarded.
    with (out/'cloud_residuals.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['source_index','x_m','y_m','z_m','distance_mm','signed_distance_mm','region','over_5mm'])
        for i in range(len(points)):w.writerow([i,*points[i],pm[i]*1000,signed[i]*1000,regions[i],bool(over[i])])
    palette=np.tile([58,166,130,255],(len(points),1));palette[over]=[229,65,60,255]
    trimesh.points.PointCloud(points,colors=palette).export(out/'residuals_over_5mm.ply')
    structure_pass=(structural['visual_watertight'] and structural['visual_winding_consistent'] and
        structural['visual_connected_components']==1 and structural['visual_volume_m3']>0 and
        all(x['convex'] and x['watertight'] and x['winding_consistent'] and x['volume_m3']>0 for x in checks.values()) and
        abs(structural['base_z_m'])<1e-7 and np.max(np.abs(structural['base_centre_xy_m']))<1e-7 and
        all_bounds_error<1e-6 and structural.get('underside',{}).get('passed',True) and
        structural.get('bottom_fillet',{}).get('passed',True))
    fit_pass=balanced['p95_mm']<=5
    ground_truth_pass=all(d['ruler_check']=='PASS' for d in dimensions)
    # 5 mm is a diagnostic acceptance target; raw noise can exceed it.
    report=dict(round=a.round,status='PASS' if structure_pass and fit_pass and ground_truth_pass else 'NOT_FULLY_VERIFIED',
        structural_pass=bool(structure_pass),ruler_2mm_pass=bool(ground_truth_pass),
        approximate_nominal_dimensions_2mm_pass=all(d['nominal_2mm_check']=='PASS' for d in dimensions),
        raw_symmetric_chamfer_p95_le_5mm=bool(fit_pass),
        parameter_export_2mm_pass=all(abs(d['export_error_mm'])<=2 for d in dimensions),
        dimensions=dimensions,structural=structural,chamfer=balanced,directional=direction,
        cloud_to_mesh_by_nearest_region=regions_report,coverage_and_outliers=missing,
        gripper=dict(max_opening_mm=80,rim_short_clearance_mm=80-g['rim_short_mm'],
            rim_long_clearance_mm=80-g['rim_long_mm'],
            body_short_clearance_mm=80-g['body_top_short_mm'],
            body_long_clearance_mm=80-g['body_top_long_mm'],
            note='Geometric opening only; no Panda grasp, force/friction or collision-free approach certified.'),
        assumptions=c['assumptions'])
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    history=out/'verification_rounds';history.mkdir(exist_ok=True)
    (history/f'round_{a.round}.json').write_text(json.dumps(report,indent=2)+'\n')
    (history/f'parameters_{a.round}.json').write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','structural_pass','chamfer','directional','coverage_and_outliers']},indent=2))
    if not structure_pass:raise SystemExit('Structural export checks FAILED')


if __name__=='__main__':main()
