"""Anchor physical size to approximate user measurements; fit pose/rounding.

Cloud coordinates are never scaled. Incompatible size evidence is reported,
not hidden by deforming the reference or shrinking the model to ghost surfaces.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares

from fit import fit_cloud, exterior_band, local_xy, rectangle_sdf


def refine(cloud,measurements):
    original=fit_cloud(cloud,bootstrap=0,fit_band_mm=4,rim_percentile=97.5)
    measured=json.loads(measurements.read_text())
    g=original['geometry'].copy();g.update(measured['dimensions_mm'])
    g.update(measured.get('bottom_geometry_estimates',{}))
    # User's 2.5 mm describes the complete flat top assembly, not foil thickness.
    g['lid_thickness_mm']=.15
    g['rim_thickness_mm']=g['top_assembly_thickness_mm']-g['lid_thickness_mm']
    points=np.asarray(trimesh.load(cloud).vertices)*1000
    T=np.array(original['cloud_to_model']);yaw=np.arctan2(T[0,1],T[0,0])
    centre=-T[:2,:2].T@T[:2,3]*1000
    band=exterior_band(points,4)
    sides=points[band&(points[:,2]>8)&(points[:,2]<g['height_mm']-7)]
    bottom_z=g.get('bottom_round_height_mm',0)
    t=np.clip((sides[:,2]-bottom_z)/(g['height_mm']-g['top_assembly_thickness_mm']-bottom_z),0,1)
    lengths=g['base_long_mm']+(g['body_top_long_mm']-g['base_long_mm'])*t
    widths=g['base_short_mm']+(g['body_top_short_mm']-g['base_short_mm'])*t
    # Roundness priors come from the original exterior fit, capped to avoid
    # turning size disagreement into nearly semicircular, unobserved corners.
    rb=min(original['geometry']['base_corner_radius_mm'],g['base_short_mm']*.25)
    rt=min(original['geometry']['body_top_corner_radius_mm'],g['body_top_short_mm']*.25)
    seed=np.r_[centre,yaw,rb,rt]
    low=np.r_[centre-2,yaw-np.deg2rad(5),max(1,rb-2),max(1,rt-2)]
    high=np.r_[centre+2,yaw+np.deg2rad(5),min(rb+2,g['base_short_mm']*.25),min(rt+2,g['body_top_short_mm']*.25)]
    # Normalizing the scan block prevents point density from overwhelming the
    # regularizers. No independent uncertainty estimate is implied by weights.
    def residual(v):
        xy=local_xy(sides,*v[:3]);r=v[3]+(v[4]-v[3])*t
        sdf=rectangle_sdf(xy,lengths,widths,r)
        robust=np.sign(sdf)*np.sqrt(2*9*(np.sqrt(1+(sdf/3)**2)-1))
        prior=(v-seed)/np.array([1,1,np.deg2rad(2),1,1])
        return np.r_[robust/np.sqrt(len(sides)),prior]
    result=least_squares(residual,seed,bounds=(low,high))
    cx,cy,angle,rb,rt=result.x
    g['base_corner_radius_mm']=float(rb);g['body_top_corner_radius_mm']=float(rt)
    g['rim_corner_radius_mm']=float(min(rt+min(g['rim_long_mm']-g['body_top_long_mm'],g['rim_short_mm']-g['body_top_short_mm'])/2,g['rim_short_mm']*.2))
    ca,sa=np.cos(angle),np.sin(angle);newT=np.eye(4);newT[:2,:2]=[[ca,sa],[-sa,ca]]
    newT[:2,3]=-newT[:2,:2]@np.array([cx,cy])/1000
    dims=original['geometry'].copy()
    original.update(schema_version=2,status='APPROXIMATE_MEASUREMENT_ANCHORED_UNVERIFIED',
        geometry=g,cloud_to_model=newT.tolist(),measurements_mm=measured['dimensions_mm'],
        measurement_quality=measured['quality'],measurement_metadata=measured,
        fit=dict(method='Approximate user dimensions held nominal; robust exterior pose and regularized corner-radius fit.',
                 nominal_dimensions_held=True,cloud_scale=1.0,cloud_rigid_alignment_only=True,
                 side_points=len(sides),total_points=len(points),fit_envelope_band_mm=4,
                 reference_cloud_dimensions_mm=dims,
                 observed_top_z_p90_p99_mm=np.percentile(points[:,2],[90,99]).tolist(),
                 radius_prior_mm=[float(seed[3]),float(seed[4])],
                 pose_refinement_xy_mm=(result.x[:2]-centre).tolist(),pose_refinement_yaw_deg=float(np.rad2deg(angle-yaw)),
                 min_observed_z_mm=float(points[:,2].min())),
        assumptions=[
          'User dimensions are approximate nominal anchors, not certified ruler ground truth; their uncertainty was not supplied.',
          'Total height is 63.5 mm, the midpoint of the reported 63-64 mm range.',
          'The 2.5 mm flat top assembly includes rim plastic and film; it does not mean 2.5 mm film.',
          'Film thickness 0.15 mm and film inset 1.0 mm remain unresolved appearance/geometry assumptions; rim plastic is derived as 2.35 mm.',
          'Flat base at z=0; no underside recess, hidden cavity, contents, ribs, or unseen tab geometry.',
          'Centered rounded-rectangle sections taper linearly from the stated upper body to base dimensions.',
          'Corners use a regularized exterior-cloud estimate; lateral size disagreement prevents reliable corner metrology.',
          'Raw cloud is rigidly aligned in XY/yaw only; no isotropic or anisotropic rescaling, Z offset, cropping or residual trimming.',
          'Significant lateral disagreement with raw fused depth remains; repeated internal surfaces and incomplete coverage do not fully explain physical scale.',
          'Nominal dimensions are retained instead of shrinking to incompatible depth; actual dimensions may differ from these approximate readings.'
        ])
    if 'bottom_geometry_estimates' in measured:
        original.update(schema_version=3,revision=measured.get('model_revision','bottom_photo_v3'),bottom_reference=measured['bottom_reference'])
        original['assumptions']=[s for s in original['assumptions'] if 'Flat base' not in s and 'taper linearly' not in s]+measured['bottom_assumptions']
    return original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cloud',type=Path,required=True)
    p.add_argument('--measurements',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();c=refine(a.cloud,a.measurements);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps(dict(source=str(a.cloud),geometry=c['geometry']),indent=2))
