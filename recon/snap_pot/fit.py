#!/usr/bin/env python3
"""Fit a tapered rounded rectangle to metric depth, without rescaling the cloud."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull


def local_xy(points, cx, cy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return (points[:, :2] - [cx, cy]) @ np.array([[c, -s], [s, c]])


def rectangle_sdf(xy, length, width, radius):
    q = np.abs(xy) - np.stack(np.broadcast_arrays(length / 2 - radius, width / 2 - radius), axis=-1)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - radius


def decode(v):
    cx, cy, yaw, bl, bs, dl, ds, rb, rt = v
    return cx, cy, yaw, bl, bs, bl + dl, bs + ds, rb, rt


def fit_sides(points, body_height, initial=None, max_nfev=400):
    z = np.clip(points[:, 2] / body_height, 0, 1)

    def residual(v):
        cx, cy, yaw, bl, bs, tl, ts, rb, rt = decode(v)
        xy = local_xy(points, cx, cy, yaw)
        length, width, radius = bl + (tl - bl)*z, bs + (ts - bs)*z, rb + (rt-rb)*z
        sdf = rectangle_sdf(xy, length, width, radius)
        # Radius cannot exceed half the short side at either end.
        return np.r_[sdf, 10*max(rb-bs/2+.1, 0), 10*max(rt-ts/2+.1, 0)]

    lower = [-30, -30, -8, 20, 20, 0, 0, 1, 1]
    upper = [30, 30, 8, 110, 110, 45, 45, 25, 25]
    starts = [initial] if initial is not None else [
        [*np.median(points[:, :2], axis=0), a, 50, 40, 12, 12, 8, 10]
        for a in np.linspace(0, np.pi, 8, endpoint=False)
    ]
    # Cauchy loss prevents the duplicate interior sheets present in this scan
    # from shrinking the exterior. Verification always uses every raw point.
    results = [least_squares(residual, v, bounds=(lower, upper), loss='cauchy',
                             f_scale=1.5, max_nfev=max_nfev) for v in starts]
    result = min(results, key=lambda x: x.cost)
    v = result.x
    if v[3] + v[5] < v[4] + v[6]:
        v[[3, 4]] = v[[4, 3]]
        v[[5, 6]] = v[[6, 5]]
        v[2] += np.pi/2
    v[2] = (v[2] + np.pi/2) % np.pi - np.pi/2
    return v, residual(v)[:-2]


def exterior_band(points, width_mm=4.0):
    """Fit-only mask: retain points near each 5 mm slice's convex envelope.

    This rejects duplicate interior reconstruction sheets, not validation points.
    All input points remain in the distance report and side-by-side renders.
    """
    mask=np.zeros(len(points),dtype=bool)
    for low in np.arange(5,points[:,2].max(),5):
        ids=np.where((points[:,2]>=low)&(points[:,2]<low+5))[0]
        if len(ids)<8:continue
        eq=ConvexHull(points[ids,:2]).equations
        inward=-(points[ids,:2] @ eq[:,:2].T + eq[:,2]).max(axis=1)
        mask[ids]=inward<=width_mm
    return mask


def fit_cloud(path, bootstrap=30, fit_band_mm=4.0, rim_percentile=95):
    raw = np.asarray(trimesh.load(path).vertices)
    if not np.isfinite(raw).all() or len(raw) < 100:
        raise ValueError('Expected at least 100 finite metric cloud points')
    points = raw * 1000
    # Central high surface determines height; do not use raw extrema.
    h0 = np.percentile(points[:, 2], 95)
    envelope_mask=exterior_band(points,fit_band_mm)
    sides = points[envelope_mask & (points[:, 2] > 8) & (points[:, 2] < h0-7)]
    v, residual = fit_sides(sides, h0-1.35)
    cx, cy, yaw, bl, bs, tl, ts, rb, rt = decode(v)
    xy = local_xy(points, cx, cy, yaw)
    central_top = (points[:, 2] > h0-5) & (np.abs(xy[:, 0]) < tl*.32) & (np.abs(xy[:, 1]) < ts*.32)
    height = float(np.median(points[central_top, 2]))
    body_height = height - 1.35
    sides = points[envelope_mask & (points[:, 2] > 8) & (points[:, 2] < height-7)]
    v, residual = fit_sides(sides, body_height, initial=v)
    cx, cy, yaw, bl, bs, tl, ts, rb, rt = decode(v)
    xy = local_xy(points, cx, cy, yaw)
    top = xy[points[:, 2] > height-5]
    # Estimate opposing rim supports via medians of the outer 20% of each side.
    # The full rim perimeter is sparsely sampled: document this estimator.
    rim_l = float(np.percentile(top[:, 0], rim_percentile) - np.percentile(top[:, 0], 100-rim_percentile))
    rim_s = float(np.percentile(top[:, 1], rim_percentile) - np.percentile(top[:, 1], 100-rim_percentile))
    rim_l, rim_s = max(rim_l, tl+2), max(rim_s, ts+2)
    geom = dict(height_mm=height, body_top_long_mm=tl, body_top_short_mm=ts,
                base_long_mm=bl, base_short_mm=bs, base_corner_radius_mm=rb,
                body_top_corner_radius_mm=rt, rim_long_mm=rim_l, rim_short_mm=rim_s,
                rim_corner_radius_mm=rt+min(rim_l-tl, rim_s-ts)/2,
                rim_thickness_mm=1.2, lid_thickness_mm=.15, lid_inset_mm=1.0,
                visual_corner_segments=24, collision_corner_segments=12)
    # Sampling uncertainty only. Independent point resampling understates correlated
    # reconstruction errors, so never interpret these as absolute accuracy intervals.
    rng = np.random.default_rng(1729)
    samples = []
    for _ in range(bootstrap):
        vv, _ = fit_sides(sides[rng.integers(len(sides), size=len(sides))], body_height, v, 120)
        samples.append(decode(vv)[3:])
    names = ['base_long_mm','base_short_mm','body_top_long_mm','body_top_short_mm',
             'base_corner_radius_mm','body_top_corner_radius_mm']
    intervals = {n: np.percentile(np.array(samples)[:,i], [2.5,97.5]).tolist()
                 for i,n in enumerate(names)} if samples else {}
    c,s=np.cos(yaw),np.sin(yaw)
    transform = np.eye(4)
    transform[:2,:2] = [[c,s],[-s,c]]
    transform[:2,3] = -transform[:2,:2] @ (np.array([cx,cy])/1000)
    return dict(schema_version=1, status='POINT_CLOUD_ESTIMATE_NOT_RULER_VERIFIED',
                source_cloud=str(path), source_cloud_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                units='mm in geometry, metres in exports', geometry=geom,
                cloud_to_model=transform.tolist(),
                measurements_mm={n:None for n in ['height_mm','body_top_long_mm','body_top_short_mm',
                                                'rim_long_mm','rim_short_mm','base_long_mm','base_short_mm']},
                fit=dict(side_points=len(sides), total_points=len(points), fit_envelope_band_mm=fit_band_mm,
                         rim_support_percentile=rim_percentile,
                         rejected_for_fit_only=int((~envelope_mask).sum()),
                         side_abs_residual_mean_mm=float(np.abs(residual).mean()),
                         side_abs_residual_p95_mm=float(np.percentile(np.abs(residual),95)),
                         bootstrap_95pct_sampling_only=intervals,
                         min_observed_z_mm=float(points[:,2].min()),
                         cloud_rigid_alignment_only=True, scale=1.0),
                assumptions=[
                    'User authorized dimensions estimated from point cloud; ruler values unavailable.',
                    'Selected beans_fuse_K_depth: metric RealSense TSDF, intrinsics and depth conditioned poses.',
                    'Base is flat at table z=0; footprint extrapolated through missing lowest depth samples.',
                    'Body cross sections are centered rounded rectangles, with linearly varying dimensions and radius.',
                    'Lid is planar; scan curvature is not reproduced as physical film geometry.',
                    'Rim thickness 1.2 mm, film thickness 0.15 mm, film inset 1.0 mm are unresolved modeling assumptions.',
                    'No interior cavity, contents, bottom recess, ribs, or unobserved pull-tab geometry modeled.',
                    'Closed outer envelope used for rigid collision; it does not describe plastic wall thickness.',
                    f'Rim outline estimated from {100-rim_percentile:g}th/{rim_percentile:g}th support percentiles of top points; edge is sparsely observed.',
                    'Duplicate interior sheets observed in scan; fit uses outer slice envelopes, but verification uses ALL original points.',
                    'Bootstrap intervals describe sampling stability only, not metric accuracy or correlated scan bias.'
                ])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cloud',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--bootstrap',type=int,default=30)
    p.add_argument('--fit-band-mm',type=float,default=4.0)
    p.add_argument('--rim-percentile',type=float,default=95)
    a=p.parse_args()
    config=fit_cloud(a.cloud,a.bootstrap,a.fit_band_mm,a.rim_percentile)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config['geometry'],indent=2))
    print(json.dumps(config['fit'],indent=2))


if __name__=='__main__':
    main()
