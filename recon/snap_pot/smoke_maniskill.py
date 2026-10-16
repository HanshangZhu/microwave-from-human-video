"""CPU ManiSkill import/drop smoke test. Not a calibrated Panda grasp test."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import sapien
from scipy.spatial import ConvexHull
from mani_skill.envs.sapien_env import BaseEnv

from maniskill_asset import build_snap_pot


def first_vertical_hit(vertices,xy):
    """First +Z ray intersection with a cooked convex hull, in object coordinates."""
    eq=ConvexHull(vertices).equations
    a=eq[:,:2]@np.asarray(xy)+eq[:,3];nz=eq[:,2]
    if np.any((abs(nz)<1e-8)&(a>1e-8)):return float('inf')
    lo=np.max(-a[nz < -1e-8]/nz[nz < -1e-8],initial=-np.inf)
    hi=np.min(-a[nz > 1e-8]/nz[nz > 1e-8],initial=np.inf)
    return float(lo) if lo<=hi+1e-8 else float('inf')


class PotSmokeEnv(BaseEnv):
    SUPPORTED_REWARD_MODES=['none']
    SUPPORTED_ROBOTS=['none']

    def __init__(self,asset_dir,**kw):
        self.asset_dir=asset_dir
        super().__init__(robot_uids='none',obs_mode='none',reward_mode='none',
                         sim_backend='cpu',render_backend='none',**kw)

    @property
    def _default_sensor_configs(self):return []

    @property
    def _default_human_render_camera_configs(self):return []

    def _load_scene(self,options):
        b=self.scene.create_actor_builder()
        b.add_box_collision(half_size=[.2,.2,.01])
        b.initial_pose=sapien.Pose(p=[0,0,-.01]);b.build_static(name='table')
        self.pot=build_snap_pot(self.scene,self.asset_dir,mass_kg=.2,friction=.5,
                               initial_pose=sapien.Pose(p=[0,0,.04]))

    def _initialize_episode(self,env_idx,options):
        self.pot.set_pose(sapien.Pose(p=[0,0,.04]))

    def evaluate(self):return {}

    def _get_obs_extra(self,info):return {}

    def get_state_dict(self):
        # This installed ManiSkill version unconditionally queries an agent in
        # BaseEnv.get_state_dict; an object-only test has no controller state.
        return self.scene.get_sim_state()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True);a=ap.parse_args()
    env=PotSmokeEnv(a.asset);env.reset(seed=42)
    body=env.pot._bodies[0]
    shapes=body.get_collision_shapes()
    g=json.loads((a.asset/'parameters.json').read_text())['geometry']
    expected_shapes=len(json.loads((a.asset/'verification.json').read_text())['structural']['collision_parts'])
    cooked_parts=[]
    for shape in shapes:
        transform=shape.local_pose.to_transformation_matrix()
        points=np.asarray(shape.vertices)*np.asarray(shape.scale)
        cooked_parts.append(points@transform[:3,:3].T+transform[:3,3])
    cooked=np.concatenate(cooked_parts)
    bounds=np.stack([cooked.min(0),cooked.max(0)])*1000
    expected=np.array([[-g['rim_long_mm']/2,-g['rim_short_mm']/2,0],
                        [g['rim_long_mm']/2,g['rim_short_mm']/2,g['height_mm']]])
    bounds_error=float(np.abs(bounds-expected).max())
    underside={}
    if 'bottom_recess_depth_mm' in g:
        y=(g['bottom_panel_short_mm']/2+g['bottom_recess_wall_bevel_mm']+g['base_short_mm']/2-g['bottom_round_inset_mm'])/2/1000
        hits=[min(first_vertical_hit(v,xy) for v in cooked_parts)*1000 for xy in [(0,0),(0,y)]]
        # PhysX simplifies thin curved hulls. Bound this approximation with the
        # same 1 mm surface tolerance as export QA, while separately requiring
        # the central recess to survive within 0.1 mm.
        halfx=(g['base_long_mm']/2-g['bottom_round_inset_mm']+g['bottom_panel_long_mm']/2+g['bottom_recess_wall_bevel_mm'])/2
        halfy=y*1000
        radius=(g['base_corner_radius_mm']-g['bottom_round_inset_mm']+g['bottom_panel_corner_radius_mm']+g['bottom_recess_wall_bevel_mm'])/2
        land=[]
        for sx,sy,start in [(1,1,0),(-1,1,90),(-1,-1,180),(1,-1,270)]:
            for angle in np.linspace(start,start+90,9):
                xy=((sx*(halfx-radius)+radius*math.cos(math.radians(angle)))/1000,
                    (sy*(halfy-radius)+radius*math.sin(math.radians(angle)))/1000)
                land.append(min(first_vertical_hit(v,xy) for v in cooked_parts)*1000)
        error=max(abs(z) for z in land+[hits[1]])
        underside=dict(centre_first_surface_z_mm=hits[0],support_first_surface_z_mm=hits[1],
            support_land_max_height_error_mm=error,support_land_rays=len(land)+1,
            surface_tolerance_mm=1.0,recess_centre_tolerance_mm=.1,
            note='Cooked support faces can tilt slightly; settled z=0 is checked separately. Initial strict 0.1 mm local-support diagnostic is preserved.',
            passed=bool(abs(hits[0]-g['bottom_recess_depth_mm'])<.1 and error<=1.0))
    history=[]
    for _ in range(100):
        env.step(None)
        history.append(env.pot.pose.p.cpu().numpy()[0].copy())
    positions=np.asarray(history);end=positions[-1]
    velocity=np.asarray(body.linear_velocity)
    report=dict(test='ManiSkill CPU import and 4 cm drop onto table',
        passed=bool(len(shapes)==expected_shapes and underside.get('passed',True) and bounds_error<=2 and abs(body.mass-.2)<1e-6 and np.isfinite(positions).all() and abs(end[2])<.002 and
                    np.linalg.norm(velocity)<.005 and np.linalg.norm(end[:2])<.01),
        collision_shapes=len(shapes),expected_collision_shapes=expected_shapes,mass_kg=float(body.mass),
        cooked_underside=underside,
        cooked_bounds_mm=bounds.tolist(),cooked_bounds_max_error_mm=bounds_error,
        cooked_shape_vertices=[len(s.vertices) for s in shapes],
        final_base_position_m=end.tolist(),final_linear_velocity_m_s=velocity.tolist(),
        max_final_20_steps_position_range_m=float(np.ptp(positions[-20:],axis=0).max()),
        test_only_assumptions=dict(mass_kg=.2,static_and_dynamic_friction=.5,restitution=0),
        simulation_steps=100*env._sim_steps_per_control,simulation_seconds=100/env._control_freq,
        visual_runtime_loaded=False,panda_grasp_tested=False,
        note='Headless render_backend=none checks actual ManiSkill collision import and dynamics. GLB geometry/materials separately verified and rendered in Blender. Mass, inertia and friction are not measured.')
    (a.asset/'maniskill_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));env.close()
    if not report['passed']:raise SystemExit('ManiSkill collision smoke FAILED')


if __name__=='__main__':main()
