"""Load the metre / Z-up asset into an existing ManiSkill scene."""
import json
from pathlib import Path

import sapien


def build_snap_pot(scene, asset_dir, *, mass_kg, friction, initial_pose=None,
                   name='heinz_snap_pot', restitution=0.0):
    """Mass/friction must come from the task; the scan cannot measure them.

    Inertia is a uniform-filled-envelope proxy. This does not model flexible
    plastic, liquid/sloshing contents, lid peeling or a hollow empty container.
    """
    root=Path(asset_dir).resolve()
    report=json.loads((root/'verification.json').read_text())
    if not report['structural_pass']:
        raise ValueError('Asset failed structural verification')
    if mass_kg<=0 or friction<0 or not 0<=restitution<=1:
        raise ValueError('Invalid physical parameters')
    volume=sum(x['volume_m3'] for x in report['structural']['collision_parts'].values())
    builder=scene.create_actor_builder()
    mat=sapien.physx.PhysxMaterial(friction,friction,restitution)
    builder.add_multiple_convex_collisions_from_file(
        filename=str(root/'collision.obj'),scale=[1,1,1],material=mat,
        density=mass_kg/volume,decomposition='none')
    builder.add_visual_from_file(filename=str(root/'visual.glb'),scale=[1,1,1])
    proxy=report['structural']['uniform_filled_proxy']
    # PhysX cooking may simplify hulls; preserve the caller's explicit mass and
    # the documented proxy inertia instead of inferring mass from cooked volume.
    builder.set_mass_and_inertia(mass_kg,sapien.Pose(p=proxy['centre_of_mass_m']),
        [mass_kg*x for x in proxy['principal_inertia_per_kg_m2']])
    builder.initial_pose=initial_pose if initial_pose is not None else sapien.Pose(p=[0,0,.001])
    return builder.build(name=name)

