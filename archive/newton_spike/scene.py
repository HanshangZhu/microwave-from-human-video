"""Newton (MuJoCo-Warp solver) rebuild of MicrowavePot-v0 (sim/microwave_env.py) for a replay spike.

Mirrors the ManiSkill/SAPIEN scene:
  - Panda panda_v2.urdf at the world origin, gravity off on all robot links (ManiSkill's
    balance_passive_force=True disables link gravity) -> MuJoCo body gravcomp = 1.
  - PD drives: arm & fingers kp=1e3, kd=1e2, effort limit 100 (MuJoCo position actuators,
    implicitfast integrator treats kd implicitly, PhysX drives are implicit too).
  - Fingers: two independent drives with the same target (ManiSkill's mimic *controller*);
    the URDF <mimic> equality constraint Newton creates is disabled (SAPIEN does not simulate it).
  - Microwave URDF fixed root at T_W_MW; door_hinge damping 0.05 (Newton maps URDF damping to an
    actuator kd with kp=0, i.e. a pure damper) and frictionloss 0.05 (set by hand: the URDF importer
    parses but drops <dynamics friction>).
  - Pot: single convex hull of collision.obj, mass 0.25, com (0,0,0.03), inertia diag 4e-4.
  - Counter = ground plane z = 0.
Friction: MuJoCo combines per-geom friction with max(), PhysX averages. We choose per-geom mu so
that max() reproduces the PhysX *static* averages on the pairs that matter:
  finger-pot  avg(2.0, 1.0)=1.5  -> finger mu 1.5
  pot-ground / pot-turntable avg(1.0, 0.3)=0.65 -> pot mu 0.65 (others 0.3)
  (finger-handle becomes max(1.5,0.3)=1.5 vs PhysX 1.15 -- the residual mismatch)
"""
import copy
import json
import os

import numpy as np
import warp as wp

import newton

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANDA_URDF = os.path.expanduser(
    "~/miniforge3/envs/maniskill/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf")
MW_URDF = os.path.join(ROOT, "assets/microwave/microwave.urdf")
MW_PARAMS = os.path.join(ROOT, "assets/microwave/parameters.json")
POT_OBJ = os.path.join(ROOT, "assets/heinz_snap_pot/collision.obj")

ROBOT_IN_MW = np.array([0.25, -0.70])
R_W_MW = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T_W_MW_P = -R_W_MW @ np.r_[ROBOT_IN_MW, 0.0]
POT_MASS = 0.25
PARAMS = json.load(open(MW_PARAMS))["derived"]


def world_to_mw(p):
    return (np.asarray(p, float) - T_W_MW_P) @ R_W_MW


def load_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                f.append([idx[0], idx[k], idx[k + 1]])
    return np.array(v, np.float32), np.array(f, np.int32).reshape(-1)


def load_obj_groups(path):
    """Split an OBJ by its 'o' objects (collision.obj holds 35 named convex parts: body, lid rim, feet...).
    SAPIEN's add_multiple_convex_collisions_from_file makes one convex per part; one hull of the whole
    file would wrongly fatten the 5.5 cm body to the 7.4 cm lid-rim width."""
    verts, groups, cur = [], [], None
    for line in open(path):
        if line.startswith("o ") or line.startswith("g "):
            cur = []
            groups.append(cur)
        elif line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            if cur is None:
                cur = []
                groups.append(cur)
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                cur.append([idx[0], idx[k], idx[k + 1]])
    verts = np.array(verts, np.float32)
    out = []
    for g in groups:
        if not g:
            continue
        g = np.array(g)
        used = np.unique(g)
        remap = -np.ones(len(verts), np.int64)
        remap[used] = np.arange(len(used))
        out.append((verts[used], remap[g].astype(np.int32).reshape(-1)))
    return out


def build_world(cfg):
    """One world. cfg: dict of knobs (finger_mu, pot_mu, default_mu)."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    # contact softness: ke/kd -> MuJoCo solref (timeconst = 2/kd, dampratio = kd/2*sqrt(1/ke));
    # default ke=2500, kd=100 -> solref (0.02, 1.0), MuJoCo's default
    tc = cfg.get("contact_timeconst", 0.02)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=cfg.get("default_mu", 0.3), restitution=0.0,
                                                mu_torsional=0.0, mu_rolling=0.0,
                                                kd=2.0 / tc, ke=(1.0 / tc) ** 2)
    b.default_shape_cfg = shape_cfg

    # ---- Panda ----
    b.add_urdf(PANDA_URDF, xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), floating=False,
               enable_self_collisions=False, collapse_fixed_joints=False)
    n_panda_bodies = b.body_count
    # disable the URDF mimic equality (SAPIEN/ManiSkill use two independent drives)
    for i in range(len(b.constraint_mimic_enabled)):
        b.constraint_mimic_enabled[i] = False

    # ---- microwave ----
    q_w_mw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2)
    b.add_urdf(MW_URDF, xform=wp.transform(tuple(T_W_MW_P), q_w_mw), floating=False,
               enable_self_collisions=False)

    # ---- pot ----
    pot_cfg = copy.deepcopy(shape_cfg)
    pot_cfg.mu = cfg.get("pot_mu", 0.65)
    pot_cfg.density = 0.0
    inertia = wp.mat33(4e-4, 0, 0, 0, 4e-4, 0, 0, 0, 4e-4)
    pot = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), mass=POT_MASS,
                     com=wp.vec3(0.0, 0.0, 0.03), inertia=inertia, lock_inertia=True, label="pot")
    if cfg.get("pot_single_hull", False):
        v, f = load_obj(POT_OBJ)
        b.add_shape_convex_hull(pot, mesh=newton.Mesh(v, f, compute_inertia=False), cfg=pot_cfg)
    else:
        for v, f in load_obj_groups(POT_OBJ):
            b.add_shape_convex_hull(pot, mesh=newton.Mesh(v, f, compute_inertia=False), cfg=pot_cfg)

    # ---- indices / per-shape materials ----
    def body(name):
        return next(i for i, l in enumerate(b.body_label) if l.split("/")[-1] == name)

    fingers = {body("panda_leftfinger"), body("panda_rightfinger")}
    for s, bi in enumerate(b.shape_body):
        if bi in fingers:
            b.shape_material_mu[s] = cfg.get("finger_mu", 1.5)

    # joint dofs: panda 9 (7 arm + 2 fingers), microwave door 1, pot free (7 q / 6 qd)
    def joint(name):
        return next(i for i, l in enumerate(b.joint_label) if l.split("/")[-1] == name)

    door_j = joint("door_hinge")
    door_dof = b.joint_qd_start[door_j]
    door_q = b.joint_q_start[door_j]
    b.joint_friction[door_dof] = 0.05
    # drives
    for d in range(9):
        b.joint_target_ke[d] = 1e3
        b.joint_target_kd[d] = 1e2
        b.joint_effort_limit[d] = 100.0
        b.joint_armature[d] = cfg.get("armature", 0.0) if d < 7 else cfg.get("finger_armature", 0.0)
        b.joint_friction[d] = 0.0
    # gravity off on robot links
    grav = b.custom_attributes["mujoco:gravcomp"] if "mujoco:gravcomp" in b.custom_attributes else None
    info = dict(n_panda_bodies=n_panda_bodies, pot_body=pot, door_q=door_q, door_dof=door_dof,
                tcp_body=body("panda_hand_tcp"), pot_q=b.joint_q_start[joint("pot_free_joint")],
                n_bodies=b.body_count, n_q=b.joint_coord_count, n_qd=b.joint_dof_count,
                gravcomp_attr=grav)
    return b, info


def build_model(n_worlds, cfg):
    wb, info = build_world(cfg)
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(wb, n_worlds)
    tc = cfg.get("contact_timeconst", 0.02)
    scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=cfg.get("default_mu", 0.3), mu_torsional=0.0,
                                                               mu_rolling=0.0, kd=2.0 / tc, ke=(1.0 / tc) ** 2))
    model = scene.finalize()
    # gravcomp = 1 on all panda bodies in every world
    gc = model.mujoco.gravcomp.numpy()
    nb = info["n_bodies"]
    for w in range(n_worlds):
        gc[w * nb: w * nb + info["n_panda_bodies"]] = 1.0
    model.mujoco.gravcomp.assign(gc)
    return model, info
