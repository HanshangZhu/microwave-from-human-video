"""MicrowavePot-v0: put the Snap Pot into the microwave and shut the door.

The scene is the measured digital twin: the microwave URDF built from tape measurements
with its scanned colours (assets/microwave/microwave_textured.urdf) and the Snap Pot
(assets/heinz_snap_pot), on a counter (the ground plane, z = 0). A Panda stands on the
counter at the world origin facing +x (mplib plans for a base at the origin).

Two frames matter:
  world   ManiSkill's: Panda base at the origin, +x forward, z up
  mw      the microwave's own frame, in which everything about the task was measured and
          extracted from the human takes (recon/DEMOS.md): origin on the counter under the
          front-left corner, +x along the front, +y into the microwave, z up.
The microwave is placed so the Panda stands at mw (ROBOT_IN_MW), facing the front (+y_mw).

Success: the pot rests inside the cavity on the turntable and the door is shut (< 5 deg).
"""
import json
import os

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MW_URDF = os.path.join(ROOT, "assets/microwave/microwave_textured.urdf")
MW_PARAMS = os.path.join(ROOT, "assets/microwave/parameters.json")
POT_DIR = os.path.join(ROOT, "assets/heinz_snap_pot")
ROBOT_IN_MW = np.array([0.25, -0.70])         # Panda base, in the microwave's frame: the door arc and the whole
                                               # pick-carry-insert are each one continuous IK path from here (base scan)
POT_MASS = 0.25                                # full 200 g pot + packaging (guess)
# the Panda's "rest" keyframe puts its hand where the microwave stands; tuck it back:
# tcp at (0.36, 0, 0.35), in front of the front face and above the top
INIT_QPOS = [0.0, -0.6, 0.0, -2.6, 0.0, 2.0, np.pi / 4, 0.04, 0.04]

# mw -> world: the microwave turned -90 deg about z, so +y_mw (into it) is +x_world
R_W_MW = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T_W_MW = np.eye(4)
T_W_MW[:3, :3] = R_W_MW
T_W_MW[:3, 3] = -R_W_MW @ np.r_[ROBOT_IN_MW, 0.0]
Q_W_MW = np.array([np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4)])   # wxyz, -90 deg about z


def set_robot_in_mw(xy):
    """Move where the Panda stands relative to the microwave (call before gym.make)."""
    global ROBOT_IN_MW
    ROBOT_IN_MW = np.asarray(xy, float)
    T_W_MW[:3, 3] = -R_W_MW @ np.r_[ROBOT_IN_MW, 0.0]


def mw_to_world(p):
    return np.asarray(p, float) @ R_W_MW.T + T_W_MW[:3, 3]


def world_to_mw(p):
    return (np.asarray(p, float) - T_W_MW[:3, 3]) @ R_W_MW


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


@register_env("MicrowavePot-v0", max_episode_steps=1500)
class MicrowavePotEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", pot_mw=(0.50, -0.25, np.pi / 2), pot_region=None,
                 pot_z=0.0, views=False, backdrop=None, dr_seed=None, dr_style="procedural", **kwargs):
        self.pot_mw = np.asarray(pot_mw, float)          # x, y, yaw of the pot in mw
        self.pot_z = float(pot_z)                         # start height above the counter (0: on it)
        # backdrop: a visual-only mesh in the mw frame (recon/kitchen_backdrop.py -- the scanned
        # kitchen); the cameras then see it instead of the grey ground plane (physics unchanged)
        self.backdrop = backdrop
        self.dr_seed = dr_seed      # sim/dr.py: a random visual scene + lighting (training images)
        self.dr_style = dr_style
        self.views = views          # sim/views.py: True = front + wrist cameras, "hq" = the same, supersampled
        # pot_region ((x0, y0), (x1, y1)): sample the pot's x, y per episode instead
        self.pot_region = None if pot_region is None else np.asarray(pot_region, float)
        self.params = json.load(open(MW_PARAMS))["derived"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        if self.views:
            from sim.views import camera_configs
            return camera_configs(self, hq=self.views == "hq")
        pose = sapien_utils.look_at(eye=[-0.15, -0.55, 0.65], target=[0.45, 0.0, 0.12])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-0.25, -0.75, 0.8], target=[0.5, 0.1, 0.12])
        return CameraConfig("render_camera", pose, 960, 640, np.deg2rad(55), 0.01, 100)

    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        if getattr(self, "_dr_tint_pending", False):
            from sim import dr
            dr.tint(self, self.dr_seed)
            self._dr_tint_pending = False

    def _load_lighting(self, options: dict):
        if self.dr_seed is not None:
            from sim import dr
            dr.set_lighting(self.scene, self.dr_seed)
        else:
            super()._load_lighting(options)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        if self.backdrop or self.dr_seed is not None:
            # the counter as a collision plane only (build_ground's plane, without its grey grid)
            gb = self.scene.create_actor_builder()
            gb.add_plane_collision(sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0]))
            self.ground = gb.build_static(name="ground")
        else:
            self.ground = build_ground(self.scene)
        if self.backdrop:
            b = self.scene.create_actor_builder()
            b.add_visual_from_file(self.backdrop)
            # ManiSkill's static actors need a collision shape: a tiny box 5 m under the counter
            b.add_box_collision(sapien.Pose(p=[0, 0, -5]), [0.01, 0.01, 0.01])
            b.initial_pose = sapien.Pose(p=T_W_MW[:3, 3], q=Q_W_MW)
            self.kitchen = b.build_static(name="backdrop")
        if self.dr_seed is not None:
            from sim import dr
            walls = True if self.dr_style == "procedural" else bool(np.random.default_rng(self.dr_seed).uniform() < 0.4)
            self.dr_scene = dr.build_scene(self.scene, self.dr_seed, T_W_MW, Q_W_MW, walls=walls)
            if self.dr_style == "doorman":
                dr.set_environment(self.scene, self.dr_seed)
                self._dr_tint_pending = True
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        builder = loader.parse(MW_URDF)["articulation_builders"][0]
        builder.initial_pose = sapien.Pose(p=T_W_MW[:3, 3], q=Q_W_MW)
        self.mw = builder.build(name="microwave")
        self.door_joint = self.mw.active_joints[0]

        b = self.scene.create_actor_builder()
        mat = sapien.pysapien.physx.PhysxMaterial(static_friction=1.0, dynamic_friction=0.8, restitution=0)
        b.add_multiple_convex_collisions_from_file(os.path.join(POT_DIR, "collision.obj"),
                                                   decomposition="none", material=mat)
        b.add_visual_from_file(os.path.join(POT_DIR, "visual.glb"))
        b.set_mass_and_inertia(POT_MASS, sapien.Pose(p=[0, 0, 0.03]), [4e-4, 4e-4, 4e-4])
        b.initial_pose = sapien.Pose(p=mw_to_world([*self.pot_mw[:2], 0.002 + self.pot_z]))
        self.pot = b.build(name="pot")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.agent.robot.set_qpos(torch.tensor(INIT_QPOS, dtype=torch.float32).repeat(b, 1))
            self.agent.robot.set_pose(sapien.Pose([0, 0, 0]))
            self.mw.set_qpos(torch.zeros((b, 1)))
            x, y, yaw = self.pot_mw
            q_w = yaw_quat(yaw - np.pi / 2)                  # yaw in mw -> yaw in world
            xyz = np.tile([x, y, 0.001 + self.pot_z], (b, 1))
            if self.pot_region is not None:
                lo, hi = self.pot_region
                xyz[:, :2] = self._episode_rng.uniform(lo, hi, size=(b, 2))
            self.pot.set_pose(Pose.create_from_pq(
                torch.tensor(mw_to_world(xyz), dtype=torch.float32),
                torch.tensor(q_w, dtype=torch.float32).repeat(b, 1)))

    # ---- task state, in the microwave's frame ---------------------------------------
    def door_angle(self):
        return self.mw.get_qpos()[:, 0]

    def pot_in_mw(self):
        R = torch.tensor(R_W_MW, dtype=torch.float32, device=self.device)
        t = torch.tensor(T_W_MW[:3, 3], dtype=torch.float32, device=self.device)
        return (self.pot.pose.p - t) @ R

    def tcp_in_mw(self):
        R = torch.tensor(R_W_MW, dtype=torch.float32, device=self.device)
        t = torch.tensor(T_W_MW[:3, 3], dtype=torch.float32, device=self.device)
        return (self.agent.tcp.pose.p - t) @ R

    def evaluate(self):
        p = self.pot_in_mw()
        (x0, x1), (y0, y1) = self.params["cavity_x_range"], self.params["cavity_y_range"]
        inside = (p[:, 0] > x0 + 0.03) & (p[:, 0] < x1 - 0.03) & (p[:, 1] > y0 + 0.03) \
            & (p[:, 1] < y1) & (torch.abs(p[:, 2] - self.params["turntable_top_z"]) < 0.02)
        shut = self.door_angle() < np.deg2rad(5)
        return dict(success=inside & shut, pot_inside=inside, door_shut=shut,
                    door_deg=torch.rad2deg(self.door_angle()))

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose, pot_pose=self.pot.pose.raw_pose,
                    door_qpos=self.mw.get_qpos())

    def compute_dense_reward(self, obs, action, info):
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
