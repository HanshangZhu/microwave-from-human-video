"""MicrowavePotG1-v0: MicrowavePot-v0 with ManiSkill's Unitree G1 upper body instead of the Panda.

Feasibility spike (sim/g1_spike/). Same microwave, same pot, same counter (the ground plane,
z = 0), same task frames (sim/microwave_env.py: "mw" = the microwave's frame, the robot's
root at the world origin facing +x_world = +y_mw), same success test.

The G1 is ManiSkill's `unitree_g1_simplified_upper_body`: a fixed root (pelvis), a torso yaw
joint, two 5-DoF arms (shoulder pitch / roll / yaw, elbow pitch, elbow roll -- NO wrist) and
three-finger hands. Its legs have no collision shapes; with the counter as the ground plane
they are simply below the counter top (the robot stands at the counter's edge).

Placement: the pelvis at ROBOT_IN_MW (x, y) in the microwave's frame, height ROOT_Z above
the counter top, yawed by ROOT_YAW about z (0 = facing into the microwave, +y_mw).
Set them with set_g1_stance() before gym.make.
"""
import os
import sys

import numpy as np
import sapien
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import sim.microwave_env as M  # noqa: E402
from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBody  # noqa: E402,F401
from mani_skill.envs.sapien_env import BaseEnv  # noqa: E402
from mani_skill.sensors.camera import CameraConfig  # noqa: E402
from mani_skill.utils import sapien_utils  # noqa: E402
from mani_skill.utils.registration import register_env  # noqa: E402
from mani_skill.utils.structs.pose import Pose  # noqa: E402
from mani_skill.utils.structs.types import SceneConfig, SimConfig  # noqa: E402

# defaults, overwritten by set_g1_stance (chosen by sim/g1_spike/reach.py)
STANCE = dict(xy=(0.30, -0.40), z=0.0, yaw=0.0)

# arm joints (active indices of the 25 G1 joints)
TORSO = 0
LEFT_ARM = [1, 3, 5, 7, 9]
RIGHT_ARM = [2, 4, 6, 8, 10]
LEFT_FINGERS = [11, 12, 13, 17, 18, 19, 23]
RIGHT_FINGERS = [14, 15, 16, 20, 21, 22, 24]
# left: upper arm a little forward and out, forearm turned outwards (hand beside the body, left
# of the hinge); right: upper arm out sideways, forearm raised (hand up beside the head, clear
# of the pot and the microwave)
INIT_QPOS = np.zeros(25)
INIT_QPOS[[1, 2]] = -0.6, 0.0  # shoulder pitch
INIT_QPOS[[3, 4]] = 0.3, -1.3  # shoulder roll
INIT_QPOS[[5, 6]] = 1.0, -1.5  # shoulder yaw
INIT_QPOS[7] = 0.2             # left elbow


# video camera (eye, target) in the microwave frame: in front, to the right, looking at the open
# door's side of the cavity and at the robot's front
RENDER_CAM = ([1.00, -0.65, 0.95], [0.20, -0.10, 0.05])


def set_g1_stance(xy, z, yaw=0.0):
    """Where the pelvis stands in the microwave frame (call before gym.make)."""
    STANCE.update(xy=tuple(xy), z=float(z), yaw=float(yaw))
    M.set_robot_in_mw(xy)


def root_pose():
    """The G1's root (pelvis) in the world: at the origin (M's convention), z up, yawed."""
    y = STANCE["yaw"]
    return sapien.Pose(p=[0, 0, STANCE["z"]], q=[np.cos(y / 2), 0, 0, np.sin(y / 2)])


@register_env("MicrowavePotG1-v0", max_episode_steps=4000)
class MicrowavePotG1Env(M.MicrowavePotEnv):
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body"]

    def __init__(self, *args, robot_uids="unitree_g1_simplified_upper_body", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # as UnitreeG1PlaceAppleInBowl: the dexterous fingers make many near contacts
        return SimConfig(scene_config=SceneConfig(contact_offset=0.01))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.3, 0.0, 0.75], target=[0.4, 0.0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        eye, c = M.mw_to_world(RENDER_CAM[0]), M.mw_to_world(RENDER_CAM[1])
        pose = sapien_utils.look_at(eye=eye.tolist(), target=c.tolist())
        return CameraConfig("render_camera", pose, 960, 640, np.deg2rad(50), 0.01, 100)

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(self, options, root_pose())

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # The robot stands at the counter's edge, but the counter here is an infinite plane: with
        # the shoulders 17 cm above the top, the torso and the upper arms (which in reality hang
        # in front of the edge) would be inside it. They are excluded from collisions with the
        # counter (group-2 bit 4: shapes sharing a group-2 bit do not collide); the forearms,
        # hands and fingers still collide with it.
        self.ground.set_collision_group_bit(2, 4, 1)
        for name in ["torso_link"] + [f"{s}_{l}" for s in ("left", "right")
                                      for l in ("shoulder_roll_link", "shoulder_yaw_link", "elbow_pitch_link")]:
            self.agent.robot.links_map[name].set_collision_group_bit(2, 4, 1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.agent.robot.set_qpos(torch.tensor(INIT_QPOS, dtype=torch.float32).repeat(b, 1))
            self.agent.robot.set_pose(root_pose())
            self.mw.set_qpos(torch.zeros((b, 1)))
            x, y, yaw = self.pot_mw
            q_w = M.yaw_quat(yaw - np.pi / 2)
            xyz = np.tile([x, y, 0.001], (b, 1))
            if self.pot_region is not None:
                lo, hi = self.pot_region
                xyz[:, :2] = self._episode_rng.uniform(lo, hi, size=(b, 2))
            self.pot.set_pose(Pose.create_from_pq(
                torch.tensor(M.mw_to_world(xyz), dtype=torch.float32),
                torch.tensor(q_w, dtype=torch.float32).repeat(b, 1)))

    def tcp_in_mw(self):
        raise NotImplementedError("two hands: see left_tcp / right_tcp")

    def _get_obs_extra(self, info: dict):
        return dict(left_tcp_pose=self.agent.left_tcp.pose.raw_pose,
                    right_tcp_pose=self.agent.right_tcp.pose.raw_pose,
                    pot_pose=self.pot.pose.raw_pose, door_qpos=self.mw.get_qpos())
